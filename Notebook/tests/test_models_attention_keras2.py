r"""
tests/test_models_attention_keras2.py -- Approach 2 on Keras 2 as well as 3.

NEW FILE.  Nothing else is modified: this only pins down that
`src/mscad/models_attention.py` never reaches for an API that exists solely in
Keras 3, because Steps 7-13 have models saved by the installed TensorFlow/Keras
and that version must keep loading.

Two ways of asking the same question:

  * statically -- the module's source carries no `keras` numerical-namespace
    import at all;
  * dynamically -- the layer builds, computes, saves and reloads with that
    namespace made unavailable, first on the installed Keras with the
    attribute hidden (runs everywhere), then on a genuine Keras 2 if
    `tf_keras` happens to be installed (skipped otherwise).

Tiny synthetic data, no training, everything inside tmp_path.
Never C:\\MSCAD2022, and no pathlib patching.

    cd /d C:\\Users\\banot\\MSCAD22-Multi-Stage-Cyber-Attack-Detection\\Notebook
    python -m pytest tests -v -W ignore::DeprecationWarning
"""

import io
import os
import re

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import models as md
from src.mscad import models_attention as ma

N_LATENT = 8
N_CLASSES = 3

# the names that only Keras 3 answers to
KERAS3_ONLY = (
    re.compile(r"\bfrom\s+keras\s+import\s+(?:[\w\s,]*\b)?ops\b"),
    re.compile(r"\bkeras\s*\.\s*ops\b"),
    re.compile(r"\bimport\s+keras\.ops\b"),
)


def _source() -> str:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "src", "mscad", "models_attention.py")
    return io.open(path, encoding="utf-8").read()


# ===========================================================================
# statically
# ===========================================================================
def test_the_module_never_imports_the_keras_3_only_ops_namespace():
    """A grep, as a test, so the regression cannot come back quietly."""
    source = _source()

    offenders = [pattern.pattern for pattern in KERAS3_ONLY
                 if pattern.search(source)]
    assert offenders == [], (
        "models_attention.py still refers to Keras 3's ops namespace: %s"
        % offenders)


def test_the_module_imports_without_keras_and_the_layer_is_lazy():
    """Importing the module must not build or import anything Keras-version
    specific -- the class is built on first use, which is what lets the
    version question be answered at call time."""
    assert "models_attention" in str(ma.__name__)
    assert callable(ma.register_attention)
    assert callable(ma.custom_objects)


# ===========================================================================
# dynamically -- the installed Keras with `ops` hidden
# ===========================================================================
class _KerasWithoutOps:
    """The installed Keras, with its Keras-3-only `ops` namespace removed.

    A proxy rather than a `delattr`: patching the real module would leave the
    rest of the process (Keras's own internals included) without it.
    """

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        if name == "ops":
            raise AttributeError(
                "module 'keras' has no attribute 'ops' (simulated Keras 2)")
        return getattr(self._inner, name)


@pytest.fixture
def fresh_attention_class(monkeypatch):
    """Let a test build the layer class against another Keras.

    The class is cached in a module global, so it is reset before the test and
    -- monkeypatch's doing -- restored afterwards, leaving the rest of the
    suite with the class it already had.
    """
    monkeypatch.setattr(ma, "_ATTENTION_CLASS", None)
    yield
    ma._ATTENTION_CLASS = None


def _exercise(keras_module, tmp_path, tag):
    """Build, compute, save and reload Approach 2 under one Keras module."""
    keras_module.utils.set_random_seed(0)
    model = ma.build_cnn_attention_lstm((N_LATENT, 1), N_CLASSES)

    X = np.random.default_rng(3).uniform(
        0, 1, (6, N_LATENT, 1)).astype("float32")

    # alpha is a distribution over the positions
    alpha = ma.attention_weights(model, X)
    assert alpha.shape == (6, N_LATENT // config.CNN_POOL)
    assert (alpha >= 0).all()
    assert alpha.sum(axis=1) == pytest.approx(np.ones(6), abs=1e-9)

    # and it is the layer's own arithmetic, not a second implementation
    import tensorflow as tf
    sequence = keras_module.Model(model.inputs, ma.attention_layer(model).input)
    theirs = np.asarray(
        ma.attention_layer(model).scores(
            tf.convert_to_tensor(sequence.predict(X, verbose=0))),
        dtype="float64")
    assert alpha == pytest.approx(theirs, abs=1e-6)

    # logits of the documented shape
    logits = model.predict(X, verbose=0)
    assert logits.shape == (6, N_CLASSES)

    # the saved model reloads and predicts identically
    path = tmp_path / ("cnn_attention_lstm_%s.keras" % tag)
    model.save(path)
    reloaded = keras_module.models.load_model(
        path, custom_objects=ma.custom_objects())
    assert reloaded.predict(X, verbose=0) == pytest.approx(logits, abs=1e-5)
    return model


def test_the_layer_builds_and_round_trips_with_no_ops_attribute(
        tmp_path, monkeypatch, fresh_attention_class):
    """The Keras 2 code path, on whatever Keras is installed.

    The `ops` attribute is made to raise AttributeError, so anything that
    reached for it would fail here rather than on the user's machine.
    """
    keras = md._keras()
    shim = _KerasWithoutOps(keras)
    monkeypatch.setattr(md, "_keras", lambda: shim)

    with pytest.raises(AttributeError):
        shim.ops                                  # the simulation is real

    model = _exercise(shim, tmp_path, "noops")
    assert ma.attention_layer(model).name == ma.ATTENTION_LAYER_NAME


def test_the_registered_name_is_found_however_this_keras_spells_it(
        monkeypatch, fresh_attention_class):
    """`register_keras_serializable` lives on `keras.saving` in Keras 3 and on
    `keras.utils` in Keras 2.  Either alone must be enough."""
    keras = md._keras()

    class OnlyUtils(_KerasWithoutOps):
        def __getattr__(self, name):
            if name == "saving":
                raise AttributeError("no keras.saving (simulated Keras 2.12)")
            return super().__getattr__(name)

    shim = OnlyUtils(keras)
    monkeypatch.setattr(md, "_keras", lambda: shim)

    cls = ma.register_attention()
    objects = ma.custom_objects()

    assert set(objects) == {"AdditiveAttention", "mscad>AdditiveAttention"}
    assert objects["AdditiveAttention"] is cls


# ===========================================================================
# dynamically -- a genuine Keras 2, when one is installed
# ===========================================================================
def test_the_layer_builds_and_round_trips_on_a_real_keras_2(
        tmp_path, fresh_attention_class, monkeypatch):
    """`tf_keras` is the Keras 2 API packaged for TensorFlow 2.16+.

    Skipped where it is not installed; where it is, this is the user's failing
    configuration exactly -- a Keras whose `ops` namespace does not exist.
    """
    k2 = pytest.importorskip("tf_keras",
                             reason="no Keras 2 available in this environment")
    assert not hasattr(k2, "ops"), "tf_keras unexpectedly has an ops namespace"

    monkeypatch.setattr(md, "_keras", lambda: k2)

    model = _exercise(k2, tmp_path, "keras2")
    assert type(ma.attention_layer(model)).__name__ == "AdditiveAttention"

    # and the config round trip, on this version too
    layer = ma.attention_layer(model)
    clone = type(layer).from_config(layer.get_config())
    assert clone.get_config()["units"] == layer.get_config()["units"]
