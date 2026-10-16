r"""
tests/test_rare_class_warning.py -- Step 9's rare-class warning.

NEW FILE, and the function it tests is new and additive: it reads the weight
table Step 9 already builds, writes nothing, and prints only when a class has
fewer than `config.RARE_CLASS_VALIDATION_MINIMUM` validation rows.

Why the warning exists: early stopping monitors the validation macro-F1, so a
class with one or two validation rows contributes a per-class F1 estimated from
one or two flows -- and the Step 13 fusion later weights a model by that same
per-class reliability. Nothing can recover information the split does not have;
the honest response is to say so beside every number that depends on it.
"""

import os

import pandas as pd
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import config
from src.mscad import imbalance as im


def weight_table(**validation_rows):
    """The shape Step 9's weight_table() produces, including its TOTAL row."""
    rows = []
    for i, (label, validation) in enumerate(validation_rows.items()):
        rows.append({"Label": label, "encoded": i,
                     "train rows": validation * 10,
                     "fit rows": validation * 9, "val rows": validation})
    rows.append({"Label": "TOTAL", "encoded": "",
                 "train rows": sum(r["train rows"] for r in rows),
                 "fit rows": sum(r["fit rows"] for r in rows),
                 "val rows": sum(r["val rows"] for r in rows)})
    return pd.DataFrame(rows)


def test_the_warning_triggers_for_a_class_below_the_minimum():
    table = weight_table(Brute_Force=163, Normal=52, Port_Scan=21,
                         HTTP_DDoS=2, ICMP_Flood=1, Web_Crwling=1)

    out = im.rare_class_report(table, verbose=False)

    assert out["ok"] is False
    assert out["rare"] == ["HTTP_DDoS", "ICMP_Flood", "Web_Crwling"]
    assert out["minimum"] == config.RARE_CLASS_VALIDATION_MINIMUM == 5
    warned = out["table"][out["table"]["result"] == "[WARN]"]
    assert len(warned) == 3
    for _i, row in warned.iterrows():
        assert "fewer than 5 validation rows" in row["note"]
        assert "estimated from %d flow(s)" % row["validation rows"] in row["note"]


def test_no_warning_when_every_class_has_enough_validation_rows():
    table = weight_table(Brute_Force=163, Normal=52, Port_Scan=21, HTTP_DDoS=9)

    out = im.rare_class_report(table, verbose=False)

    assert out["ok"] is True
    assert out["rare"] == []
    assert out["note"] == ""
    assert set(out["table"]["result"]) == {"[OK]"}
    assert set(out["table"]["note"]) == {""}


def test_the_boundary_is_at_the_minimum_itself():
    """Exactly `minimum` rows is enough; one fewer is not."""
    table = weight_table(A=5, B=4)

    out = im.rare_class_report(table, verbose=False)

    by_label = dict(zip(out["table"]["Label"], out["table"]["result"]))
    assert by_label["A"] == "[OK]"
    assert by_label["B"] == "[WARN]"
    assert out["rare"] == ["B"]


def test_the_threshold_can_be_overridden_for_a_what_if():
    table = weight_table(A=10, B=7)

    assert im.rare_class_report(table, minimum=5, verbose=False)["rare"] == []
    assert im.rare_class_report(table, minimum=8, verbose=False)["rare"] == ["B"]


def test_the_note_names_every_rare_class_and_what_depends_on_it():
    table = weight_table(Brute_Force=100, ICMP_Flood=1, Web_Crwling=2)

    note = im.rare_class_report(table, verbose=False)["note"]

    assert "ICMP_Flood" in note and "Web_Crwling" in note
    assert "Brute_Force" not in note
    assert "ICMP_Flood: 1" in note and "Web_Crwling: 2" in note
    assert "early stopping" in note.lower()
    assert "Step 13 fusion" in note
    assert "indicative" in note


def test_the_total_row_is_not_treated_as_a_class():
    table = weight_table(A=10, B=10)

    out = im.rare_class_report(table, verbose=False)

    assert "TOTAL" not in list(out["table"]["Label"])
    assert len(out["table"]) == 2


def test_the_warning_prints_only_when_there_is_something_to_warn_about(capsys):
    im.rare_class_report(weight_table(A=10, B=10), verbose=True)
    quiet = capsys.readouterr().out
    im.rare_class_report(weight_table(A=10, B=1), verbose=True)
    loud = capsys.readouterr().out

    assert "[WARN]" not in quiet.splitlines()[-1]
    assert "every class has at least 5 validation rows" in quiet
    assert "[WARN]" in loud
    assert "1 class(es) below 5 validation rows: B" in loud


def test_step_9s_own_two_row_check_is_untouched():
    """The existing `at least 2 validation rows per class` check still exists;
    this warning is a second, stricter one, not a replacement."""
    source = (im.__file__)
    text = open(source, encoding="utf-8").read()

    assert "at least 2 validation rows per class" in text
    assert "rare_class_report" in text


def test_run_step9_returns_the_rare_class_block():
    """The key run_step9 puts it under, so the notebook and Part 1 summary can
    read it without recomputing anything."""
    import inspect

    source = inspect.getsource(im.run_step9)

    assert "rare_class_report(table" in source
    assert '"rare_classes": rare' in source
