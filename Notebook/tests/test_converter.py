"""
tests/test_converter.py -- pytest tests for src/converter.py.

These tests need NO real captures, NO Wireshark and NO scapy.  Everything is
built from temporary files and small DataFrames, so they run in a second:

    cd /d C:\\MSCAD2022\\Notebook
    "C:\\Users\\banot\\AppData\\Local\\Programs\\Python\\Python313\\python.exe" -m pytest tests -v
"""

from pathlib import Path

import pandas as pd
import pytest

import config
from src import converter


# ===========================================================================
# helpers
# ===========================================================================
def make_capture(tmp_path: Path, rel: str, size: int = 10) -> Path:
    """Create a fake capture file (its CONTENT is irrelevant to these tests)."""
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00" * size)
    return p


def raw_flow_frame(n_rows: int = 3, **overrides) -> pd.DataFrame:
    """A minimal RAW extractor frame: every RENAME_MAP key, in seconds.

    Values are deliberately simple (1.0 everywhere) so a test can assert an
    exact number after conversion.
    """
    data = {snake: [1.0] * n_rows for snake in config.RENAME_MAP}
    # the identifier columns need realistic values
    data["src_ip"] = ["192.168.159.1"] * n_rows
    data["dst_ip"] = ["192.168.159.20"] * n_rows
    data["src_port"] = [12345] * n_rows
    data["dst_port"] = [80] * n_rows
    data["protocol"] = [6] * n_rows
    data["timestamp"] = ["2022-01-01 00:00:00"] * n_rows
    data["tot_fwd_pkts"] = [10] * n_rows
    data.update(overrides)
    return pd.DataFrame(data)


# ===========================================================================
# 1. file discovery
# ===========================================================================
def test_find_captures_is_recursive_and_case_insensitive(tmp_path):
    make_capture(tmp_path, "N-0.pcapng")
    make_capture(tmp_path, "Scan-1.PCAPNG")        # upper-case extension
    make_capture(tmp_path, "sub/App-01.pcap")      # a subfolder, already .pcap
    make_capture(tmp_path, "notes.txt")            # must be ignored
    make_capture(tmp_path, "sub/readme.md")        # must be ignored

    found = converter.find_captures(tmp_path)

    # sorted by relative path, lower-cased: n-0 < scan-1 < sub/app-01
    assert [c.stem for c in found] == ["N-0", "Scan-1", "App-01"]
    assert {c.ext for c in found} == {".pcapng", ".pcap"}
    assert all(isinstance(c.path, Path) for c in found)


def test_find_captures_normalises_extension_case(tmp_path):
    make_capture(tmp_path, "Scan-1.PCAPNG")
    (cap,) = converter.find_captures(tmp_path)
    assert cap.ext == ".pcapng"          # lower-cased
    assert cap.path.suffix == ".PCAPNG"  # the file on disk is untouched
    assert cap.is_already_pcap is False


def test_find_captures_skips_venv_and_returns_empty_for_missing_folder(tmp_path):
    make_capture(tmp_path, ".venv/lib/site-packages/thing.pcap")
    assert converter.find_captures(tmp_path) == []
    assert converter.find_captures(tmp_path / "does_not_exist") == []


def test_capture_metadata(tmp_path):
    make_capture(tmp_path, "sub/N-0.pcapng", size=2 * 1024 * 1024)
    (cap,) = converter.find_captures(tmp_path)
    assert cap.size_mb == 2.0
    assert cap.rel_path in ("sub/N-0.pcapng", "sub\\N-0.pcapng")
    assert cap.expected_label == "Normal"


def test_unknown_stem_gets_no_invented_label(tmp_path):
    make_capture(tmp_path, "mystery.pcapng")
    (cap,) = converter.find_captures(tmp_path)
    assert "UNKNOWN" in cap.expected_label
    assert cap.stem not in config.CAPTURE_LABELS


# ===========================================================================
# 2. output-name generation, including duplicate stems
# ===========================================================================
def test_output_name_is_plain_when_stems_are_unique(tmp_path):
    make_capture(tmp_path, "N-0.pcapng")
    make_capture(tmp_path, "W-B-02.pcapng")
    caps = converter.find_captures(tmp_path)
    names = {c.stem: converter.output_pcap_name(c, caps) for c in caps}
    assert names == {"N-0": "N-0.pcap", "W-B-02": "W-B-02.pcap"}


def test_duplicate_stems_are_prefixed_with_the_subfolder(tmp_path):
    make_capture(tmp_path, "N-0.pcapng")
    make_capture(tmp_path, "extra/N-0.PCAPNG")
    make_capture(tmp_path, "Scan-1.pcapng")
    caps = converter.find_captures(tmp_path)

    dupes = converter.duplicate_stems(caps)
    assert set(dupes) == {"n-0"}

    names = [converter.output_pcap_name(c, caps) for c in caps]
    assert len(set(names)) == len(names), "output names must be unique: %s" % names
    assert "extra_N-0.pcap" in names
    assert "root_N-0.pcap" in names
    assert "Scan-1.pcap" in names        # untouched: its stem is unique


def test_three_way_duplicate_still_yields_unique_names(tmp_path):
    make_capture(tmp_path, "a/N-0.pcapng")
    make_capture(tmp_path, "b/N-0.pcapng")
    make_capture(tmp_path, "c/N-0.pcapng")
    caps = converter.find_captures(tmp_path)
    names = [converter.output_pcap_name(c, caps) for c in caps]
    assert sorted(names) == ["a_N-0.pcap", "b_N-0.pcap", "c_N-0.pcap"]


# ===========================================================================
# 3. external-command construction (list args, spaces survive)
# ===========================================================================
def test_editcap_command_is_a_list_and_keeps_spaces():
    exe = r"C:\Program Files\Wireshark\editcap.exe"
    src = r"C:\MSCAD2022\PCAPNG ORIGINAL\N-0.pcapng"
    dst = r"C:\MSCAD2022\PCAPS\N-0.pcap"

    cmd = converter.build_editcap_cmd(src, dst, editcap=exe)

    assert isinstance(cmd, list)
    assert cmd == [exe, "-F", "pcap", src, dst]
    # each path stays ONE argument even though it contains spaces
    assert cmd[0] == exe and " " in cmd[0]
    assert cmd[3] == src and " " in cmd[3]
    assert all(isinstance(a, str) for a in cmd)


def test_tshark_commands():
    exe = r"C:\Program Files\Wireshark\tshark.exe"
    f = r"C:\MSCAD2022\PCAPS\My Capture.pcap"

    total = converter.build_tshark_count_cmd(f, tshark=exe)
    ipv4 = converter.build_tshark_ip_count_cmd(f, tshark=exe)

    assert total == [exe, "-r", f, "-T", "fields", "-e", "frame.number"]
    assert ipv4 == [exe, "-r", f, "-Y", "ip", "-T", "fields", "-e", "frame.number"]
    assert "-Y" in ipv4 and ipv4[ipv4.index("-Y") + 1] == "ip"


def test_capinfos_command():
    exe = r"C:\Program Files\Wireshark\capinfos.exe"
    cmd = converter.build_capinfos_count_cmd(r"C:\MSCAD2022\PCAPS\N-0.pcap", capinfos=exe)
    assert cmd == [exe, "-c", "-M", r"C:\MSCAD2022\PCAPS\N-0.pcap"]


def test_commands_accept_path_objects():
    cmd = converter.build_editcap_cmd(Path("/tmp/a b/in.pcapng"), Path("/tmp/out.pcap"),
                                      editcap="editcap")
    assert all(isinstance(a, str) for a in cmd)   # subprocess needs plain strings


# ===========================================================================
# 4. RENAME_MAP completeness
# ===========================================================================
def test_rename_map_covers_every_output_column():
    needed = set(config.FLOW_COLUMNS) - {config.LABEL_COLUMN}
    targets = set(config.RENAME_MAP.values())
    assert needed <= targets, "no RENAME_MAP entry for: %s" % sorted(needed - targets)


def test_rename_map_has_no_duplicate_targets():
    values = list(config.RENAME_MAP.values())
    dupes = sorted({v for v in values if values.count(v) > 1})
    assert dupes == [], "two snake_case names map to the same column: %s" % dupes


def test_flow_columns_shape_and_order():
    assert len(config.FLOW_COLUMNS) == 73           # 6 + 66 + 1
    assert config.FLOW_COLUMNS[:6] == config.ID_COLUMNS
    assert config.FLOW_COLUMNS[-1] == config.LABEL_COLUMN
    assert len(config.KAGGLE_FEATURES) == 66
    assert len(set(config.FLOW_COLUMNS)) == 73      # no duplicates


def test_cwe_flag_count_keeps_cicflowmeters_spelling():
    assert config.RENAME_MAP["cwr_flag_count"] == "CWE Flag Count"
    assert "CWE Flag Count" in config.FLOW_COLUMNS
    assert "CWR Flag Count" not in config.FLOW_COLUMNS


def test_the_four_tricky_renames_from_the_spec():
    assert config.RENAME_MAP["tot_fwd_pkts"] == "Tot Fwd Pkts"
    assert config.RENAME_MAP["flow_byts_s"] == "Flow Byts/s"
    assert config.RENAME_MAP["init_bwd_win_byts"] == "Init Bwd Win Byts"
    assert config.RENAME_MAP["fwd_act_data_pkts"] == "Fwd Act Data Pkts"


# ===========================================================================
# 5. column reorder / drop
# ===========================================================================
def test_strip_column_names_removes_whitespace():
    df = pd.DataFrame({" flow_duration ": [1], "tot_fwd_pkts\t": [2]})
    out = converter.strip_column_names(df)
    assert list(out.columns) == ["flow_duration", "tot_fwd_pkts"]


def test_extra_columns_are_dropped_and_listed():
    df = pd.DataFrame({
        "Src IP": ["1.1.1.1"],
        "Flow Duration": [1.0],
        "fwd_psh_flags": [0],        # a real cicflowmeter column MSCAD drops
        "init_fwd_win_byts": [8192],
    })
    out, dropped = converter.split_extra_columns(df)
    assert sorted(dropped) == ["fwd_psh_flags", "init_fwd_win_byts"]
    assert list(out.columns) == ["Src IP", "Flow Duration"]


def test_reorder_puts_columns_in_flow_columns_order():
    shuffled = list(reversed(config.FLOW_COLUMNS))
    df = pd.DataFrame({c: [0] for c in shuffled})
    out = converter.reorder_columns(df)
    assert list(out.columns) == config.FLOW_COLUMNS


def test_postprocess_produces_the_exact_header():
    df, info = converter.postprocess(raw_flow_frame(4), "Scan-1")
    assert list(df.columns) == config.FLOW_COLUMNS
    assert len(df) == 4
    assert info["flows"] == 4
    assert info["dropped_columns"] == []      # the fixture has no extra columns


def test_postprocess_reports_dropped_columns():
    raw = raw_flow_frame(2)
    raw["fwd_seg_size_min"] = 20              # an extra cicflowmeter column
    raw["bwd_blk_rate_avg"] = 0
    df, info = converter.postprocess(raw, "Scan-1")
    assert sorted(info["dropped_columns"]) == ["bwd_blk_rate_avg", "fwd_seg_size_min"]
    assert list(df.columns) == config.FLOW_COLUMNS


# ===========================================================================
# 6. missing-column error -- never fabricate a value
# ===========================================================================
def test_missing_column_raises_and_names_the_column():
    raw = raw_flow_frame(2).drop(columns=["idle_max"])
    with pytest.raises(converter.MissingColumnError) as err:
        converter.postprocess(raw, "Scan-1")
    assert "Idle Max" in str(err.value)
    assert "RENAME_MAP" in str(err.value)


def test_check_required_columns_lists_every_missing_column():
    df = pd.DataFrame({"Src IP": ["1.1.1.1"]})
    with pytest.raises(converter.MissingColumnError) as err:
        converter.check_required_columns(df)
    msg = str(err.value)
    assert "Flow Duration" in msg and "Dst IP" in msg


def test_unknown_capture_cannot_be_labelled():
    df = pd.DataFrame({"Src IP": ["1.1.1.1"], "Dst IP": ["2.2.2.2"]})
    with pytest.raises(converter.ConverterError):
        converter.label_flows(df, "not-a-real-capture", attackers=[])


# ===========================================================================
# 7. attacker-threshold logic
# ===========================================================================
def test_attackers_are_the_senders_at_or_above_half_the_top():
    df = pd.DataFrame({
        "Src IP": ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"],
        "Tot Fwd Pkts": [1000, 600, 500, 499],   # top = 1000, cutoff = 500
    })
    attackers, totals = converter.detect_attackers(df, threshold=0.5)
    assert attackers == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]   # 499 is below
    assert list(totals.index[:2]) == ["10.0.0.1", "10.0.0.2"]  # sorted descending
    assert totals.iloc[0] == 1000


def test_attacker_packets_are_summed_per_ip():
    df = pd.DataFrame({
        "Src IP": ["10.0.0.1", "10.0.0.1", "10.0.0.2"],
        "Tot Fwd Pkts": [300, 300, 500],         # .1 totals 600, so .1 is the top
    })
    attackers, totals = converter.detect_attackers(df, threshold=0.5)
    assert totals["10.0.0.1"] == 600
    assert attackers == ["10.0.0.1", "10.0.0.2"]


def test_threshold_is_configurable_and_default_is_half():
    df = pd.DataFrame({"Src IP": ["a", "b"], "Tot Fwd Pkts": [100, 40]})
    assert config.ATTACKER_THRESHOLD == 0.5
    assert converter.detect_attackers(df)[0] == ["a"]                  # 40 < 50
    assert converter.detect_attackers(df, threshold=0.3)[0] == ["a", "b"]


def test_detect_attackers_on_empty_frame():
    df = pd.DataFrame({"Src IP": [], "Tot Fwd Pkts": []})
    attackers, totals = converter.detect_attackers(df)
    assert attackers == [] and len(totals) == 0


# ===========================================================================
# 8. Benign labelling
# ===========================================================================
def test_non_attacker_flows_are_benign():
    df = pd.DataFrame({
        "Src IP": ["192.168.159.1", "192.168.159.50", "192.168.159.60"],
        "Dst IP": ["192.168.159.20", "192.168.159.1", "192.168.159.70"],
    })
    out = converter.label_flows(df, "Scan-1", attackers=["192.168.159.1"])
    assert list(out[config.LABEL_COLUMN]) == ["Port_Scan", "Port_Scan", "Benign"]


def test_attacker_as_destination_also_counts():
    df = pd.DataFrame({"Src IP": ["10.0.0.9"], "Dst IP": ["192.168.159.10"]})
    out = converter.label_flows(df, "App-01", attackers=["192.168.159.10"])
    assert out[config.LABEL_COLUMN].iloc[0] == "DoS slowloris"


def test_normal_capture_labels_every_flow_normal():
    df = pd.DataFrame({"Src IP": ["a", "b"], "Dst IP": ["c", "d"]})
    out = converter.label_flows(df, "N-0", attackers=[])
    assert set(out[config.LABEL_COLUMN]) == {"Normal"}


def test_label_spellings_are_the_official_mscad_ones():
    assert config.CAPTURE_LABELS == {
        "N-0": "Normal",
        "Scan-1": "Port_Scan",
        "App-01": "DoS slowloris",
        "App-02": "ICMP_Flood",
        "W-B-01": "Web_Crwling",     # the dataset's own spelling of "Crawling"
        "W-B-02": "Brute_Force",
    }
    assert config.BENIGN_LABEL == "Benign"


def test_postprocess_labels_icmp_flood_capture_end_to_end():
    raw = raw_flow_frame(3)
    raw["src_ip"] = ["192.168.159.14", "192.168.159.14", "192.168.159.99"]
    raw["dst_ip"] = ["192.168.159.20", "192.168.159.21", "192.168.159.22"]
    raw["tot_fwd_pkts"] = [5000, 5000, 1]     # .99 is far below half the top
    df, info = converter.postprocess(raw, "App-02")
    assert info["attackers"] == ["192.168.159.14"]
    assert df[config.LABEL_COLUMN].tolist() == ["ICMP_Flood", "ICMP_Flood", "Benign"]
    assert info["label_counts"] == {"ICMP_Flood": 2, "Benign": 1}


# ===========================================================================
# 9. seconds -> microseconds
# ===========================================================================
def test_to_microseconds_scales_only_the_time_columns():
    df = pd.DataFrame({
        "Flow Duration": [1.5],        # seconds
        "Flow IAT Mean": [0.25],
        "Idle Max": [2.0],
        "Tot Fwd Pkts": [10],          # NOT a time column
        "Flow Byts/s": [1000.0],       # NOT a time column
    })
    out = converter.to_microseconds(df)
    assert out["Flow Duration"].iloc[0] == 1_500_000
    assert out["Flow IAT Mean"].iloc[0] == 250_000
    assert out["Idle Max"].iloc[0] == 2_000_000
    assert out["Tot Fwd Pkts"].iloc[0] == 10
    assert out["Flow Byts/s"].iloc[0] == 1000.0


def test_all_23_time_columns_are_converted():
    assert len(config.MICROSECOND_COLUMNS) == 23
    df = pd.DataFrame({c: [1.0] for c in config.MICROSECOND_COLUMNS})
    out = converter.to_microseconds(df)
    for c in config.MICROSECOND_COLUMNS:
        assert out[c].iloc[0] == 1_000_000, c


def test_to_microseconds_does_not_mutate_the_input():
    df = pd.DataFrame({"Flow Duration": [1.0]})
    converter.to_microseconds(df)
    assert df["Flow Duration"].iloc[0] == 1.0       # original untouched


def test_every_microsecond_column_is_a_real_output_column():
    assert set(config.MICROSECOND_COLUMNS) <= set(config.FLOW_COLUMNS)


def test_postprocess_converts_time_but_not_counts():
    df, _info = converter.postprocess(raw_flow_frame(1), "Scan-1")
    assert df["Flow Duration"].iloc[0] == 1_000_000     # 1.0 s -> 1e6 us
    assert df["Fwd IAT Tot"].iloc[0] == 1_000_000
    assert df["Tot Fwd Pkts"].iloc[0] == 10            # a count, unscaled


# ===========================================================================
# 10. ICMP flows: ports 0, protocol 1
# ===========================================================================
def test_icmp_flow_key_has_zero_ports_and_protocol_1():
    rec = converter.PacketRecord(
        time=1.0, cap_len=98, src_ip="192.168.159.14", dst_ip="192.168.159.20",
        src_port=0, dst_port=0, protocol=converter.PROTO_ICMP)
    key = converter.flow_key(rec)
    assert key == ("192.168.159.14", "192.168.159.20", 0, 0, 1)
    assert key[2] == 0 and key[3] == 0        # ports
    assert key[4] == 1                        # protocol


def test_icmp_ports_are_forced_to_zero_even_if_set():
    rec = converter.PacketRecord(
        time=1.0, cap_len=98, src_ip="a", dst_ip="b",
        src_port=9999, dst_port=8888, protocol=converter.PROTO_ICMP)
    assert converter.flow_key(rec)[2:4] == (0, 0)


def test_tcp_and_udp_keep_their_ports():
    tcp = converter.PacketRecord(time=1.0, cap_len=60, src_ip="a", dst_ip="b",
                                 src_port=12345, dst_port=80, protocol=converter.PROTO_TCP)
    udp = converter.PacketRecord(time=1.0, cap_len=60, src_ip="a", dst_ip="b",
                                 src_port=5353, dst_port=53, protocol=converter.PROTO_UDP)
    assert converter.flow_key(tcp) == ("a", "b", 12345, 80, 6)
    assert converter.flow_key(udp) == ("a", "b", 5353, 53, 17)


def test_other_ip_protocols_also_get_zero_ports():
    rec = converter.PacketRecord(time=1.0, cap_len=60, src_ip="a", dst_ip="b",
                                 src_port=1, dst_port=2, protocol=47)   # GRE
    assert converter.flow_key(rec) == ("a", "b", 0, 0, 47)


def test_reverse_flow_key_swaps_both_ends():
    rec = converter.PacketRecord(time=1.0, cap_len=60, src_ip="a", dst_ip="b",
                                 src_port=1234, dst_port=80, protocol=6)
    assert converter.reverse_flow_key(rec) == ("b", "a", 80, 1234, 6)
    assert converter.reverse_flow_key(rec) != converter.flow_key(rec)


def test_tcp_flags_are_read_from_the_tcp_flag_byte():
    """Fix #4: stock cicflowmeter read the IP flags field instead of the TCP one."""
    fin_ack = converter.PacketRecord(time=1.0, cap_len=60, src_ip="a", dst_ip="b",
                                     src_port=1, dst_port=2, protocol=6, flags=0x11)
    assert fin_ack.fin is True and fin_ack.ack is True
    assert fin_ack.syn is False and fin_ack.rst is False
    syn = converter.PacketRecord(time=1.0, cap_len=60, src_ip="a", dst_ip="b",
                                 src_port=1, dst_port=2, protocol=6, flags=0x02)
    assert syn.syn is True and syn.fin is False
    # a non-TCP record has no flags at all
    icmp = converter.PacketRecord(time=1.0, cap_len=98, src_ip="a", dst_ip="b",
                                  src_port=0, dst_port=0, protocol=1)
    assert not any([icmp.fin, icmp.syn, icmp.rst, icmp.psh, icmp.ack, icmp.urg])


# ===========================================================================
# extras: config sanity and the Stage C guard
# ===========================================================================
def test_config_settings_match_the_spec():
    assert config.FLOW_TIMEOUT_S == 120
    assert config.IDLE_TIMEOUT_S == 240
    assert config.RANDOM_SEED == 42
    assert config.PROGRESS_EVERY == 100_000
    assert config.OVERWRITE is False
    assert config.ATTACKER_THRESHOLD == 0.5


def test_read_only_folders_are_never_in_the_create_list():
    for ro in config.READ_ONLY_DIRS:
        assert ro not in config.DIRS_TO_CREATE


def test_extract_flows_says_clearly_that_it_is_stage_c():
    with pytest.raises(NotImplementedError) as err:
        converter.extract_flows("anything.pcap")
    assert "Stage C" in str(err.value)


def test_postprocess_keeps_inf_and_nan_untouched():
    """Cleaning is a later step: inf and NaN must survive Step 1 unchanged."""
    import numpy as np
    raw = raw_flow_frame(2)
    raw.loc[0, "flow_byts_s"] = np.inf          # zero-duration flow
    raw.loc[1, "flow_pkts_s"] = np.nan
    df, _info = converter.postprocess(raw, "Scan-1")
    assert np.isinf(df["Flow Byts/s"].iloc[0])
    assert pd.isna(df["Flow Pkts/s"].iloc[1])
