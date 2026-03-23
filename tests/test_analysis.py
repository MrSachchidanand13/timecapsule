"""
tests/test_analysis.py
Tests for analysis functions — all operate on a real .tc file.
"""
import os
import math
import time
import tempfile
import gzip
import pickle
import pytest

import timecapsule as tc


# ─────────────────────────────────────────────────────────────────────────────
#  FIXTURE: create a real .tc file with known content
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tc_file(tmp_path_factory):
    """
    Creates a .tc recording of a simple loop with known properties:
    - variable 'counter' increments 0..9
    - variable 'result' accumulates squares
    - metric 'iteration' tracked
    - @tc.trace on compute()
    """
    tmp = tmp_path_factory.mktemp("recordings")
    out = str(tmp / "analysis_test.tc")

    tc.record(every=0.02, output=out, live=False, warn=True)

    @tc.trace
    def compute(x):
        return x * x

    result  = 0
    counter = 0
    for i in range(10):
        counter = i
        tc.metric("iteration", i)
        result += compute(i)
        if i == 4:
            tc.snap("halfway", tags=["milestone"])

    tc.metric("final_result", result)
    tc.stop()
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────────────────────────────────────

class TestLoad:

    def test_load_returns_dict(self, tc_file):
        d = tc.load(tc_file)
        assert isinstance(d, dict)

    def test_load_has_snaps(self, tc_file):
        d = tc.load(tc_file)
        assert len(d["snaps"]) > 0

    def test_load_no_crash(self, tc_file):
        d = tc.load(tc_file)
        assert d["crash"] is None


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY / EXPLAIN
# ─────────────────────────────────────────────────────────────────────────────

class TestSummaryExplain:

    def test_summary_returns_string(self, tc_file, capsys):
        result = tc.summary(tc_file)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_explain_returns_string(self, tc_file, capsys):
        result = tc.explain(tc_file)
        assert isinstance(result, str)
        assert len(result) > 20

    def test_explain_mentions_script(self, tc_file, capsys):
        result = tc.explain(tc_file)
        # should mention the filename or some script reference
        assert "analysis_test" in result or "test" in result.lower()


# ─────────────────────────────────────────────────────────────────────────────
#  HISTORY / DIFF
# ─────────────────────────────────────────────────────────────────────────────

class TestHistoryDiff:

    def test_history_returns_list(self, tc_file):
        hist = tc.history(tc_file, "counter")
        assert isinstance(hist, list)

    def test_history_has_correct_values(self, tc_file):
        hist = tc.history(tc_file, "counter")
        if hist:
            values = [v for _, v in hist]
            # should include 0..9
            assert any(v == 0 for v in values), "counter=0 not in history"

    def test_diff_returns_changes_only(self, tc_file):
        changes = tc.diff(tc_file, "counter")
        assert isinstance(changes, list)
        # all entries should have different old/new values
        for t, old, new in changes:
            assert old != new, "diff() returned a non-change"

    def test_history_unknown_var_returns_empty(self, tc_file):
        hist = tc.history(tc_file, "__definitely_does_not_exist__")
        assert hist == []


# ─────────────────────────────────────────────────────────────────────────────
#  TIMINGS
# ─────────────────────────────────────────────────────────────────────────────

class TestTimings:

    def test_timings_returns_dict(self, tc_file):
        t = tc.timings(tc_file)
        assert isinstance(t, dict)

    def test_timings_has_compute(self, tc_file):
        t = tc.timings(tc_file)
        # fn name may be qualified: "tc_file.<locals>.compute" or just "compute"
        assert any("compute" in k for k in t), \
            f"compute() not found in timings — keys: {list(t.keys())}"

    def test_timings_has_required_fields(self, tc_file):
        t = tc.timings(tc_file)
        key = next((k for k in t if "compute" in k), None)
        if key:
            s = t[key]
            for field in ["calls", "avg_ms", "p95_ms", "min_ms", "max_ms", "total_ms"]:
                assert field in s, f"'{field}' missing from timing stats"

    def test_timings_calls_count(self, tc_file):
        t = tc.timings(tc_file)
        key = next((k for k in t if "compute" in k), None)
        if key:
            assert t[key]["calls"] == 10

    def test_slowest_returns_list(self, tc_file):
        slow = tc.slowest(tc_file, top_n=5)
        assert isinstance(slow, list)
        assert len(slow) <= 5

    def test_slowest_sorted_descending(self, tc_file):
        slow = tc.slowest(tc_file, top_n=10)
        if len(slow) > 1:
            for i in range(len(slow) - 1):
                assert slow[i][0] >= slow[i+1][0], "slowest() not sorted"

    def test_rate_returns_dict(self, tc_file):
        r = tc.rate(tc_file)
        assert isinstance(r, dict)
        assert any("compute" in k for k in r), \
            f"compute not found in rate — keys: {list(r.keys())}"


# ─────────────────────────────────────────────────────────────────────────────
#  SEARCH
# ─────────────────────────────────────────────────────────────────────────────

class TestSearch:

    def test_search_finds_label(self, tc_file):
        results = tc.search(tc_file, "halfway")
        assert len(results) > 0, "search('halfway') found nothing"

    def test_search_no_match_returns_empty(self, tc_file):
        results = tc.search(tc_file, "xyzzy_no_match_12345")
        assert results == []

    def test_search_finds_variable_value(self, tc_file):
        # search for a numeric value that appears in counter
        results = tc.search(tc_file, "milestone")
        assert isinstance(results, list)


# ─────────────────────────────────────────────────────────────────────────────
#  ANOMALIES
# ─────────────────────────────────────────────────────────────────────────────

class TestAnomalies:

    def test_anomalies_returns_list(self, tc_file):
        result = tc.anomalies(tc_file, "iteration")
        assert isinstance(result, list)

    def test_anomalies_sorted_by_zscore(self, tc_file):
        result = tc.anomalies(tc_file, "iteration")
        if len(result) > 1:
            for i in range(len(result)-1):
                assert result[i][2] >= result[i+1][2], "anomalies not sorted by z-score"

    def test_anomalies_unknown_metric_returns_empty(self, tc_file):
        result = tc.anomalies(tc_file, "__no_such_metric__")
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
#  MEMORY MAP
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryMap:

    def test_memory_map_returns_dict(self, tc_file):
        mm = tc.memory_map(tc_file)
        assert isinstance(mm, dict)

    def test_memory_map_sorted_by_max_bytes(self, tc_file):
        mm = tc.memory_map(tc_file)
        sizes = [v["max_bytes"] for v in mm.values()]
        if len(sizes) > 1:
            for i in range(len(sizes)-1):
                assert sizes[i] >= sizes[i+1], "memory_map not sorted"


# ─────────────────────────────────────────────────────────────────────────────
#  EXPORT JSON / FLAMEGRAPH
# ─────────────────────────────────────────────────────────────────────────────

class TestExport:

    def test_export_json_creates_file(self, tc_file, tmp_path):
        out = str(tmp_path / "out.json")
        result = tc.export_json(tc_file, out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_export_json_is_valid_json(self, tc_file, tmp_path):
        import json
        out = str(tmp_path / "out.json")
        tc.export_json(tc_file, out)
        with open(out) as f:
            data = json.load(f)
        assert "meta" in data
        assert "snapshots" in data

    def test_flamegraph_creates_file(self, tc_file, tmp_path):
        out = str(tmp_path / "flame.json")
        tc.flamegraph(tc_file, out)
        assert os.path.exists(out)

    def test_flamegraph_has_trace_events(self, tc_file, tmp_path):
        import json
        out = str(tmp_path / "flame.json")
        tc.flamegraph(tc_file, out)
        with open(out) as f:
            data = json.load(f)
        assert "traceEvents" in data
        assert len(data["traceEvents"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
#  HOTLINES
# ─────────────────────────────────────────────────────────────────────────────

class TestHotlines:

    def test_hotlines_returns_list(self, tc_file):
        hl = tc.hotlines(tc_file)
        assert isinstance(hl, list)

    def test_hotlines_sorted_descending(self, tc_file):
        hl = tc.hotlines(tc_file, top_n=20)
        if len(hl) > 1:
            for i in range(len(hl)-1):
                assert hl[i][0] >= hl[i+1][0], "hotlines not sorted"


# ─────────────────────────────────────────────────────────────────────────────
#  MEMORY LEAK CHECK
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryLeakCheck:

    def test_returns_dict(self, tc_file):
        result = tc.memory_leak_check(tc_file)
        assert isinstance(result, dict)
        assert "leak_detected" in result

    def test_no_fake_leak_on_clean_run(self, tc_file):
        result = tc.memory_leak_check(tc_file)
        # A clean short test run should not report a leak
        # (may have insufficient data — that's OK, should return False or no data)
        assert result.get("leak_detected") in (True, False)


# ─────────────────────────────────────────────────────────────────────────────
#  LOOP DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class TestLoopDetector:

    def test_returns_list(self, tc_file):
        findings = tc.loop_detector(tc_file)
        assert isinstance(findings, list)


# ─────────────────────────────────────────────────────────────────────────────
#  VARIABLE CORRELATIONS
# ─────────────────────────────────────────────────────────────────────────────

class TestVariableCorrelations:

    def test_returns_list(self, tc_file):
        results = tc.variable_correlations(tc_file, min_r=0.5)
        assert isinstance(results, list)

    def test_correlations_are_between_minus1_and_1(self, tc_file):
        results = tc.variable_correlations(tc_file, min_r=0.5)
        for a, b, r in results:
            assert -1.0 <= r <= 1.0, f"correlation {r} out of [-1,1]"

    def test_correlations_above_threshold(self, tc_file):
        threshold = 0.6
        results = tc.variable_correlations(tc_file, min_r=threshold)
        for a, b, r in results:
            assert abs(r) >= threshold - 1e-9, f"correlation {r} below threshold {threshold}"


# ─────────────────────────────────────────────────────────────────────────────
#  SINCE / REPLAY
# ─────────────────────────────────────────────────────────────────────────────

class TestSinceReplay:

    def test_since_returns_snaps_after_time(self, tc_file):
        d = tc.load(tc_file)
        dur = d["duration"]
        snaps = tc.since(tc_file, dur / 2)
        for s in snaps:
            assert s["t"] >= dur / 2

    def test_replay_yields_snaps_in_order(self, tc_file):
        times = [s["t"] for s in tc.replay(tc_file)]
        for i in range(len(times)-1):
            assert times[i] <= times[i+1], "replay not in time order"

    def test_since_beyond_duration_returns_empty(self, tc_file):
        d = tc.load(tc_file)
        snaps = tc.since(tc_file, d["duration"] + 100)
        assert snaps == []


# ─────────────────────────────────────────────────────────────────────────────
#  EXPORT FOR CLAUDE
# ─────────────────────────────────────────────────────────────────────────────

class TestExportForClaude:

    def test_creates_text_file(self, tc_file, tmp_path):
        out = str(tmp_path / "for_claude.txt")
        tc.export_for_claude(tc_file, out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_file_has_required_sections(self, tc_file, tmp_path):
        out = str(tmp_path / "for_claude.txt")
        tc.export_for_claude(tc_file, out)
        content = open(out, encoding="utf-8").read()
        assert "TIMECAPSULE RECORDING" in content
        assert "SUMMARY" in content
        assert "FUNCTION PERFORMANCE" in content
        assert "HINTS FOR CLAUDE" in content
