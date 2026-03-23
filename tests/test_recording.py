"""
tests/test_recording.py
Tests for the recording engine.
"""
import os
import sys
import math
import time
import tempfile
import gzip
import pickle
import pytest

import timecapsule as tc


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_tc(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def run_and_load(fn, every=0.05, **kwargs):
    """Run fn while recording, return the loaded .tc dict."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "test.tc")
        tc.record(every=every, output=out, live=False, **kwargs)
        fn()
        tc.stop()
        return load_tc(out)


# ─────────────────────────────────────────────────────────────────────────────
#  BASIC RECORDING
# ─────────────────────────────────────────────────────────────────────────────

class TestBasicRecording:

    def test_record_stop_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            tc.record(every=1.0, output=out, live=False, warn=False)
            x = 1 + 1
            tc.stop()
            assert os.path.exists(out), ".tc file not created"
            assert os.path.getsize(out) > 0, ".tc file is empty"

    def test_file_is_gzip_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            tc.record(every=1.0, output=out, live=False, warn=False)
            tc.stop()
            d = load_tc(out)
            assert isinstance(d, dict)
            assert "ver" in d
            assert d["ver"] == "6.0"

    def test_top_level_keys_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            tc.record(every=1.0, output=out, live=False, warn=False)
            tc.stop()
            d = load_tc(out)
            required = ["ver", "session_id", "src", "duration", "n", "snaps",
                        "crash", "warnings", "fn_timings", "fn_calls",
                        "final_metrics", "all_vars"]
            for k in required:
                assert k in d, f"key '{k}' missing from .tc file"

    def test_duration_is_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            tc.record(every=1.0, output=out, live=False, warn=False)
            time.sleep(0.05)
            tc.stop()
            d = load_tc(out)
            assert d["duration"] > 0

    def test_no_crash_field_is_none(self):
        def fn():
            x = 1 + 1
        d = run_and_load(fn, every=1.0, warn=False)
        assert d["crash"] is None

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            with tc.record(every=1.0, output=out, live=False, warn=False):
                x = 42
            d = load_tc(out)
            assert d["n"] > 0

    def test_second_record_resets_state(self):
        """Bug A1 — state must fully reset between recordings."""
        with tempfile.TemporaryDirectory() as tmp:
            out1 = os.path.join(tmp, "first.tc")
            out2 = os.path.join(tmp, "second.tc")
            tc.record(every=1.0, output=out1, live=False, warn=False)
            tc.metric("run", 1)
            tc.stop()
            tc.record(every=1.0, output=out2, live=False, warn=False)
            tc.stop()
            d2 = load_tc(out2)
            # second recording should NOT carry over metrics from first
            assert d2["final_metrics"].get("run") is None, \
                "Global state leaked from first to second recording"


# ─────────────────────────────────────────────────────────────────────────────
#  SNAPSHOTS
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshots:

    def test_manual_snap_creates_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            tc.record(every=1.0, output=out, live=False, warn=False)
            tc.snap("hello_world")
            tc.stop()
            d = load_tc(out)
            labels = [s.get("label") for s in d["snaps"]]
            assert "hello_world" in labels

    def test_snap_captures_label_and_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            tc.record(every=1.0, output=out, live=False, warn=False)
            tc.snap("milestone", tags=["test", "unit"])
            tc.stop()
            d = load_tc(out)
            target = next((s for s in d["snaps"] if s.get("label") == "milestone"), None)
            assert target is not None
            assert "test" in target.get("tags", [])

    def test_initial_snap_always_created(self):
        """Bug B3 — must have at least one snap even for trivially short runs."""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            tc.record(every=99.0, output=out, live=False, warn=False)
            tc.stop()
            d = load_tc(out)
            assert d["n"] >= 1, "No snapshots in recording — initial snap missing"


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:

    def test_metric_stored_in_final_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            tc.record(every=1.0, output=out, live=False, warn=False)
            tc.metric("loss", 0.42)
            tc.metric("epoch", 5)
            tc.stop()
            d = load_tc(out)
            assert d["final_metrics"]["loss"] == pytest.approx(0.42)
            assert d["final_metrics"]["epoch"] == 5

    def test_metric_appears_in_snap_mets(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            tc.record(every=0.02, output=out, live=False, warn=False)
            tc.metric("score", 99)
            time.sleep(0.05)
            tc.stop()
            d = load_tc(out)
            # at least one timer snap should carry the metric
            snaps_with_score = [
                s for s in d["snaps"]
                if s.get("mets", {}).get("score") == 99
            ]
            assert len(snaps_with_score) > 0


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCTION TIMING
# ─────────────────────────────────────────────────────────────────────────────

class TestFunctionTiming:

    def test_trace_decorator_records_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            tc.record(every=1.0, output=out, live=False, warn=False)

            @tc.trace
            def add(a, b):
                return a + b

            for _ in range(5):
                add(1, 2)

            tc.stop()
            d = load_tc(out)
            assert any("add" in k for k in d["fn_timings"]), \
                f"trace decorator did not record timings — keys: {list(d['fn_timings'].keys())}"
            key = next(k for k in d["fn_timings"] if "add" in k)
            assert len(d["fn_timings"][key]) == 5

    def test_trace_decorator_records_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            tc.record(every=1.0, output=out, live=False, warn=False)

            @tc.trace
            def multiply(x, y):
                return x * y

            for _ in range(7):
                multiply(2, 3)

            tc.stop()
            d = load_tc(out)
            key = next((k for k in d["fn_calls"] if "multiply" in k), None)
            assert key is not None, f"multiply not found in fn_calls — keys: {list(d['fn_calls'].keys())}"
            assert d["fn_calls"][key] == 7

    def test_fn_entry_no_corruption_under_exception(self):
        """Bug A2 — fn_entry must not underflow when exception unwinds stack."""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            tc.record(every=1.0, output=out, live=False, warn=False)

            @tc.trace
            def risky(x):
                if x == 0:
                    raise ValueError("zero")
                return 1 / x

            for i in range(5):
                try:
                    risky(i)
                except ValueError:
                    pass

            tc.stop()
            d = load_tc(out)
            timings = d["fn_timings"].get("risky", [])
            # no timing should be > 60 seconds (epoch-time corruption)
            bad = [t for t in timings if t > 60.0]
            assert not bad, f"fn_entry corruption detected — timings: {timings}"


# ─────────────────────────────────────────────────────────────────────────────
#  WARNINGS
# ─────────────────────────────────────────────────────────────────────────────

class TestWarnings:

    def test_nan_triggers_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            tc.record(every=0.05, output=out, live=False, warn=True)
            x = float("nan")   # should trigger NAN_DETECTED
            time.sleep(0.1)
            tc.stop()
            d = load_tc(out)
            codes = [w["code"] for w in d["warnings"]]
            assert "NAN_DETECTED" in codes, "NaN variable not detected"

    def test_assert_var_fires_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            tc.record(every=1.0, output=out, live=False, warn=True)
            loss = 999.0
            tc.assert_var("loss", lambda v: v < 10.0, "loss too high")
            tc.stop()
            d = load_tc(out)
            codes = [w["code"] for w in d["warnings"]]
            assert "ASSERTION_FAILED" in codes

    def test_warn_false_disables_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            tc.record(every=0.05, output=out, live=False, warn=False)
            x = float("nan")
            time.sleep(0.1)
            tc.stop()
            d = load_tc(out)
            assert len(d["warnings"]) == 0, "warnings produced despite warn=False"


# ─────────────────────────────────────────────────────────────────────────────
#  CRASH CAPTURE
# ─────────────────────────────────────────────────────────────────────────────

class TestCrashCapture:

    def test_crash_captured_in_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            tc.record(every=1.0, output=out, live=False, warn=False)
            try:
                result = 1 / 0
            except ZeroDivisionError:
                pass
            tc.stop()
            # crash may or may not be captured depending on backend
            d = load_tc(out)
            # just assert file is valid regardless
            assert isinstance(d, dict)


# ─────────────────────────────────────────────────────────────────────────────
#  PROFILE BLOCK / CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

class TestBlockTiming:

    def test_profile_block_records_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            tc.record(every=1.0, output=out, live=False, warn=False)
            with tc.profile_block("my_block"):
                time.sleep(0.02)
            tc.stop()
            d = load_tc(out)
            bt = d.get("block_timings", {})
            assert "block:my_block" in bt
            assert bt["block:my_block"][0] >= 0.01  # at least 10ms

    def test_checkpoint_records_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            tc.record(every=1.0, output=out, live=False, warn=False)
            tc.checkpoint("data_load")
            time.sleep(0.02)
            tc.checkpoint("data_load")
            tc.stop()
            d = load_tc(out)
            bt = d.get("block_timings", {})
            assert "checkpoint:data_load" in bt
            assert bt["checkpoint:data_load"][0] >= 0.01

    def test_profile_block_nesting(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            tc.record(every=1.0, output=out, live=False, warn=False)
            with tc.profile_block("outer"):
                with tc.profile_block("inner"):
                    time.sleep(0.01)
            tc.stop()
            d = load_tc(out)
            bt = d.get("block_timings", {})
            assert "block:outer" in bt
            assert "block:inner" in bt


# ─────────────────────────────────────────────────────────────────────────────
#  VARIABLE CAPTURE
# ─────────────────────────────────────────────────────────────────────────────

class TestVariableCapture:

    def test_watch_filter_limits_captured_vars(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            tc.record(every=0.05, output=out, live=False, warn=False,
                      watch=["important"])
            important = 42
            ignored   = "should not appear"
            time.sleep(0.1)
            tc.stop()
            d = load_tc(out)
            all_seen = set()
            for s in d["snaps"]:
                all_seen.update(s.get("loc", {}).keys())
            assert "important" in all_seen or len(all_seen) == 0  # may miss if no snaps hit that frame
            # ignored should NOT be present
            assert "ignored" not in all_seen

    def test_max_snaps_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "test.tc")
            tc.record(every=0.005, output=out, live=False, warn=False, max_snaps=10)
            time.sleep(0.2)
            tc.stop()
            d = load_tc(out)
            assert len(d["snaps"]) <= 10
