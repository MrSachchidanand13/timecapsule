"""
tests/test_cli.py
Tests for the CLI (python -m timecapsule).
"""
import os
import sys
import time
import gzip
import pickle
import tempfile
import subprocess
import pytest

import timecapsule as tc


@pytest.fixture(scope="module")
def tc_file(tmp_path_factory):
    """Minimal .tc file for CLI tests."""
    tmp = tmp_path_factory.mktemp("cli")
    out = str(tmp / "cli_test.tc")
    tc.record(every=0.05, output=out, live=False, warn=False)
    tc.metric("x", 42)
    tc.snap("test_snap")
    time.sleep(0.1)
    tc.stop()
    return out


def run_cli(*args, tc_path=None):
    """Run python -m timecapsule with args, return (returncode, stdout, stderr)."""
    cmd = [sys.executable, "-m", "timecapsule"]
    if tc_path:
        cmd.append(tc_path)
    cmd.extend(args)
    result = subprocess.run(
        cmd, capture_output=True, timeout=30,
        encoding="utf-8", errors="replace",   # force UTF-8 on Windows
    )
    return result.returncode, result.stdout, result.stderr


class TestCLIBasic:

    def test_cli_runs_with_tc_file(self, tc_file):
        code, out, err = run_cli(tc_path=tc_file)
        assert code == 0, f"CLI exited {code}\nstdout: {out}\nstderr: {err}"

    def test_cli_outputs_something(self, tc_file):
        code, out, err = run_cli(tc_path=tc_file)
        assert len(out) > 100, "CLI produced no output"

    def test_cli_report_contains_session(self, tc_file):
        code, out, err = run_cli(tc_path=tc_file)
        # should contain the session id somewhere
        d = tc.load(tc_file)
        assert d["session_id"] in out

    def test_cli_summary_flag(self, tc_file):
        code, out, err = run_cli("--summary", tc_path=tc_file)
        assert code == 0
        assert "snaps" in out.lower() or "snap" in out.lower()

    def test_cli_explain_flag(self, tc_file):
        code, out, err = run_cli("--explain", tc_path=tc_file)
        assert code == 0
        assert len(out) > 20

    def test_cli_export_json_flag(self, tc_file, tmp_path):
        out_json = str(tmp_path / "out.json")
        code, out, err = run_cli("--json", tc_path=tc_file)
        assert code == 0

    def test_cli_hotlines_flag(self, tc_file):
        code, out, err = run_cli("--hotlines", tc_path=tc_file)
        assert code == 0

    def test_cli_loops_flag(self, tc_file):
        code, out, err = run_cli("--loops", tc_path=tc_file)
        assert code == 0

    def test_cli_leak_check_flag(self, tc_file):
        code, out, err = run_cli("--leak-check", tc_path=tc_file)
        assert code == 0

    def test_cli_correlations_flag(self, tc_file):
        code, out, err = run_cli("--correlations", tc_path=tc_file)
        assert code == 0

    def test_cli_deadcode_flag(self, tc_file):
        code, out, err = run_cli("--deadcode", tc_path=tc_file)
        assert code == 0

    def test_cli_for_claude_creates_file(self, tc_file, tmp_path):
        os.chdir(tmp_path)
        code, out, err = run_cli("--for-claude", tc_path=tc_file)
        assert code == 0
        # should create a *_for_claude.txt file
        txt_files = list(tmp_path.glob("*_for_claude.txt"))
        assert len(txt_files) > 0, "export_for_claude file not created"

    def test_cli_speedtest(self):
        code, out, err = run_cli("--speedtest")
        assert code == 0
        assert "speedtest" in out.lower() or "baseline" in out.lower()


class TestCLISearch:

    def test_cli_search_flag(self, tc_file):
        code, out, err = run_cli("--search", "test_snap", tc_path=tc_file)
        assert code == 0

    def test_cli_search_no_results(self, tc_file):
        code, out, err = run_cli("--search", "xyzzy_no_match_99999", tc_path=tc_file)
        assert code == 0
        assert "0 result" in out.lower() or "result" in out.lower()


class TestCLIInvalidInput:

    def test_cli_nonexistent_file(self, tmp_path):
        code, out, err = run_cli(tc_path=str(tmp_path / "does_not_exist.tc"))
        assert code != 0

    def test_cli_invalid_file(self, tmp_path):
        bad = tmp_path / "bad.tc"
        bad.write_bytes(b"not a valid tc file")
        code, out, err = run_cli(tc_path=str(bad))
        assert code != 0


class TestCLIRegression:

    def test_baseline_and_diff(self, tc_file, tmp_path):
        os.chdir(tmp_path)
        # Save a baseline
        code, out, err = run_cli("--baseline", tc_path=tc_file)
        assert code == 0, f"baseline failed: {err}"
        # Compare against it
        code2, out2, err2 = run_cli("--diff-baseline", tc_path=tc_file)
        assert code2 == 0, f"diff-baseline failed: {err2}"