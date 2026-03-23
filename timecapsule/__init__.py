"""
timecapsule — Python program execution recorder and forensic analyser
======================================================================

The world's first Python tool that records complete variable state,
function timings, system metrics and crash forensics into a single
replayable .tc file — then gives you 15 analysis functions to
understand exactly what your program did.

Quick start
-----------
    import timecapsule as tc

    tc.record(every=0.1)       # start recording
    # ... your code ...
    tc.stop()                  # saves recording.tc

    tc.report("recording.tc")  # full forensic report

Recording API
-------------
    tc.record(every, watch, output, max_snaps, flush_every,
              warn, tags, live, include_files, exclude_files, mode)
    tc.snap(label, tags)
    tc.tag(*tags)
    tc.metric(name, value)
    tc.assert_var(name, condition_fn, label)
    tc.watch_expr(name, expr_str)
    tc.checkpoint(name, max_elapsed_s)
    tc.profile_block(name)        — context manager
    tc.stop()
    tc.reset()
    tc.watch(*var_names)
    tc.trace                      — decorator
    tc.breakpoint(label)
    tc.watch_condition(expr)

Analysis API
------------
    tc.report(path)               — master 15-section terminal report
    tc.summary(path)
    tc.explain(path)
    tc.history(path, var)
    tc.diff(path, var)
    tc.timings(path)
    tc.hotlines(path)
    tc.slowest(path)
    tc.anomalies(path, metric)
    tc.search(path, query)
    tc.memory_map(path)
    tc.export_json(path)
    tc.flamegraph(path)
    tc.callgraph(path)
    tc.coverage_report(path)
    tc.loop_detector(path)
    tc.memory_leak_check(path)
    tc.variable_correlations(path)
    tc.exception_chain(path)
    tc.patch_summary(before, after)
    tc.regression_check(path)
    tc.export_for_claude(path)
    tc.baseline(path)
    tc.diff_baseline(path)
    tc.heatmap(path, var)
    tc.deadcode(path)
    tc.tail(path)
    tc.speedtest()

CLI
---
    python -m timecapsule recording.tc
    python -m timecapsule recording.tc --callgraph
    python -m timecapsule recording.tc --correlations
    python -m timecapsule recording.tc --tail
    python -m timecapsule recording.tc --for-claude
"""

from timecapsule.recorder import (
    # ── recording ──────────────────────────────────────────────────────
    record,
    snap,
    tag,
    metric,
    assert_var,
    watch_expr,
    checkpoint,
    profile_block,
    stop,
    reset,
    watch,
    trace,
    breakpoint,
    watch_condition,
    profile,
    speedtest,

    # ── analysis ───────────────────────────────────────────────────────
    load,
    report,
    dump,
    summary,
    explain,
    history,
    diff,
    timings,
    hotlines,
    slowest,
    rate,
    anomalies,
    search,
    since,
    replay,
    memory_map,
    export_json,
    flamegraph,
    callgraph,
    coverage_report,
    loop_detector,
    memory_leak_check,
    variable_correlations,
    exception_chain,
    patch_summary,
    export_for_claude,
    regression_check,
    baseline,
    diff_baseline,
    heatmap,
    deadcode,
    tail,
    profile_report,

    # ── version ────────────────────────────────────────────────────────
    __version__,
)

__all__ = [
    # recording
    "record", "snap", "tag", "metric", "assert_var", "watch_expr",
    "checkpoint", "profile_block", "stop", "reset", "watch", "trace",
    "breakpoint", "watch_condition", "profile", "speedtest",
    # analysis
    "load", "report", "dump", "summary", "explain", "history", "diff",
    "timings", "hotlines", "slowest", "rate", "anomalies", "search",
    "since", "replay", "memory_map", "export_json", "flamegraph",
    "callgraph", "coverage_report", "loop_detector", "memory_leak_check",
    "variable_correlations", "exception_chain", "patch_summary",
    "export_for_claude", "regression_check", "baseline", "diff_baseline",
    "heatmap", "deadcode", "tail", "profile_report",
    # version
    "__version__",
]
