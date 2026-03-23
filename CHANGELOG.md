# Changelog

All notable changes to timecapsule are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [6.0.0] — 2026-03-23

### First public release

timecapsule v6 is the first version published to PyPI.

#### Recording engine
- `tc.record()` — start recording with timer-based snapshots
- `tc.record(mode="sample")` — stack-sampling mode, <1% overhead
- `tc.record(include_files=["src/*.py"])` — multi-file project support
- `tc.snap()`, `tc.tag()`, `tc.metric()` — manual instrumentation
- `tc.assert_var()` — live assertions that auto-snapshot on failure
- `tc.watch_expr()` — evaluate expressions on every snapshot
- `tc.checkpoint()` — named timing intervals
- `tc.profile_block()` — context manager for block timing
- `tc.watch()` — live variable change printer (replaces print debugging)
- `@tc.trace` — decorator for zero-boilerplate per-call recording
- `tc.breakpoint()` — soft breakpoint, never pauses execution
- `tc.watch_condition()` — auto-trigger snapshots on condition

#### Cross-platform
- Linux: `/proc/PID/status` for RAM, `/proc/stat` for CPU
- macOS: `resource.getrusage()` for RAM, `ps` subprocess for CPU
- Windows: `ctypes` → `psapi.GetProcessMemoryInfo`, zero extra deps
- GPU: auto-detects torch CUDA/MPS or pynvml, cached at startup

#### Bug fixes vs v5
- State never fully reset between recordings — fixed (complete `_S` clear)
- `fn_entry` stack corrupted under recursion + exceptions — fixed (frame-id keying)
- `excepthook` chained infinitely on repeated `record()` calls — fixed (wraps `sys.__excepthook__` once)
- Mixed `time.time()` / `time.perf_counter()` clocks — fixed (unified `perf_counter` everywhere)
- `list.pop(0)` O(n) snap buffer — fixed (`collections.deque` O(1))
- Flush held the lock during disk I/O — fixed (flush outside lock)
- Single-file source assumption — fixed (`include_files` glob patterns)

#### Analysis functions (15 sections in terminal report)
- `tc.report()` — master report with sparklines, heat bars, ASCII flame graph
- `tc.explain()` — plain-English paragraph summary
- `tc.callgraph()` — ASCII call graph
- `tc.coverage_report()` — annotated source with hit counts
- `tc.memory_leak_check()` — linear regression on RAM samples
- `tc.variable_correlations()` — Pearson correlation matrix
- `tc.exception_chain()` — root cause analysis beyond crash line
- `tc.loop_detector()` — O(n²) growth and stuck loop detection
- `tc.patch_summary()` — before/after plain-English diff
- `tc.regression_check()` — CI/CD pass/fail with `sys.exit(1)`
- `tc.export_for_claude()` — structured export for AI analysis
- `tc.baseline()` / `tc.diff_baseline()` — performance baseline tracking
- `tc.heatmap()` — ASCII heatmap of numeric variables
- `tc.deadcode()` — find never-called functions
- `tc.tail()` — live-tail a recording as it grows

#### CLI
- `python -m timecapsule recording.tc` — full report
- All analysis functions available as flags (`--callgraph`, `--correlations`, etc.)
