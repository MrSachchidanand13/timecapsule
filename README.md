# timecapsule

**The Python tool that records what your program was thinking.**

Every other profiler tells you *where time went*. timecapsule tells you *what your program was doing* — variable values, function timings, system metrics, crash forensics — all captured into a single replayable `.tc` file.

```
pip install timecapsule
```

---

## The difference

| Tool | What it captures |
|---|---|
| cProfile, py-spy, Scalene | Function timing only |
| viztracer | Call tree + timing |
| Sentry | Exception at crash moment only |
| **timecapsule** | Variable values + timing + metrics + system state + crash + **everything** |

No other Python tool captures variable state. timecapsule is the first.

---

## Quick start

```python
import timecapsule as tc

tc.record(every=0.1)          # start — snapshot every 100ms

# ... your code runs here ...

x = 0
for i in range(100):
    x += i * 2
    tc.metric("progress", i)  # track custom metrics

tc.stop()                     # saves  script_20260323_142501.tc
```

Then in your terminal:

```bash
python -m timecapsule script_20260323_142501.tc
```

You get a 15-section report showing exactly what your program did, what was slow, what changed, and what to fix.

---

## Installation

### Minimal (zero dependencies)
```bash
pip install timecapsule
```

### With better system metrics (RAM/CPU via psutil)
```bash
pip install timecapsule[system]
```

### With GPU profiling (pynvml)
```bash
pip install timecapsule[gpu]
```

### Everything
```bash
pip install timecapsule[all]
```

**Requires Python 3.8+. Works on Linux, macOS, Windows.**

---

## Recording API

### `tc.record()` — start recording

```python
tc.record(
    every        = 1.0,      # seconds between timer snapshots
    watch        = None,     # list of var names to capture, or None for all
    output       = None,     # .tc output path (auto-named if None)
    max_snaps    = 10000,    # in-memory snapshot cap
    flush_every  = 50,       # crash-safe flush interval (0 = off)
    warn         = True,     # enable automatic warnings
    tags         = None,     # tags applied to every snapshot
    live         = False,    # print live status line to stderr
    include_files = None,    # glob patterns for multi-file tracking
    exclude_files = None,    # glob patterns to exclude
    mode         = "full",   # "full" or "sample" (<1% overhead)
)
```

Works as a context manager too:
```python
with tc.record(every=0.1):
    run_training()
# automatically calls tc.stop()
```

### `tc.snap()` — manual snapshot

```python
tc.snap("after_data_load")
tc.snap("epoch_complete", tags=["milestone"])
```

### `tc.metric()` — track a value

```python
tc.metric("loss", 0.342)
tc.metric("accuracy", 0.891)
tc.metric("epoch", 5)
```

Metrics appear in every subsequent snapshot and in the final report.

### `tc.watch()` — live variable printer

```python
tc.watch("loss", "accuracy")
# [TC WATCH] t=1.234s  loss:  0.8234 → 0.7891  (Δ -0.0343)
# [TC WATCH] t=1.234s  accuracy:  0.61 → 0.64
```

Replaces scattered `print()` debugging. Prints to stderr as variables change.

### `@tc.trace` — decorator

```python
@tc.trace
def train_step(batch):
    ...

@tc.trace(warn_slow_ms=500)
def load_dataset(path):
    ...
```

Auto-snapshots on every call and return. Shows args, return value, duration.

### `tc.breakpoint()` — soft breakpoint

```python
tc.breakpoint("after_transform")
tc.breakpoint(condition=lambda v: v["loss"] > 5.0)
```

Takes a snapshot and continues. Never pauses execution. Safe in production.

### `tc.assert_var()` — live assertion

```python
tc.assert_var("loss", lambda v: v < 10.0, "loss explosion!")
# fires ASSERTION_FAILED warning + auto-snapshot if loss >= 10
```

### `tc.watch_condition()` — auto-trigger

```python
tc.watch_condition("loss > 5.0", label="loss_spike")
tc.watch_condition("len(results) == 0", snap_mode="once")
```

Evaluated on every timer tick. Near-zero overhead when False.

### `tc.watch_expr()` — track an expression

```python
tc.watch_expr("loss_delta", "loss - prev_loss")
tc.watch_expr("ratio", "correct / total * 100")
```

### `tc.checkpoint()` — named timing intervals

```python
tc.checkpoint("data_load")
load_data()
tc.checkpoint("data_load")   # records elapsed, warns if > limit
```

### `tc.profile_block()` — context manager timing

```python
with tc.profile_block("preprocessing"):
    df = preprocess(raw)
# [TC] block 'preprocessing' → 142.3ms
```

### `tc.stop()` — stop and save

```python
tc.stop()
# [TC] ✓ saved  2611 snaps  4KB  35 vars  crash=no  warn=34  → recording.tc
```

### `tc.reset()` — clear without stopping

```python
# useful between training epochs
tc.reset()
```

---

## Analysis API

All analysis functions accept a `.tc` file path. `None` auto-loads the most recent `.tc` in the current directory.

### `tc.report()` — master report

```python
tc.report("recording.tc")
tc.report("recording.tc", export="report.txt")
tc.report("recording.tc", full=True, top_n=25)
```

15 sections:
1. Header with session info
2. Plain-English explanation of what the code did
3. Warnings with timestamps
4. Crash forensics (if crashed)
5. Function performance table with sparkline bars
6. Custom metrics with sparklines + anomaly markers
7. Memory (RAM / CPU / GPU / threads over time)
8. Variable timelines (every change, with old/new values)
9. Hot lines (most-executed source lines)
10. Return values
11. Stack sample profile (in `mode="sample"`)
12. Async task activity
13. Optimization suggestions
14. ASCII flame graph
15. Snapshot timeline

### `tc.explain()` — plain English

```python
tc.explain("recording.tc")
# order_pipeline.py ran for 4.629s, recording 2611 snapshots across 35 variables.
# Dominant function: process_payment() × 44 (avg 20.67ms/call).
# Final metrics: total_revenue=19087.33, failed_count=5, success_rate_pct=90.0.
# RAM grew 33→40MB (+7MB). 34 warning(s): TRACE_SLOW, ASSERTION_FAILED, REPEATED_CALL.
# No crashes.
```

### `tc.history()` — variable value history

```python
hist = tc.history("recording.tc", "loss")
# [(0.1, 0.9), (0.2, 0.87), (0.3, 0.84), ...]
for t, value in hist:
    print(f"t={t:.2f}s  loss={value}")
```

### `tc.diff()` — where a variable changed

```python
changes = tc.diff("recording.tc", "status")
# [(1.23, 'pending', 'processing'), (2.45, 'processing', 'complete')]
```

### `tc.timings()` — per-function stats

```python
stats = tc.timings("recording.tc")
# {
#   "process_payment": {calls:44, avg_ms:20.67, p95_ms:37.5, min_ms:12.7, max_ms:46.5, ...},
#   "generate_dispatch": {calls:45, avg_ms:4.33, ...},
# }
```

### `tc.anomalies()` — metric spike detection

```python
spikes = tc.anomalies("recording.tc", "gateway_latency_ms", z_threshold=2.5)
# [(2.254s, 42.51ms, z=3.2), (1.628s, 40.02ms, z=2.8)]
```

### `tc.search()` — full-text search

```python
hits = tc.search("recording.tc", "fraud")
hits = tc.search("recording.tc", "ERROR")
hits = tc.search("recording.tc", "ORD-2026")
```

### `tc.memory_leak_check()` — leak detection

```python
result = tc.memory_leak_check("recording.tc")
# {'leak_detected': False, 'growth_mb_per_min': 0.12, 'r_squared': 0.31, 'verdict': 'STABLE'}
```

Uses linear regression on RAM samples. Flags if growth is linear with R² > 0.7.

### `tc.variable_correlations()` — Pearson correlation

```python
corrs = tc.variable_correlations("recording.tc", min_r=0.75)
# [('total_revenue', 'orders_done', 0.999),
#  ('failed_count', 'gateway_latency_ms', 0.82)]
```

### `tc.exception_chain()` — root cause analysis

```python
tc.exception_chain("recording.tc")
```

Traces bad variable values back to where they were introduced — not just the crash line.

### `tc.loop_detector()` — O(n²) and stuck loop detection

```python
findings = tc.loop_detector("recording.tc")
# [{'type': 'O(N²)', 'severity': 'HIGH', 'fn': 'process', ...}]
```

### `tc.patch_summary()` — before/after diff

```python
tc.patch_summary("before_refactor.tc", "after_refactor.tc")
# [FASTER]  Total 4.389s → 2.1s (-52.0%)
# [FASTER]  process_payment()  20.67→14.2ms (-31.3%)
# [FIX WRN] Fixed: TRACE_SLOW:generate_dispatch
```

### `tc.regression_check()` — CI/CD

```python
tc.regression_check("recording.tc", max_slowdown_pct=20.0, exit_code=True)
# exits with code 1 if any function regressed > 20%
```

Use in CI:
```yaml
- run: python -m timecapsule recording.tc --regression-check
```

### `tc.baseline()` / `tc.diff_baseline()`

```python
tc.baseline("before.tc", name="v1.2")
# ... make changes ...
tc.diff_baseline("after.tc", name="v1.2")
```

Saves/loads from `~/.tc_baseline.json`.

### `tc.callgraph()` — ASCII call graph

```python
tc.callgraph("recording.tc")
```

### `tc.coverage_report()` — annotated source

```python
tc.coverage_report("recording.tc", source_path="my_script.py")
```

### `tc.heatmap()` — ASCII heatmap

```python
tc.heatmap("recording.tc", "loss")
```

### `tc.deadcode()` — never-called functions

```python
tc.deadcode("recording.tc")
```

### `tc.tail()` — live tail

```python
tc.tail("recording.tc")   # Ctrl+C to stop
```

### `tc.export_for_claude()` — AI analysis export

```python
tc.export_for_claude("recording.tc")
# → recording_for_claude.txt
```

Produces a structured document optimised for feeding to Claude or another LLM. Paste it into a chat and ask "what went wrong?".

### `tc.export_json()` — JSON export

```python
tc.export_json("recording.tc", "output.json")
```

### `tc.flamegraph()` — Chrome/SpeedScope export

```python
tc.flamegraph("recording.tc", "flame.json")
# open at speedscope.app
```

---

## CLI

```bash
# Full 15-section report
python -m timecapsule recording.tc

# Individual analyses
python -m timecapsule recording.tc --callgraph
python -m timecapsule recording.tc --coverage
python -m timecapsule recording.tc --loops
python -m timecapsule recording.tc --correlations
python -m timecapsule recording.tc --exception-chain
python -m timecapsule recording.tc --leak-check
python -m timecapsule recording.tc --deadcode
python -m timecapsule recording.tc --heatmap loss
python -m timecapsule recording.tc --hotlines
python -m timecapsule recording.tc --flamegraph
python -m timecapsule recording.tc --json

# Search
python -m timecapsule recording.tc --search fraud
python -m timecapsule recording.tc --history loss
python -m timecapsule recording.tc --anomaly gateway_latency_ms

# Comparison
python -m timecapsule before.tc --patch-summary after.tc
python -m timecapsule recording.tc --baseline
python -m timecapsule recording.tc --diff-baseline
python -m timecapsule recording.tc --regression-check

# Live
python -m timecapsule recording.tc --tail

# Export
python -m timecapsule recording.tc --export report.txt
python -m timecapsule recording.tc --for-claude

# Utility
python -m timecapsule recording.tc --summary
python -m timecapsule recording.tc --explain
python -m timecapsule --speedtest
```

---

## The .tc file format

`.tc` files are **gzip + Python pickle**. They survive program crashes because of the crash-safe flush mechanism (`flush_every=50`).

```python
d = tc.load("recording.tc")

d["ver"]            # "6.0"
d["session_id"]     # unique 8-char ID per recording
d["src"]            # source file path
d["duration"]       # elapsed seconds
d["n"]              # total snapshot count
d["crash"]          # None, or crash dict
d["warnings"]       # list of warning dicts
d["fn_timings"]     # {fn_name: [duration_seconds, ...]}
d["fn_calls"]       # {fn_name: call_count}
d["final_metrics"]  # {metric_name: value}
d["all_vars"]       # list of all variable names seen
d["snaps"]          # list of snapshot dicts
d["block_timings"]  # {block_name: [duration_seconds, ...]}
```

Each snapshot:
```python
snap = d["snaps"][42]

snap["i"]     # snapshot index
snap["t"]     # elapsed seconds from recording start
snap["evt"]   # "TIMER" | "LINE" | "TRACE:call:fn" | "SNAP:label" | "CRASH" | ...
snap["fn"]    # function name
snap["ln"]    # line number
snap["file"]  # filename (basename)
snap["loc"]   # {var_name: {"r": repr_str, "t": type_name, "size": bytes, "chg": bool}}
snap["stk"]   # call stack [{fn, file, ln, mod}, ...]
snap["sys"]   # {ram_mb, cpu_pct, threads, gpu_mb}
snap["mets"]  # {metric_name: value}
snap["tags"]  # [string, ...]
snap["label"] # string or None
```

---

## Automatic warnings

timecapsule detects these automatically (controlled by `warn=True`):

| Code | Meaning |
|---|---|
| `NAN_DETECTED` | A float variable is NaN |
| `INF_DETECTED` | A float variable is Inf |
| `NAN_IN_ARRAY` | A numpy array contains NaN |
| `VERY_LARGE` | A float exceeds 1e15 |
| `VAR_STUCK` | A variable hasn't changed for 5+ seconds |
| `UNBOUNDED_GROWTH` | A list/dict more than doubled in size |
| `RAM_GROWING` | RAM grew > 50MB from baseline |
| `FD_LEAK` | Open file descriptors grew by > 20 |
| `DEEP_RECURSION` | A function recursed > 50 levels |
| `REPEATED_CALL` | Same function called 50+ times with same args |
| `METRIC_PLATEAU` | A metric stagnated for 5 readings |
| `STUCK_LOOP` | Execution stuck at same line for 3× the timer interval |
| `ASSERTION_FAILED` | `tc.assert_var()` condition returned False |
| `CHECKPOINT_SLOW` | `tc.checkpoint()` elapsed time exceeded limit |
| `BUFFER_OVERFLOW` | C extension event buffer near capacity |

---

## Real-world example

```python
import timecapsule as tc

# Instrument an order processing pipeline
tc.record(
    every      = 0.1,
    live       = True,           # live status line
    flush_every= 25,             # crash-safe
)
tc.watch("total_revenue", "failed_count")
tc.watch_condition("failed_count > 5", label="high_failure_rate")

@tc.trace(warn_slow_ms=50)
def process_order(order):
    validate(order)
    charge(order)
    dispatch(order)

for order in orders:
    tc.metric("orders_done", done)
    process_order(order)
    done += 1

tc.stop()
```

After running:
```bash
python -m timecapsule order_pipeline_20260323.tc
python -m timecapsule order_pipeline_20260323.tc --correlations
python -m timecapsule order_pipeline_20260323.tc --for-claude
```

---

## Sampling mode

For tight loops where `every=0.1` is too much overhead:

```python
tc.record(mode="sample", sample_rate_hz=1000)
```

A separate thread wakes 1000 times/second and walks all Python frames. Zero per-call overhead. The report shows a statistical sample profile — which functions and lines were running most often.

---

## Multi-file projects

```python
tc.record(
    include_files=["src/*.py", "lib/utils.py"],
    exclude_files=["tests/*", "*/site-packages/*"],
)
```

---

## Platform notes

**Linux** — full metrics via `/proc`. Best experience.

**macOS** — RAM via `resource.getrusage()`, CPU via `ps`. All analysis functions work.

**Windows** — RAM via `ctypes` + `psapi.GetProcessMemoryInfo`. Terminal colors work with Windows 10 1903+ (ANSI mode). ANSI is auto-enabled.

**GPU** — auto-detected at startup:
1. `torch.cuda` if available
2. `torch.mps` (Apple Silicon) if available
3. `pynvml` if installed
4. None (no GPU overhead)

---

## How it compares to viztracer

| | viztracer | timecapsule |
|---|---|---|
| Variable capture | ❌ | ✅ every snapshot |
| Crash forensics | ❌ | ✅ root cause chain |
| NaN / leak / stuck warnings | ❌ | ✅ automatic |
| Plain-English explain | ❌ | ✅ |
| AI export | ❌ | ✅ |
| Metric sparklines | ❌ | ✅ |
| CI regression check | ❌ | ✅ |
| Coverage annotated source | ❌ | ✅ |
| Per-function timing | ✅ | ✅ |
| Chrome flame graph export | ✅ | ✅ |
| Multi-process tracing | ✅ | ❌ planned |
| Cross-platform C extension | ✅ | ⚠ Windows only, pure-Python fallback |

viztracer is a profiler. timecapsule is a forensic recorder. Different tools, different questions.

---

## Architecture

timecapsule is intentionally a **single-file module** with zero required dependencies.

```
timecapsule/
├── __init__.py    public API (imports only)
├── __main__.py    CLI entry point
└── recorder.py   complete implementation (~3400 lines, pure stdlib)
```

The recording engine uses:
- `sys.monitoring` (Python 3.12+) — low-overhead per-call/return hooks
- `sys.settrace` (Python 3.8–3.11) — fallback pure-Python tracing
- A background timer thread for periodic snapshots
- `deque` with `maxlen` for O(1) snapshot buffer
- `perf_counter` as the unified clock throughout
- `gzip` + `pickle` for the `.tc` file format
- Frame-ID keyed `fn_entry` dict to prevent timing corruption under recursion

---

## License

MIT — see [LICENSE](LICENSE).

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
