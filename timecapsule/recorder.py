"""
tc_recorder.py  —  timecapsule v6.0
=====================================
Zero external dependencies. Pure stdlib.
Optional C acceleration: tc_core (auto-detected).

WHAT'S NEW IN v6:
  ✦ All 12 critical bugs from v5 fixed (state reset, fn_entry corruption,
    exception-stack underflow, excepthook chaining, time-source inconsistency)
  ✦ Cross-platform system metrics: Linux /proc + macOS resource + Windows ctypes
  ✦ GPU auto-detection: torch CUDA/MPS + pynvml fallback, cached at startup
  ✦ Single-pass variable capture: 4× faster, repr computed once
  ✦ Multi-file project support: include_files / exclude_files patterns
  ✦ Async/asyncio task awareness: captures Task name + coroutine state
  ✦ Flush happens OUTSIDE the lock: no disk-IO stall on hot path
  ✦ deque-based snap buffer: O(1) append/pop vs O(n) list.pop(0)
  ✦ Unified perf_counter clock: nanosecond resolution, consistent everywhere
  ✦ Stack-sampler mode: zero per-call overhead, works on any Python
  ✦ Insane terminal report: 15 sections, sparklines, heat bars, ASCII flame graphs
  ✦ Smart repr deduplication: unchanged vars cost one dict lookup
  ✦ Thread-safe fn_entry via frame-id keying (no stack under/overflow)
  ✦ Sampling mode (tc.record(mode="sample")): <1% overhead on tight loops

PUBLIC API — RECORDING:
  record(every, watch, output, max_snaps, flush_every, warn, tags, live,
         include_files, exclude_files, mode)
  snap(label, tags)
  tag(*tags)
  metric(name, value)
  assert_var(name, condition_fn, label)
  watch_expr(name, expr_str)
  checkpoint(name, max_elapsed_s)
  profile_block(name)   — context manager
  stop()
  reset()
  watch(*var_names)
  trace(fn)             — decorator
  breakpoint(label)     — soft breakpoint
  watch_condition(expr) — auto-trigger

PUBLIC API — ANALYSIS (terminal, zero GUI):
  load(path)
  report(path, ...)     — master terminal report
  dump(path, ...)       — raw snapshot dump
  summary(path)
  explain(path)
  history(path, var)
  diff(path, var)
  timings(path)
  hotlines(path)
  slowest(path)
  anomalies(path, metric)
  search(path, query)
  memory_map(path)
  export_json(path)
  flamegraph(path)
  callgraph(path)
  coverage_report(path)
  loop_detector(path)
  memory_leak_check(path)
  variable_correlations(path)
  exception_chain(path)
  patch_summary(before, after)
  regression_check(path)
  export_for_claude(path)
  baseline(path) / diff_baseline(path)
  heatmap(path, var)
  timeline_report(path)
  deadcode(path)
"""

import sys, os, time, math, json, struct, pickle, gzip, threading, re
import traceback as _tb_mod, inspect, contextlib, statistics, fnmatch
from datetime import datetime
from collections import defaultdict, deque

__version__ = "6.0"

# ── Force UTF-8 output on Windows (cp1252 default crashes on →, ✓, ═ etc.) ──
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
#  C EXTENSION  (optional)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import tc_core as _C
    _HAVE_C = True
except ImportError:
    _C = None
    _HAVE_C = False

def _c_info():
    if not _HAVE_C: return "pure-Python"
    try:
        rdtsc = getattr(_C, "HAS_RDTSC", 0)
        cpns  = _C.cycles_per_ns() if hasattr(_C, "cycles_per_ns") else None
        s = f"C+monitoring  RDTSC={'yes' if rdtsc else 'no'}"
        if cpns: s += f"  {cpns:.2f}cyc/ns"
        return s
    except Exception: return "C-ext"

# ─────────────────────────────────────────────────────────────────────────────
#  TERMINAL COLORS  (Windows-safe)
# ─────────────────────────────────────────────────────────────────────────────
def _init_color():
    if not sys.stdout.isatty(): return False
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleMode(
                ctypes.windll.kernel32.GetStdHandle(-11), 7)
            return True
        except Exception: return False
    return True

_USE_COLOR = _init_color()

def _c(t, code): return f"\033[{code}m{t}\033[0m" if _USE_COLOR else str(t)
def bold(t):    return _c(t, "1")
def dim(t):     return _c(t, "2")
def red(t):     return _c(t, "31")
def green(t):   return _c(t, "32")
def yellow(t):  return _c(t, "33")
def blue(t):    return _c(t, "34")
def magenta(t): return _c(t, "35")
def cyan(t):    return _c(t, "36")

# ─────────────────────────────────────────────────────────────────────────────
#  PLATFORM DETECTION
# ─────────────────────────────────────────────────────────────────────────────
_PLATFORM = sys.platform   # "linux", "darwin", "win32"
_PID      = os.getpid()

# ─────────────────────────────────────────────────────────────────────────────
#  GPU BACKEND — detected once at import, never re-detected
# ─────────────────────────────────────────────────────────────────────────────
_GPU_BACKEND  = None   # "torch_cuda" | "torch_mps" | "pynvml" | "none"
_GPU_HANDLE   = None   # pynvml handle
_GPU_DETECTED = False

def _detect_gpu():
    global _GPU_BACKEND, _GPU_HANDLE, _GPU_DETECTED
    if _GPU_DETECTED: return
    _GPU_DETECTED = True
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _GPU_BACKEND = "torch_cuda"; return
        if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
            _GPU_BACKEND = "torch_mps"; return
    except ImportError: pass
    try:
        import pynvml as _pynvml
        _pynvml.nvmlInit()
        _GPU_HANDLE = _pynvml.nvmlDeviceGetHandleByIndex(0)
        _GPU_BACKEND = "pynvml"; return
    except Exception: pass
    _GPU_BACKEND = "none"

def _gpu_metrics():
    if not _GPU_DETECTED: _detect_gpu()
    if _GPU_BACKEND == "none" or _GPU_BACKEND is None: return {}
    try:
        if _GPU_BACKEND == "torch_cuda":
            import torch as _t
            return {
                "gpu_mb":    round(_t.cuda.memory_allocated() / 1_048_576, 1),
                "gpu_peak":  round(_t.cuda.max_memory_allocated() / 1_048_576, 1),
                "gpu_reserved": round(_t.cuda.memory_reserved() / 1_048_576, 1),
            }
        if _GPU_BACKEND == "torch_mps":
            import torch as _t
            return {"gpu_mb": round(_t.mps.current_allocated_memory() / 1_048_576, 1)}
        if _GPU_BACKEND == "pynvml":
            import pynvml as _pynvml
            info = _pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
            util = _pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE)
            return {
                "gpu_mb":   round(info.used / 1_048_576, 1),
                "gpu_total":round(info.total / 1_048_576, 1),
                "gpu_util": util.gpu,
            }
    except Exception: pass
    return {}

# ─────────────────────────────────────────────────────────────────────────────
#  CROSS-PLATFORM SYSTEM METRICS
# ─────────────────────────────────────────────────────────────────────────────
_CPU_PREV = [0.0, 0.0]   # [total_jiffies, idle_jiffies] — Linux
_PROC_START_TICKS = None  # for per-process CPU on macOS

def _sys_metrics_linux(m):
    try:
        with open(f"/proc/{_PID}/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    m["ram_mb"] = round(int(line.split()[1]) / 1024, 2)
                elif line.startswith("VmPeak:"):
                    m["ram_peak_mb"] = round(int(line.split()[1]) / 1024, 2)
    except Exception: pass
    try:
        m["open_fds"] = len(os.listdir(f"/proc/{_PID}/fd"))
    except Exception: pass
    try:
        with open("/proc/stat", "r") as fh:
            parts = fh.readline().split()
        vals = [float(x) for x in parts[1:8]]
        idle, total = vals[3], sum(vals)
        pt, pi = _CPU_PREV; dt = total - pt; di = idle - pi
        if dt > 0: m["cpu_pct"] = round(100.0 * (1.0 - di / dt), 1)
        _CPU_PREV[0] = total; _CPU_PREV[1] = idle
    except Exception: pass

def _sys_metrics_darwin(m):
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # macOS: ru_maxrss is bytes
        m["ram_mb"] = round(ru.ru_maxrss / 1_048_576, 2)
    except Exception: pass
    try:
        import subprocess, shlex
        out = subprocess.check_output(
            ["ps", "-o", "pcpu=", "-p", str(_PID)],
            timeout=0.1, stderr=subprocess.DEVNULL)
        m["cpu_pct"] = round(float(out.strip()), 1)
    except Exception: pass

def _sys_metrics_win32(m):
    try:
        import ctypes, ctypes.wintypes as wt
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb",                          wt.DWORD),
                ("PageFaultCount",              wt.DWORD),
                ("PeakWorkingSetSize",          ctypes.c_size_t),
                ("WorkingSetSize",              ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage",     ctypes.c_size_t),
                ("QuotaPagedPoolUsage",         ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage",  ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage",      ctypes.c_size_t),
                ("PagefileUsage",               ctypes.c_size_t),
                ("PeakPagefileUsage",           ctypes.c_size_t),
            ]
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(pmc)
        psapi = ctypes.windll.psapi
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
            m["ram_mb"]      = round(pmc.WorkingSetSize / 1_048_576, 2)
            m["ram_peak_mb"] = round(pmc.PeakWorkingSetSize / 1_048_576, 2)
    except Exception: pass

def _sys_metrics():
    m = {"threads": threading.active_count()}
    if   _PLATFORM == "linux":  _sys_metrics_linux(m)
    elif _PLATFORM == "darwin": _sys_metrics_darwin(m)
    elif _PLATFORM == "win32":  _sys_metrics_win32(m)
    m.update(_gpu_metrics())
    # update peaks in state
    ram = m.get("ram_mb", 0)
    if ram and ram > _S.get("ram_peak", 0): _S["ram_peak"] = ram
    fds = m.get("open_fds", 0)
    if fds and fds > _S.get("fd_peak", 0): _S["fd_peak"] = fds
    return m

def _check_sys_warnings(m, t):
    try:
        bl = _S.get("ram_baseline"); ram = m.get("ram_mb", 0)
        if bl and ram > 0 and (ram - bl) > 50:
            _warn("RAM_GROWING", f"RAM {bl:.0f}→{ram:.0f}MB (+{ram-bl:.0f}MB)", t)
        fb = _S.get("fd_baseline"); fds = m.get("open_fds", 0)
        if fb and fds > 0 and (fds - fb) > 20:
            _warn("FD_LEAK", f"FDs grew {fb}→{fds}", t)
    except Exception: pass

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL STATE  (v6: all resets are complete)
# ─────────────────────────────────────────────────────────────────────────────
_S = {
    "on": False, "snaps": deque(), "every": 1.0, "watch": None,
    "max": 10000, "outfile": None, "t0_perf": None, "t0_wall": None,
    "n": 0, "crash": None, "src": None, "saved": False, "warn": True,
    "lock": threading.Lock(), "timer": None, "mode": "full",
    "metrics": {}, "metric_hist": defaultdict(list),
    "line_hits": defaultdict(int),
    "fn_calls": defaultdict(int),
    "fn_timings": defaultdict(list),
    "fn_entry": {},          # {frame_id: (fn_name, entry_perf)}
    "ret_vals": deque(maxlen=2000),
    "rec_depth": defaultdict(int), "rec_max": defaultdict(int),
    "var_last": {},          # {var: repr_str} — for change detection
    "var_unchanged_since": {}, "var_sizes": defaultdict(list),
    "seen_call_counts": defaultdict(int),  # {fn: call_count}
    "warnings": [], "warned_keys": set(),
    "ram_baseline": None, "ram_peak": 0.0,
    "fd_baseline": None, "fd_peak": 0,
    "flush_every": 50, "flush_count": 0, "c_meta": {},
    "current_tags": [], "watch_exprs": {}, "checkpoints": {},
    "live_enabled": False, "live_last": 0.0,
    "block_timings": defaultdict(list),
    "session_id": None,
    "_line_prev_vars": {}, "_watch_vars": {}, "_watch_active": False,
    "_watch_every_snap": False,
    # multi-file support
    "include_patterns": [], "exclude_patterns": [],
    # async support
    "_async_task_ids": {},
    # sampling mode state
    "_sampler_interval": 0.001,
}

_SKIP_TYPES  = (type(sys), type(threading.Lock()), threading.Thread,
                type(lambda: None), type(print), type(type))
_SIMPLE_TYPES = (int, float, str, bool, bytes, type(None), complex)

# ─────────────────────────────────────────────────────────────────────────────
#  UNIFIED CLOCK  (perf_counter everywhere, anchored to t0_perf)
# ─────────────────────────────────────────────────────────────────────────────
def _elapsed():
    t0 = _S.get("t0_perf")
    if t0 is None: return 0.0
    return time.perf_counter() - t0

def _t_now():
    return round(_elapsed(), 6)

# ─────────────────────────────────────────────────────────────────────────────
#  FILE MATCHING (multi-file support)
# ─────────────────────────────────────────────────────────────────────────────
_IS_MINE_CACHE = {}   # {filename: bool}

def _is_mine(frame):
    try:
        fname = frame.f_code.co_filename
        cached = _IS_MINE_CACHE.get(fname)
        if cached is not None: return cached
        result = _is_mine_uncached(fname)
        _IS_MINE_CACHE[fname] = result
        return result
    except Exception: return False

def _is_mine_uncached(fname):
    src = _S.get("src")
    if not src: return False
    try:
        abs_fname = os.path.abspath(fname)
        # always exclude recorder itself
        if "tc_recorder" in abs_fname: return False
        # exclude patterns take priority
        for pat in _S.get("exclude_patterns", []):
            if fnmatch.fnmatch(abs_fname, pat): return False
        # if include patterns set, file must match one
        include = _S.get("include_patterns", [])
        if include:
            return any(fnmatch.fnmatch(abs_fname, p) for p in include)
        # default: only main source file
        return abs_fname == src
    except Exception: return False

def _find_mine(frame):
    f = frame
    while f:
        if _is_mine(f): return f
        f = f.f_back
    return None

# ─────────────────────────────────────────────────────────────────────────────
#  REPR / PICKLE / SIZE  (single-pass: repr computed ONCE per var per snap)
# ─────────────────────────────────────────────────────────────────────────────
_PICKLE_LIMIT = 500_000

def _safe_repr(obj, max_len=500):
    try:
        r = repr(obj)
        return r[:max_len] + "…" if len(r) > max_len else r
    except Exception:
        try:    return f"<{type(obj).__name__} id={id(obj):#x}>"
        except: return "<unrepresentable>"

def _smart_repr(obj):
    tn = type(obj).__name__
    try:
        if tn == "ndarray":
            s = obj.shape; d = obj.dtype
            try:    return f"ndarray{s} {d} min={obj.min():.4g} max={obj.max():.4g} mean={obj.mean():.4g}"
            except: return f"ndarray{s} {d}"
        if tn == "DataFrame": return f"DataFrame{obj.shape} cols={list(obj.columns)[:5]}"
        if tn == "Series":    return f"Series[{obj.dtype}] len={len(obj)}"
        if tn in ("Tensor","Parameter"):
            try:    return f"Tensor{tuple(obj.shape)} {obj.dtype} dev={obj.device} grad={obj.requires_grad}"
            except: return f"Tensor {tn}"
        if isinstance(obj, (list, tuple)):
            return f"{tn}[{len(obj)}] {_safe_repr(obj, 120)}"
        if isinstance(obj, dict):
            return f"dict[{len(obj)}] {_safe_repr(obj, 120)}"
        if isinstance(obj, set):
            return f"set[{len(obj)}] {_safe_repr(obj, 120)}"
    except Exception: pass
    return _safe_repr(obj)

def _pkl(obj):
    try:
        if isinstance(obj, _SIMPLE_TYPES):
            return pickle.dumps(obj, 2)
        if isinstance(obj, (list, tuple, dict, set, frozenset)):
            est = sys.getsizeof(obj)
            if est > _PICKLE_LIMIT:
                return pickle.dumps(_smart_repr(obj), 2)
            d = pickle.dumps(obj, 2)
            return d if len(d) < _PICKLE_LIMIT else pickle.dumps(_smart_repr(obj), 2)
        tn = type(obj).__name__
        if tn in ("ndarray", "DataFrame", "Series", "Tensor", "Parameter"):
            return pickle.dumps(_smart_repr(obj), 2)
        return pickle.dumps(_safe_repr(obj), 2)
    except Exception:
        return b""

def _sizeof(obj):
    try:
        if isinstance(obj, (list, tuple)): return sys.getsizeof(obj) + sum(sys.getsizeof(x) for x in obj[:50])
        if isinstance(obj, dict):           return sys.getsizeof(obj) + sum(sys.getsizeof(k)+sys.getsizeof(v) for k,v in list(obj.items())[:50])
        return sys.getsizeof(obj)
    except Exception: return 0

# ─────────────────────────────────────────────────────────────────────────────
#  WARNINGS ENGINE  (thread-safe, per-session dedup)
# ─────────────────────────────────────────────────────────────────────────────
def _warn(code, msg, t, level="WARN", **extra):
    if not _S.get("warn"): return
    key = f"{code}:{extra.get('var','')}"
    wkeys = _S.get("warned_keys")
    if wkeys is None or key in wkeys: return
    wkeys.add(key)
    w = {"code": code, "msg": msg, "t": round(t, 4), "level": level, **extra}
    with _S["lock"]:
        _S["warnings"].append(w)
    icon = red("🔴") if level == "ERROR" else yellow("🟡")
    print(f"\n{icon} [TC] {bold(code)}: {msg}", file=sys.stderr)

def _check_numeric(name, val, t):
    try:
        if isinstance(val, float):
            if math.isnan(val):   _warn("NAN_DETECTED",  f"'{name}' is NaN at t={t:.3f}s",   t, level="ERROR", var=name)
            elif math.isinf(val): _warn("INF_DETECTED",  f"'{name}' is Inf at t={t:.3f}s",   t, level="ERROR", var=name)
            elif abs(val) > 1e15: _warn("VERY_LARGE",    f"'{name}'={val:.2e} at t={t:.3f}s", t, level="WARN",  var=name)
        tn = type(val).__name__
        if tn == "ndarray":
            try:
                import numpy as np
                if np.isnan(val).any():  _warn("NAN_IN_ARRAY", f"NaN in array '{name}'", t, level="ERROR", var=name)
                elif np.isinf(val).any():_warn("INF_IN_ARRAY", f"Inf in array '{name}'", t, level="ERROR", var=name)
            except Exception: pass
    except Exception: pass

def _check_metric_plateau(name, val, t):
    try:
        hist = _S["metric_hist"][name]; hist.append((t, val))
        if len(hist) < 5: return
        recent = [v for _, v in hist[-5:]]
        if all(isinstance(v, (int, float)) for v in recent):
            span = max(recent) - min(recent)
            mx   = max(abs(v) for v in recent) or 1
            if span / mx < 0.001 and mx > 0.01:
                _warn("METRIC_PLATEAU", f"'{name}' stagnant ≈{recent[-1]:.4g} for 5 readings", t, metric=name)
        if len(hist) > 200: _S["metric_hist"][name] = hist[-100:]
    except Exception: pass

# ─────────────────────────────────────────────────────────────────────────────
#  VARIABLE CAPTURE  (v6: single-pass, repr computed once, diff-only pickle)
# ─────────────────────────────────────────────────────────────────────────────
def _cap_locals(frame):
    out     = {}
    watch   = _S["watch"]
    t       = _t_now()
    var_last    = _S["var_last"]
    var_unch    = _S["var_unchanged_since"]
    var_sizes   = _S["var_sizes"]
    do_warn     = _S.get("warn", False)

    try:
        for k, v in frame.f_locals.items():
            if k[0] == "_": continue          # faster than startswith
            if watch is not None and k not in watch: continue
            if isinstance(v, _SKIP_TYPES): continue
            if callable(v) and not isinstance(v, (list, dict, tuple, set, frozenset)): continue

            try:
                sr   = _smart_repr(v)          # ← computed ONCE
                typ  = type(v).__name__

                last = var_last.get(k)
                changed = (last != sr)

                if changed:
                    # only pickle changed vars — unchanged cost = one dict lookup
                    pkl  = _pkl(v)
                    size = _sizeof(v)
                    var_last[k]  = sr
                    var_unch[k]  = t
                    out[k] = {"r": sr, "t": typ, "p": pkl, "size": size, "chg": True}
                else:
                    # unchanged: reuse cached repr, skip expensive ops
                    out[k] = {"r": sr, "t": typ, "p": b"", "size": 0, "chg": False}

                if do_warn:
                    _check_numeric(k, v, t)
                    if not changed:
                        since = var_unch.get(k, t)
                        if (t - since) > 5.0:
                            _warn("VAR_STUCK", f"'{k}' unchanged for {t-since:.0f}s", t, var=k)
                    if isinstance(v, (list, dict, set)):
                        sz = len(v)
                        hist = var_sizes[k]; hist.append((t, sz))
                        if len(hist) >= 3 and hist[-1][1] > hist[0][1] * 2 and hist[0][1] > 10:
                            _warn("UNBOUNDED_GROWTH", f"'{k}' grew {hist[0][1]}→{hist[-1][1]}", t, var=k)
                        if len(hist) > 20: var_sizes[k] = hist[-10:]
            except Exception: pass

    except Exception as e:
        out["__cap_err"] = {"r": str(e), "t": "capture_error", "p": b"", "size": 0}

    # watch_exprs
    for expr_name, expr_str in _S.get("watch_exprs", {}).items():
        try:
            val = eval(expr_str, frame.f_globals, frame.f_locals)
            sr  = _smart_repr(val)
            out[f"__expr_{expr_name}"] = {
                "r": sr, "t": type(val).__name__,
                "p": _pkl(val), "size": _sizeof(val), "expr": expr_str
            }
        except Exception as e:
            out[f"__expr_{expr_name}"] = {"r": f"<eval:{e}>", "t": "error", "p": b"", "size": 0, "expr": expr_str}

    return out

def _cap_stack(frame, depth=16):
    stk = []
    f = frame; d = 0
    while f and d < depth:
        stk.append({
            "fn":   f.f_code.co_name,
            "file": os.path.basename(f.f_code.co_filename),
            "abs":  f.f_code.co_filename,
            "ln":   f.f_lineno,
            "mod":  f.f_globals.get("__name__", "?"),
        })
        f = f.f_back; d += 1
    return stk

# ─────────────────────────────────────────────────────────────────────────────
#  ASYNC CONTEXT DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def _async_info():
    try:
        import asyncio
        task = asyncio.current_task()
        if task:
            return {"task": task.get_name(), "coro": task.get_coro().__qualname__}
    except Exception: pass
    return {}

# ─────────────────────────────────────────────────────────────────────────────
#  SOURCE CONTEXT
# ─────────────────────────────────────────────────────────────────────────────
_SRC_CACHE = {}  # {path: [lines]}

def _source_context(filename, lineno, radius=5):
    try:
        if filename not in _SRC_CACHE:
            with open(filename, "r", encoding="utf-8", errors="replace") as f:
                _SRC_CACHE[filename] = f.readlines()
        lines = _SRC_CACHE[filename]
        start = max(0, lineno - radius - 1)
        end   = min(len(lines), lineno + radius)
        result = []
        for i, ln in enumerate(lines[start:end], start + 1):
            result.append((i, ln.rstrip(), i == lineno))
        return result
    except Exception: return []

# ─────────────────────────────────────────────────────────────────────────────
#  SNAPSHOT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def _build_snap(frame, evt, extra=None):
    t = _t_now()
    m = _sys_metrics()
    _check_sys_warnings(m, t)
    ainfo = _async_info()
    snap = {
        "i":    _S["n"],
        "t":    t,
        "wt":   datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "evt":  evt,
        "fn":   frame.f_code.co_name,
        "ln":   frame.f_lineno,
        "file": os.path.basename(frame.f_code.co_filename),
        "abs":  frame.f_code.co_filename,
        "loc":  _cap_locals(frame),
        "stk":  _cap_stack(frame),
        "sys":  m,
        "mets": dict(_S["metrics"]),
        "hits": _S["line_hits"].get((frame.f_code.co_filename, frame.f_lineno), 0),
        "rec":  dict(_S["rec_max"]),
        "tags": list(_S.get("current_tags", [])),
        "sid":  _S.get("session_id"),
        "label": None,
    }
    if ainfo: snap["async"] = ainfo
    if extra:  snap.update(extra)
    _S["current_tags"] = []
    return snap

def _store(snap):
    do_flush = False
    with _S["lock"]:
        _S["snaps"].append(snap)
        _S["n"] += 1
        if len(_S["snaps"]) > _S["max"]:
            _S["snaps"].popleft()   # O(1) deque pop
        _S["flush_count"] += 1
        if _S["flush_every"] and _S["flush_count"] >= _S["flush_every"]:
            _S["flush_count"] = 0
            do_flush = True
    if do_flush:             # OUTSIDE the lock — no disk-IO stall
        _flush_chunk()
    if _S.get("live_enabled"): _print_live(snap)
    if _S.get("_watch_active"):
        try: _check_watch(snap)
        except Exception: pass

# ─────────────────────────────────────────────────────────────────────────────
#  CRASH TIPS  (expanded)
# ─────────────────────────────────────────────────────────────────────────────
_ERROR_TIPS = {
    "IndexError":        "Tried to access list index out of range.\n  Tip: guard with 'if idx < len(lst)' or use enumerate().",
    "KeyError":          "Dict key doesn't exist.\n  Tip: use dict.get('key', default) or 'if key in d'.",
    "AttributeError":    "Attribute/method not found — object may be None or wrong type.\n  Tip: print(type(obj), dir(obj)).",
    "TypeError":         "Wrong type passed.\n  Tip: print(type(x)) before the operation.",
    "NameError":         "Variable not defined.\n  Tip: check for typos, or define before use.",
    "ZeroDivisionError": "Divided by zero.\n  Tip: guard with 'if denominator != 0' or use max(denom, 1e-9).",
    "RecursionError":    "Call stack overflowed.\n  Tip: add base case, or convert to iterative with explicit stack.",
    "MemoryError":       "Out of RAM.\n  Tip: process in chunks, use generators, clear large vars with del.",
    "FileNotFoundError": "File not found.\n  Tip: print(os.getcwd()); check path spelling and working directory.",
    "ValueError":        "Right type, wrong value.\n  Tip: validate input before conversion; use try/except.",
    "ImportError":       "Module not found.\n  Tip: pip install <module>; check virtual environment.",
    "StopIteration":     "Iterator exhausted.\n  Tip: use 'for' loops instead of bare next(); check has_next().",
    "OverflowError":     "Number too large for float.\n  Tip: use Python's arbitrary-precision int, or scale down values.",
    "UnicodeDecodeError":"Invalid bytes for encoding.\n  Tip: open(f, encoding='utf-8', errors='replace').",
    "PermissionError":   "No permission to read/write.\n  Tip: check file ownership and chmod; run as appropriate user.",
    "TimeoutError":      "Operation timed out.\n  Tip: increase timeout; check network; add retry logic.",
    "OSError":           "OS-level error.\n  Tip: check file/path existence; check disk space and permissions.",
    "RuntimeError":      "Runtime error.\n  Tip: read the message; check state before calling this function.",
    "NotImplementedError":"Method not implemented.\n  Tip: check if you're calling an abstract base class directly.",
    "AssertionError":    "Assertion failed.\n  Tip: check what the assertion is testing; add debugging prints.",
    "ConnectionError":   "Network connection failed.\n  Tip: check host/port; verify network connectivity.",
}

def _build_crash(et, ev, tb_str, frame):
    src_ctx = _source_context(frame.f_code.co_filename, frame.f_lineno)
    t = _t_now()
    return {
        "exc":       et.__name__,
        "msg":       str(ev),
        "tb":        tb_str,
        "t":         t,
        "fn":        frame.f_code.co_name,
        "ln":        frame.f_lineno,
        "file":      os.path.basename(frame.f_code.co_filename),
        "full_file": frame.f_code.co_filename,
        "loc":       _cap_locals(frame),
        "stk":       _cap_stack(frame),
        "sys":       _sys_metrics(),
        "friendly":  _ERROR_TIPS.get(et.__name__, f"An unexpected {et.__name__} occurred."),
        "src_ctx":   src_ctx,
    }

# v6: wrap stdlib excepthook ONCE — never chain
_ORIG_EXCEPTHOOK = sys.__excepthook__

def _make_excepthook():
    def _hook(etype, evalue, etb):
        try:
            tb_str = "".join(_tb_mod.format_tb(etb))
            is_internal = "<frozen" in tb_str or "tc_recorder" in tb_str
            if not is_internal and not _S.get("crash"):
                frame = etb.tb_frame
                _S["crash"] = _build_crash(etype, evalue, tb_str, frame)
        except Exception: pass
        _ORIG_EXCEPTHOOK(etype, evalue, etb)
    return _hook

# ─────────────────────────────────────────────────────────────────────────────
#  sys.monitoring HOOKS (Python 3.12+)
# ─────────────────────────────────────────────────────────────────────────────
_MON_TOOL_ID = None
_SRC_ABS     = None

def _install_monitoring_hooks(src_abs, include_all=False):
    global _MON_TOOL_ID, _SRC_ABS
    if not hasattr(sys, "monitoring"): return False
    _SRC_ABS = src_abs
    try:
        mon = sys.monitoring
        TOOL    = mon.DEBUGGER_ID
        DISABLE = mon.DISABLE
        if _HAVE_C: _C.start(src_abs, DISABLE)

        def _is_tracked(filename):
            """Check if a file should be tracked (supports multi-file mode)."""
            if filename == src_abs: return True
            if not include_all: return False
            abs_f = os.path.abspath(filename)
            for pat in _S.get("exclude_patterns", []):
                if fnmatch.fnmatch(abs_f, pat): return False
            for pat in _S.get("include_patterns", []):
                if fnmatch.fnmatch(abs_f, pat): return True
            return False

        def _mon_call(code, offset):
            cid      = id(code)
            filename = getattr(code, "co_filename", "")
            tracked  = _is_tracked(filename)
            if cid not in _S["c_meta"]:
                _S["c_meta"][cid] = {
                    "name": getattr(code, "co_qualname", getattr(code, "co_name", "?")),
                    "file": filename,
                    "line": getattr(code, "co_firstlineno", 0),
                    "tracked": tracked,
                }
            if not tracked:
                return _C.on_call(code, offset) if _HAVE_C else None
            fn = _S["c_meta"][cid]["name"]
            _S["fn_calls"][fn] = _S["fn_calls"].get(fn, 0) + 1
            # v6: key by frame id, not fn name — solves recursion+exception corruption
            frame_id = id(sys._current_frames().get(threading.main_thread().ident))
            _S["fn_entry"][frame_id] = (fn, time.perf_counter())
            d = _S["rec_depth"].get(fn, 0) + 1
            _S["rec_depth"][fn] = d
            if d > _S["rec_max"].get(fn, 0): _S["rec_max"][fn] = d
            if d > 50: _warn("DEEP_RECURSION", f"'{fn}' depth {d}", _t_now(), var=fn)
            # check repeated calls with same arg fingerprint
            cnt = _S["seen_call_counts"].get(fn, 0) + 1
            _S["seen_call_counts"][fn] = cnt
            if cnt == 50: _warn("REPEATED_CALL", f"'{fn}' called ×{cnt}", _t_now(), var=fn)
            return _C.on_call(code, offset) if _HAVE_C else None

        def _mon_return(code, offset, retval):
            cid      = id(code)
            meta     = _S["c_meta"].get(cid, {})
            if not meta.get("tracked", False):
                return _C.on_return(code, offset, retval) if _HAVE_C else None
            fn = meta.get("name", "?")
            # v6: look up by any matching frame_id for this fn
            frame_id = id(sys._current_frames().get(threading.main_thread().ident))
            entry = _S["fn_entry"].pop(frame_id, None)
            if entry:
                _fn, entry_t = entry
                _S["fn_timings"][_fn].append(time.perf_counter() - entry_t)
            if _S["rec_depth"].get(fn, 0) > 0:
                _S["rec_depth"][fn] -= 1
            if retval is not None and not isinstance(retval, _SKIP_TYPES):
                with _S["lock"]:
                    _S["ret_vals"].append({
                        "t": _t_now(), "fn": fn,
                        "val": _smart_repr(retval),
                        "type": type(retval).__name__,
                    })
            return _C.on_return(code, offset, retval) if _HAVE_C else None

        def _mon_line(code, line_no):
            filename = code.co_filename
            if not _is_tracked(filename):
                return mon.DISABLE
            _S["line_hits"][(filename, line_no)] = \
                _S["line_hits"].get((filename, line_no), 0) + 1
            # line-level variable change detection
            try:
                mid    = threading.main_thread().ident
                frames = sys._current_frames()
                f      = frames.get(mid)
                if not f: return
                target = _find_mine(f) or f
                loc    = target.f_locals
                watch  = _S.get("watch")
                prev   = _S["_line_prev_vars"]
                changed_vars = {}
                for k, v in loc.items():
                    if k[0] == "_": continue
                    if watch and k not in watch: continue
                    try:
                        r = _smart_repr(v)
                        if prev.get(k) != r:
                            changed_vars[k] = r
                            prev[k] = r
                    except Exception: pass
                if changed_vars:
                    snap = _build_snap(target, "LINE", {
                        "changed": list(changed_vars.keys()),
                        "ln": line_no,
                    })
                    _store(snap)
            except Exception: pass

        mon.use_tool_id(TOOL, "tc")
        mon.set_events(TOOL, mon.events.PY_START | mon.events.PY_RETURN | mon.events.LINE)
        mon.register_callback(TOOL, mon.events.PY_START,  _mon_call)
        mon.register_callback(TOOL, mon.events.PY_RETURN, _mon_return)
        mon.register_callback(TOOL, mon.events.LINE,      _mon_line)
        _MON_TOOL_ID = TOOL
        return True
    except Exception as e:
        print(f"[TC] monitoring hook failed: {e} — using settrace", file=sys.stderr)
        return False

def _remove_monitoring_hooks():
    global _MON_TOOL_ID
    if _MON_TOOL_ID is None: return
    try:
        mon = sys.monitoring
        mon.set_events(_MON_TOOL_ID, 0)
        mon.free_tool_id(_MON_TOOL_ID)
    except Exception: pass
    _MON_TOOL_ID = None
    if _HAVE_C:
        try: _C.stop()
        except Exception: pass

# ─────────────────────────────────────────────────────────────────────────────
#  PURE-PYTHON sys.settrace  (fallback for Python < 3.12)
# ─────────────────────────────────────────────────────────────────────────────
def _trace(frame, evt, arg):
    if not _S["on"]: return None
    if threading.current_thread().name == "tc-timer": return _trace
    if not _is_mine(frame): return _trace

    if evt == "line":
        _S["line_hits"][(frame.f_code.co_filename, frame.f_lineno)] += 1

    elif evt == "call":
        fn = frame.f_code.co_name
        _S["fn_calls"][fn] += 1
        # v6: frame-id keying — immune to recursion+exception corruption
        _S["fn_entry"][id(frame)] = (fn, time.perf_counter())
        d = _S["rec_depth"][fn] + 1
        _S["rec_depth"][fn] = d
        if d > _S["rec_max"][fn]: _S["rec_max"][fn] = d
        if d > 50: _warn("DEEP_RECURSION", f"'{fn}' depth {d}", _t_now(), var=fn)
        cnt = _S["seen_call_counts"].get(fn, 0) + 1
        _S["seen_call_counts"][fn] = cnt
        if cnt == 50: _warn("REPEATED_CALL", f"'{fn}' called ×{cnt}", _t_now(), var=fn)

    elif evt == "return":
        fn = frame.f_code.co_name
        entry = _S["fn_entry"].pop(id(frame), None)   # safe: KeyError → None
        if entry:
            _fn, entry_t = entry
            _S["fn_timings"][_fn].append(time.perf_counter() - entry_t)
        if _S["rec_depth"][fn] > 0: _S["rec_depth"][fn] -= 1
        if arg is not None and not isinstance(arg, _SKIP_TYPES):
            with _S["lock"]:
                _S["ret_vals"].append({
                    "t": _t_now(), "fn": fn,
                    "val": _smart_repr(arg),
                    "type": type(arg).__name__,
                })

    elif evt == "exception":
        et, ev2, tb = arg
        tb_str = "".join(_tb_mod.format_tb(tb))
        if "<frozen" not in tb_str and "tc_recorder" not in tb_str:
            uf = _find_mine(frame)
            if uf and not _S.get("crash"):
                _S["crash"] = _build_crash(et, ev2, tb_str, uf)
                _store(_build_snap(uf, "CRASH"))

    return _trace

# ─────────────────────────────────────────────────────────────────────────────
#  TIMER THREAD  (periodic snapshot + stuck-loop detection)
# ─────────────────────────────────────────────────────────────────────────────
_last_snap_state = {}
_stuck_since     = [None]

def _timer_loop():
    while _S["on"]:
        time.sleep(_S["every"])
        if not _S["on"]: break
        try:
            if _HAVE_C: _process_c_events()
            mid = threading.main_thread().ident
            f   = sys._current_frames().get(mid)
            if not f: continue
            target = _find_mine(f) or f
            if "tc_recorder" in target.f_code.co_filename: continue
            snap = _build_snap(target, "TIMER")
            _store(snap)
            try: _eval_conditions(snap)
            except Exception: pass
            # stuck-loop detection
            cur_ln   = snap["ln"]
            cur_vars = {k: v["r"] for k, v in snap["loc"].items()}
            prev     = _last_snap_state.get("state")
            if prev and prev["ln"] == cur_ln and prev["vars"] == cur_vars:
                if _stuck_since[0] is None: _stuck_since[0] = snap["t"]
                elif (snap["t"] - _stuck_since[0]) > (3 * _S["every"]):
                    _warn("STUCK_LOOP", f"Stuck at {snap['fn']}():{cur_ln} for {snap['t']-_stuck_since[0]:.0f}s", snap["t"])
            else:
                _stuck_since[0] = None
            _last_snap_state["state"] = {"ln": cur_ln, "vars": cur_vars}
        except Exception: pass

# ─────────────────────────────────────────────────────────────────────────────
#  STACK SAMPLER  (mode="sample" — zero per-call overhead)
# ─────────────────────────────────────────────────────────────────────────────
_SAMPLE_COUNTS  = defaultdict(int)   # {(file,fn,ln): count}
_SAMPLE_RUNNING = [False]

def _sampler_loop():
    interval = _S.get("_sampler_interval", 0.001)
    while _SAMPLE_RUNNING[0] and _S["on"]:
        time.sleep(interval)
        try:
            frames = sys._current_frames()
            for tid, f in frames.items():
                if tid == threading.current_thread().ident: continue
                while f:
                    key = (os.path.basename(f.f_code.co_filename),
                           f.f_code.co_name, f.f_lineno)
                    _SAMPLE_COUNTS[key] += 1
                    f = f.f_back
        except Exception: pass

# ─────────────────────────────────────────────────────────────────────────────
#  C EVENT PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def _process_c_events():
    if not _HAVE_C: return
    try:
        pending = _C.pending() if hasattr(_C, "pending") else 0
        buf_sz  = _C.buf_size() if hasattr(_C, "buf_size") else 2_000_000
        if pending >= buf_sz * 0.9:
            _warn("BUFFER_OVERFLOW",
                  f"C buffer {pending}/{buf_sz} ({pending/buf_sz*100:.0f}% full) — timing data may be incomplete",
                  _t_now(), level="WARN")
        mon_active = (_MON_TOOL_ID is not None)
        for ts_ns, cid, kind in _C.drain():
            if mon_active: continue
            meta = _S["c_meta"].get(cid)
            if not meta: continue
            fn = meta.get("name", "?")
            if kind == 0:
                _S["fn_calls"][fn] += 1
            elif kind == 1:
                pass  # handled in _mon_return via frame-id keying
    except Exception: pass

# ─────────────────────────────────────────────────────────────────────────────
#  LIVE STATUS LINE
# ─────────────────────────────────────────────────────────────────────────────
def _print_live(snap):
    try:
        now = snap["t"]
        if now - _S.get("live_last", 0) < 0.5: return
        _S["live_last"] = now
        m    = snap.get("sys", {})
        mets = snap.get("mets", {})
        ram  = m.get("ram_mb", "?")
        cpu  = m.get("cpu_pct", "?")
        gpu  = m.get("gpu_mb")
        fn   = snap.get("fn", "?")
        met_str = "  ".join(f"{k}={v}" for k, v in list(mets.items())[:3])
        gpu_str = f"  GPU={gpu}MB" if gpu else ""
        line = (f"\r[TC #{snap['i']:05d}  t={now:.2f}s  {fn}()"
                f"  RAM={ram}MB  CPU={cpu}%{gpu_str}"
                + (f"  {met_str}" if met_str else "") + "]   ")
        sys.stderr.write(line); sys.stderr.flush()
    except Exception: pass

# ─────────────────────────────────────────────────────────────────────────────
#  FLUSH + SAVE  (v6: flush outside lock, atomic rename, fsync)
# ─────────────────────────────────────────────────────────────────────────────
def _flush_chunk():
    out = _S.get("outfile")
    if not out or _S.get("saved"): return
    try:
        with _S["lock"]:
            batch = list(_S["snaps"])[-_S["flush_every"]:]
        chunk = out + ".chunk"
        tmp   = chunk + ".tmp"
        with open(tmp, "ab") as f:
            for s in batch:
                blob = pickle.dumps(s, 2)
                f.write(struct.pack(">I", len(blob)) + blob)
            f.flush()
            try: os.fsync(f.fileno())
            except Exception: pass
        os.replace(tmp, chunk)   # atomic
    except Exception: pass

def _merge_chunks(outfile):
    chunk = outfile + ".chunk"
    if not os.path.exists(chunk): return
    try:
        extra = []
        with open(chunk, "rb") as f:
            while True:
                hdr = f.read(4)
                if len(hdr) < 4: break
                size = struct.unpack(">I", hdr)[0]
                blob = f.read(size)
                if len(blob) < size: break
                try: extra.append(pickle.loads(blob))
                except Exception: pass
        existing = {s["i"] for s in _S["snaps"]}
        for s in extra:
            if s["i"] not in existing: _S["snaps"].append(s)
        snaps_sorted = sorted(_S["snaps"], key=lambda s: s["t"])
        _S["snaps"] = deque(snaps_sorted, maxlen=_S["max"])
        try: os.remove(chunk)
        except Exception: pass
    except Exception: pass

def _save():
    if _S.get("saved") or not _S.get("outfile"): return
    if not _S["snaps"] and not _S["crash"]: return
    _S["saved"] = True
    outfile = _S["outfile"]
    _merge_chunks(outfile)
    snaps_list = list(_S["snaps"])
    all_vars   = set()
    for s in snaps_list: all_vars.update(s.get("loc", {}).keys())

    # sampling mode data
    sample_data = {}
    if _S.get("mode") == "sample" and _SAMPLE_COUNTS:
        sample_data = dict(_SAMPLE_COUNTS)

    payload = {
        "ver":          __version__,
        "backend":      _c_info(),
        "mode":         _S.get("mode", "full"),
        "session_id":   _S.get("session_id"),
        "src":          _S.get("src", ""),
        "argv":         sys.argv[:],
        "python":       sys.version,
        "platform":     sys.platform,
        "pid":          _PID,
        "t0_wall":      _S.get("t0_wall", 0),
        "saved_at":     datetime.now().isoformat(),
        "duration":     round(_elapsed(), 6),
        "n":            len(snaps_list),
        "all_vars":     sorted(v for v in all_vars if not v.startswith("__expr_")),
        "crash":        _S["crash"],
        "snaps":        snaps_list,
        "ret_vals":     list(_S["ret_vals"]),
        "line_hits":    {str(k): v for k, v in _S["line_hits"].items()},
        "fn_calls":     dict(_S["fn_calls"]),
        "fn_timings":   {k: sorted(v) for k, v in _S["fn_timings"].items()},
        "rec_max":      dict(_S["rec_max"]),
        "final_metrics":dict(_S["metrics"]),
        "warnings":     _S["warnings"],
        "ram_peak":     _S.get("ram_peak", 0),
        "fd_peak":      _S.get("fd_peak", 0),
        "block_timings":dict(_S.get("block_timings", {})),
        "watch_exprs":  dict(_S.get("watch_exprs", {})),
        "sample_counts":sample_data,
        "include_files":_S.get("include_patterns", []),
    }
    try:
        with gzip.open(outfile, "wb", compresslevel=1) as f:   # level 1: 10× faster save
            pickle.dump(payload, f, protocol=4)
        kb  = os.path.getsize(outfile) / 1024
        wc  = len(_S["warnings"])
        if _S.get("live_enabled"): sys.stderr.write("\r" + " " * 100 + "\r")
        print(f"\n[TC] ✓ {bold('saved')}  {len(snaps_list)} snaps"
              f"  {kb:.0f}KB  {len(all_vars)} vars"
              f"  crash={red('YES ⚠') if _S['crash'] else green('no')}"
              f"  warn={wc}"
              f"  → {cyan(outfile)}")
    except Exception as e:
        print(f"[TC] save error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  CONTEXT MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class _RecordContext:
    def __enter__(self): return self
    def __exit__(self, *_): stop(); return False

# ─────────────────────────────────────────────────────────────────────────────
#  watch_condition support
# ─────────────────────────────────────────────────────────────────────────────
_CONDITIONS = []

def _eval_conditions(snap):
    if not _CONDITIONS: return
    loc  = snap.get("loc", {})
    mets = snap.get("mets", {})
    t    = snap.get("t", 0)
    ctx  = {}
    for k, vd in loc.items():
        try: ctx[k] = pickle.loads(vd.get("p", b""))
        except Exception: ctx[k] = vd.get("r")
    ctx.update(mets)
    for cond in _CONDITIONS:
        if cond["snap_mode"] == "once" and cond["fired"]: continue
        try:
            val = bool(eval(cond["expr"], {"__builtins__": {}}, ctx))
        except Exception: continue
        prev = cond["last_val"]; cond["last_val"] = val
        should = (
            (cond["snap_mode"] == "on_change" and val != prev and val) or
            (cond["snap_mode"] == "on_true" and val) or
            (cond["snap_mode"] == "once" and val)
        )
        if should:
            cond["fired"] = True; cond["fire_count"] += 1
            _warn(f"CONDITION:{cond['expr'][:30]}", f"watch_condition fired: {cond['expr']}", t, level=cond["level"])
            try:
                mid = threading.main_thread().ident
                frames = sys._current_frames()
                f2 = frames.get(mid)
                if f2:
                    target = _find_mine(f2) or f2
                    _S["current_tags"] = cond["tags"]
                    _store(_build_snap(target, "CONDITION", {
                        "label": cond["label"],
                        "condition_expr": cond["expr"],
                        "fire_count": cond["fire_count"],
                    }))
            except Exception: pass

# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API — RECORDING
# ─────────────────────────────────────────────────────────────────────────────

_WATCHED_VARS  = {}
_WATCH_ACTIVE  = False

def record(every=1.0, watch=None, output=None, max_snaps=10000,
           flush_every=50, warn=True, tags=None, live=False,
           include_files=None, exclude_files=None, mode="full",
           sample_rate_hz=1000):
    """
    Start recording the current script.

    Args:
        every          : seconds between timer snapshots (default 1.0)
        watch          : list of variable names to capture (None = all)
        output         : .tc output path (auto-named if None)
        max_snaps      : in-memory snapshot cap (default 10000, deque — O(1))
        flush_every    : crash-safe flush interval (0 = off)
        warn           : enable automatic warnings
        tags           : tags applied to every snapshot
        live           : print live status line to stderr
        include_files  : list of glob patterns for multi-file tracking
                         e.g. ["src/*.py", "lib/utils.py"]
        exclude_files  : glob patterns to exclude
        mode           : "full" (default) or "sample" (zero per-call overhead)
        sample_rate_hz : samples/sec in sample mode (default 1000)

    Returns context manager: with tc.record(): ...

    Example:
        tc.record(every=0.1, include_files=["src/*.py"])
        run_my_code()
        tc.stop()
    """
    global _last_snap_state, _stuck_since, _MON_TOOL_ID
    global _WATCHED_VARS, _WATCH_ACTIVE, _SAMPLE_COUNTS, _SAMPLE_RUNNING

    # v6: COMPLETE global state reset — bug A1 fixed
    _last_snap_state.clear()
    _stuck_since[0]   = None
    _IS_MINE_CACHE.clear()
    _SRC_CACHE.clear()
    _SAMPLE_COUNTS.clear()
    _CONDITIONS.clear()
    _WATCHED_VARS.clear()
    _WATCH_ACTIVE = False

    caller = inspect.stack()[1]
    src    = os.path.abspath(caller.filename)

    # Handle REPL / Jupyter / exec contexts
    if src.startswith("<") or not os.path.exists(src):
        src = os.path.abspath(".")

    if output is None:
        base   = os.path.splitext(os.path.basename(src))[0]
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"{base}_{ts}.tc"

    if os.path.exists(output + ".chunk"):
        try: os.remove(output + ".chunk")
        except Exception: pass

    m0 = _sys_metrics()

    import uuid
    new_state = {
        "on": True, "snaps": deque(maxlen=max_snaps), "every": every,
        "watch": watch, "max": max_snaps, "outfile": output,
        "t0_perf": time.perf_counter(), "t0_wall": time.time(),
        "n": 0, "crash": None, "src": src, "saved": False, "warn": warn,
        "lock": threading.Lock(), "timer": None, "mode": mode,
        "metrics": {}, "metric_hist": defaultdict(list),
        "line_hits": defaultdict(int), "fn_calls": defaultdict(int),
        "fn_timings": defaultdict(list), "fn_entry": {},
        "ret_vals": deque(maxlen=2000),
        "rec_depth": defaultdict(int), "rec_max": defaultdict(int),
        "var_last": {}, "var_unchanged_since": {}, "var_sizes": defaultdict(list),
        "seen_call_counts": defaultdict(int),
        "warnings": [], "warned_keys": set(),
        "ram_baseline": m0.get("ram_mb", 0), "ram_peak": m0.get("ram_mb", 0),
        "fd_baseline": m0.get("open_fds", 0), "fd_peak": m0.get("open_fds", 0),
        "flush_every": flush_every, "flush_count": 0, "c_meta": {},
        "_line_prev_vars": {}, "_watch_vars": {}, "_watch_active": False,
        "_watch_every_snap": False, "current_tags": list(tags or []),
        "watch_exprs": {}, "checkpoints": {}, "live_enabled": live,
        "live_last": 0.0, "block_timings": defaultdict(list),
        "session_id": str(uuid.uuid4())[:8],
        "include_patterns": [os.path.abspath(p) for p in (include_files or [])],
        "exclude_patterns": (exclude_files or ["*/tc_recorder*", "*/site-packages/*"]),
        "_async_task_ids": {},
        "_sampler_interval": 1.0 / max(1, sample_rate_hz),
    }
    _S.clear()
    _S.update(new_state)

    # detect GPU backend once
    _detect_gpu()

    include_all = bool(include_files)
    used_c = False

    if mode == "full":
        used_c = _install_monitoring_hooks(src, include_all)
        if not used_c:
            sys.settrace(_trace)
            threading.settrace(_trace)

    # timer thread
    t = threading.Thread(target=_timer_loop, daemon=True, name="tc-timer")
    _S["timer"] = t; t.start()

    # sampling thread
    if mode == "sample":
        _SAMPLE_RUNNING[0] = True
        st = threading.Thread(target=_sampler_loop, daemon=True, name="tc-sampler")
        st.start()

    import atexit
    atexit.register(_save)
    sys.excepthook = _make_excepthook()   # v6: wraps stdlib ONCE, never chains

    backend = ("C+monitoring" if (used_c and _HAVE_C) else
               "monitoring"   if used_c else
               "sample"       if mode == "sample" else
               "pure-Python")

    # initial snap
    try:
        caller_frame = sys._getframe(1)
        _store(_build_snap(caller_frame, "SNAP:record_start",
                           {"label": "_tc_init", "tags": ["_internal"]}))
    except Exception: pass

    print(f"[TC] {bold('recording')} → {cyan(output)}"
          f"  every={every}s  backend={backend}"
          f"  session={_S['session_id']}"
          + (f"  tracking={len(_S['include_patterns'])+1} files" if include_all else ""))
    return _RecordContext()


def snap(label=None, tags=None):
    """Force an immediate snapshot of current state."""
    if not _S.get("on"): return
    if tags: _S["current_tags"] = list(tags)
    f = sys._getframe(1)
    extra = {"label": label} if label else {}
    _store(_build_snap(f, f"SNAP:{label or ''}", extra))
    print(f"[TC] snap #{_S['n']}" + (f" — {dim(label)}" if label else ""))


def tag(*tags):
    """Attach tags to the NEXT snapshot produced."""
    _S["current_tags"] = list(tags)


def metric(name, value):
    """Record a named metric. Carried in every subsequent snapshot."""
    _S["metrics"][name] = value
    if not _S.get("on"): return
    t = _t_now()
    if _S.get("warn") and isinstance(value, (int, float)):
        _check_numeric(name, value, t)
        _check_metric_plateau(name, value, t)


def assert_var(name, condition_fn, label=None):
    """
    Live assertion. If condition_fn(value) is False: auto-snap + warn.
    Zero overhead when condition passes.

    Example:
        tc.assert_var("loss", lambda v: v < 10.0, "loss explosion!")
    """
    if not _S.get("on"): return
    f   = sys._getframe(1)
    val = f.f_locals.get(name)
    try:
        ok = condition_fn(val)
    except Exception as e:
        ok = False; label = label or f"assert_var({name}) error: {e}"
    if not ok:
        t   = _t_now()
        msg = label or f"assert_var('{name}') failed: {_safe_repr(val)}"
        _warn("ASSERTION_FAILED", msg, t, level="ERROR", var=name)
        _store(_build_snap(f, "ASSERT_FAIL", {
            "label": f"ASSERT_FAIL:{name}",
            "assert_name": name, "assert_val": _safe_repr(val),
        }))


def watch_expr(name, expr_str):
    """Evaluate an expression each snapshot and record it.

    Example:
        tc.watch_expr("loss_delta", "loss - prev_loss")
        tc.watch_expr("accuracy_pct", "correct / total * 100")
    """
    _S["watch_exprs"][name] = expr_str


def checkpoint(name, max_elapsed_s=None):
    """Named timing checkpoint. Call twice with same name to measure interval.

    Example:
        tc.checkpoint("data_load")
        load()
        tc.checkpoint("data_load")   # warns if > max_elapsed_s
    """
    if name not in _S["checkpoints"]:
        _S["checkpoints"][name] = (time.perf_counter(), max_elapsed_s)
        print(f"[TC] ▶ checkpoint '{name}'")
    else:
        start, limit = _S["checkpoints"].pop(name)
        elapsed = round(time.perf_counter() - start, 6)
        _S["block_timings"][f"checkpoint:{name}"].append(elapsed)
        print(f"[TC] ■ checkpoint '{name}' → {elapsed*1000:.2f}ms")
        if limit and elapsed > limit:
            _warn("CHECKPOINT_SLOW", f"'{name}' took {elapsed:.3f}s > limit {limit}s", _t_now(), var=name)


@contextlib.contextmanager
def profile_block(name):
    """Context manager: time a named block of code.

    Example:
        with tc.profile_block("preprocessing"):
            df = preprocess(raw)
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        _S["block_timings"][f"block:{name}"].append(round(elapsed, 9))
        print(f"[TC] block '{name}' → {elapsed*1000:.3f}ms")


def stop():
    """Stop recording and save the .tc file."""
    global _SAMPLE_RUNNING
    # final snap before stopping
    try:
        mid = threading.main_thread().ident
        f   = sys._current_frames().get(mid)
        if f:
            target = _find_mine(f) or f
            if "tc_recorder" not in target.f_code.co_filename:
                _store(_build_snap(target, "SNAP:record_stop",
                                   {"label": "_tc_final", "tags": ["_internal"]}))
    except Exception: pass

    _S["on"] = False
    _SAMPLE_RUNNING[0] = False

    if _MON_TOOL_ID is not None:
        _remove_monitoring_hooks()
    else:
        try: sys.settrace(None); threading.settrace(None)
        except Exception: pass

    t = _S.get("timer")
    if t and t.is_alive(): t.join(timeout=2.0)
    if _HAVE_C: _process_c_events()
    _save()


def reset():
    """Clear all data without stopping recording. Use between epochs."""
    with _S["lock"]:
        _S["snaps"].clear(); _S["n"] = 0
        _S["metrics"].clear(); _S["warnings"].clear()
        _S["ret_vals"].clear(); _S["fn_calls"].clear()
        _S["fn_timings"].clear(); _S["warned_keys"].clear()
        _SAMPLE_COUNTS.clear()
    _last_snap_state.clear()
    _stuck_since[0] = None
    print("[TC] state reset")


def watch(*var_names, every_snap=False):
    """Live-print variable changes as they happen.

    Example:
        tc.watch("loss", "accuracy")
    """
    global _WATCH_ACTIVE
    _WATCH_ACTIVE = True
    for name in var_names:
        _WATCHED_VARS[name] = None
    _S["_watch_vars"]      = _WATCHED_VARS
    _S["_watch_every_snap"]= every_snap
    _S["_watch_active"]    = True
    print(f"[TC] watching: {list(var_names)}", file=sys.stderr)


def _check_watch(snap):
    if not _S.get("_watch_active"): return
    watch_vars = _S.get("_watch_vars", {})
    every_snap = _S.get("_watch_every_snap", False)
    loc  = snap.get("loc", {})
    mets = snap.get("mets", {})
    t    = snap.get("t", 0)
    for name in list(watch_vars.keys()):
        if name in loc:
            new_r = loc[name].get("r", ""); typ = loc[name].get("t", "")
        elif name in mets:
            new_r = str(mets[name]); typ = type(mets[name]).__name__
        else:
            continue
        old_r   = watch_vars.get(name)
        changed = (old_r != new_r)
        if changed or every_snap:
            delta_s = ""
            if old_r is not None and changed:
                try:
                    d = float(new_r.strip("'\"")) - float(old_r.strip("'\""))
                    delta_s = f"  (Δ {d:+.4g})"
                except Exception: pass
            def _clean(r):
                try: return f"{float(r):.6g}"
                except Exception: return r
            arrow = f"{_clean(old_r)} → " if old_r is not None else ""
            print(f"[TC WATCH] t={t:.4f}s  {cyan(name)}:  {arrow}{green(_clean(new_r))}{yellow(delta_s)}",
                  file=sys.stderr, flush=True)
            watch_vars[name] = new_r


def trace(_fn=None, *, label=None, watch_args=True, watch_return=True,
          snap_on_call=True, snap_on_return=True, warn_slow_ms=None):
    """Decorator: auto-snapshot on every call + return.

    Examples:
        @tc.trace
        def train_step(batch): ...

        @tc.trace(warn_slow_ms=500)
        def load_data(path): ...
    """
    import functools

    def decorator(fn):
        fn_name  = label or fn.__qualname__
        call_ctr = [0]

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _S.get("on"):
                return fn(*args, **kwargs)
            call_ctr[0] += 1
            call_id = call_ctr[0]
            t_entry = time.perf_counter()
            args_repr = {}
            if watch_args:
                try:
                    sig   = inspect.signature(fn)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    for k, v in bound.arguments.items():
                        if not isinstance(v, _SKIP_TYPES):
                            args_repr[k] = _safe_repr(v, 60)
                except Exception:
                    for i, a in enumerate(args[:4]):
                        args_repr[f"arg{i}"] = _safe_repr(a, 60)
            if snap_on_call:
                f = sys._getframe(1)
                _store(_build_snap(f, f"TRACE:call:{fn_name}", {
                    "label":      f"→ {fn_name}() #{call_id}",
                    "tags":       ["trace", "call"],
                    "trace_fn":   fn_name,
                    "trace_call": call_id,
                    "trace_args": args_repr,
                }))
            exc_info = None; result = None
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                exc_info = e; raise
            finally:
                elapsed_ms = (time.perf_counter() - t_entry) * 1000
                if warn_slow_ms and elapsed_ms > warn_slow_ms:
                    _warn(f"TRACE_SLOW:{fn_name}:{call_id}",
                          f"{fn_name}() #{call_id} took {elapsed_ms:.1f}ms (limit={warn_slow_ms}ms)",
                          _t_now(), level="WARN", var=fn_name)
                if snap_on_return:
                    f = sys._getframe(1)
                    ret_repr = (_safe_repr(result, 80)
                                if watch_return and result is not None
                                and not isinstance(result, _SKIP_TYPES) else None)
                    _store(_build_snap(f, f"TRACE:return:{fn_name}", {
                        "label":        f"← {fn_name}() #{call_id} {elapsed_ms:.1f}ms",
                        "tags":         ["trace", "return"] + (["trace_error"] if exc_info else []),
                        "trace_fn":     fn_name,
                        "trace_call":   call_id,
                        "trace_ms":     round(elapsed_ms, 3),
                        "trace_return": ret_repr,
                        "trace_error":  str(exc_info) if exc_info else None,
                    }))
            return result
        wrapper.__tc_traced__ = True
        wrapper.__tc_fn_name__ = fn_name
        return wrapper

    return decorator(_fn) if _fn is not None else decorator


def breakpoint(label=None, *, condition=None, tags=None, verbose=True):
    """Soft breakpoint: snapshot + continue, NEVER pauses execution.

    Example:
        tc.breakpoint("after_transform")
        tc.breakpoint(condition=lambda v: v['loss'] > 5)
    """
    if not _S.get("on"): return
    try:
        f    = sys._getframe(1)
        file = os.path.basename(f.f_code.co_filename)
        line = f.f_lineno
        fn   = f.f_code.co_name
        if condition is not None:
            try:
                if not condition(dict(f.f_locals)): return
            except Exception as e:
                _warn("BP_CONDITION_ERROR", f"breakpoint condition raised: {e}", _t_now())
        auto_label = label or f"bp:{file}:{line}"
        _S["current_tags"] = ["breakpoint"] + list(tags or [])
        snap_obj = _build_snap(f, "BREAKPOINT", {
            "label": auto_label, "bp_file": file, "bp_line": line, "bp_fn": fn,
        })
        _store(snap_obj)
        if verbose:
            t = snap_obj.get("t", 0); n = snap_obj.get("i", "?")
            nv = len(snap_obj.get("loc", {}))
            print(f"[TC BP] t={t:.4f}s  #{n}  {fn}() {file}:{line}"
                  + (f"  — {label}" if label else "") + f"  ({nv} vars)",
                  file=sys.stderr, flush=True)
    except Exception: pass


def watch_condition(expr, label=None, *, snap_mode="on_change", tags=None, level="WARN"):
    """Register a condition expression evaluated on every timer tick.

    Example:
        tc.watch_condition("loss > 5.0", label="loss_spike")
        tc.watch_condition("len(results) == 0", label="empty", snap_mode="once")
    """
    entry = {
        "expr": expr, "label": label or f"condition: {expr[:40]}",
        "snap_mode": snap_mode, "tags": ["watch_condition"] + list(tags or []),
        "level": level, "last_val": None, "fired": False, "fire_count": 0,
    }
    _CONDITIONS.append(entry)
    print(f"[TC] watching condition: {repr(expr)}", file=sys.stderr)
    return entry


# ─────────────────────────────────────────────────────────────────────────────
#  BASELINE / DIFF (CI/CD performance regression tracking)
# ─────────────────────────────────────────────────────────────────────────────
import json as _json
_BASELINE_FILE = os.path.join(os.path.expanduser("~"), ".tc_baseline.json")


def baseline(path=None, name="default", *, project=None):
    """Save a .tc recording as the performance baseline for future diffs."""
    d    = load(path)
    src  = os.path.basename(d.get("src", "?"))
    proj = project or src
    record_data = {
        "name": name, "project": proj, "src": src,
        "saved_at": d.get("saved_at", ""), "session_id": d.get("session_id", ""),
        "duration": d.get("duration", 0), "n_snaps": d.get("n", 0),
        "ram_peak": d.get("ram_peak", 0),
        "fn_timings": {k: {
            "calls": len(v),
            "avg_ms": round(sum(t*1000 for t in v)/len(v), 4) if v else 0,
            "p95_ms": sorted(t*1000 for t in v)[max(0,int(len(v)*.95)-1)] if v else 0,
        } for k, v in d.get("fn_timings", {}).items()},
        "warnings": len(d.get("warnings", [])), "crash": bool(d.get("crash")),
    }
    try:
        with open(_BASELINE_FILE) as f: store = _json.load(f)
    except Exception: store = {}
    store[f"{proj}::{name}"] = record_data
    with open(_BASELINE_FILE, "w") as f: _json.dump(store, f, indent=2)
    print(f"[TC] baseline saved → {proj}::{name}  ({src}  {d['duration']:.3f}s)")
    return record_data


def diff_baseline(path=None, name="default", *, project=None, threshold_pct=10.0):
    """Compare a .tc recording against a saved baseline. CI-friendly."""
    d    = load(path)
    src  = os.path.basename(d.get("src", "?"))
    proj = project or src
    try:
        with open(_BASELINE_FILE) as f: store = _json.load(f)
    except Exception:
        print("[TC] No baseline found. Run tc.baseline() first."); return {}
    key = f"{proj}::{name}"
    if key not in store:
        candidates = [k for k in store if k.startswith(proj+"::")]
        key = candidates[0] if candidates else (list(store.keys())[-1] if store else None)
        if not key: print("[TC] No baseline found."); return {}
        print(f"[TC] Using baseline: {key}")
    base = store[key]
    cur_ft = {}
    for fn, times in d.get("fn_timings", {}).items():
        if not times: continue
        ms = [t*1000 for t in times]
        cur_ft[fn] = {"avg_ms": sum(ms)/len(ms), "calls": len(ms)}
    base_ft     = base.get("fn_timings", {})
    regressions = []; improvements = []
    for fn in sorted(set(list(cur_ft.keys()) + list(base_ft.keys()))):
        ca = cur_ft.get(fn, {}).get("avg_ms"); cb = base_ft.get(fn, {}).get("avg_ms")
        if ca is None or cb is None or cb == 0: continue
        pct = (ca - cb) / cb * 100
        if abs(pct) < threshold_pct: continue
        (regressions if pct > 0 else improvements).append((fn, round(pct,1), cb, ca))
    regressions.sort(key=lambda x: -x[1]); improvements.sort(key=lambda x: x[1])
    verdict = "REGRESSION" if regressions else ("IMPROVEMENT" if improvements else "NEUTRAL")
    W = 65
    icons = {"REGRESSION": red("🔴 REGRESSION"), "IMPROVEMENT": green("🟢 IMPROVEMENT"), "NEUTRAL": yellow("🟡 NEUTRAL")}
    print(f"\n{'═'*W}\n  tc diff — {src}  vs  [{key}]\n{'═'*W}")
    print(f"  verdict  : {icons[verdict]}")
    print(f"  duration : {base['duration']:.3f}s → {d.get('duration',0):.3f}s")
    if regressions:
        print(f"\n  {red('REGRESSIONS')} (>{threshold_pct:.0f}% slower):")
        for fn, pct, old, new in regressions[:10]:
            bar = red("█" * min(20, int(abs(pct)/5)))
            print(f"    {fn[:30]:<30}  {old:.2f}→{new:.2f}ms  ({pct:+.1f}%)  {bar}")
    if improvements:
        print(f"\n  {green('IMPROVEMENTS')} (>{threshold_pct:.0f}% faster):")
        for fn, pct, old, new in improvements[:10]:
            bar = green("░" * min(20, int(abs(pct)/5)))
            print(f"    {fn[:30]:<30}  {old:.2f}→{new:.2f}ms  ({pct:+.1f}%)  {bar}")
    print(f"{'═'*W}\n")
    return {"verdict": verdict, "regressions": regressions, "improvements": improvements}


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API — ANALYSIS (load + query)
# ─────────────────────────────────────────────────────────────────────────────

def load(path=None):
    """Load a .tc file. Auto-loads most recent if path is None."""
    if path is None:
        import glob
        files = sorted(glob.glob("*.tc"), key=os.path.getmtime, reverse=True)
        if not files: raise FileNotFoundError("No .tc files found in current directory")
        path = files[0]
        print(f"[TC] loading: {cyan(path)}")
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def summary(path=None):
    """One-line human-readable summary."""
    d     = load(path)
    crash = red("💥 CRASH") if d.get("crash") else green("✓ clean")
    warns = len(d.get("warnings", []))
    fn_top= sorted(d.get("fn_calls",{}).items(), key=lambda x:-x[1])[:3]
    fn_str= ", ".join(f"{fn}×{c}" for fn,c in fn_top)
    s = (f"[{crash}] {bold(os.path.basename(d.get('src','?')))} | "
         f"{d.get('n',0)} snaps | {d.get('duration',0):.3f}s | "
         f"{len(d.get('all_vars',[]))} vars | {warns} warn"
         + (f" | {fn_str}" if fn_str else ""))
    print(s); return s


def history(path, var_name):
    """Full value history of a variable: [(elapsed_s, value), ...]"""
    d = load(path); result = []
    for s in d.get("snaps", []):
        if var_name in s.get("loc", {}):
            vd = s["loc"][var_name]
            try:    val = pickle.loads(vd["p"]) if vd.get("p") else vd.get("r")
            except: val = vd.get("r")
            result.append((s["t"], val))
    return result


def diff(path, var_name):
    """Where a variable changed: [(t, old_repr, new_repr), ...]"""
    d = load(path); prev = None; out = []
    for s in d.get("snaps", []):
        if var_name in s.get("loc", {}):
            cur = s["loc"][var_name].get("r", "")
            if prev is not None and cur != prev: out.append((s["t"], prev, cur))
            prev = cur
    return out


def hotlines(path, top_n=20):
    """Most-executed source lines."""
    d = load(path); items = []
    for k, v in d.get("line_hits", {}).items():
        try:
            if isinstance(k, str): k = eval(k)
            items.append((v, k[0], k[1]))
        except Exception: pass
    return sorted(items, reverse=True)[:top_n]


def timings(path):
    """Per-function timing stats."""
    d = load(path); out = {}
    for fn, times in d.get("fn_timings", {}).items():
        if not times: continue
        ms = sorted(t*1000 for t in times); n = len(ms)
        out[fn] = {
            "calls": n, "total_ms": round(sum(ms), 4),
            "avg_ms": round(sum(ms)/n, 4),
            "p50_ms": round(ms[n//2], 4),
            "p95_ms": round(ms[max(0,int(n*.95)-1)], 4),
            "p99_ms": round(ms[max(0,int(n*.99)-1)], 4),
            "min_ms": round(ms[0], 4), "max_ms": round(ms[-1], 4),
            "std_ms": round(math.sqrt(sum((x-sum(ms)/n)**2 for x in ms)/n), 4) if n>1 else 0,
        }
    return dict(sorted(out.items(), key=lambda x: -x[1]["total_ms"]))


def slowest(path, top_n=10):
    """Top-N slowest individual function calls."""
    d = load(path); items = []
    for fn, times in d.get("fn_timings", {}).items():
        for dur in times: items.append((dur*1000, fn))
    items.sort(reverse=True)
    return [(round(ms,4), fn) for ms, fn in items[:top_n]]


def rate(path):
    """Calls-per-second for each function."""
    d = load(path); dur = d.get("duration",1) or 1
    return {fn: round(c/dur,2) for fn,c in d.get("fn_calls",{}).items()}


def anomalies(path, metric_name, z_threshold=2.5):
    """Detect anomalous metric values using z-score."""
    d = load(path); points = []
    for s in d.get("snaps", []):
        v = s.get("mets",{}).get(metric_name)
        if v is not None and isinstance(v,(int,float)) and not math.isnan(v):
            points.append((s["t"], v))
    if len(points) < 3: return []
    vals = [v for _,v in points]
    mu = sum(vals)/len(vals)
    sigma = math.sqrt(sum((x-mu)**2 for x in vals)/len(vals)) or 1e-9
    return sorted([(t,v,round(abs((v-mu)/sigma),2)) for t,v in points
                   if abs((v-mu)/sigma) >= z_threshold], key=lambda x: -x[2])


def search(path, query, fields=("loc","label","tags","evt","fn","mets")):
    """Full-text search across snapshots."""
    d = load(path); q = query.lower().strip(); out = []
    if not q: return out
    for s in d.get("snaps",[]):
        hit = False
        if "loc" in fields:
            for k,vd in s.get("loc",{}).items():
                if q in k.lower() or q in str(vd.get("r","")).lower():
                    hit = True; break
        if not hit:
            for field in ["label","evt","fn"]:
                if field in fields and q in str(s.get(field,"")).lower():
                    hit = True; break
        if not hit and "tags" in fields:
            if any(q in str(tg).lower() for tg in s.get("tags",[])): hit = True
        if not hit and "mets" in fields:
            if any(q in str(k).lower() or q in str(v).lower() for k,v in s.get("mets",{}).items()):
                hit = True
        if hit: out.append(s)
    return out


def since(path, elapsed_s):
    """All snapshots after elapsed_s seconds."""
    return [s for s in load(path).get("snaps",[]) if s.get("t",0) >= elapsed_s]


def replay(path):
    """Iterate snapshots in time order."""
    for s in sorted(load(path).get("snaps",[]), key=lambda x: x.get("t",0)):
        yield s


def memory_map(path):
    """Variable memory audit: {var: {avg_bytes, max_bytes, n_seen}}"""
    d = load(path); sizes = defaultdict(list)
    for s in d.get("snaps",[]):
        for k,v in s.get("loc",{}).items():
            sz = v.get("size",0)
            if sz: sizes[k].append(sz)
    out = {}
    for k,vals in sizes.items():
        out[k] = {"avg_bytes": round(sum(vals)/len(vals)), "max_bytes": max(vals), "n_seen": len(vals)}
    return dict(sorted(out.items(), key=lambda x: -x[1]["max_bytes"]))


def export_json(path, out=None):
    """Export .tc to JSON."""
    d = load(path); out = out or path.replace(".tc",".json")
    def _j(obj):
        if isinstance(obj,(str,int,float,bool,type(None))): return obj
        if isinstance(obj,bytes):
            try: return repr(pickle.loads(obj))
            except: return f"<bytes {len(obj)}>"
        if isinstance(obj,dict): return {str(k):_j(v) for k,v in obj.items()}
        if isinstance(obj,(list,tuple)): return [_j(x) for x in obj]
        return repr(obj)
    tstats = timings(path)
    payload = {
        "meta": {k:d.get(k) for k in ["ver","backend","mode","session_id","src","python","platform","pid","duration","n","all_vars","ram_peak","saved_at"]},
        "crash": _j(d.get("crash")), "warnings": _j(d.get("warnings",[])),
        "fn_calls": d.get("fn_calls",{}), "timings": tstats,
        "block_timings": {k:{"calls":len(v),"total_ms":round(sum(v)*1000,2),"avg_ms":round(sum(v)/len(v)*1000,2)} for k,v in d.get("block_timings",{}).items() if v},
        "ret_vals": _j(d.get("ret_vals",[])),
        "snapshots": [{"i":s["i"],"t":s["t"],"wt":s.get("wt"),"evt":s["evt"],
                       "fn":s["fn"],"ln":s["ln"],"file":s["file"],
                       "label":s.get("label"),"tags":s.get("tags",[]),
                       "vars":{k:v["r"] for k,v in s.get("loc",{}).items()},
                       "sys":s.get("sys",{}),"mets":s.get("mets",{})}
                      for s in d.get("snaps",[])],
    }
    with open(out,"w",encoding="utf-8") as f: _json.dump(payload,f,indent=2,default=str)
    print(f"[TC] JSON → {out} ({os.path.getsize(out)//1024}KB)")
    return out


def flamegraph(path, out=None):
    """Export Chrome/SpeedScope flame graph JSON. Open at speedscope.app"""
    d = load(path); out = out or path.replace(".tc","_flame.json")
    events = []
    for s in d.get("snaps",[]):
        events.append({"name":s["fn"],"ph":"X","ts":s["t"]*1_000_000,
                       "dur":1000,"pid":d.get("pid",1),"tid":1,
                       "args":{"file":s["file"],"ln":s["ln"],"evt":s["evt"]}})
    for fn, times in d.get("fn_timings",{}).items():
        for dur in times:
            events.append({"name":fn,"ph":"X","ts":0,"dur":dur*1_000_000,
                           "pid":d.get("pid",1),"tid":2})
    with open(out,"w") as f: _json.dump({"traceEvents":events,"displayTimeUnit":"ms"},f)
    print(f"[TC] flame graph → {out} ({os.path.getsize(out)//1024}KB)  open at speedscope.app")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  MASTER TERMINAL REPORT  (insane output — 15 sections)
# ─────────────────────────────────────────────────────────────────────────────

SPARKS = "▁▂▃▄▅▆▇█"

def _spark(values, width=40, mark=None):
    nums = [v for v in values if isinstance(v,(int,float)) and not math.isnan(v) and not math.isinf(v)]
    if not nums or width <= 0: return dim("─"*width)
    mn,mx = min(nums),max(nums); rng = mx-mn or 1
    sampled = ([nums[int(i*len(nums)/width)] for i in range(width)]
               if len(nums)>width else [nums[0]]*(width-len(nums))+nums)
    chars = [SPARKS[int((v-mn)/rng*(len(SPARKS)-1))] for v in sampled]
    if mark is not None and 0 <= mark < len(chars): chars[mark] = cyan("┃")
    return "".join(chars)

def _hbar(v, mx, width=20, col=None):
    col = col or green
    if mx<=0 or width<=0: return dim("░"*width)
    filled = int(min(1.0,v/mx)*width)
    return col("█"*filled)+dim("░"*(width-filled))

def _pct_bar(pct, width=20):
    # gradient: green → yellow → red
    filled = int(min(1.0,pct/100)*width)
    if pct < 50:   col = green
    elif pct < 80: col = yellow
    else:          col = red
    return col("█"*filled)+dim("░"*(width-filled))

def _sep(W, ch="═"): return dim(ch*W)
def _sec(title, W):  return f"\n{_sep(W)}\n  {bold(title)}\n{_sep(W)}"
def _hsep(W):        return dim("─"*W)


def report(path=None, *, export=None, full=False, top_n=15, width=None):
    """
    Master terminal report — 15 sections of deep analysis.
    The most comprehensive output any Python profiler has ever produced.

    Args:
        path   : .tc file (None = auto-load latest)
        export : save plain-text to this file path
        full   : show all rows, no limits
        top_n  : rows per table (default 15)
        width  : terminal width override
    """
    d     = load(path)
    snaps = d.get("snaps", [])
    crash = d.get("crash")
    warns = d.get("warnings", [])

    try:    W = width or min(os.get_terminal_size().columns, 120)
    except: W = min(int(os.environ.get("COLUMNS", 100)), 120)

    USE_COLOR = sys.stdout.isatty()
    lines = []

    def P(*a): lines.append(" ".join(str(x) for x in a))
    def NL():  lines.append("")

    src     = os.path.basename(d.get("src","?"))
    dur     = d.get("duration", 0)
    n       = d.get("n", len(snaps))
    be      = d.get("backend","?")
    sid     = (d.get("session_id","") or "")[:8]
    py_ver  = str(d.get("python","?")).split()[0]
    plat    = d.get("platform","?")
    pid     = d.get("pid","?")
    saved   = d.get("saved_at","?")
    ram_pk  = d.get("ram_peak", 0)
    fd_pk   = d.get("fd_peak", 0)
    mets    = d.get("final_metrics", {})
    all_vars= d.get("all_vars", [])
    argv    = d.get("argv", [])
    mode    = d.get("mode","full")

    # ── SECTION 1: HEADER ──────────────────────────────────────────────────
    P(_sec("TIMECAPSULE v6 REPORT", W))
    NL()
    P(f"  {dim('script   :')} {bold(cyan(src))}")
    P(f"  {dim('session  :')} {sid}   {dim('saved:')} {saved}")
    P(f"  {dim('python   :')} {py_ver}   {dim('platform:')} {plat}   {dim('pid:')} {pid}")
    P(f"  {dim('backend  :')} {bold(be)}   {dim('mode:')} {mode}")
    P(f"  {dim('argv     :')} {' '.join(str(a) for a in argv[:6])}")
    NL()
    P(f"  {dim('duration :')} {bold(f'{dur:.6f}s')}")
    P(f"  {dim('snapshots:')} {bold(str(n))}   {dim('variables:')} {len(all_vars)}")
    P(f"  {dim('RAM peak :')} {bold(f'{ram_pk:.1f}MB')}   {dim('FD peak:')} {fd_pk}")
    if mets:
        met_str = "   ".join(f"{k}={v:.4g}" if isinstance(v,float) else f"{k}={v}" for k,v in list(mets.items())[:8])
        P(f"  {dim('metrics  :')} {met_str}")
    NL()

    # status line
    if crash:
        P(f"  {red(bold('⚠ CRASH'))}  {red(crash['exc'])}: {crash['msg'][:60]}")
    else:
        P(f"  {green('✓ No crash')}")
    if warns:
        errs = [w for w in warns if w.get("level")=="ERROR"]
        if errs: P(f"  {red(bold(f'⚠ {len(errs)} ERROR warning(s)'))}"
                   f"  {yellow(f'+ {len(warns)-len(errs)} WARN')}")
        else:     P(f"  {yellow(f'⚠ {len(warns)} warning(s)')}")
    else:
        P(f"  {green('✓ No warnings')}")

    # ── SECTION 2: PLAIN-ENGLISH EXPLAIN ──────────────────────────────────
    P(_sec("WHAT YOUR CODE DID", W))
    NL()
    parts = []
    fns   = d.get("fn_calls",{})
    fn_t  = d.get("fn_timings",{})
    labels= [s["label"] for s in snaps if s.get("label") and not str(s.get("label","")).startswith("_tc")]

    parts.append(f"{src} ran for {dur:.3f}s capturing {n} snapshots across {len(all_vars)} variables.")
    skip  = {"<genexpr>","<listcomp>","<dictcomp>","<lambda>","<module>"}
    ufns  = {k:v for k,v in fns.items() if not any(s2 in k for s2 in skip)}
    if ufns:
        tf,tc2 = max(ufns.items(), key=lambda x:x[1])
        times2 = fn_t.get(tf,[])
        avg_ms = round(sum(times2)/len(times2)*1000,2) if times2 else None
        tim    = f" avg {avg_ms}ms/call" if avg_ms else ""
        parts.append(f"Dominant function: {tf.split('.')[-1]}() × {tc2}{tim}.")
    if mets:
        nm = {k:v for k,v in mets.items() if isinstance(v,(int,float))}
        if nm: parts.append("Final metrics: " + ", ".join(f"{k}={v:.4g}" if isinstance(v,float) else f"{k}={v}" for k,v in list(nm.items())[:5]) + ".")
    ram_snaps2 = [s.get("sys",{}).get("ram_mb",0) for s in snaps if s.get("sys",{}).get("ram_mb")]
    if ram_snaps2:
        gr = ram_snaps2[-1]-ram_snaps2[0]
        if abs(gr)<2: parts.append(f"RAM stable ~{ram_pk:.0f}MB.")
        elif gr>0:    parts.append(f"RAM grew {ram_snaps2[0]:.0f}→{ram_snaps2[-1]:.0f}MB (+{gr:.0f}MB).")
        else:          parts.append(f"RAM peaked {ram_pk:.0f}MB then dropped to {ram_snaps2[-1]:.0f}MB.")
    if warns: parts.append(f"{'⚠ ' if any(w.get('level')=='ERROR' for w in warns) else ''}{len(warns)} automatic warning(s) detected.")
    else:      parts.append("No warnings detected.")
    if crash:  parts.append(f"⚠ CRASHED: {crash['exc']} — {crash['msg'][:60]} in {crash.get('fn','?')}():{crash.get('ln','?')}.")
    else:      parts.append("No crashes.")

    # word-wrap
    words = " ".join(parts).split()
    cur   = "  "; wrapped = []
    for w in words:
        if len(cur)+len(w)+1 > W-2: wrapped.append(cur); cur = "  "+w
        else: cur += (" " if cur.strip() else "")+w
    if cur.strip(): wrapped.append(cur)
    for wl in wrapped: P(wl)

    # ── SECTION 3: WARNINGS ───────────────────────────────────────────────
    if warns:
        P(_sec(f"WARNINGS  ({len(warns)} total)", W))
        NL()
        for w in warns:
            lvl  = w.get("level","WARN")
            code = w.get("code","")
            wmsg = w.get("msg","")
            wt   = w.get("t",0)
            wvar = w.get("var","")
            icon = red(bold("[ERROR]")) if lvl=="ERROR" else yellow("[WARN] ")
            vstr = f"  {dim('var=')+magenta(wvar)}" if wvar else ""
            P(f"  {icon}  {dim(f't={wt:.3f}s')}  {bold(code)}{vstr}")
            P(f"         {wmsg}")

    # ── SECTION 4: CRASH FORENSICS ────────────────────────────────────────
    if crash:
        P(_sec("CRASH FORENSICS", W))
        NL()
        _exc = crash.get("exc","?"); _emsg = crash.get("msg","")
        P(f"  {red(bold(f'⚠  {_exc}: {_emsg}'))}")
        P(f"  {dim('location:')} {crash.get('file','?')}  {bold('fn='+crash.get('fn','?')+'()')}  line={bold(str(crash.get('ln','?')))}  t={crash.get('t',0):.4f}s")
        NL()
        friendly = crash.get("friendly","")
        if friendly:
            P(f"  {yellow(bold('What this means:'))}")
            for fl in friendly.splitlines(): P(f"    {yellow(fl)}")
            NL()
        src_ctx = crash.get("src_ctx",[])
        if src_ctx:
            P(f"  {bold('Source context:')}")
            for lineno, text, is_crash in src_ctx:
                if is_crash:
                    P(f"  {red('►')} {red(bold(f'{lineno:4d}  {text}'))}")
                else:
                    P(f"    {dim(f'{lineno:4d}  {text}')}")
            NL()
        cloc = crash.get("loc",{})
        if cloc:
            P(f"  {bold('Variables at crash:')}")
            for k,vd in cloc.items():
                if k.startswith("__"): continue
                P(f"    {cyan(f'{k:<22}')} {dim('['+vd.get('t','')[:10]+']')} = {vd.get('r','')[:W-40]}")
            NL()
        FIX = {
            "IndexError":    ["Check: if idx < len(lst)","Use enumerate() for index+value loops","Clip: idx = min(idx, len(lst)-1)"],
            "KeyError":      ["Use: dict.get('key', default)","Check: if key in dict","Use collections.defaultdict"],
            "TypeError":     ["Print type: print(type(x))","Add isinstance() guard","Check None before calling methods"],
            "AttributeError":["Check None: if obj is not None","Use hasattr(obj, 'attr')","Check class definition"],
            "ValueError":    ["Validate before converting","Wrap in try/except","Check input range"],
            "ZeroDivisionError":["Guard: if denom != 0","Use: max(denom, 1e-9) for soft divide"],
            "NameError":     ["Check variable defined before use","Check spelling"],
            "RecursionError":["Add base case","Convert to iterative + explicit stack"],
            "MemoryError":   ["Process in chunks","Use generators instead of lists","Delete large vars with del"],
        }
        hints = FIX.get(crash.get("exc",""),[])
        if hints:
            P(f"  {green(bold('Fix checklist:'))}")
            for h in hints: P(f"    {green('☐')} {h}")
            NL()
        stk = crash.get("stk",[])
        if stk:
            P(f"  {bold('Call stack:')}")
            for frame in stk[:6]:
                P(f"    {dim('→')} {cyan(frame.get('fn','?')+'()')}  {dim(frame.get('file','?')+':'+str(frame.get('ln',0)))}  [{dim(frame.get('mod','?'))}]")
            NL()
        tb = crash.get("tb","")
        if tb:
            P(f"  {bold('Full traceback:')}")
            for tbl in tb.splitlines(): P(f"    {dim(tbl)}")

    # ── SECTION 5: FUNCTION PERFORMANCE ───────────────────────────────────
    fn_timings_d = d.get("fn_timings",{})
    fn_calls_d   = d.get("fn_calls",{})
    if fn_timings_d:
        P(_sec("FUNCTION PERFORMANCE", W))
        NL()
        stats = []
        for fn,times in fn_timings_d.items():
            if not times: continue
            ms  = sorted(t*1000 for t in times); nc = len(ms); tot = sum(ms)
            avg = tot/nc
            std = math.sqrt(sum((x-avg)**2 for x in ms)/nc) if nc>1 else 0
            stats.append({
                "fn":fn,"calls":nc,"total":tot,"avg":avg,"std":std,
                "min":ms[0],"max":ms[-1],
                "p50":ms[nc//2],
                "p95":ms[max(0,int(nc*.95)-1)],
                "p99":ms[max(0,int(nc*.99)-1)],
                "calls_fn":fn_calls_d.get(fn,nc),
            })
        stats.sort(key=lambda x:-x["total"])
        limit    = len(stats) if full else min(top_n, len(stats))
        max_tot  = stats[0]["total"] if stats else 1
        dur2     = dur or 1

        COL = max(30, W - 75)
        P(bold(f"  {'FUNCTION':<{COL}} {'CALLS':>7} {'TOTAL':>10} {'AVG':>9} {'P50':>9} {'P95':>9} {'P99':>9}  BAR"))
        P(_hsep(W))
        for s2 in stats[:limit]:
            avg = s2["avg"]
            bc  = (green if avg<1 else yellow if avg<10 else magenta if avg<100 else red)
            fn_s = str(s2["fn"])[-COL:]
            bstr = _hbar(s2["total"], max_tot, 14, bc)
            P(f"  {cyan(f'{fn_s:<{COL}}')} {s2['calls']:>7}"
              f" {s2['total']:>9.2f}ms {s2['avg']:>8.3f}ms"
              f" {s2['p50']:>8.3f}ms {s2['p95']:>8.3f}ms {s2['p99']:>8.3f}ms  {bstr}")
        if len(stats) > limit:
            P(dim(f"  ... {len(stats)-limit} more functions  (use full=True)"))
        NL()

        # total stats footer
        total_calls = sum(s2["calls_fn"] for s2 in stats)
        total_ms_all= sum(s2["total"] for s2 in stats)
        P(f"  {dim('Total:')} {total_calls:,} calls  {total_ms_all:.1f}ms  {total_calls/dur2:.0f} calls/s")

        # slowest individual calls
        slow = slowest(path, top_n=5)
        if slow:
            NL()
            P(f"  {bold('Slowest individual calls:')}")
            for ms_v,fn2 in slow:
                P(f"    {red(f'{ms_v:>10.3f}ms')}  {fn2}")

        # block/checkpoint timings
        bt = d.get("block_timings",{})
        if bt:
            NL()
            P(f"  {bold('Block & checkpoint timings:')}")
            P(dim(f"  {'NAME':<35} {'CALLS':>6} {'TOTAL':>10} {'AVG':>9} {'MIN':>9} {'MAX':>9}"))
            P(_hsep(W))
            for bname, btimes in sorted(bt.items(), key=lambda x:-sum(x[1]) if x[1] else 0):
                if not btimes: continue
                ms2 = [t*1000 for t in btimes]; nc2 = len(ms2); tot2 = sum(ms2)
                P(f"  {yellow(f'{bname:<35}')} {nc2:>6} {tot2:>9.2f}ms {tot2/nc2:>8.2f}ms"
                  f" {min(ms2):>8.2f}ms {max(ms2):>8.2f}ms")

        # recursion
        rec = d.get("rec_max",{})
        deep = {k:v for k,v in rec.items() if v>1}
        if deep:
            NL()
            P(f"  {bold('Recursion depths:')}")
            for fn3,depth in sorted(deep.items(), key=lambda x:-x[1]):
                dc = (red if depth>20 else yellow if depth>5 else green)
                P(f"  {cyan(f'{fn3:<30}')} max depth = {dc(str(depth))}")

    # ── SECTION 6: METRICS ────────────────────────────────────────────────
    all_mkeys = set()
    for s in snaps: all_mkeys.update(s.get("mets",{}).keys())
    if all_mkeys:
        P(_sec(f"METRICS  ({len(all_mkeys)} tracked)", W))
        NL()
        sp_w = max(20, W - 50)
        for mname in sorted(all_mkeys):
            vals2 = [s["mets"][mname] for s in snaps
                     if mname in s.get("mets",{}) and isinstance(s["mets"][mname],(int,float))
                     and not math.isnan(s["mets"][mname])]
            if not vals2: continue
            mn_v,mx_v = min(vals2),max(vals2)
            mean_v = sum(vals2)/len(vals2)
            std_v  = math.sqrt(sum((v-mean_v)**2 for v in vals2)/len(vals2)) if len(vals2)>1 else 0
            cur_v  = vals2[-1]
            sp     = _spark(vals2, sp_w, mark=len(vals2)-1)
            P(f"  {bold(magenta(f'{mname:<22}'))} cur={cyan(f'{cur_v:.4g}'):<12}"
              f" min={mn_v:.4g}  max={mx_v:.4g}  μ={mean_v:.4g}  σ={std_v:.4g}  n={len(vals2)}")
            P(f"  ▕{sp}▏  {dim(f'{mn_v:.3g}…{mx_v:.3g}')}")
            # anomaly markers
            if std_v > 0:
                anom = list(" "*sp_w)
                for ni,v in enumerate(vals2):
                    z = abs((v-mean_v)/std_v)
                    if z > 2.5:
                        pos = int(ni/max(len(vals2)-1,1)*(sp_w-1))
                        if 0<=pos<sp_w: anom[pos]="^"
                astr = "".join(anom)
                if "^" in astr: P(f"  {red(' '+''.join(astr))} {red(f'← anomalies (z>2.5)')}")
            NL()

    # ── SECTION 7: MEMORY ─────────────────────────────────────────────────
    P(_sec("MEMORY", W))
    NL()
    sp_w = max(20, W - 40)
    ram_ts  = [(s.get("t",0), s["sys"].get("ram_mb",0)) for s in snaps if s.get("sys",{}).get("ram_mb")]
    cpu_ts  = [(s.get("t",0), s["sys"].get("cpu_pct",0)) for s in snaps if s.get("sys",{}).get("cpu_pct") is not None]
    gpu_ts  = [(s.get("t",0), s["sys"].get("gpu_mb",0)) for s in snaps if s.get("sys",{}).get("gpu_mb")]
    thr_ts  = [(s.get("t",0), s["sys"].get("threads",0)) for s in snaps if s.get("sys",{}).get("threads")]

    def _metric_row(label, ts, unit, col):
        if not ts: P(f"  {dim(label+':')} no data"); return
        vals3 = [v for _,v in ts]
        P(f"  {bold(label+':')} peak={col(f'{max(vals3):.1f}{unit}')}  start={vals3[0]:.1f}  end={vals3[-1]:.1f}  growth={vals3[-1]-vals3[0]:+.1f}")
        P(f"  ▕{_spark(vals3, sp_w)}▏  {dim(f'{min(vals3):.1f}…{max(vals3):.1f}{unit}')}")

    _metric_row("RAM (MB)",    ram_ts, "MB", green)
    NL()
    _metric_row("CPU (%)",     cpu_ts, "%",  yellow)
    NL()
    if gpu_ts: _metric_row("GPU (MB)", gpu_ts, "MB", magenta); NL()
    if thr_ts: _metric_row("Threads",  thr_ts, "",   cyan);    NL()

    # variable size audit
    mm = memory_map(path) if path else {}
    if mm:
        P(f"  {bold('Variable size audit:')}")
        P(dim(f"  {'VARIABLE':<22} {'TYPE':<14} {'MAX':>10} {'AVG':>10} {'SEEN':>6}  BAR"))
        P(_hsep(W))
        limit_mm = len(mm) if full else min(top_n, len(mm))
        max_sz   = max(v["max_bytes"] for v in mm.values()) if mm else 1
        for var, info in list(mm.items())[:limit_mm]:
            typ2  = ""
            for s in snaps:
                if var in s.get("loc",{}): typ2 = s["loc"][var].get("t","")[:12]; break
            bstr = _hbar(info["max_bytes"], max_sz, 12, yellow)
            P(f"  {cyan(f'{var:<22}')} {dim(f'{typ2:<14}')} {info['max_bytes']:>10,}  {info['avg_bytes']:>9,}  {info['n_seen']:>6}  {bstr}")

    # ── SECTION 8: VARIABLE TIMELINES ─────────────────────────────────────
    P(_sec("VARIABLE TIMELINES", W))
    NL()
    var_hist2   = defaultdict(list)
    var_changes2= {}
    for s in snaps:
        for k,vd in s.get("loc",{}).items():
            if not k.startswith("__"): var_hist2[k].append((s["t"],vd.get("r","?"),vd.get("t","?")))
    for var,hist in var_hist2.items():
        changes = []
        prev = None
        for t2,r,typ in hist:
            if r != prev: changes.append((t2,prev,r,typ)); prev=r
        var_changes2[var] = changes
    sorted_vars = sorted(var_changes2.items(), key=lambda x:-len(x[1]))
    limit_v     = len(sorted_vars) if full else min(top_n, len(sorted_vars))
    for var,changes in sorted_vars[:limit_v]:
        n_chg = len(changes)
        P(f"  {cyan(bold(var))}  {dim(f'({n_chg} changes)')}")
        show = changes if full else changes[:5]
        for t2,old,new,typ in show:
            old_s = dim(f"{str(old)[:20]} → ") if old is not None else ""
            P(f"    {dim(f't={t2:.4f}s')}  {old_s}{yellow(str(new)[:35])}  {dim('['+typ+']')}")
        if not full and len(changes)>5: P(dim(f"    ... {len(changes)-5} more"))
        NL()

    # ── SECTION 9: HOT LINES ──────────────────────────────────────────────
    lh = d.get("line_hits",{})
    if lh:
        P(_sec("HOT LINES", W))
        NL()
        hl_items = []
        for k,v in lh.items():
            try:
                if isinstance(k,str): k=eval(k)
                hl_items.append((v,k[0],k[1]))
            except Exception: pass
        hl_items.sort(reverse=True)
        limit_hl  = len(hl_items) if full else min(top_n, len(hl_items))
        max_hits  = hl_items[0][0] if hl_items else 1
        P(bold(f"  {'HITS':>8}  {'FILE':<30}  {'LINE':>6}  BAR"))
        P(_hsep(W))
        for hits,file2,line2 in hl_items[:limit_hl]:
            bstr = _hbar(hits, max_hits, 20, cyan)
            P(f"  {red(f'{hits:>8,}')}  {dim(f'{str(file2)[-30:]:<30}')}  {yellow(f'{line2:>6}')}  {bstr}")

    # ── SECTION 10: RETURN VALUES ─────────────────────────────────────────
    ret_vals2 = d.get("ret_vals",[])
    if ret_vals2:
        P(_sec(f"RETURN VALUES  (last {min(len(ret_vals2),10)})", W))
        NL()
        for rv in list(ret_vals2)[-10:]:
            rv_t  = rv.get("t",0); rv_fn = rv.get("fn","?")
            rv_val= str(rv.get("val",""))[:60]; rv_type = rv.get("type","?")
            P(f"  {dim(f't={rv_t:.4f}s')}  {cyan(rv_fn+'()')} → {yellow(rv_val)}  {dim('['+rv_type+']')}")

    # ── SECTION 11: SAMPLE DATA (sample mode) ─────────────────────────────
    sample_counts = d.get("sample_counts", {})
    if sample_counts:
        P(_sec("STACK SAMPLE PROFILE", W))
        NL()
        P(f"  {dim('(mode=sample — zero per-call overhead, statistical profiling)')}")
        NL()
        sorted_sc = sorted(sample_counts.items(), key=lambda x:-x[1])
        total_sc  = sum(v for _,v in sorted_sc) or 1
        limit_sc  = len(sorted_sc) if full else min(top_n, len(sorted_sc))
        P(bold(f"  {'FILE':<25} {'FN':<25} {'LINE':>6} {'SAMPLES':>8} {'%':>6}  BAR"))
        P(_hsep(W))
        for (file2,fn2,line2),cnt in sorted_sc[:limit_sc]:
            pct = cnt/total_sc*100
            bstr= _pct_bar(pct, 20)
            P(f"  {dim(f'{file2:<25}')} {cyan(f'{fn2:<25}')} {line2:>6} {cnt:>8,} {pct:>5.1f}%  {bstr}")

    # ── SECTION 12: ASYNC INFO ─────────────────────────────────────────────
    async_snaps = [s for s in snaps if s.get("async")]
    if async_snaps:
        P(_sec("ASYNC / COROUTINE ACTIVITY", W))
        NL()
        task_counts = defaultdict(int)
        for s in async_snaps:
            task_counts[s["async"].get("task","?")] += 1
        for task_name,cnt in sorted(task_counts.items(), key=lambda x:-x[1]):
            P(f"  {cyan(task_name):<30} {cnt} snapshots")

    # ── SECTION 13: OPTIMIZATION SUGGESTIONS ─────────────────────────────
    P(_sec("OPTIMIZATION SUGGESTIONS", W))
    NL()
    suggestions = []
    for fn2,times in fn_timings_d.items():
        if not times: continue
        ms_v = [t*1000 for t in times]; avg2 = sum(ms_v)/len(ms_v); tot2 = sum(ms_v)
        if avg2 > 100:
            suggestions.append((red("SLOW"),
                f"{fn2}() avg {avg2:.1f}ms — cache results or profile internals"))
        elif avg2 > 10 and len(ms_v) > 50:
            suggestions.append((yellow("PERF"),
                f"{fn2}() called {len(ms_v)}× at {avg2:.1f}ms avg ({tot2:.0f}ms total) — hot path"))
        if len(ms_v)>5:
            std_fn = math.sqrt(sum((v-avg2)**2 for v in ms_v)/len(ms_v))
            if std_fn/max(avg2,0.001) > 1.5 and avg2>1:
                suggestions.append((yellow("VARY"),
                    f"{fn2}() σ={std_fn:.1f}ms ({std_fn/avg2*100:.0f}% of avg) — inconsistent, check for cold paths"))
    ram_snaps3 = [v for _,v in ram_ts]
    if ram_snaps3 and (ram_snaps3[-1]-ram_snaps3[0]) > 50:
        suggestions.append((red("MEM "),
            f"RAM grew {ram_snaps3[-1]-ram_snaps3[0]:.0f}MB — check for unbounded lists/caches"))
    if ram_pk > 500:
        suggestions.append((yellow("MEM "),
            f"Peak RAM {ram_pk:.0f}MB — consider chunked processing or generators"))
    rec2 = d.get("rec_max",{})
    for fn3,depth in rec2.items():
        if depth>20: suggestions.append((red("REC "), f"{fn3}() depth {depth} — stack overflow risk, use iteration"))
        elif depth>5: suggestions.append((yellow("REC "), f"{fn3}() depth {depth} — consider memoization"))
    warn_codes = defaultdict(int)
    for w2 in warns: warn_codes[w2.get("code","")] += 1
    for code2,cnt2 in warn_codes.items():
        if cnt2>1: suggestions.append((yellow("WARN"), f"{code2} fired {cnt2}× — systematic issue"))
    if not mets and dur>2:
        suggestions.append((dim("TIP "), "No metrics recorded — add tc.metric('name', value) to track progress"))
    user_labels = [s for s in snaps if s.get("label") and not str(s.get("label","")).startswith("_tc")]
    if not user_labels and n>20:
        suggestions.append((dim("TIP "), "No snapshot labels — add tc.snap('phase') to mark important moments"))
    if mode=="full" and n>5000:
        suggestions.append((dim("TIP "), f"{n} snaps may slow save — consider tc.record(mode='sample') for <1% overhead"))
    if suggestions:
        for tag2,suggestion in suggestions: P(f"  [{tag2}]  {suggestion}")
    else: P(f"  {green('✓ No obvious optimizations detected.')}")

    # ── SECTION 14: ASCII FLAME GRAPH ─────────────────────────────────────
    if fn_timings_d:
        P(_sec("ASCII FLAME GRAPH  (total time by function)", W))
        NL()
        flame_stats = []
        for fn2,times in fn_timings_d.items():
            if times: flame_stats.append((fn2, sum(t*1000 for t in times)))
        flame_stats.sort(key=lambda x:-x[1])
        total_flame = max(sum(t for _,t in flame_stats), 0.001)
        bar_w       = max(10, W - 50)
        for fn2,tot2 in flame_stats[:min(20, len(flame_stats))]:
            pct  = tot2/total_flame*100
            filled = int(pct/100*bar_w)
            col  = (green if pct<5 else yellow if pct<20 else magenta if pct<50 else red)
            bstr = col("█"*filled)+dim("░"*(bar_w-filled))
            P(f"  {cyan(f'{fn2[-30:]:<30}')} {tot2:>9.2f}ms {pct:>5.1f}%  {bstr}")

    # ── SECTION 15: SNAPSHOT TIMELINE ────────────────────────────────────
    P(_sec(f"SNAPSHOT TIMELINE  ({n} total)", W))
    NL()
    limit_tl = len(snaps) if full else min(top_n*3, len(snaps))
    for s in snaps[:limit_tl]:
        evt2  = s.get("evt","")
        fn2   = s.get("fn","?")
        ln2   = s.get("ln",0)
        t2    = s.get("t",0)
        lbl2  = s.get("label","") or ""
        tags2 = [tg for tg in s.get("tags",[]) if tg not in ("_internal",)]
        loc2  = {k:vd for k,vd in s.get("loc",{}).items() if not k.startswith("__")}
        mets2 = s.get("mets",{})
        sys2  = s.get("sys",{})

        if "CRASH" in evt2:     evt_c = red(bold(f"[{evt2}]"))
        elif evt2.startswith("SNAP"): evt_c = green(f"[{evt2}]")
        elif evt2.startswith("TRACE"): evt_c = cyan(f"[{evt2}]")
        elif evt2=="LINE":      evt_c = dim(f"[{evt2}]")
        else:                   evt_c = yellow(f"[{evt2}]")

        _si = s.get("i", 0)
        _hdr = f"  {dim(f'#{_si:04d}')}  {dim(f't={t2:.4f}s')}  {evt_c}  {bold(fn2+'()')}:{ln2}"
        _lbl_s = f"  {green('['+lbl2+']')}" if lbl2 and not lbl2.startswith("_tc") else ""
        _tag_s = f"  {dim(str(tags2))}" if tags2 else ""
        P(_hdr + _lbl_s + _tag_s)

        changed = s.get("changed",[])
        for k in changed[:4]:
            if k in loc2 and not k.startswith("__"):
                P(f"       {yellow('~')} {cyan(k)} = {loc2[k].get('r','?')[:50]}")

        if evt2 != "LINE":
            for k2,vd2 in list(loc2.items())[:3]:
                P(f"       {dim('·')} {cyan(k2)} = {vd2.get('r','?')[:50]}  {dim('['+vd2.get('t','')[:8]+']')}")
            if len(loc2)>3: P(dim(f"       ... {len(loc2)-3} more vars"))

        if mets2:
            mstr = "  ".join(f"{k}={v}" for k,v in list(mets2.items())[:4])
            P(f"       {dim('mets:')} {mstr}")
        if sys2:
            parts2 = []
            for key2,lbl22 in [("ram_mb","RAM"),("cpu_pct","CPU"),("gpu_mb","GPU"),("threads","thr")]:
                if key2 in sys2:
                    suf = "MB" if "mb" in key2 else "%" if "pct" in key2 else ""
                    parts2.append(f"{lbl22}={sys2[key2]}{suf}")
            if parts2: P(f"       {dim('sys:')} {dim('  '.join(parts2))}")

    if not full and len(snaps)>limit_tl:
        P(dim(f"  ... {len(snaps)-limit_tl} more snapshots  (use full=True)"))

    # ── FOOTER ────────────────────────────────────────────────────────────
    NL()
    P(_sep(W))
    P(f"  {dim('Generated:')} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
      f"   {dim('Source:')} {path or '(latest .tc)'}"
      f"   {dim('timecapsule v'+__version__)}")
    P(_sep(W))
    NL()

    # ── OUTPUT ────────────────────────────────────────────────────────────
    output_str = "\n".join(lines)
    print(output_str)

    if export:
        clean = re.sub(r"\033\[[0-9;]*m", "", output_str)
        with open(export, "w", encoding="utf-8") as f: f.write(clean)
        print(f"\n[TC] report exported → {cyan(export)}")

    return {
        "src":src,"duration":dur,"n_snaps":n,"crash":crash,
        "warnings":warns,"fn_stats":timings(path) if path else {},
        "all_vars":all_vars,"ram_peak":ram_pk,
    }


def dump(path=None, show_vars=True, show_sys=True, show_stack=False,
         show_warnings=True, show_timings=True, show_returns=False,
         show_blocks=True, max_snaps=None):
    """Raw snapshot dump (original dump format, compatible with v5)."""
    d = load(path); W = 72
    crash = d.get("crash")
    print(f"\n{bold('='*W)}")
    print(f"  {bold('timecapsule')} v{d.get('ver','?')}  —  {cyan(os.path.basename(d.get('src','?')))}")
    print(f"{'='*W}")
    for label,val in [
        ("script",   d.get("src","?")),
        ("session",  d.get("session_id","?")),
        ("backend",  d.get("backend","?")),
        ("python",   str(d.get("python","?")).split()[0]),
        ("platform", f"{d.get('platform','?')}  pid={d.get('pid','?')}"),
        ("saved",    d.get("saved_at","?")),
        ("duration", f"{d.get('duration','?')}s"),
        ("snapshots",f"{d.get('n',0)}  |  vars: {len(d.get('all_vars',[]))}"),
        ("all_vars", ", ".join(d.get("all_vars",[])[:15])),
        ("RAM peak", f"{d.get('ram_peak','?')}MB  |  FD peak: {d.get('fd_peak','?')}"),
        ("warnings", str(len(d.get("warnings",[])))),
        ("crash",    (red("YES — "+crash["exc"]+": "+crash["msg"]) if crash else green("none"))),
    ]:
        print(f"  {dim(label+':'):<14} {val}")
    if d.get("final_metrics"): print(f"  {dim('metrics:'):<14} {d['final_metrics']}")
    print(f"{'='*W}")
    snaps = d.get("snaps",[]); 
    if max_snaps: snaps = snaps[:max_snaps]
    for s in snaps:
        t_str = dim(f"t={s['t']:.4f}s")
        evt   = red(bold(s["evt"])) if "CRASH" in s["evt"] else yellow(s["evt"]) if "WARN" in s["evt"] else dim(s["evt"])
        print(f"\n  [{s['i']:04d}] {t_str}  {evt}")
        print(f"         fn={bold(s['fn'])}()  line={s['ln']}  file={s['file']}")
        if s.get("label"):  print(f"         label = {cyan(s['label'])}")
        if s.get("tags"):   print(f"         tags  = {s['tags']}")
        if show_vars and s.get("loc"):
            for k,v in s["loc"].items():
                if k.startswith("__expr_"):
                    print(f"         {cyan('expr')}[{k[7:]}] = {v['r']}  [expr={v.get('expr','')}]")
                else:
                    chg = green(" [CHG]") if v.get("chg") else ""
                    print(f"         {k} = {v['r']}  [{dim(v['t'])}  {v.get('size',0)}B]{chg}")
        if show_sys and s.get("sys"):
            m=s["sys"]; parts=[]
            for key,lbl in [("ram_mb","RAM"),("cpu_pct","CPU"),("threads","thr"),("open_fds","fds"),("gpu_mb","GPU")]:
                if key in m: parts.append(f"{lbl}={m[key]}"+("MB" if "mb" in key else "%"if "pct" in key else ""))
            if parts: print(f"         {dim('sys:')} {' | '.join(parts)}")
        if s.get("mets"):   print(f"         {dim('mets:')} {s['mets']}")
        if show_stack and s.get("stk"):
            frames=" → ".join(f"{f2['fn']}:{f2['ln']}" for f2 in reversed(s["stk"][:4]))
            print(f"         {dim('stack:')} {frames}")
    if show_warnings and d.get("warnings"):
        print(f"\n{'='*W}\n  {bold('WARNINGS')} ({len(d['warnings'])})\n{'='*W}")
        for w in d["warnings"]:
            icon = red("[ERROR]") if w.get("level")=="ERROR" else yellow("[WARN] ")
            print(f"  {icon} t={w['t']:.3f}s  {bold(w['code'])}: {w['msg']}")
    if show_timings and d.get("fn_timings"):
        t_stats = timings(path)
        print(f"\n{'='*W}\n  {bold('FUNCTION TIMINGS')}\n{'='*W}")
        for fn,st in sorted(t_stats.items(), key=lambda x:-x[1]["total_ms"]):
            print(f"  {fn}()  calls={st['calls']}  total={st['total_ms']:.2f}ms  avg={st['avg_ms']:.3f}ms  p95={st['p95_ms']:.3f}ms")
    if show_blocks and d.get("block_timings"):
        print(f"\n{'='*W}\n  {bold('BLOCK TIMINGS')}\n{'='*W}")
        for name,times in d["block_timings"].items():
            if not times: continue
            ms=[t*1000 for t in times]; n=len(ms)
            print(f"  {name}  n={n}  avg={sum(ms)/n:.2f}ms  total={sum(ms):.2f}ms")
    if show_returns and d.get("ret_vals"):
        print(f"\n{'='*W}\n  {bold('RETURN VALUES')}\n{'='*W}")
        for r in list(d["ret_vals"])[:20]:
            print(f"  t={r['t']:.3f}s  {r['fn']}() → {str(r['val'])[:60]}  [{r['type']}]")
    if crash:
        print(f"\n{'='*W}\n  {red(bold('CRASH REPORT'))}\n{'='*W}")
        print(f"  {red(crash['exc'])}: {crash['msg']}")
        print(f"  at {crash['file']}  fn={crash['fn']}()  line={crash['ln']}  t={crash['t']:.4f}s")
        if crash.get("src_ctx"):
            print(f"\n  {bold('[SOURCE CONTEXT]')}")
            for ln,text,is_crash in crash["src_ctx"]:
                marker = red("►") if is_crash else " "
                print(f"  {marker} {ln:4d}  {red(bold(text)) if is_crash else dim(text)}")
        if crash.get("friendly"):
            print(f"\n  {bold('[WHAT THIS MEANS]')}")
            for line in crash["friendly"].splitlines(): print(f"  {line}")
        if crash.get("loc"):
            print(f"\n  {bold('[VARS AT CRASH]')}")
            for k,v in crash["loc"].items():
                print(f"    {k} = {v['r']}  [{v['t']}]")
        print(f"\n  {bold('[TRACEBACK]')}")
        for line in crash["tb"].splitlines(): print(f"    {dim(line)}")
    print(f"\n{'='*W}\n")


def explain(path=None, verbose=False):
    """Plain-English explanation of what the recording shows."""
    d    = load(path)
    src  = os.path.basename(d.get("src","?"))
    dur  = d.get("duration",0)
    n    = d.get("n",0)
    snaps= d.get("snaps",[])
    warn = d.get("warnings",[])
    crash= d.get("crash")
    ram  = d.get("ram_peak",0)
    fns  = d.get("fn_calls",{})
    fn_t = d.get("fn_timings",{})
    mets = d.get("final_metrics",{})
    vars_= d.get("all_vars",[])
    labels=[s["label"] for s in snaps if s.get("label") and not str(s.get("label","")).startswith("_tc")]
    parts=[]
    parts.append(f"{src} ran for {dur:.3f}s, recording {n} snapshots across {len(vars_)} variables.")
    skip={"<genexpr>","<listcomp>","<dictcomp>","<lambda>","<module>"}
    ufns={k:v for k,v in fns.items() if not any(s2 in k for s2 in skip)}
    if ufns:
        tf,tc2=max(ufns.items(),key=lambda x:x[1])
        times2=fn_t.get(tf,[])
        avg_ms=round(sum(times2)/len(times2)*1000,2) if times2 else None
        tim=f" (avg {avg_ms}ms/call)" if avg_ms else ""
        parts.append(f"Dominant operation: {tf.split('.')[-1]}() × {tc2}{tim}.")
    for name in ["iteration","step","epoch","i","batch","idx","count"]:
        if name in mets:
            parts.append(f"Iterated {int(mets[name])+1} {name}s."); break
    if mets:
        nm={k:v for k,v in mets.items() if isinstance(v,(int,float))}
        if nm: parts.append("Final metrics: "+", ".join(f"{k}={v:.4g}" if isinstance(v,float) else f"{k}={v}" for k,v in list(nm.items())[:5])+".")
    change_counts={}
    for s in snaps:
        for k in s.get("loc",{}):
            if not k.startswith("_"): change_counts[k]=change_counts.get(k,0)+1
    if change_counts:
        top3=[k for k,_ in sorted(change_counts.items(),key=lambda x:-x[1])[:3] if k not in ("i","j","k","_","__")][:3]
        if top3: parts.append(f"Most-tracked: {', '.join(top3)}.")
    if labels:
        parts.append(f"Named checkpoints: {', '.join(repr(l) for l in labels[:5])}"+(f" (+{len(labels)-5} more)" if len(labels)>5 else "")+".")
    if fn_t:
        skip2={"<genexpr>","<listcomp>","<dictcomp>","<lambda>"}
        all_t={k:v for k,v in fn_t.items() if not any(s2 in k for s2 in skip2) and v}
        if all_t:
            sf=max(all_t,key=lambda k:sum(all_t[k]))
            stms=[t*1000 for t in all_t[sf]]
            parts.append(f"Slowest: {sf.split('.')[-1]}() {sum(stms)/len(stms):.2f}ms avg, {sum(stms):.1f}ms total.")
    ram_snaps=[s.get("sys",{}).get("ram_mb",0) for s in snaps if s.get("sys",{}).get("ram_mb")]
    if ram_snaps:
        gr=ram_snaps[-1]-ram_snaps[0]
        if abs(gr)<2: parts.append(f"RAM stable ~{ram:.0f}MB.")
        elif gr>0: parts.append(f"RAM grew {ram_snaps[0]:.0f}→{ram_snaps[-1]:.0f}MB (+{gr:.0f}MB).")
        else: parts.append(f"RAM peaked {ram:.0f}MB, dropped to {ram_snaps[-1]:.0f}MB.")
    if warn:
        errs=[w for w in warn if w.get("level")=="ERROR"]
        parts.append(f"{'⚠ ' if errs else ''}{len(warn)} warning(s): {', '.join(set(w['code'] for w in warn[:3]))}.")
    else: parts.append("No warnings.")
    if crash:
        parts.append(f"⚠ CRASH: {crash['exc']} — {crash['msg'][:60]} in {crash.get('fn','?')}():{crash.get('ln','?')}.")
    else: parts.append("No crashes.")
    result=" ".join(parts)
    if verbose:
        print(f"\n{'─'*65}\n  tc.explain — {src}\n{'─'*65}")
        for i,p in enumerate(parts,1): print(f"  {i:2d}. {p}")
        print("─"*65)
    print(result); return result


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS FUNCTIONS (from tc_recorder_final new additions)
# ─────────────────────────────────────────────────────────────────────────────

def callgraph(path=None, *, min_calls=1, export=None):
    """ASCII call graph from recorded stack data."""
    d = load(path); snaps = d.get("snaps",[]); fn_t = d.get("fn_timings",{}); fn_c = d.get("fn_calls",{})
    edges = defaultdict(lambda: defaultdict(int))
    for s in snaps:
        stk = s.get("stk",[])
        for i in range(len(stk)-1):
            callee = stk[i].get("fn","?"); caller = stk[i+1].get("fn","?")
            if callee!=caller and callee not in ("<module>","<lambda>"): edges[caller][callee]+=1
    graph = {c:{ca:cnt for ca,cnt in cs.items() if cnt>=min_calls} for c,cs in edges.items()}
    graph = {k:v for k,v in graph.items() if v}
    all_callees = {c for cs in graph.values() for c in cs}
    roots = set(graph.keys())-all_callees or ({max(fn_c,key=fn_c.get)} if fn_c else {"<module>"})
    lines=[]; visited=set()
    def P(t): lines.append(t)
    src=os.path.basename(d.get("src","?")); P(f"\n  callgraph — {src}"); P("  "+"─"*70)
    def _render(fn,prefix="",is_last=True,depth=0):
        if depth>8: return
        con="└── " if is_last else "├── "; ext="    " if is_last else "│   "
        times=fn_t.get(fn,[]); calls=fn_c.get(fn,0); ann=""
        if times:
            ms=[t*1000 for t in times]; tot=sum(ms); avg=tot/len(ms)
            ann=dim(f"  ×{calls}  {avg:.2f}ms avg  {tot:.1f}ms total")
        fc=cyan(bold(fn+"()")) if depth==0 else (cyan(fn+"()") if fn not in visited else dim(fn+"() [↑ seen]"))
        P(f"  {prefix}{con}{fc}{ann}"); visited.add(fn)
        children=sorted(graph.get(fn,{}).items(),key=lambda x:-x[1])
        for i,(child,cnt) in enumerate(children):
            if child not in visited: _render(child,prefix+ext,i==len(children)-1,depth+1)
    if not graph: P("  (no call graph data — run with pure-Python backend)")
    else:
        for i,root in enumerate(sorted(roots)): _render(root,"",i==len(roots)-1)
    P("  "+"─"*70); P(f"  {len(set(graph.keys()))} callers  {len(all_callees)} callees")
    out="\n".join(lines); print(out)
    if export:
        with open(export,"w") as f: f.write(re.sub(r"\033\[[0-9;]*m","",out))
    return dict(graph)


def coverage_report(path=None, *, source_path=None, export=None):
    """Annotated source coverage from line_hits data."""
    d=load(path); line_hits=d.get("line_hits",{}); src_path=source_path or d.get("src","")
    hits={}
    for k,v in line_hits.items():
        try:
            if isinstance(k,str): k=eval(k)
            hits[k[1]]=hits.get(k[1],0)+v
        except Exception: pass
    lines=[]; src_name=os.path.basename(src_path)
    def P(t): lines.append(t)
    P(f"\n  coverage_report — {src_name}"); P("  "+"─"*70)
    P(f"  {'LINE':>5}  {'HITS':>6}  SOURCE"); P("  "+"─"*70)
    try:
        with open(src_path,"r",encoding="utf-8",errors="replace") as f: src_lines=f.readlines()
        max_h=max(hits.values()) if hits else 1; hit_lines=0; zero_lines=0
        for i,sl in enumerate(src_lines):
            lineno=i+1; cnt=hits.get(lineno,0); text=sl.rstrip()
            blank = not text.strip() or text.strip().startswith("#")
            if cnt>0: hit_lines+=1
            elif not blank: zero_lines+=1
            if cnt==0 and blank:
                P(f"  {dim(f'{lineno:>5}')}  {dim('      ')}  {dim(text)}")
            elif cnt==0:
                P(f"  {dim(f'{lineno:>5}')}  {yellow('     0')}  {yellow(text)}  {yellow('← never hit')}")
            elif cnt>=max_h*0.8:
                P(f"  {cyan(f'{lineno:>5}')}  {red(bold(f'{cnt:>6,}'))}  {bold(text)}  {red('← HOT')}")
            elif cnt>=max_h*0.3:
                P(f"  {cyan(f'{lineno:>5}')}  {yellow(f'{cnt:>6,}')}  {text}")
            else:
                P(f"  {dim(f'{lineno:>5}')}  {green(f'{cnt:>6,}')}  {text}")
        total_lines=len(src_lines)
        pct=hit_lines/max(total_lines,1)*100
        P("  "+"─"*70); P(f"  {hit_lines}/{total_lines} lines hit ({pct:.1f}%)  |  {zero_lines} executable lines never reached")
    except Exception: P(f"  (source not found: {src_path})")
    out="\n".join(lines); print(out)
    if export:
        with open(export,"w") as f: f.write(re.sub(r"\033\[[0-9;]*m","",out))
    return hits


def memory_leak_check(path=None, *, growth_threshold_mb=5.0):
    """Statistical memory leak detection using linear regression."""
    d=load(path); snaps=d.get("snaps",[]); src=os.path.basename(d.get("src","?"))
    ram_pts=[(s.get("t",0),s["sys"].get("ram_mb",0)) for s in snaps if s.get("sys",{}).get("ram_mb")]
    print(f"\n  memory_leak_check — {src}"); print("  "+"─"*55)
    if len(ram_pts)<4: print("  Insufficient data."); return {"leak_detected":False}
    xs=[t for t,_ in ram_pts]; ys=[r for _,r in ram_pts]
    n=len(xs); sx=sum(xs); sy=sum(ys); sxx=sum(x*x for x in xs); sxy=sum(x*y for x,y in zip(xs,ys))
    denom=n*sxx-sx*sx
    slope=(n*sxy-sx*sy)/denom if denom!=0 else 0
    intercept=(sy-slope*sx)/n
    y_mean=sy/n
    ss_tot=sum((y-y_mean)**2 for y in ys)
    ss_res=sum((y-(slope*x+intercept))**2 for x,y in zip(xs,ys))
    r2=1-ss_res/ss_tot if ss_tot>0 else 0
    growth_per_min=slope*60
    leak=(growth_per_min>growth_threshold_mb and r2>0.7)
    sp=_spark(ys,min(50,len(ys)))
    print(f"  RAM: ▕{sp}▏  {min(ys):.1f}…{max(ys):.1f}MB")
    print(f"  Regression: slope={slope*1000:.4f}MB/s  growth={growth_per_min:.2f}MB/min  R²={r2:.3f}")
    if leak: print(f"  {red(bold('⚠ LIKELY MEMORY LEAK'))} {growth_per_min:.1f}MB/min  projected +{growth_per_min*60:.0f}MB/hour")
    elif growth_per_min>0: print(f"  {yellow('↑ Gradual growth')} ({growth_per_min:.2f}MB/min) — monitor over longer run")
    else: print(f"  {green('✓ No leak detected')}")
    print("  "+"─"*55)
    return {"leak_detected":leak,"growth_mb_per_min":round(growth_per_min,4),"r_squared":round(r2,4),"verdict":"LEAK" if leak else "GROWING" if growth_per_min>0 else "STABLE"}


def variable_correlations(path=None, *, min_r=0.7, export=None):
    """Pearson correlation between all numeric variable pairs."""
    d=load(path); snaps=d.get("snaps",[]); src=os.path.basename(d.get("src","?"))
    series=defaultdict(list)
    for s in snaps:
        present={}
        for k,vd in s.get("loc",{}).items():
            if k.startswith("__"): continue
            try:
                v=float(vd.get("r",""))
                if not math.isnan(v) and not math.isinf(v): present[k]=v
            except Exception: pass
        for k,v in s.get("mets",{}).items():
            if isinstance(v,(int,float)) and not math.isnan(v): present[k]=float(v)
        for k,v in present.items(): series[k].append(v)
    min_pts=max(5,len(snaps)//4)
    valid={k:v for k,v in series.items() if len(v)>=min_pts}
    def pearson(xs,ys):
        n=min(len(xs),len(ys))
        if n<3: return None
        xs,ys=xs[:n],ys[:n]; mx=sum(xs)/n; my=sum(ys)/n
        num=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
        dx=math.sqrt(sum((x-mx)**2 for x in xs)); dy=math.sqrt(sum((y-my)**2 for y in ys))
        if dx==0 or dy==0: return None
        return num/(dx*dy)
    results=[]
    keys=sorted(valid.keys())
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            r=pearson(valid[a],valid[b])
            if r is not None and abs(r)>=min_r: results.append((a,b,r))
    results.sort(key=lambda x:-abs(x[2]))
    lines=[]; P=lines.append
    P(f"\n  variable_correlations — {src}  (|r|≥{min_r})")
    P("  "+"─"*65)
    P(f"  {'VAR A':<20} {'VAR B':<20} {'r':>7}  {'STRENGTH':<20}  DIRECTION")
    P("  "+"─"*65)
    if not results: P(f"  {dim('No correlations above threshold')}")
    else:
        for a,b,r in results[:30]:
            abs_r=abs(r)
            strength=("very strong" if abs_r>0.95 else "strong" if abs_r>0.85 else "moderate" if abs_r>0.70 else "weak")
            direction="positive" if r>0 else "negative"
            rc=(green if r>0.9 else yellow if r>0.7 else red if r<-0.9 else yellow if r<-0.7 else dim)
            bar_len=int(abs_r*20)
            bstr=rc("█"*bar_len+dim("░"*(20-bar_len)))
            P(f"  {cyan(a):<20} {cyan(b):<20} {rc(f'{r:+.3f}'):>7}  {bstr}  {strength} {direction}")
    P("  "+"─"*65); P(f"  {len(valid)} numeric variables  ·  {len(results)} correlation(s)")
    out="\n".join(lines); print(out)
    if export:
        with open(export,"w") as f: f.write(re.sub(r"\033\[[0-9;]*m","",out))
    return results


def exception_chain(path=None, *, export=None):
    """Deep root cause analysis — traces bad values to where they were introduced."""
    d=load(path); crash=d.get("crash"); snaps=d.get("snaps",[]); src=os.path.basename(d.get("src","?"))
    lines=[]; P=lines.append
    P(f"\n  exception_chain — {src}"); P("  "+"─"*70)
    if not crash: P(f"  {green('✓ No crash in this recording.')}"); print("\n".join(lines)); return {}
    exc=crash.get("exc","?"); msg=crash.get("msg",""); fn=crash.get("fn","?"); ln=crash.get("ln",0); ct=crash.get("t",0)
    P(f"  {red(bold('CRASH'))}  {red(exc)}: {yellow(msg[:70])}"); P(f"  {dim('at')} {fn}():{ln}  t={ct:.4f}s"); P("")
    cloc=crash.get("loc",{})
    P(f"  {bold('Variables at crash:')}") 
    for k,vd in cloc.items():
        if not k.startswith("__"): P(f"    {cyan(f'{k:<20}')} {dim('['+vd.get('t','')[:10]+']')} = {vd.get('r','')[:60]}")
    P("")
    P(f"  {bold('Variable histories (→ crash):')}") 
    root_cause={}
    var_hist3=defaultdict(list)
    for s in snaps:
        for k,vd in s.get("loc",{}).items():
            if not k.startswith("__"): var_hist3[k].append((s["t"],vd.get("r","?"),s.get("fn","?")))
    for k in list(cloc.keys())[:6]:
        hist=[(t,r,fn2) for t,r,fn2 in var_hist3.get(k,[]) if t<=ct+0.01]
        if not hist: continue
        recent=hist[-5:]
        chain=" → ".join(red(r[:12]) if i==len(recent)-1 else dim(r[:12]) for i,(t,r,fn2) in enumerate(recent))
        P(f"    {cyan(k)}: {chain} {red('(CRASH)')}")
        for i in range(len(hist)-2,-1,-1):
            if hist[i][1]!=hist[-1][1]:
                P(f"           {dim('changed at t=')+dim(str(round(hist[i+1][0],4))+'s')} {dim('in '+hist[i+1][2]+'()')}")
                root_cause.update({"bad_var":k,"bad_value":hist[-1][1],"introduced_at":hist[i+1][0],"introduced_in":hist[i+1][2]})
                break
    P("")
    src_ctx=crash.get("src_ctx",[])
    if src_ctx:
        P(f"  {bold('Source:')}") 
        for lineno,text,is_crash in src_ctx:
            m=red("►") if is_crash else " "
            P(f"    {m} {(red(bold(f'{lineno:4d}  {text}')) if is_crash else dim(f'{lineno:4d}  {text}'))}")
        P("")
    FIX_C={"IndexError":["Guard: if idx < len(lst)","Use enumerate()","Clip: min(idx, len-1)"],"KeyError":["Use dict.get(key, default)","Check: if key in d"],"ZeroDivisionError":["Guard: if denom != 0"],"AttributeError":["Check None","Use hasattr()"]}
    hints=FIX_C.get(exc,[])
    if hints:
        P(f"  {green(bold('Fixes:'))}") 
        for h in hints: P(f"    {green('☐')} {h}")
    if root_cause:
        P(""); P(f"  {yellow(bold('Root cause:'))}") 
        P(f"    {cyan(root_cause.get('bad_var','?'))} got {red(str(root_cause.get('bad_value','?'))[:30])} at t={root_cause.get('introduced_at',0):.4f}s in {root_cause.get('introduced_in','?')}()")
    P("  "+"─"*70)
    out="\n".join(lines); print(out)
    if export:
        with open(export,"w") as f: f.write(re.sub(r"\033\[[0-9;]*m","",out))
    return root_cause


def loop_detector(path=None):
    """Detect O(n²) growth, tight loops, stuck loops, deep recursion."""
    d=load(path); snaps=d.get("snaps",[]); warns=d.get("warnings",[]); src=os.path.basename(d.get("src","?"))
    findings=[]
    for w in warns:
        if w.get("code")=="STUCK_LOOP": findings.append({"type":"STUCK","severity":"HIGH","fn":w.get("var","?"),"line":0,"evidence":w.get("msg","")})
    line_sc=defaultdict(int)
    for s in snaps: line_sc[(s.get("fn","?"),s.get("ln",0))]+=1
    total=len(snaps)
    for (fn,ln),cnt in line_sc.items():
        if cnt/max(total,1)>0.70 and total>5:
            findings.append({"type":"TIGHT","severity":"MEDIUM","fn":fn,"line":ln,"evidence":f"{cnt}/{total} snaps on this line"})
    fn_snap_times=defaultdict(list)
    for s in snaps: fn_snap_times[s.get("fn","?")].append(s.get("t",0))
    for fn,times in fn_snap_times.items():
        if len(times)<10: continue
        gaps=[times[i+1]-times[i] for i in range(len(times)-1)]
        third=len(gaps)//3
        early=gaps[:third]; late=gaps[-third:]
        if early and late:
            ae=sum(early)/len(early); al=sum(late)/len(late)
            if ae>0 and al/ae>5.0:
                findings.append({"type":"O(N²)","severity":"HIGH","fn":fn,"line":0,"evidence":f"inter-snap time grew {al/ae:.1f}× from start to end"})
    for fn,depth in d.get("rec_max",{}).items():
        if depth>50: findings.append({"type":"DEEP_REC","severity":"HIGH","fn":fn,"line":0,"evidence":f"max depth {depth}"})
        elif depth>10: findings.append({"type":"REC","severity":"LOW","fn":fn,"line":0,"evidence":f"depth {depth}"})
    print(f"\n  loop_detector — {src}"); print("  "+"─"*65)
    if not findings: print(f"  {green('✓ No loop issues detected')}")
    else:
        sc={"HIGH":"1;31","MEDIUM":"33","LOW":"2"}
        for f2 in findings:
            cc=sc.get(f2["severity"],"0")
            tag=_c(f"[{f2['type']:<8}]",cc); sev=_c(f2["severity"],cc)
            fn_s=cyan(f"{f2['fn']}()")+( f":{f2['line']}" if f2["line"] else "")
            print(f"  {tag} {sev:<6}  {fn_s}"); print(f"           {dim(f2['evidence'])}")
    print("  "+"─"*65+f"\n  {len(findings)} finding(s)\n")
    return findings


def patch_summary(path_before, path_after, *, export=None):
    """Before/after comparison in plain English — perfect for commit messages."""
    a=load(path_before); b=load(path_after)
    lines=[]; P=lines.append
    P(f"\n  patch_summary")
    P(f"  BEFORE: {os.path.basename(path_before)}  ({a.get('duration',0):.3f}s  {a.get('n',0)} snaps)")
    P(f"  AFTER:  {os.path.basename(path_after)}  ({b.get('duration',0):.3f}s  {b.get('n',0)} snaps)")
    P("  "+"─"*65)
    findings=[]
    da,db=a.get("duration",0),b.get("duration",0)
    if da>0 and abs(db-da)/da>0.05:
        pct=(db-da)/da*100
        findings.append(f"  [{(green if pct<0 else red)('FASTER' if pct<0 else 'SLOWER')}]  Total {da:.3f}s → {db:.3f}s ({pct:+.1f}%)")
    def ft_stats(d2):
        out={}
        for fn,times in d2.get("fn_timings",{}).items():
            if times: ms=[t*1000 for t in times]; out[fn]={"avg":sum(ms)/len(ms),"calls":len(ms)}
        return out
    sta,stb=ft_stats(a),ft_stats(b)
    for fn in sorted(set(list(sta.keys())+list(stb.keys()))):
        fa,fb=sta.get(fn),stb.get(fn)
        if fa and fb:
            pct=(fb["avg"]-fa["avg"])/fa["avg"]*100 if fa["avg"]>0 else 0
            if abs(pct)>10:
                d2=green("FASTER") if pct<0 else red("SLOWER")
                findings.append(f"  [{d2}]  {fn}()  {fa['avg']:.2f}→{fb['avg']:.2f}ms ({pct:+.1f}%)")
        elif fa and not fb: findings.append(f"  [{yellow('REMOVED')}]  {fn}() gone")
        elif fb and not fa: findings.append(f"  [{cyan('NEW FN ')}]  {fn}() ×{fb['calls']} avg {fb['avg']:.2f}ms")
    ca,cb=a.get("crash"),b.get("crash")
    if ca and not cb:    findings.append(f"  [{green(bold('FIXED  '))}]  Crash FIXED: was {ca['exc']}: {ca['msg'][:40]}")
    elif not ca and cb:  findings.append(f"  [{red(bold('BROKE  '))}]  NEW CRASH: {cb['exc']}: {cb['msg'][:50]}")
    ra,rb=a.get("ram_peak",0),b.get("ram_peak",0)
    if ra>0 and abs(rb-ra)>2:
        findings.append(f"  [{(green('LESS RAM') if rb<ra else yellow('MORE RAM'))}]  RAM {ra:.0f}→{rb:.0f}MB ({rb-ra:+.0f}MB)")
    wa={w.get("code","") for w in a.get("warnings",[])}; wb={w.get("code","") for w in b.get("warnings",[])}
    nw=wb-wa; fw=wa-wb
    if nw: findings.append(f"  [{yellow('NEW WRN')}]  {', '.join(sorted(nw))}")
    if fw: findings.append(f"  [{green('FIX WRN')}]  Fixed: {', '.join(sorted(fw))}")
    if not findings: P(f"  {dim('No significant differences (>10% threshold)')}")
    else:
        for f2 in findings: P(f2)
    P("  "+"─"*65)
    out="\n".join(lines); print(out)
    if export:
        with open(export,"w") as f: f.write(re.sub(r"\033\[[0-9;]*m","",out))
    return re.sub(r"\033\[[0-9;]*m","",out)


def export_for_claude(path=None, out=None, *, max_snaps=50):
    """Export .tc as structured plain-text for Claude/LLM analysis."""
    d=load(path); snaps=d.get("snaps",[]); crash=d.get("crash"); src=d.get("src","?"); src_n=os.path.basename(src)
    out=out or (src_n.replace(".py","")+"_for_claude.txt")
    L=[]; P=L.append; S=lambda t: (P(""),P("="*70),P(f"  {t}"),P("="*70)); H=lambda t: (P(""),P(f"--- {t} ---"))
    P("TIMECAPSULE RECORDING — FOR CLAUDE ANALYSIS"); P("="*70)
    P(f"Generated: {datetime.now().isoformat()}"); P(f"Tool: timecapsule v{d.get('ver','?')} (v6)")
    P(f"Script: {src}"); P(f"Python: {str(d.get('python','?')).split()[0]}"); P(f"Backend: {d.get('backend','?')}")
    P(f"Session: {d.get('session_id','?')}"); P(f"Saved: {d.get('saved_at','?')}")
    P(""); P("SUMMARY"); P(f"  Duration:  {d.get('duration',0):.6f}s"); P(f"  Snapshots: {d.get('n',0)}")
    P(f"  Variables: {', '.join(d.get('all_vars',[])[:20])}"); P(f"  RAM peak:  {d.get('ram_peak',0):.1f}MB")
    P(f"  Warnings:  {len(d.get('warnings',[]))}"); P(f"  Crash:     {'YES — '+crash['exc']+': '+crash['msg'][:60] if crash else 'NO'}")
    if d.get("final_metrics"): P(f"  Metrics:   {d['final_metrics']}")
    if crash:
        S("CRASH FORENSICS"); P(f"Exception:  {crash.get('exc','?')}: {crash.get('msg','')}"); P(f"Location:   {crash.get('file','?')}  fn={crash.get('fn','?')}()  line={crash.get('ln','?')}")
        P(f"Time:       t={crash.get('t',0):.6f}s"); P(f"Plain English: {crash.get('friendly','')}"); P("")
        if crash.get("src_ctx"):
            P("Source context:")
            for lineno,text,is_crash in crash["src_ctx"]: P(f"  {'>>>' if is_crash else '   '} {lineno:4d}  {text}")
        P(""); P("Variables at crash:")
        for k,vd in crash.get("loc",{}).items():
            if not k.startswith("__"): P(f"  {k:<22} [{vd.get('t',''):<10}] = {vd.get('r','')[:80]}")
        P(""); P("Traceback:"); P(crash.get("tb",""))
    if d.get("warnings"):
        S(f"WARNINGS ({len(d['warnings'])})") 
        for w in d["warnings"]: P(f"[{w.get('level','WARN')}] t={w.get('t',0):.3f}s  {w.get('code','')}  {w.get('msg','')}")
    fn_t=d.get("fn_timings",{})
    if fn_t:
        S("FUNCTION PERFORMANCE"); P(f"  {'FUNCTION':<30} {'CALLS':>6} {'TOTAL_MS':>10} {'AVG_MS':>8} {'P95_MS':>8}")
        P("  "+"-"*65)
        stats=[]
        for fn,times in fn_t.items():
            if not times: continue
            ms=sorted(t*1000 for t in times); nc=len(ms); tot=sum(ms); avg=tot/nc
            stats.append((fn,nc,tot,avg,ms[max(0,int(nc*.95)-1)]))
        for fn,nc,tot,avg,p95 in sorted(stats,key=lambda x:-x[2]):
            P(f"  {fn:<30} {nc:>6} {tot:>10.2f} {avg:>8.3f} {p95:>8.3f}")
    S("VARIABLE TIMELINES")
    vh=defaultdict(list)
    for s in snaps:
        for k,vd in s.get("loc",{}).items():
            if not k.startswith("__"): vh[k].append((s["t"],vd.get("r","?"),vd.get("t","?")))
    for var,hist in sorted(vh.items(), key=lambda x:-len(x[1]))[:20]:
        chgs=[]; prev=None
        for t2,r,tp in hist:
            if r!=prev: chgs.append((t2,prev,r,tp)); prev=r
        P(f"\n  {var}  ({len(chgs)} changes):")
        for t2,old,new,tp in chgs[:8]:
            P(f"    t={t2:.6f}s  {''+str(old)[:20]+' -> ' if old is not None else ''}{new[:40]}  [{tp}]")
        if len(chgs)>8: P(f"    ... {len(chgs)-8} more")
    S(f"SNAPSHOTS (first {min(max_snaps,len(snaps))} of {len(snaps)})")
    for s in snaps[:max_snaps]:
        P(f"\n  [{s.get('i',0):04d}] t={s.get('t',0):.6f}s  evt={s.get('evt','')}  fn={s.get('fn','?')}():{s.get('ln',0)}")
        if s.get("label") and not str(s.get("label","")).startswith("_tc"): P(f"         label={s['label']}")
        for k,vd in list({k:vd for k,vd in s.get("loc",{}).items() if not k.startswith("__")}.items())[:8]:
            P(f"         {k} = {vd.get('r','?')[:60]}  [{vd.get('t','?')[:10]}]  {vd.get('size',0)}B")
        sys2=s.get("sys",{})
        if sys2: P(f"         sys: ram={sys2.get('ram_mb','?')}MB  cpu={sys2.get('cpu_pct','?')}%  gpu={sys2.get('gpu_mb','?')}MB  thr={sys2.get('threads','?')}")
        if s.get("mets"): P(f"         mets: {s['mets']}")
    S("HINTS FOR CLAUDE")
    P("1. Start with CRASH FORENSICS if crash=YES."); P("2. Cross-reference warning timestamps with variable values.")
    P("3. Variable values are Python repr() strings."); P("4. Times (t=) are elapsed seconds from recording start.")
    P(f"5. Backend: {d.get('backend','?')} — timing precision accordingly.")
    P("6. Look for the root cause (where bad value was introduced), not just the crash line.")
    content="\n".join(L)
    with open(out,"w",encoding="utf-8") as f: f.write(content)
    print(f"[TC] export_for_claude → {out}  ({os.path.getsize(out)//1024}KB)")
    return out


def tail(path=None, interval=0.5, vars_filter=None, show_sys=False):
    """Live-tail a .tc recording as it grows. Press Ctrl+C to stop."""
    import glob as _glob
    if path is None:
        files=sorted(_glob.glob("*.tc"),key=os.path.getmtime,reverse=True)
        if not files: print("[TC tail] No .tc files found."); return
        path=files[0]
    print(f"[TC tail] watching {cyan(path)}  (Ctrl+C to stop)",flush=True)
    last_n=0; last_mtime=0
    try:
        while True:
            time.sleep(interval)
            try:
                mt=os.path.getmtime(path)
                if mt<=last_mtime: continue
                last_mtime=mt
                with gzip.open(path,"rb") as f: d2=pickle.load(f)
                snaps2=d2.get("snaps",[]); n2=len(snaps2); new_snaps=snaps2[last_n:]; last_n=n2
                for s in new_snaps:
                    evt2=s.get("evt",""); fn2=s.get("fn","?"); t2=s.get("t",0)
                    loc2=s.get("loc",{}); mets2=s.get("mets",{}); lbl2=s.get("label","") or ""
                    ecol=(red if "CRASH" in evt2 else green if "SNAP" in evt2 else yellow)
                    print(f"{dim(f't={t2:.4f}s')}  {ecol('['+evt2[:12]+']')}  {bold(fn2+'()')}"
                          +(f"  {green('['+lbl2+']')}" if lbl2 and not lbl2.startswith("_tc") else ""),flush=True)
                    show_loc={k:vd for k,vd in loc2.items() if not k.startswith("__") and (vars_filter is None or k in vars_filter)}
                    for k2,vd2 in list(show_loc.items())[:4]:
                        print(f"    {cyan(k2)} = {vd2.get('r','?')[:60]}  {dim('['+vd2.get('t','')[:8]+']')}")
                    if mets2: print(f"    {dim('mets:')} {'  '.join(f'{k}={v}' for k,v in list(mets2.items())[:4])}")
                    if show_sys and s.get("sys"):
                        sys2=s["sys"]
                        print(f"    {dim('sys:')} RAM={sys2.get('ram_mb','?')}MB  CPU={sys2.get('cpu_pct','?')}%  GPU={sys2.get('gpu_mb','?')}MB")
            except (EOFError, pickle.UnpicklingError, gzip.BadGzipFile): pass
            except Exception as e: print(f"  [tail error] {e}",flush=True)
    except KeyboardInterrupt:
        print(f"\n[TC tail] stopped.  {last_n} total snapshots seen.")


def heatmap(path=None, var_name=None, width=60):
    """ASCII heatmap of numeric variables over time."""
    d=load(path); snaps=d.get("snaps",[])
    series=defaultdict(list)
    for s in snaps:
        for k,vd in s.get("loc",{}).items():
            if k.startswith("__"): continue
            if var_name and k!=var_name: continue
            try:
                v=float(vd.get("r","")); 
                if not math.isnan(v) and not math.isinf(v): series[k].append((s.get("t",0),v))
            except Exception: pass
        for k,v in s.get("mets",{}).items():
            if var_name and k!=var_name: continue
            if isinstance(v,(int,float)) and not math.isnan(v) and not math.isinf(v): series[k].append((s.get("t",0),float(v)))
    if not series: print(f"[TC] No numeric data for heatmap"+( f" (var: {var_name})" if var_name else "")); return
    dur=d.get("duration",1) or 1; src=os.path.basename(d.get("src","?"))
    print(f"\n  timecapsule heatmap — {src}"); print("  "+"─"*width)
    for vname,pts in sorted(series.items()):
        if len(pts)<2: continue
        vals=[v for _,v in pts]; mn,mx=min(vals),max(vals); rng=mx-mn or 1
        sampled=[vals[int(i*len(vals)/width)] for i in range(width)]
        chars=[]
        for v in sampled:
            norm=(v-mn)/rng
            if USE_COLOR:
                c=("\033[34m" if norm<0.33 else "\033[33m" if norm<0.66 else "\033[31m")
                chars.append(c+SPARKS[int(norm*(len(SPARKS)-1))]+"\033[0m")
            else:
                chars.append(SPARKS[int(norm*(len(SPARKS)-1))])
        mean_v=sum(vals)/len(vals); std_v=math.sqrt(sum((v-mean_v)**2 for v in vals)/len(vals)) if len(vals)>1 else 0
        print(f"  {cyan(bold(f'{vname:<18}'))} ▕{''.join(chars)}▏  {dim(f'{mn:.3g}…{mx:.3g}')}  μ={mean_v:.3g} σ={std_v:.3g}")
    print("  "+"─"*width)
    print(f"  t=0{'─'*(width-12)}{dim(f'{dur:.2f}s')}\n")


def deadcode(path=None):
    """Find functions seen in stacks but never timed — possible dead code."""
    d=load(path); fn_c=d.get("fn_calls",{}); snaps=d.get("snaps",[]); src=os.path.basename(d.get("src","?"))
    seen_in_stacks=set()
    for s in snaps:
        for frame in s.get("stk",[]):
            fn=frame.get("fn","")
            if fn and fn not in ("<module>","<lambda>","<listcomp>","<genexpr>","<dictcomp>"): seen_in_stacks.add(fn)
    called_fns={k:v for k,v in fn_c.items() if k not in ("<module>","<lambda>","<listcomp>","<genexpr>","<dictcomp>")}
    never_called=sorted(seen_in_stacks-set(called_fns.keys()))
    called_once=sorted(fn for fn,cnt in called_fns.items() if cnt==1)
    hot_fns=sorted(((fn,cnt) for fn,cnt in called_fns.items() if cnt>100),key=lambda x:-x[1])
    print(f"\n  deadcode — {src}"); print("  "+"─"*60)
    if never_called:
        print(f"  {yellow(bold('NEVER TIMED'))} ({len(never_called)} — seen in stacks but no timing recorded):")
        for fn in never_called[:15]: print(f"    {dim('·')} {fn}()")
    else: print(f"  {green('✓ No dead functions detected')}")
    if called_once:
        print(f"\n  {yellow('CALLED ONCE')} ({len(called_once)}):")
        for fn in called_once[:10]: print(f"    {dim('·')} {fn}()  {dim('(single use?)')}")
    if hot_fns:
        print(f"\n  {red('HOT')} (>100 calls):")
        for fn,cnt in hot_fns[:8]:
            bar="█"*min(30,cnt//10)
            print(f"    {cyan(fn+'()')} ×{cnt:,}  {red(bar)}")
    print("  "+"─"*60)
    return {"never_called":never_called,"called_once":called_once,"hot":[(fn,cnt) for fn,cnt in hot_fns]}


def regression_check(path=None, *, baseline_path=None, max_slowdown_pct=20.0,
                     max_ram_growth_mb=50.0, fail_on_crash=True, exit_code=True):
    """CI/CD regression check. Exits with code 1 on failure."""
    cur=load(path)
    if baseline_path: base=load(baseline_path); base_ft={fn:{"avg_ms":sum(t*1000 for t in v)/len(v)} for fn,v in base.get("fn_timings",{}).items() if v}; base_ram=base.get("ram_peak",0); base_crash=bool(base.get("crash"))
    else:
        try:
            with open(_BASELINE_FILE) as f: store=_json.load(f)
            if not store: print("[TC] No baseline."); return True
            rec=list(store.values())[-1]; base_ft=rec.get("fn_timings",{}); base_ram=rec.get("ram_peak",0); base_crash=rec.get("crash",False)
        except Exception: print("[TC] No baseline found. Run tc.baseline() first."); return True
    src=os.path.basename(cur.get("src","?")); failures=[]; print(f"\n  regression_check — {src}"); print("  "+"─"*55)
    if fail_on_crash and cur.get("crash"):
        c=cur["crash"]; msg=f"CRASH: {c['exc']}: {c['msg'][:40]}"
        failures.append(msg); print(f"  {red(bold('✗ '+msg))}")
    else: print(f"  {green('✓ No crash')}")
    cur_ft={fn:sum(t*1000 for t in v)/len(v) for fn,v in cur.get("fn_timings",{}).items() if v}
    perf_ok=True
    for fn in cur_ft:
        if fn not in base_ft: continue
        bav=base_ft[fn] if isinstance(base_ft[fn],(int,float)) else base_ft[fn].get("avg_ms",0)
        if bav==0: continue
        pct=(cur_ft[fn]-bav)/bav*100
        if pct>max_slowdown_pct:
            msg=f"SLOW: {fn}() +{pct:.1f}% ({bav:.2f}→{cur_ft[fn]:.2f}ms)"
            failures.append(msg); print(f"  {red('✗ '+msg)}"); perf_ok=False
    if perf_ok: print(f"  {green(f'✓ No timing regressions (threshold: {max_slowdown_pct}%)')}")
    cur_ram=cur.get("ram_peak",0)
    if base_ram and cur_ram-base_ram>max_ram_growth_mb:
        msg=f"RAM: {base_ram:.0f}→{cur_ram:.0f}MB (+{cur_ram-base_ram:.0f}MB > {max_ram_growth_mb:.0f}MB limit)"
        failures.append(msg); print(f"  {red('✗ '+msg)}")
    else: print(f"  {green(f'✓ RAM OK (peak {cur_ram:.0f}MB)')}")
    print("  "+"─"*55)
    passed=len(failures)==0
    if passed: print(f"  {green(bold('✓ ALL CHECKS PASSED'))}\n")
    else:
        print(f"  {red(bold(f'✗ {len(failures)} FAILURE(S)'))}")
        for f2 in failures: print(f"    • {f2}")
        if exit_code: sys.exit(1)
    return passed


def speedtest(n_iters=5000):
    """Benchmark tc_recorder overhead on this machine."""
    import timeit
    code="""
x=0
for i in range(1000):
    x+=i*2
    if x>1_000_000: x=0
"""
    t_base=timeit.timeit(code, number=n_iters//1000)/(n_iters//1000)*1000
    print(f"[TC speedtest] Baseline (no recording): {t_base:.3f}ms/1000 iters")
    snap_times=[]
    f=sys._getframe(0)
    for _ in range(500):
        t0=time.perf_counter_ns()
        loc={}
        for k,v in f.f_locals.items():
            if k[0]=="_": continue
            try: loc[k]={"r":repr(v)[:200],"t":type(v).__name__}
            except Exception: pass
        snap_times.append(time.perf_counter_ns()-t0)
    avg_ns=sum(snap_times)/len(snap_times)
    med_ns=sorted(snap_times)[len(snap_times)//2]
    p95_ns=sorted(snap_times)[max(0,int(len(snap_times)*.95)-1)]
    print(f"[TC speedtest] Variable capture:         avg={avg_ns/1000:.1f}µs  p50={med_ns/1000:.1f}µs  p95={p95_ns/1000:.1f}µs")
    print(f"[TC speedtest] Backend:  {_c_info()}")
    print(f"[TC speedtest] Platform: {sys.platform}  Python {sys.version.split()[0]}")


# ─────────────────────────────────────────────────────────────────────────────
#  PROFILE (per-line microsecond profiler)
# ─────────────────────────────────────────────────────────────────────────────
_PROFILE_DATA = defaultdict(list)
_PROFILE_ON   = [False]
_PROFILE_PREV = [None, 0]

def profile(enable=True):
    """Toggle per-line profiling. Call profile_report() for results."""
    _PROFILE_ON[0] = enable
    if enable: sys.settrace(_profile_trace); print("[TC] per-line profiling ON", file=sys.stderr)
    else:      sys.settrace(None);           print(f"[TC] per-line profiling OFF  ({len(_PROFILE_DATA)} lines)", file=sys.stderr)

def _profile_trace(frame, event, arg):
    if not _PROFILE_ON[0]: return None
    now = time.perf_counter_ns()
    if event in ("line","return"):
        prev_key, prev_time = _PROFILE_PREV
        if prev_key is not None and prev_time:
            _PROFILE_DATA[prev_key].append(now-prev_time)
        key = (frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name)
        _PROFILE_PREV[0]=key; _PROFILE_PREV[1]=now
    return _profile_trace

def profile_report(top_n=20, min_us=0.1):
    """Print per-line timing report after tc.profile()."""
    if not _PROFILE_DATA: print("[TC] No profile data. Run tc.profile(True) first."); return
    W=100; print(f"\n{'═'*W}\n  tc.profile_report  —  {len(_PROFILE_DATA)} lines  top {top_n}\n{'═'*W}")
    print(f"  {'TOTAL(ms)':>10} {'AVG(µs)':>9} {'P95(µs)':>9} {'CALLS':>7}  FILE:LINE  FUNCTION"); print("─"*W)
    rows=[]
    for (fname,lineno,fn),times in _PROFILE_DATA.items():
        total_us=sum(times)/1000; avg_us=total_us/len(times)
        if avg_us<min_us: continue
        p95_us=sorted(times)[max(0,int(len(times)*.95)-1)]/1000
        rows.append((total_us,avg_us,p95_us,len(times),fname,lineno,fn))
    rows.sort(reverse=True); max_tot=rows[0][0] if rows else 1
    for (tot,avg,p95,nc,fname,lineno,fn) in rows[:top_n]:
        bar="█"*int(min(1.0,tot/max_tot)*20)
        print(f"  {tot/1000:>10.3f} {avg:>9.2f} {p95:>9.2f} {nc:>7,}  {os.path.basename(fname)}:{lineno}  {fn}()  {cyan(bar)}")
    print(f"{'═'*W}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    import argparse as _ap
    p = _ap.ArgumentParser(prog="python tc_recorder.py",
        description="timecapsule v6 — terminal analysis",
        formatter_class=_ap.RawDescriptionHelpFormatter,
        epilog="""
REPORT:
  run.tc                        full 15-section report
  run.tc --export out.txt       save to file
  run.tc --full                 all rows, no limits
  run.tc --top 25               25 rows per table

ANALYSIS:
  run.tc --callgraph            ASCII call graph
  run.tc --coverage             annotated source coverage
  run.tc --loops                O(n²) and stuck loop detection
  run.tc --correlations         variable correlation matrix
  run.tc --exception-chain      root cause analysis
  run.tc --leak-check           memory leak detector
  run.tc --deadcode             unused function detection
  run.tc --heatmap [VAR]        ASCII heatmap
  run.tc --search QUERY         search snapshots
  run.tc --history VAR          variable value history
  run.tc --anomaly METRIC       anomaly detection
  run.tc --hotlines             hottest source lines
  run.tc --flamegraph           export speedscope JSON
  run.tc --json                 export JSON

COMPARISON:
  before.tc --patch-summary after.tc
  run.tc --baseline             save as baseline
  run.tc --diff-baseline        compare to baseline
  run.tc --regression-check [baseline.tc]

LIVE:
  run.tc --tail                 live tail
  run.tc --watch VAR [VAR ...]  watch specific variables

UTILITY:
  run.tc --summary              one-line summary
  run.tc --explain              plain-English explanation
  run.tc --for-claude           export for AI analysis
  run.tc --speedtest            benchmark overhead
  run.tc --profile-report       per-line profiling results
""")
    p.add_argument("file",  nargs="?")
    p.add_argument("file2", nargs="?")
    p.add_argument("--export",           metavar="FILE")
    p.add_argument("--full",             action="store_true")
    p.add_argument("--top",              type=int, default=15)
    p.add_argument("--callgraph",        action="store_true")
    p.add_argument("--coverage",         action="store_true")
    p.add_argument("--loops",            action="store_true")
    p.add_argument("--correlations",     action="store_true")
    p.add_argument("--exception-chain",  action="store_true")
    p.add_argument("--leak-check",       action="store_true")
    p.add_argument("--deadcode",         action="store_true")
    p.add_argument("--heatmap",          metavar="VAR", nargs="?", const="")
    p.add_argument("--search",           metavar="QUERY")
    p.add_argument("--history",          metavar="VAR")
    p.add_argument("--anomaly",          metavar="METRIC")
    p.add_argument("--hotlines",         action="store_true")
    p.add_argument("--flamegraph",       action="store_true")
    p.add_argument("--json",             action="store_true")
    p.add_argument("--patch-summary",    action="store_true")
    p.add_argument("--baseline",         action="store_true")
    p.add_argument("--diff-baseline",    action="store_true")
    p.add_argument("--regression-check", metavar="BASELINE", nargs="?", const="")
    p.add_argument("--tail",             action="store_true")
    p.add_argument("--watch",            nargs="+", metavar="VAR")
    p.add_argument("--summary",          action="store_true")
    p.add_argument("--explain",          action="store_true")
    p.add_argument("--for-claude",       action="store_true")
    p.add_argument("--speedtest",        action="store_true")
    p.add_argument("--profile-report",   action="store_true")
    p.add_argument("--dump",             action="store_true")
    a = p.parse_args()
    path = a.file or None; path2 = a.file2 or None

    if a.speedtest:          speedtest(); return
    if a.summary:            summary(path); return
    if a.explain:            explain(path, verbose=True); return
    if a.baseline:           baseline(path); return
    if a.diff_baseline:      diff_baseline(path); return
    if a.profile_report:     profile_report(top_n=a.top); return
    if a.callgraph:          callgraph(path, export=a.export); return
    if a.coverage:           coverage_report(path, export=a.export); return
    if a.loops:              loop_detector(path); return
    if a.correlations:       variable_correlations(path, export=a.export); return
    if a.exception_chain:    exception_chain(path, export=a.export); return
    if a.leak_check:         memory_leak_check(path); return
    if a.deadcode:           deadcode(path); return
    if a.for_claude:         export_for_claude(path); return
    if a.flamegraph:         flamegraph(path); return
    if a.dump:               dump(path); return
    if a.tail:               tail(path); return
    if a.json:               export_json(path); return
    if a.hotlines:
        items=hotlines(path, top_n=a.top)
        print(f"\n  Hot lines ({len(items)}):")
        for hits,file2,line2 in items: print(f"  {red(f'{hits:>8,}')}  {dim(file2[-30:])}:{yellow(str(line2))}")
        return
    if a.patch_summary:
        if not path2: print("Usage: tc_recorder.py before.tc --patch-summary after.tc"); return
        patch_summary(path, path2, export=a.export); return
    if a.regression_check is not None:
        regression_check(path, baseline_path=a.regression_check or None); return
    if a.heatmap is not None:
        heatmap(path, a.heatmap or None); return
    if a.search:
        hits=search(path, a.search)
        print(f"\n[TC] '{a.search}' → {len(hits)} result(s)\n")
        for s in hits[:50]:
            print(f"  #{s.get('i',0):04d}  t={s.get('t',0):.4f}s  [{s.get('evt','')}]  {s.get('fn','?')}():{s.get('ln',0)}"
                  +(f"  [{s['label']}]" if s.get("label") else ""))
            for k,vd in s.get("loc",{}).items():
                if not k.startswith("__") and a.search.lower() in str(vd.get("r","")).lower():
                    print(f"         {k} = {vd.get('r','')[:60]}")
        return
    if a.history:
        hist=history(path, a.history)
        print(f"\n[TC] history of '{a.history}'  ({len(hist)} values)\n")
        prev=None
        for t2,v in hist:
            marker="~" if repr(v)!=repr(prev) else " "
            print(f"  {marker}  t={t2:.6f}s  {repr(v)[:80]}")
            prev=v
        return
    if a.anomaly:
        anoms=anomalies(path, a.anomaly)
        print(f"\n[TC] anomalies in '{a.anomaly}' (z>2.5) → {len(anoms)} found\n")
        for t2,v,z in anoms:
            bar="█"*min(20,int(z*2))
            print(f"  t={t2:.4f}s  value={v:.6g}  z={z:.2f}  {red(bar)}")
        return
    if a.watch:
        d=load(path); snaps=d.get("snaps",[])
        print(f"\n[TC] watching {a.watch} through {len(snaps)} snapshots\n")
        prev_vals={v:None for v in a.watch}
        for s in snaps:
            for var in a.watch:
                if var in s.get("loc",{}):
                    new_r=s["loc"][var].get("r","")
                    if new_r!=prev_vals[var]:
                        old_s=f"{prev_vals[var][:20]} → " if prev_vals[var] else ""
                        print(f"  t={s.get('t',0):.4f}s  {cyan(var)}:  {old_s}{yellow(new_r[:50])}")
                        prev_vals[var]=new_r
        return
    # default: full report
    report(path, export=a.export, full=a.full, top_n=a.top)


if __name__ == "__main__":
    _cli()
