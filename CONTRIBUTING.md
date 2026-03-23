# Contributing to timecapsule

Thank you for wanting to make timecapsule better.

## Setup

```bash
git clone https://github.com/yourusername/timecapsule
cd timecapsule
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
pytest tests/test_recording.py -v   # specific file
pytest --cov=timecapsule            # with coverage
```

## Code style

```bash
black timecapsule/ tests/
ruff check timecapsule/ tests/
```

## Project structure

```
timecapsule/
├── timecapsule/
│   ├── __init__.py      public API surface (imports only)
│   ├── __main__.py      python -m timecapsule CLI entry
│   └── recorder.py      entire implementation (single file, zero deps)
├── tests/
│   ├── test_recording.py
│   ├── test_analysis.py
│   └── test_cli.py
├── docs/
│   ├── README.md        full documentation
│   ├── api.md           API reference
│   ├── internals.md     architecture notes
│   └── examples/        runnable example scripts
├── pyproject.toml
├── CHANGELOG.md
└── CONTRIBUTING.md
```

## What to contribute

**Most wanted:**
- C extension builds for Linux (`.so`) and macOS (`.dylib`)
- `psutil` integration as optional drop-in for `_sys_metrics`
- `asyncio` task-aware recording
- pytest plugin (`pip install timecapsule[pytest]`)

**Not currently wanted:**
- A web UI / HTML viewer (planned for a separate package)
- Changes to the `.tc` file format without a migration path
- New external dependencies in the core `recorder.py`

## Pull request checklist

- [ ] Tests pass (`pytest`)
- [ ] New behaviour has tests
- [ ] `black` and `ruff` pass
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No new required dependencies added to `recorder.py`

## Reporting bugs

Open a GitHub issue with:
1. Python version and OS
2. Minimal reproduction script
3. The `.tc` file if the bug is in analysis (attach or gist it)

## Questions

Open a GitHub Discussion — not an issue.
