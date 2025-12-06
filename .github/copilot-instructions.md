**Repository Overview**

- **Purpose**: `quant_analytics_torch` is a PyTorch-based quantitative analytics library (derivative pricing, model calibration, and ML hedging). Top-level package is `quant_analytics_torch` with focused subpackages: `analytics`, `calculators`, `instruments`, `interpolators`, `marketdata`, `models`, `modules`, and `sandbox`.

- **Where to look**: key code lives under `quant_analytics_torch/` and example notebooks are under `examples/`. Documentation sources are under `docs/source/`.

**Typical Workflows & Commands**

- **Install deps (dev)**: `pip install -r requirements.txt` (note: `nvmath-python[cu12]` is CUDA-specific; pick the correct package for your GPU/CPU environment).
- **Run tests**: `pytest --cov-report term --cov=quant_analytics_torch tests/ --html=./test-reports/report.html --cov-report=html:./test-reports/coverage --profile` (this exact command is included in `README.md`).
- **Build docs**: from repo root: `sphinx-apidoc -o docs/source/ quant_analytics_torch` then `make -C docs html` (or use Codespaces workflow if available).
- **Install package locally**: `pip install -e .` or `python setup.py develop`.

**Project Conventions & Patterns**

- Package layout follows common Python package conventions (setuptools `find_packages()` in `setup.py`). Keep modules inside `quant_analytics_torch/<subpkg>` and export public APIs via their `__init__.py` when appropriate.
- Test structure mirrors package structure under `tests/`. Use `pytest` style simple asserts (see `tests/basic_test.py`). Keep tests small and fast; run the whole test suite before opening PRs.
- Notebooks in `examples/` and `docs/source/examples/` demonstrate API usage and are considered canonical examples — update them when public API changes.

**Code Style & Implementation Notes**

- The code is idiomatic Python without type annotations; prefer clear, explicit function names and small helpers (look at modules like `quant_analytics_torch/calculators/*` for examples).
- Numerical code often relies on `numpy`/`torch` and SciPy; ensure operations are numerically stable and GPU/CPU behaviour is considered (tests assume CPU unless CUDA packages are installed).

**Integration Points & External Dependencies**

- Core ML backend: `torch` (PyTorch). Tests/development use `torch` from `requirements.txt`.
- Optional/accelerated math: `nvmath-python[cu12]` — environment dependent. Document any GPU-only behavior in PRs.
- Docs: Sphinx with `nbsphinx` for notebooks, `furo` theme.

**What to check before editing or opening PRs**

- Run unit tests (`pytest ...`) and ensure coverage doesn't drop for modified modules.
- If public API changes, update `docs/source/` examples and regenerate API docs via `sphinx-apidoc`.
- If you change package metadata, update `setup.py` version accordingly.

**Examples (quick references)**

- Small change to calculators: edit `quant_analytics_torch/calculators/europeanoptioncalculator.py` → run subset of tests for calculators: `pytest tests/calculators/ -q`.
- To add a new example notebook: place it under `examples/`, update `docs/source/examples` index, and ensure `nbsphinx` can render it.

**Notes for AI coding agents**

- No existing `.github/copilot-instructions.md` or AGENT files were found — create instructions that are concise, actionable, and reference the files above.
- Prefer minimal, focused edits. When adding functionality, include unit tests under `tests/` that mirror the package path.
- Use the `README.md` test command and `requirements.txt` as authoritative examples for running and installing the project.

If any part of the codebase seems unclear or you need a deeper walkthrough of a subpackage (e.g., `analytics` vs `calculators` responsibilities), tell me which module and I'll extract a short, focused guide.
