# CLAUDE.md — PySHS Development Guide

## Project overview

PySHS is a Python library that simplifies survey data analysis for social science researchers. It targets French-speaking users and prioritizes **simplicity**, **readability**, and **reproducibility**.

## Core principles

- **Simple API**: Each function takes a DataFrame + column names and returns a DataFrame. No complex objects or chained methods.
- **Readable output**: Results are formatted tables ready for publication (labels, totals, percentages). French is the default language (`langue = "fr"`).
- **Reproducibility**: Weighted computations are first-class (every statistical function supports a `poids` parameter). Rounding is explicit via the `ro` parameter.

## Architecture

- All logic lives in `pyshs/_core.py` — single flat module, no class hierarchy.
- `pyshs/__init__.py` re-exports public functions and defines `__version__`.
- Keep it flat: prefer adding functions to `_core.py` over creating submodules.

## Coding conventions

- Function and parameter names are in **French** (e.g. `tri_a_plat`, `poids`, `tableau_croise`).
- Docstrings follow **numpydoc** format and are written in French.
- Type hints are used on function signatures (use `str | None` union syntax).
- Input validation: raise `Exception` or `warnings.warn` with a French message when inputs are invalid.
- When no weight is provided, functions internally create a unit weight column (`df["poids"] = 1`) rather than branching logic.

## Key patterns

- DataFrames are never mutated in place — use `df = df.copy()` before modifying.
- Statistical outputs combine absolute values and percentages in formatted strings (e.g. `"42 (13.5%)"`).
- The `langue` module-level variable controls output labels (`"fr"` or `"en"`).
- Column names with special characters are handled via `Q('...')` (patsy quoting).

## Dependencies

pandas, numpy, statsmodels, scipy, plotly, samplics, matplotlib, seaborn, adjustText, openpyxl, regex.

Do not add new dependencies without strong justification.

## Testing

- Tests use **pytest** with fixtures. Run: `pytest test/`
- Tests are basic (type/shape checks). New functions should include at least a smoke test.

## Build & publish

- Build system: **flit** (configured in `pyproject.toml`)
- Install in dev mode: `pip install -e .`
