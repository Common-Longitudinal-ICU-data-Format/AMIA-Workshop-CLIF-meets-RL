# dev/

Marimo dev versions of the workshop notebooks. Use these for fast iteration;
the canonical Colab-ready Jupyter notebooks live in [`../code/`](../code/).

| File | Mirror of |
|---|---|
| `01_clif_rl_cohort_wide_dev.py` | `code/01_clif_rl_cohort_wide.ipynb` |
| `02_clif_rl_training_dev.py`    | `code/02_clif_rl_training.ipynb` |
| `03_icu_readmission_dev.py`     | `code/03_icu_readmission_demo.ipynb` |

## Run

```bash
# Interactive (recommended for iteration)
uv run marimo edit dev/01_clif_rl_cohort_wide_dev.py

# As a script (smoke test — runs every cell top to bottom, no UI)
uv run python dev/01_clif_rl_cohort_wide_dev.py

# Lint
uvx marimo check dev/01_clif_rl_cohort_wide_dev.py
```

Both files share the same `config.json` + demo fallback resolution as the .ipynbs.

## Why marimo

- **Reactive.** Editing a cell re-runs only downstream cells.
- **Pure Python files.** Diffs cleanly in git; no JSON merge conflicts; no risk
  of committing cell outputs.
- **Script-mode execution.** A single `python dev/0X_*.py` runs the full
  pipeline end-to-end, which is the simplest possible smoke test.

For attendees who don't use marimo: the `.ipynb` files in `code/` are the
canonical deliverable. The dev files are author-side scaffolding.
