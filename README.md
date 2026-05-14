# AMIA TRI08 — CLIF Meets RL Workshop

A hands-on workshop on using the [Common Longitudinal ICU data Format (CLIF)](https://clif-icu.com/) and the [`clifpy`](https://common-longitudinal-icu-data-format.github.io/clifpy/) Python package to do ICU data science, with a path forward into reinforcement learning for mechanical ventilation.

**Session:** AMIA Informatics Summit 2026 — TRI08
**Presenters:** Kaveri Chhikara (University of Chicago) · Saki Amagai (Northwestern University) · Yikuan Li (George Mason University)
**Slide deck:** [`slides/TRI08.pdf`](slides/TRI08.pdf)

## CLIF version

2.1

## Objective

Teach workshop attendees how to:

1. Load CLIF tables and run built-in Data Quality Assessment (DQA) with one `ClifOrchestrator` call.
2. Identify a cohort using `clifpy`'s utilities.
3. Build a wide, hourly-resampled dataset that is suitable as RL state input.

## What's in this repo

| Path | Purpose |
|---|---|
| [`code/`](code/) | Production Colab-ready Jupyter notebooks. Run these. |
| [`dev/`](dev/) | Marimo dev versions of the same notebooks for fast iteration. |
| [`config/`](config/) | Site config template (`config_template.json`). |
| [`utils/`](utils/) | Python helpers (`config.py`). |
| [`slides/`](slides/) | The TRI08 deck. |
| [`output/`](output/) | Local outputs (gitignored). |

## Workshop arc

1. **Notebook 01 — Build the CLIF-RL cohort.** Live with Kaveri. Tour of `clifpy` from raw tables to two analysis-ready dataframes (`hourly_df` + `static_df`).
2. **Saki's deep-dive demo** (separate session).
3. **Notebook 02 — RL training setup + ATS validation.** A walk-through of the action/reward/network/trainer code (not executed on demo data) and the three-site forest plot from the ATS submission.
4. **Notebook 03 — ICU readmission cohort** (bonus). A different cohort flavour from the RL pipeline — purely descriptive, showing `clifpy` for a non-RL use case.

## Required CLIF tables

| Table | Used in |
|---|---|
| `patient` | both notebooks |
| `hospitalization` | both notebooks |
| `adt` | 01 |
| `respiratory_support` | 01 + 02 |
| `vitals` | 01 + 02 |
| `labs` | 01 + 02 |
| `hospital_diagnosis` | 01 (Charlson CCI) |

For the live workshop attendees use the demo dataset bundled inside `clifpy`, which contains all of the above (see *Quick start — Colab* below).

## Quick start — Google Colab (zero setup)

Both notebooks default to the CLIF demo dataset that ships inside `clifpy` (~3 MB, derived from MIMIC-IV, openly accessible). No PhysioNet credentials, no Drive mount, no `git clone` — just open and run.

| Notebook | Open in Colab |
|---|---|
| 01 — Build the CLIF-RL cohort | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Common-Longitudinal-ICU-data-Format/AMIA-Workshop-CLIF-meets-RL/blob/main/code/01_clif_rl_cohort_wide.ipynb) |
| 02 — RL training setup + ATS results | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Common-Longitudinal-ICU-data-Format/AMIA-Workshop-CLIF-meets-RL/blob/main/code/02_clif_rl_training.ipynb) |
| 03 — ICU readmission cohort (bonus) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Common-Longitudinal-ICU-data-Format/AMIA-Workshop-CLIF-meets-RL/blob/main/code/03_icu_readmission_demo.ipynb) |

Once a notebook opens in Colab: **Runtime → Run all**. The first cell (`!pip install -q ...`) takes ~20 s the first time; everything else runs in under a minute on the free CPU runtime.

## Local install — to run on your own CLIF data

The repo ships a `pyproject.toml` + `uv.lock` pinning the same dependency versions used by the upstream [`CLIF-RL` project that is WIP](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF-RL/tree/ats-submission) where they overlap (torch, pandas, numpy, pyarrow, duckdb, marimo), so the workshop training notebook is bit-for-bit reproducible against that pipeline.

**Recommended — [uv](https://docs.astral.sh/uv/):**

```bash
git clone https://github.com/Common-Longitudinal-ICU-data-Format/AMIA-Workshop-CLIF-meets-RL.git
cd AMIA-Workshop-CLIF-meets-RL

# Runtime deps only (notebooks)
uv sync

# Or with dev tools (marimo, nbstripout, pre-commit)
uv sync --extra dev

# Activate so `jupyter`, `python`, etc. resolve from .venv/
source .venv/bin/activate
# (or prefix every command with `uv run`, e.g. `uv run jupyter notebook`)
```

**Alternative — plain venv + pip** (reads dependencies from `pyproject.toml`; gets the floor versions, not the exact `uv.lock` pins):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install .
```

**Set up nbstripout so notebook outputs never get committed:**

```bash
pre-commit install
nbstripout --install
```

Point the notebooks at your CLIF parquet directory:

```bash
cp config/config_template.json config/config.json
# edit config/config.json — set tables_path to your CLIF directory
```

Then `jupyter notebook code/01_clif_rl_cohort_wide.ipynb`. The notebooks auto-detect `config/config.json` and use its `tables_path`; if `config.json` is absent or `use_demo: true`, they fall back to the bundled demo. See [`config/README.md`](config/README.md) for the schema.

## What each notebook does

### 01 — Build the CLIF-RL cohort

Mirrors [`CLIF-RL/code/00_cohort.py`](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF-RL/blob/ats-submission/code/00_cohort.py). Cohort:

- Adults ≥ 18 at admission
- At least one invasive mechanical ventilation (IMV) record
- No tracheostomy at the time of the first IMV record
- *(Production CLIF-RL also requires height + weight for IBW; we skip that filter on the demo path because the demo dataset has height for only ~24 % of hospitalizations — applying it would gut the cohort.)*

Pipeline: load tables → STROBE waterfall → respiratory-support waterfall → wide dataset → hourly grid → LOCF + normal-value imputation → baseline SOFA + Charlson CCI → static dataframe.

Outputs: `hourly_df` (RL state grid, 0 NaN, 22 columns) and `static_df` (hospitalization-level demographics + outcome + SOFA + CCI). Plus a STROBE figure, Table 1, episode/severity/LOS histograms, vent set-point distributions, and a sample patient trajectory.

### 02 — RL training setup + ATS validation

Lifted from the [`ats-submission` branch of CLIF-RL](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF-RL/tree/ats-submission). Walks through (without executing) the production training contract:

- **Action**: 2 × 2 grid encoded 0–3 — mode (controlled vs uncontrolled) × oxygenation support (low vs high PEEP/FiO₂).
- **State**: 20 features on the demo (44 in the ATS submission, the demo dataset doesn't carry every continuous-med flag).
- **Reward**: terminal only (+1 survived, −1 died).
- **Agent**: dueling Q-network + Double DQN with Huber loss, soft target update.
- **Eval**: physician–agent per-hospitalization concordance → adjusted logistic regression of in-hospital mortality.

Then renders the ATS submission's three-site external-validation forest plot:

> Across three CLIF-standardized ICUs, every +10 percentage-point increase in clinician–agent concordance was associated with a **3–8 % lower adjusted odds of in-hospital mortality** (all p ≤ 0.006).

The forest plot data is inline in the notebook (Fig 1A of the poster); the slide deck has the higher-resolution version.

### 03 — ICU readmission cohort (bonus)

Adapted from the [CLIF ICU Readmission project](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF_icu_readmission). A different cohort flavour, purely descriptive:

- Adults ≥ 18 at admission
- At least one ICU stay during the hospitalization
- Exclude: died during index ICU stay
- Exclude: discharged immediately after index ICU stay

Outputs: STROBE waterfall, Table 1 by readmission status, patient-journey Sankey, inpatient-mortality comparison bar chart.

## A note on dates

MIMIC and CLIF-MIMIC use de-identified per-patient date shifts. The notebooks deliberately do **not** filter by absolute calendar year — that would be meaningless on MIMIC. Time intervals within a hospitalization are still real and used freely (vent episode duration, hours since first IMV, etc.).

## Author / maintainer

[Kaveri Chhikara](mailto:kaveri.chhikara@gmail.com), University of Chicago — questions / issues welcome on this repo's GitHub issue tracker.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).

## Related projects

- [`clifpy`](https://github.com/Common-Longitudinal-ICU-data-Format/CLIFpy) — Python interface to CLIF
- [`CLIF-RL`](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF-RL) — the full RL pipeline this workshop teases
- [`CLIF_icu_readmission`](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF_icu_readmission) — readmission methodology
