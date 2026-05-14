# code/

Production workshop notebooks (Colab-ready). Run them top-to-bottom.

| Notebook | What it builds |
|---|---|
| [`01_clif_rl_cohort_wide.ipynb`](01_clif_rl_cohort_wide.ipynb) | Builds the CLIF-RL ventilation cohort end-to-end: STROBE waterfall, respiratory-support waterfall, wide dataset, hourly state grid (zero NaN after imputation), baseline SOFA, Charlson CCI, Table 1, cohort-characterization figures, sample patient trajectory. Outputs `hourly_df` + `static_df`. |
| [`02_clif_rl_training.ipynb`](02_clif_rl_training.ipynb) | RL training setup walkthrough — action encoding, terminal reward, dueling Q-net, Double DQN trainer, concordance evaluation — none executed on demo data. Then renders the ATS submission's three-site external-validation forest plot. |
| [`03_icu_readmission_demo.ipynb`](03_icu_readmission_demo.ipynb) | Bonus notebook: ICU-readmission cohort on the demo data, with Table 1 by readmission status, a patient-journey Sankey, and an inpatient-mortality comparison. Adapted from the [CLIF ICU Readmission project](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF_icu_readmission). |

## How they pick a data source

Both notebooks call a small loader (`resolve_data_dir`) that:

1. Reads `config/config.json` (one level up) if present and `use_demo` is not `true`,
   and uses its `tables_path`.
2. Otherwise falls back to the CLIF demo dataset bundled inside `clifpy`.

So the same notebooks run as-is on Google Colab (no `config.json` → demo data)
and as-is after `git clone` with a `config/config.json` pointing at your CLIF
parquet directory.

## Dev versions

The marimo dev versions live in [`../dev/`](../dev/) and stay in sync with these
.ipynbs.

## Outputs

Notebooks render results inline. `nbstripout` (installed at the repo root)
strips cell outputs on commit, so the .ipynb files stay clean in git.
