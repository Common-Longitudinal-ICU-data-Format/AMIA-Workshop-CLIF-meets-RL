"""Dev marimo: build the CLIF-RL cohort + hourly wide dataset on demo data.

This is the opening notebook for the AMIA TRI08 workshop. It walks attendees
from raw CLIF tables to two finished dataframes:

    * `hourly_df`  — RL-ready state grid, zero NaN, ~5,000 rows.
    * `static_df`  — hospitalization-level descriptors with baseline SOFA + CCI.

Includes narrative sections, a STROBE diagram, Table 1, and cohort-characterization
figures so the notebook serves as a stand-alone teaching artifact.

Run as script:    python dev/01_clif_rl_cohort_wide_dev.py
Run interactive:  uv run marimo edit dev/01_clif_rl_cohort_wide_dev.py
"""

import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
    # Building a CLIF-RL cohort with `clifpy`

    **AMIA TRI08 — CLIF Meets RL Workshop**
    *Kaveri Chhikara · University of Chicago*

    Welcome. In the next ~30 minutes we'll go from raw CLIF tables to the kind of
    analysis-ready dataset that a downstream reinforcement-learning pipeline can
    consume — and we'll do it with code that's <100 lines per step, because
    `clifpy` packages the heavy lifting.

    ### What you'll leave with

    1. A working mental model of CLIF (federated CDM for ICU data) and `clifpy`
       (the consortium-standard Python interface).
    2. A reusable recipe for cohort building: load tables → apply
       inclusion/exclusion → STROBE waterfall.
    3. Exposure to `clifpy`'s higher-level utilities: respiratory-support
       waterfall, wide-dataset construction, hourly aggregation, SOFA computation,
       Charlson CCI.
    4. Two finished dataframes (`hourly_df`, `static_df`) ready for Saki's
       deep-dive in the next session.

    This notebook runs **as-is on Google Colab** — it uses the CLIF demo dataset
    bundled inside `clifpy`. No PhysioNet credentialing, no `git clone`, no Drive
    mount.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Why CLIF, and why `clifpy`?

    **CLIF** (Common Longitudinal ICU Format) is an open-source common data model
    for critical-care data, shared across 20+ health systems. It's federated by
    design — *code travels between sites; patient data does not*. Every CLIF
    site has the same 14 tables with the same column names and the same
    minimum Common ICU Data Elements (mCIDE) vocabulary, so an analysis script
    written once can be distributed and run anywhere.

    **`clifpy`** is the Python interface. One install gives you:

    | What | Why it matters |
    |---|---|
    | Typed table objects (`Patient`, `Vitals`, `Labs`, …) | Schema-validated I/O; no more `pd.read_parquet` boilerplate. |
    | `ClifOrchestrator` | Load N tables in one call; consistent timezone handling. |
    | DQA framework (`validate_all`) | 20+ rules, 2,000+ checks (Kahn et al. 2016 adapted). |
    | `process_resp_support_waterfall` | Clean + scaffold the messy `respiratory_support` table. |
    | `create_wide_dataset` + `convert_wide_to_hourly` | Pivot, join, time-window-filter into a regular state grid. |
    | `compute_sofa_polars` | SOFA scores, one row per encounter, fully vectorized. |
    | `calculate_cci` | Charlson Comorbidity Index from ICD-10-CM codes. |

    The rest of the notebook is a tour of these.
    """
    )
    return


@app.cell
def _():
    import clifpy
    from clifpy import ClifOrchestrator
    from pathlib import Path
    import json, sys
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    import polars as pl
    from tableone import TableOne

    def resolve_data_dir():
        """Find the CLIF parquet directory.

        Priority: config/config.json if present (local clone), otherwise the bundled
        clifpy demo dataset (Colab / first-run default).
        """
        candidates = [
            Path.cwd() / "config" / "config.json",
            Path.cwd().parent / "config" / "config.json",
        ]
        for cp in candidates:
            if cp.exists():
                with open(cp) as fh:
                    cfg = json.load(fh)
                if not cfg.get("use_demo", False):
                    return (
                        cfg["tables_path"],
                        cfg.get("file_type", "parquet"),
                        cfg.get("timezone", "US/Eastern"),
                        str(cp),
                    )
        return (
            str(Path(clifpy.__file__).parent / "data" / "clif_demo"),
            "parquet",
            "US/Eastern",
            "bundled_clifpy_demo",
        )

    DATA_DIR, FILE_TYPE, TIMEZONE, _src = resolve_data_dir()
    print(f"Data source: {_src}")
    print(f"DATA_DIR    = {DATA_DIR}")
    return (
        ClifOrchestrator, DATA_DIR, FILE_TYPE, TIMEZONE,
        FancyBboxPatch, TableOne, np, pd, pl, plt,
    )


@app.cell
def _(mo):
    mo.md(
        """
    ## 1. Load CLIF tables

    For this pipeline we need seven CLIF tables. With `clifpy` it's one
    `ClifOrchestrator.initialize(...)` call — typed objects, parsed timezones,
    schema validation, all in one.

    | Table | Why we need it |
    |---|---|
    | `patient` | demographics + `death_dttm` (for mortality outcome) |
    | `hospitalization` | admission/discharge timestamps + `age_at_admission` |
    | `adt` | ICU/ward/ED location timeline |
    | `respiratory_support` | device, mode, set-points (the action proxy) |
    | `vitals` | HR, RR, BP, SpO₂, temp (physiology state) |
    | `labs` | ABG, electrolytes, hemoglobin (lab state) |
    | `hospital_diagnosis` | ICD-10-CM codes for Charlson CCI |
    """
    )
    return


@app.cell
def _(ClifOrchestrator, DATA_DIR, FILE_TYPE, TIMEZONE):
    co = ClifOrchestrator(
        data_directory=DATA_DIR,
        filetype=FILE_TYPE,
        timezone=TIMEZONE,
    )
    co.initialize(tables=[
        "patient", "hospitalization", "adt",
        "respiratory_support", "vitals", "labs", "hospital_diagnosis",
    ])
    print("Loaded tables:", co.get_loaded_tables())
    print(f"\nDemo dataset shape:")
    print(f"  patient              {co.patient.df.shape}")
    print(f"  hospitalization      {co.hospitalization.df.shape}")
    print(f"  adt                  {co.adt.df.shape}")
    print(f"  respiratory_support  {co.respiratory_support.df.shape}")
    print(f"  vitals               {co.vitals.df.shape}")
    print(f"  labs                 {co.labs.df.shape}")
    print(f"  hospital_diagnosis   {co.hospital_diagnosis.df.shape}")
    return (co,)


@app.cell
def _(mo):
    mo.md(
        """
    ## 2. Build the CLIF-RL cohort

    The CLIF-RL paper studies adults receiving invasive mechanical ventilation
    (IMV) for whom the agent can recommend hourly settings. The cohort definition
    is:

    | Criterion | Rationale |
    |---|---|
    | **Age ≥ 18 at admission** | Adult ventilation guidelines differ from pediatric |
    | **At least one IMV record** | Defines the population of interest |
    | **No tracheostomy at first IMV** | Excludes chronic trach-dependent patients; we want intubation-driven ventilation |

    We track N at each step and produce a STROBE-style waterfall plot.

    > **What we skip on the demo path:** the published CLIF-RL also requires
    > recorded height + weight (for IBW-based tidal-volume normalization). The
    > demo dataset has height for only ~24 % of hospitalizations, so applying it
    > would gut the cohort and obscure the teaching points. The full-MIMIC
    > pipeline applies all four criteria.
    """
    )
    return


@app.cell
def _(co, pd):
    hosp = co.hospitalization.df.copy()
    resp = co.respiratory_support.df.copy()

    waterfall = []

    def record(step, df):
        n = df["hospitalization_id"].nunique()
        waterfall.append({"step": step, "n_hospitalizations": n})
        print(f"  {step:<55s} n = {n:,}")
        return df

    print("STROBE waterfall:")
    hosp = record("All hospitalizations in dataset", hosp)
    hosp = record("Age >= 18 at admission", hosp[hosp["age_at_admission"] >= 18])

    imv_records = resp[resp["device_category"].str.upper() == "IMV"].copy()
    imv_records = imv_records.sort_values(["hospitalization_id", "recorded_dttm"])
    imv_ids = set(imv_records["hospitalization_id"])
    hosp = record("Any IMV record present", hosp[hosp["hospitalization_id"].isin(imv_ids)])

    first_imv = imv_records.drop_duplicates("hospitalization_id", keep="first")[
        ["hospitalization_id", "recorded_dttm", "tracheostomy"]
    ].rename(columns={"recorded_dttm": "first_imv_dttm", "tracheostomy": "trach_at_first_imv"})
    hosp = hosp.merge(first_imv, on="hospitalization_id", how="left")
    no_trach = hosp["trach_at_first_imv"].isin([False, 0]) | hosp["trach_at_first_imv"].isna()
    hosp = record("No tracheostomy at first IMV", hosp[no_trach])

    last_imv = imv_records.drop_duplicates("hospitalization_id", keep="last")[
        ["hospitalization_id", "recorded_dttm"]
    ].rename(columns={"recorded_dttm": "last_imv_dttm"})
    hosp = hosp.merge(last_imv, on="hospitalization_id", how="left")

    cohort = hosp.reset_index(drop=True)
    print(f"\nFinal CLIF-RL cohort: {cohort['hospitalization_id'].nunique():,} hospitalizations.")
    waterfall_df = pd.DataFrame(waterfall)
    return cohort, waterfall_df


@app.cell
def _(mo):
    mo.md(
        """
    ### STROBE diagram

    A visual of the waterfall — for the workshop, and as a sanity check that
    each step removes a sensible fraction of the cohort.
    """
    )
    return


@app.cell
def _(FancyBboxPatch, plt, waterfall_df):
    _fig, _ax = plt.subplots(figsize=(8, 4.5))
    _n_steps = len(waterfall_df)
    _y_positions = list(range(_n_steps, 0, -1))

    _excluded = []
    for _i in range(_n_steps):
        if _i == 0:
            _excluded.append(0)
        else:
            _excluded.append(
                waterfall_df.iloc[_i - 1]["n_hospitalizations"]
                - waterfall_df.iloc[_i]["n_hospitalizations"]
            )

    for _i, _row in waterfall_df.iterrows():
        _y = _y_positions[_i]
        _box = FancyBboxPatch(
            (0.1, _y - 0.35), 0.55, 0.7,
            boxstyle="round,pad=0.02",
            linewidth=1.2, edgecolor="#2c3e50", facecolor="#ecf0f1",
        )
        _ax.add_patch(_box)
        _ax.text(0.375, _y, f"{_row['step']}\nN = {_row['n_hospitalizations']:,}",
                 ha="center", va="center", fontsize=10, fontweight="bold")
        if _i > 0:
            _ax.annotate("", xy=(0.375, _y + 0.4), xytext=(0.375, _y + 0.6),
                         arrowprops=dict(arrowstyle="->", lw=1.4, color="#2c3e50"))
            _ax.text(0.78, _y + 0.5, f"− {_excluded[_i]:,} excluded",
                     ha="left", va="center", fontsize=9, color="#c0392b")

    _ax.set_xlim(0, 1.05)
    _ax.set_ylim(0.5, _n_steps + 0.5)
    _ax.axis("off")
    _ax.set_title("STROBE waterfall — CLIF-RL cohort", fontsize=12, pad=10)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 3. Cohort time windows + respiratory-support waterfall

    Two things in this step:

    1. **Vent-episode windows.** Build a `cohort_df` with three columns —
       `hospitalization_id`, `start_time` (first IMV), `end_time` (last IMV).
       Downstream calls use this window to filter physiology data to only the
       intubation period.
    2. **Waterfall.** `clifpy.process_resp_support_waterfall` is the consortium's
       answer to a notorious data-quality problem: respiratory-support tables
       across sites are sparse, mode-dependent, and inconsistently timestamped.
       The waterfall inserts hourly scaffold rows, applies a device/mode
       hierarchy, and forward-fills set-points within mode blocks. After it
       runs, every hour of the encounter has a coherent vent-state row.

    Reassigning `co.respiratory_support.df` to the cleaned version means the
    downstream `create_wide_dataset` call automatically uses the waterfall'd data.
    """
    )
    return


@app.cell
def _(cohort, co):
    from clifpy import process_resp_support_waterfall
    import pandas as _pd

    cohort_df = cohort[["hospitalization_id", "first_imv_dttm", "last_imv_dttm"]].rename(
        columns={"first_imv_dttm": "start_time", "last_imv_dttm": "end_time"}
    )
    cohort_df["hospitalization_id"] = cohort_df["hospitalization_id"].astype(str)
    cohort_df["episode_hours"] = (
        (cohort_df["end_time"] - cohort_df["start_time"]).dt.total_seconds() / 3600
    )
    print("Episode duration (hours):")
    print(cohort_df["episode_hours"].describe().round(1))

    raw_resp_rows = len(co.respiratory_support.df)
    resp_clean = process_resp_support_waterfall(co.respiratory_support.df.copy(), verbose=False)
    # Waterfall returns object-dtype Timestamps (mixed offsets after the hourly scaffold);
    # coerce to a single tz dtype so downstream clifpy reads it cleanly.
    resp_clean["recorded_dttm"] = (
        _pd.to_datetime(resp_clean["recorded_dttm"], utc=True).dt.tz_convert(co.timezone)
    )
    co.respiratory_support.df = resp_clean
    print(f"\nrespiratory_support: {raw_resp_rows:,} (raw) → {len(resp_clean):,} (post-waterfall)")
    return (cohort_df,)


@app.cell
def _(mo):
    mo.md(
        """
    ## 4. Build the wide dataset

    `ClifOrchestrator.create_wide_dataset` is the heart of this notebook. In one
    call it:

    1. Filters every event table to the cohort hospitalization-ids and the
       supplied time window.
    2. **Pivots** narrow tables (vitals, labs) so each `vital_category` / `lab_category`
       becomes a column.
    3. **Keeps** wide tables (respiratory_support) as-is, selecting only the
       columns we asked for.
    4. Joins everything on `hospitalization_id` + event timestamp into a long
       event-stream table.

    Behind the scenes it uses DuckDB for the heavy joins, so this stays fast on
    the full MIMIC-IV CLIF dataset (~3 M hourly rows in our tests).

    ⚠️ `create_wide_dataset` stores the result on `co.wide_df` rather than
    returning it. (One of those clifpy quirks. Easy to miss the first time.)
    """
    )
    return


@app.cell
def _(co, cohort_df):
    category_filters = {
        "vitals": [
            "heart_rate", "respiratory_rate", "sbp", "dbp", "map", "spo2", "temp_c",
        ],
        "labs": [
            "ph_arterial", "pco2_arterial", "po2_arterial", "bicarbonate",
            "sodium", "potassium", "creatinine", "lactate", "hemoglobin",
        ],
        "respiratory_support": [
            "device_category", "mode_category",
            "fio2_set", "peep_set", "tidal_volume_set", "resp_rate_set",
            "peak_inspiratory_pressure_set", "plateau_pressure_obs",
        ],
    }

    co.create_wide_dataset(
        category_filters=category_filters,
        cohort_df=cohort_df[["hospitalization_id", "start_time", "end_time"]],
        show_progress=False,
    )
    wide_df = co.wide_df
    print(f"wide_df shape: {wide_df.shape}")
    print(f"columns: {list(wide_df.columns)}")
    return (wide_df,)


@app.cell
def _(mo):
    mo.md(
        """
    ## 5. Collapse to an hourly state grid

    `convert_wide_to_hourly` does the regularization step: each
    `(hospitalization, hour)` pair gets one row; per column we choose **mean**
    (continuous) or **first** (categorical) aggregation.

    With `fill_gaps=True`, every hour between window 0 and the last observed
    window exists as a row — gaps that have no data become NaN, which the next
    section's imputation cascade fills.
    """
    )
    return


@app.cell
def _(co, wide_df):
    aggregation_config = {
        "mean": [
            "heart_rate", "respiratory_rate", "sbp", "dbp", "map", "spo2", "temp_c",
            "ph_arterial", "pco2_arterial", "po2_arterial", "bicarbonate",
            "sodium", "potassium", "creatinine", "lactate", "hemoglobin",
            "fio2_set", "peep_set", "tidal_volume_set", "resp_rate_set",
            "peak_inspiratory_pressure_set", "plateau_pressure_obs",
        ],
        "first": ["device_category", "mode_category"],
    }
    hourly_raw = co.convert_wide_to_hourly(
        wide_df=wide_df,
        aggregation_config=aggregation_config,
        hourly_window=1,
        fill_gaps=True,
    )
    print(f"hourly_raw shape: {hourly_raw.shape}")
    print(f"hours per hospitalization: median = "
          f"{int(hourly_raw.groupby('hospitalization_id').size().median())}")
    return (hourly_raw,)


@app.cell
def _(mo):
    mo.md(
        """
    ## 6. Imputation pipeline

    A raw hourly grid will have lots of NaN — labs aren't drawn every hour,
    vitals are charted at irregular intervals, vent set-points only get
    documented when something changes. For an RL agent's state vector we need
    a clean, fully-filled matrix.

    The CLIF-RL pipeline uses a three-stage cascade (from
    `CLIF-RL/code/01_analysis_data.py`):

    1. **LOCF** within `hospitalization_id` — forward-fill the last observed
       value. Captures the "vitals haven't changed since last chart" assumption.
    2. **Backfill** for slow-changing measurements (`temp_c`, height, weight)
       so leading NaNs before the first observation get filled.
    3. **Normal-value fill** for whatever remains — population-level normals
       (pH 7.40, lactate 1.25, MAP 80, FiO₂ 0.21, etc.). Used as a last resort
       when a hospitalization is *completely* missing a column.

    At the end we assert zero NaN in the state columns. That's the contract for
    the RL state vector.
    """
    )
    return


@app.cell
def _(hourly_raw):
    # Normal values from CLIF-RL/code/01_analysis_data.py
    NORMAL_VALUES = {
        # labs
        "bicarbonate_mean":     25.0,
        "creatinine_mean":       0.9,
        "hemoglobin_mean":      15.0,
        "lactate_mean":          1.25,
        "pco2_arterial_mean":   40.0,
        "ph_arterial_mean":      7.40,
        "po2_arterial_mean":    90.0,
        "potassium_mean":        4.25,
        "sodium_mean":         140.0,
        # vitals (population norms)
        "heart_rate_mean":      80.0,
        "respiratory_rate_mean":16.0,
        "sbp_mean":            120.0,
        "dbp_mean":             70.0,
        "map_mean":             80.0,
        "spo2_mean":            98.0,
        "temp_c_mean":          36.8,
        # vent set-points (room-air defaults for cells before vent initiation)
        "fio2_set_mean":         0.21,
        "peep_set_mean":         5.0,
        "tidal_volume_set_mean": 450.0,
        "resp_rate_set_mean":   16.0,
        "peak_inspiratory_pressure_set_mean": 20.0,
        "plateau_pressure_obs_mean":          20.0,
    }
    BFILL_COLS = ["temp_c_mean"]

    _df = hourly_raw.sort_values(["hospitalization_id", "window_number"]).copy()
    state_cols = list(NORMAL_VALUES.keys())

    print("Missing-value % BEFORE imputation (top 10):")
    miss_before = (_df[state_cols].isna().mean() * 100).round(1).sort_values(ascending=False)
    print(miss_before.head(10).to_string())

    _df[state_cols] = _df.groupby("hospitalization_id")[state_cols].ffill()
    _df[BFILL_COLS] = _df.groupby("hospitalization_id")[BFILL_COLS].bfill()
    for _col, _val in NORMAL_VALUES.items():
        _df[_col] = _df[_col].fillna(_val)

    for _cat_col, _default in [("device_category_first", "Room Air"),
                                ("mode_category_first", "unknown")]:
        if _cat_col in _df.columns:
            _df[_cat_col] = _df.groupby("hospitalization_id")[_cat_col].ffill()
            _df[_cat_col] = _df[_cat_col].fillna(_default)

    print("\nMissing-value % AFTER imputation (top 10):")
    miss_after = (_df[state_cols].isna().mean() * 100).round(2).sort_values(ascending=False)
    print(miss_after.head(10).to_string())

    assert _df[state_cols].isna().sum().sum() == 0, "NaN remaining in state columns"
    print("\n✓ zero NaN in state columns after imputation")
    hourly_df = _df
    return hourly_df, state_cols


@app.cell
def _(mo):
    mo.md(
        """
    ## 7. Baseline SOFA score

    SOFA (Sequential Organ Failure Assessment) is the standard ICU severity
    score — a 0–24 scale summed across six organ systems (respiratory,
    cardiovascular, hepatic, coagulation, renal, neurological).

    `clifpy.compute_sofa_polars` reads the CLIF parquet files directly and
    returns one row per hospitalization with each component score + total,
    computed as the **worst** value within a supplied time window. We use the
    **first 24 hours** of each vent episode as the baseline-severity window.

    SOFA gets joined into `static_df` and becomes one of the adjustment
    covariates in the downstream concordance analysis.
    """
    )
    return


@app.cell
def _(DATA_DIR, FILE_TYPE, TIMEZONE, cohort_df, pd, pl):
    from clifpy import compute_sofa_polars

    baseline = cohort_df[["hospitalization_id", "start_time"]].copy()
    baseline["start_dttm"] = baseline["start_time"]
    baseline["end_dttm"] = baseline["start_time"] + pd.Timedelta(hours=24)
    baseline = baseline[["hospitalization_id", "start_dttm", "end_dttm"]]
    baseline["hospitalization_id"] = baseline["hospitalization_id"].astype(str)

    sofa_df = compute_sofa_polars(
        data_directory=DATA_DIR,
        cohort_df=pl.from_pandas(baseline),
        filetype=FILE_TYPE,
        timezone=TIMEZONE,
    )
    sofa_pdf = sofa_df.to_pandas()
    _sofa_keep = [c for c in sofa_pdf.columns if c.startswith("sofa_") or c == "hospitalization_id"]
    sofa_pdf = sofa_pdf[_sofa_keep]
    print(f"SOFA computed for {len(sofa_pdf)} hospitalizations.")
    print(f"\nsofa_total distribution (worst score in first 24h):")
    print(sofa_pdf["sofa_total"].describe().round(1))
    return (sofa_pdf,)


@app.cell
def _(mo):
    mo.md(
        """
    ## 8. Charlson Comorbidity Index

    CCI captures chronic disease burden from ICD codes (`hospital_diagnosis`
    table). `clifpy.calculate_cci` maps ICD-10-CM codes to 17 weighted
    conditions (Quan 2011 adaptation), returns a `cci_score` per
    hospitalization.

    Hospitalizations with no ICD-10-CM rows (some demo hospitalizations don't
    have any diagnosis codes) default to 0.
    """
    )
    return


@app.cell
def _(co, cohort_df):
    from clifpy import calculate_cci

    cci_df = calculate_cci(co.hospital_diagnosis.df)
    cci_df["hospitalization_id"] = cci_df["hospitalization_id"].astype(str)
    cci_df = cci_df[
        cci_df["hospitalization_id"].isin(cohort_df["hospitalization_id"].astype(str))
    ].reset_index(drop=True)
    cci_df = cci_df[["hospitalization_id", "cci_score"]]
    print(f"CCI rows (cohort with ICD codes): {len(cci_df)} / {len(cohort_df)}")
    print(f"\ncci_score distribution:")
    print(cci_df["cci_score"].describe().round(1))
    return (cci_df,)


@app.cell
def _(mo):
    mo.md(
        """
    ## 9. Build the static (hospitalization-level) dataframe

    Combine everything into a one-row-per-hospitalization table:

    | Block | Columns |
    |---|---|
    | Identity | `hospitalization_id`, `patient_id` |
    | Demographics | `age_at_admission`, `sex_category`, `race_category`, `ethnicity_category` |
    | Episode | `start_time`, `end_time`, `episode_hours`, `admission_dttm`, `discharge_dttm`, `hospital_los_days` |
    | Outcome | `inpatient_mortality` |
    | Severity | `sofa_total` + 6 component scores |
    | Comorbidity | `cci_score` |

    This is what an analyst would join onto the RL trajectory for Table 1,
    Kaplan-Meier survival, or covariate-adjusted regression.
    """
    )
    return


@app.cell
def _(cci_df, co, cohort_df, sofa_pdf):
    pat = co.patient.df[["patient_id", "sex_category", "race_category",
                          "ethnicity_category", "death_dttm"]].copy()

    base = cohort_df.copy()
    base["hospitalization_id"] = base["hospitalization_id"].astype(str)

    hosp_min = co.hospitalization.df[
        ["hospitalization_id", "patient_id", "admission_dttm", "discharge_dttm",
         "age_at_admission"]
    ].copy()
    hosp_min["hospitalization_id"] = hosp_min["hospitalization_id"].astype(str)

    static_df = (
        base.merge(hosp_min, on="hospitalization_id", how="left")
            .merge(pat, on="patient_id", how="left")
    )
    static_df["hospital_los_days"] = (
        (static_df["discharge_dttm"] - static_df["admission_dttm"]).dt.total_seconds() / 86400
    )
    static_df["inpatient_mortality"] = (
        static_df["death_dttm"].notna()
        & (static_df["death_dttm"] <= static_df["discharge_dttm"])
    ).astype(int)
    static_df = static_df.merge(sofa_pdf, on="hospitalization_id", how="left")
    static_df = static_df.merge(cci_df, on="hospitalization_id", how="left")
    static_df["cci_score"] = static_df["cci_score"].fillna(0).astype(int)

    print(f"static_df shape: {static_df.shape}")
    static_df.head()
    return (static_df,)


@app.cell
def _(mo):
    mo.md(
        """
    ## 10. Cohort characterization

    Now that we have `static_df`, we can describe the cohort and compare
    survivors vs non-survivors. This is the standard "Table 1" that goes at the
    top of any clinical paper.

    We use the `tableone` package — it's automated and produces publication-ready
    output with appropriate medians/IQRs for non-normal continuous variables.
    """
    )
    return


@app.cell
def _(TableOne, static_df):
    t1_cols = [
        "age_at_admission", "sex_category", "race_category",
        "episode_hours", "hospital_los_days",
        "sofa_total", "sofa_resp", "sofa_cv_97", "sofa_renal", "sofa_cns",
        "cci_score",
    ]
    t1_cat = ["sex_category", "race_category"]
    t1 = TableOne(
        static_df,
        columns=t1_cols,
        categorical=t1_cat,
        groupby="inpatient_mortality",
        pval=False,
        rename={
            "age_at_admission": "Age at admission",
            "sex_category": "Sex",
            "race_category": "Race",
            "episode_hours": "Vent episode (hours)",
            "hospital_los_days": "Hospital LOS (days)",
            "sofa_total": "SOFA (baseline 24h)",
            "sofa_resp": "  SOFA — Respiratory",
            "sofa_cv_97": "  SOFA — Cardiovascular",
            "sofa_renal": "  SOFA — Renal",
            "sofa_cns": "  SOFA — Neurological",
            "cci_score": "Charlson CCI",
        },
    )
    t1
    return


@app.cell
def _(mo):
    mo.md(
        """
    ### Figures: cohort characteristics

    Three quick figures — episode-length distribution, SOFA distribution by
    mortality, and the action-relevant vent set-point distributions across all
    hourly rows.
    """
    )
    return


@app.cell
def _(plt, static_df):
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 3.8))

    # Panel A — episode hours histogram
    _axes[0].hist(static_df["episode_hours"], bins=20,
                  color="#2980b9", edgecolor="white")
    _axes[0].set_xlabel("Vent episode (hours)")
    _axes[0].set_ylabel("Hospitalizations")
    _axes[0].set_title("A. Episode length distribution")

    # Panel B — SOFA by mortality
    _surv = static_df.loc[static_df["inpatient_mortality"] == 0, "sofa_total"]
    _died = static_df.loc[static_df["inpatient_mortality"] == 1, "sofa_total"]
    _axes[1].hist([_surv, _died], bins=range(0, int(static_df["sofa_total"].max()) + 2),
                  label=["Survived", "Died"], color=["#2980b9", "#c0392b"],
                  edgecolor="white", stacked=False)
    _axes[1].set_xlabel("SOFA total (worst, first 24h)")
    _axes[1].set_ylabel("Hospitalizations")
    _axes[1].set_title("B. Baseline severity by outcome")
    _axes[1].legend()

    # Panel C — hospital LOS histogram
    _axes[2].hist(static_df["hospital_los_days"], bins=20,
                  color="#16a085", edgecolor="white")
    _axes[2].set_xlabel("Hospital LOS (days)")
    _axes[2].set_ylabel("Hospitalizations")
    _axes[2].set_title("C. Hospital length of stay")

    plt.tight_layout()
    _fig
    return


@app.cell
def _(hourly_df, plt):
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 3.8))

    _axes[0].hist(hourly_df["fio2_set_mean"], bins=20, color="#e67e22", edgecolor="white")
    _axes[0].set_xlabel("FiO₂ set")
    _axes[0].set_ylabel("Hourly rows")
    _axes[0].set_title("A. FiO₂ across all hours")

    _axes[1].hist(hourly_df["peep_set_mean"], bins=20, color="#8e44ad", edgecolor="white")
    _axes[1].set_xlabel("PEEP set (cmH₂O)")
    _axes[1].set_ylabel("Hourly rows")
    _axes[1].set_title("B. PEEP across all hours")

    _axes[2].hist(hourly_df["tidal_volume_set_mean"], bins=20, color="#34495e", edgecolor="white")
    _axes[2].set_xlabel("Tidal volume set (mL)")
    _axes[2].set_ylabel("Hourly rows")
    _axes[2].set_title("C. Tidal volume across all hours")

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        """
    ### Sample patient trajectory

    The point of building `hourly_df` is to have a regular grid of every state
    feature, every hour, for every patient. Here's what one looks like for the
    patient with the longest vent episode in our cohort.
    """
    )
    return


@app.cell
def _(hourly_df, plt):
    counts = hourly_df.groupby("hospitalization_id").size()
    pick_id = counts.idxmax()
    one = hourly_df[hourly_df["hospitalization_id"] == pick_id].sort_values("window_number")

    _fig, _axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    _axes[0].plot(one["window_number"], one["heart_rate_mean"], label="HR", color="#c0392b")
    _axes[0].plot(one["window_number"], one["map_mean"], label="MAP", color="#2980b9")
    _axes[0].set_ylabel("bpm / mmHg")
    _axes[0].legend(loc="upper right")
    _axes[0].set_title(f"Hospitalization {pick_id} — physiology + vent over time")

    _axes[1].plot(one["window_number"], one["spo2_mean"], color="#16a085", label="SpO₂")
    _axes[1].set_ylabel("SpO₂ (%)")
    _axes[1].set_ylim(80, 102)
    _axes[1].legend(loc="lower right")

    _axes[2].plot(one["window_number"], one["fio2_set_mean"], color="#e67e22", label="FiO₂ set")
    _axes[2].plot(one["window_number"], one["peep_set_mean"] / 10, color="#8e44ad",
                  label="PEEP set / 10")
    _axes[2].set_ylabel("vent set-point")
    _axes[2].set_xlabel("Hour from first IMV record")
    _axes[2].legend(loc="upper right")

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Two outputs, ready for the next step

    ✅ **`hourly_df`** — the RL state grid. One row per `(hospitalization, hour)`,
    fully imputed, **zero NaN** in 22 state columns spanning vitals, ABG, labs,
    and ventilator set-points.

    ✅ **`static_df`** — one row per hospitalization with demographics, episode
    timing, mortality outcome, baseline SOFA components + total, and Charlson
    CCI.

    These two dataframes are what Saki's deep-dive notebook (next session)
    consumes. After that we'll come back for a look at the RL training pipeline
    in [`02_clif_rl_training.ipynb`](./02_clif_rl_training.ipynb).

    **Run on full CLIF-MIMIC.** Available from PhysioNet at
    [mimic-iv-ext-clif/1.1.0](https://physionet.org/content/mimic-iv-ext-clif/1.1.0/)
    (credentialed access — CITI training + DUA). Once you have the parquet files
    locally, copy `config/config_template.json` to `config/config.json`, point
    `tables_path` at your CLIF directory, and re-run.
    """
    )
    return


if __name__ == "__main__":
    app.run()
