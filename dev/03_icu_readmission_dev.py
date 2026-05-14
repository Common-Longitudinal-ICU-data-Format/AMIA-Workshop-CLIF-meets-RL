"""Dev marimo: ICU readmission demo on bundled CLIF demo data.

Mirrors the methodology of the CLIF ICU readmission project
(https://github.com/Common-Longitudinal-ICU-data-Format/CLIF_icu_readmission)
but rewritten on top of clifpy. Companion to the AMIA TRI08 workshop — kept as
a bonus notebook showing a different cohort flavour (purely descriptive, no RL).

Run as script:    python dev/03_icu_readmission_dev.py
Run interactive:  uv run marimo edit dev/03_icu_readmission_dev.py
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
    # CLIF in Action: Identifying an ICU Readmission Cohort

    **AMIA TRI08 — CLIF Meets RL Workshop (bonus notebook)**
    *Kaveri Chhikara · University of Chicago*

    A different cohort flavour from the RL pipeline — purely descriptive. The pattern:

    1. Load CLIF tables with `clifpy` in one call.
    2. Build an ICU-readmission cohort (adults, ICU stay, exclusions).
    3. Describe the cohort with a Table 1, a patient-journey Sankey, and a mortality comparison.

    This notebook runs **as-is on Google Colab** — it uses the CLIF demo dataset
    bundled inside `clifpy`. To re-run on full CLIF-MIMIC after the workshop,
    populate `config/config.json` (see `config/README.md`).
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Why CLIF, and why `clifpy`?

    **CLIF** (Common Longitudinal ICU Format) is an open-source CDM for
    critical-care data, shared across 20+ health systems. Code travels between
    sites; patient data does not.

    **`clifpy`** is the consortium-standard Python interface to CLIF —
    typed table objects, timezone-aware datetime handling, a built-in Data
    Quality Assessment layer, and utilities for wide datasets and clinical
    scores.
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
    from tableone import TableOne
    import plotly.graph_objects as go

    def resolve_data_dir():
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
    return ClifOrchestrator, DATA_DIR, FILE_TYPE, TIMEZONE, TableOne, go, np, pd, plt


@app.cell
def _(mo):
    mo.md(
        """
    ## 1. Load CLIF tables

    Before `clifpy`, this is what you'd write for each table: `pd.read_parquet(...)`,
    then manually parse timezones, manually validate column names against the
    CLIF data dictionary, manually map mCIDE categories. For three tables that's
    already 30+ lines of boilerplate.

    `ClifOrchestrator` collapses all of that to one call.
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
    co.initialize(tables=["patient", "hospitalization", "adt"])
    print("Loaded tables:", co.get_loaded_tables())
    return (co,)


@app.cell
def _(co):
    # clifpy ships a built-in Data Quality Assessment layer.
    # This is the same DQA framework that powers the CLIF Quality Reports.
    co.validate_all()

    print("\nLocation categories present in the ADT table:")
    print(co.adt.df["location_category"].value_counts(dropna=False))

    print("\nFirst few hospitalizations:")
    co.hospitalization.df.head()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 2. Build the ICU readmission cohort

    **Cohort definition** (from the [CLIF ICU Readmission project](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF_icu_readmission)):

    - Adults **≥ 18** at admission
    - At least one **ICU stay** during the hospitalization
    - Exclude: died during the index ICU stay
    - Exclude: discharged immediately after the index ICU stay (no further inpatient time)

    We print a STROBE waterfall as we go so you can see how many hospitalizations
    survive each criterion.

    > *Note on dates.* MIMIC timestamps are de-identified per-patient shifts, so
    > the absolute calendar year is meaningless. The published
    > `CLIF_icu_readmission` project applies a 2020–2021 window for real-data
    > sites; we drop that window here because the dataset is MIMIC-derived.
    """
    )
    return


@app.cell
def _(co, pd):
    hosp = co.hospitalization.df.copy()
    adt = co.adt.df.copy()
    patient = co.patient.df.copy()

    waterfall = []

    def record(step, df):
        n_hosp = df["hospitalization_id"].nunique()
        waterfall.append({"step": step, "n_hospitalizations": n_hosp})
        print(f"  {step:<55s} n_hospitalizations = {n_hosp:,}")
        return df

    print("STROBE waterfall:")
    hosp = record("All hospitalizations in dataset", hosp)
    hosp = record("Age >= 18 at admission", hosp[hosp["age_at_admission"] >= 18])

    icu_hosp_ids = set(
        adt.loc[adt["location_category"].str.lower() == "icu", "hospitalization_id"]
    )
    hosp = record("Has at least one ICU stay", hosp[hosp["hospitalization_id"].isin(icu_hosp_ids)])

    # Compute the index (first) ICU stay per hospitalization.
    icu_adt = adt[adt["location_category"].str.lower() == "icu"].copy()
    icu_adt = icu_adt.sort_values(["hospitalization_id", "in_dttm"])
    first_icu = (
        icu_adt.drop_duplicates("hospitalization_id", keep="first")[
            ["hospitalization_id", "in_dttm", "out_dttm"]
        ].rename(columns={"in_dttm": "index_icu_in", "out_dttm": "index_icu_out"})
    )
    hosp = hosp.merge(first_icu, on="hospitalization_id", how="left")
    hosp = hosp.merge(patient[["patient_id", "death_dttm"]], on="patient_id", how="left")

    died_in_index_icu = (
        hosp["death_dttm"].notna()
        & (hosp["death_dttm"] >= hosp["index_icu_in"])
        & (hosp["death_dttm"] <= hosp["index_icu_out"])
    )
    hosp = record("Did not die during index ICU stay", hosp[~died_in_index_icu])

    gap_hours = (hosp["discharge_dttm"] - hosp["index_icu_out"]).dt.total_seconds() / 3600
    immediate = gap_hours <= 1
    hosp = record("Not discharged immediately after index ICU", hosp[~immediate])

    cohort = hosp.reset_index(drop=True)
    print(f"\nFinal cohort: {cohort['hospitalization_id'].nunique():,} hospitalizations.")
    return adt, cohort, patient


@app.cell
def _(adt, cohort, pd):
    # Classify each surviving hospitalization as readmitted-to-ICU or not.
    # Readmitted = a later ICU stay AFTER the index ICU's out_dttm in the same hospitalization.
    icu_visits = (
        adt[adt["location_category"].str.lower() == "icu"]
        .merge(cohort[["hospitalization_id", "index_icu_out"]],
               on="hospitalization_id", how="inner")
    )
    icu_visits["is_post_index"] = icu_visits["in_dttm"] > icu_visits["index_icu_out"]
    readmitted_ids = set(icu_visits.loc[icu_visits["is_post_index"], "hospitalization_id"])
    cohort["readmitted"] = cohort["hospitalization_id"].isin(readmitted_ids)

    n = len(cohort)
    n_re = int(cohort["readmitted"].sum())
    if n:
        print(f"Readmitted to ICU: {n_re:,} / {n:,} ({100 * n_re / n:.1f}%)")
    else:
        print("Cohort is empty.")

    pd.DataFrame([
        {"readmitted": False, "n": int((~cohort["readmitted"]).sum())},
        {"readmitted": True,  "n": int(cohort["readmitted"].sum())},
    ])
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 3. Describe the cohort

    Three quick descriptive views: a Table 1 split by readmission status, a
    Sankey of the patient journey through the hospital, and a side-by-side
    inpatient-mortality comparison.
    """
    )
    return


@app.cell
def _(TableOne, cohort, patient):
    describe_df = cohort.merge(
        patient[["patient_id", "race_category", "ethnicity_category", "sex_category"]],
        on="patient_id", how="left",
    )
    describe_df["hospital_los_days"] = (
        (describe_df["discharge_dttm"] - describe_df["admission_dttm"]).dt.total_seconds() / 86400
    )
    describe_df["inpatient_mortality"] = describe_df["death_dttm"].notna() & (
        describe_df["death_dttm"] <= describe_df["discharge_dttm"]
    )

    columns = [
        "age_at_admission", "sex_category", "race_category", "ethnicity_category",
        "hospital_los_days", "inpatient_mortality",
    ]
    categorical = ["sex_category", "race_category", "ethnicity_category", "inpatient_mortality"]

    t1 = TableOne(
        describe_df,
        columns=columns,
        categorical=categorical,
        groupby="readmitted",
        pval=False,
    )
    t1
    return (describe_df,)


@app.cell
def _(adt, cohort, go, pd):
    # Sankey of the first 4 ADT stops per cohort hospitalization.
    LOC_MAP = {
        "icu": "ICU", "stepdown": "ICU",
        "ward": "Ward", "ed": "ED",
        "procedural": "Procedural",
        "psych": "Other", "other": "Other",
    }
    adt_clean = adt.copy()
    adt_clean["loc"] = (
        adt_clean["location_category"].str.lower().str.strip().map(LOC_MAP).fillna("Other")
    )

    traj = (
        adt_clean[adt_clean["hospitalization_id"].isin(cohort["hospitalization_id"])]
        .sort_values(["hospitalization_id", "in_dttm"])
        .groupby("hospitalization_id")["loc"]
        .apply(list)
    )

    N_STOPS = 4

    def pad(stops):
        stops = stops[:N_STOPS]
        while len(stops) < N_STOPS:
            stops.append("Discharge")
        return stops

    traj_df = pd.DataFrame(
        traj.apply(pad).tolist(),
        columns=[f"stop_{i}" for i in range(N_STOPS)],
    )

    labels = []
    label_to_idx = {}

    def node_id(stop_i, label):
        key = (stop_i, label)
        if key not in label_to_idx:
            label_to_idx[key] = len(labels)
            labels.append(f"{label} (stop {stop_i + 1})")
        return label_to_idx[key]

    sources, targets, values = [], [], []
    for _i in range(N_STOPS - 1):
        _flows = traj_df.groupby([f"stop_{_i}", f"stop_{_i+1}"]).size().reset_index(name="n")
        for _, _row in _flows.iterrows():
            sources.append(node_id(_i, _row[f"stop_{_i}"]))
            targets.append(node_id(_i + 1, _row[f"stop_{_i+1}"]))
            values.append(int(_row["n"]))

    sankey_fig = go.Figure(data=[
        go.Sankey(
            node=dict(label=labels, pad=20, thickness=20),
            link=dict(source=sources, target=targets, value=values),
        )
    ])
    sankey_fig.update_layout(
        title="Patient journey: first 4 ADT locations (cohort)",
        height=500,
    )
    sankey_fig
    return


@app.cell
def _(describe_df, plt):
    rates = (
        describe_df.groupby("readmitted")
        .agg(n=("hospitalization_id", "nunique"),
             inpatient_mortality_rate=("inpatient_mortality", "mean"))
        .reset_index()
    )
    rates["inpatient_mortality_pct"] = rates["inpatient_mortality_rate"] * 100
    print(rates[["readmitted", "n", "inpatient_mortality_pct"]].to_string(index=False))

    _fig, _ax = plt.subplots(figsize=(5, 4))
    _ax.bar(
        rates["readmitted"].map({True: "Readmitted", False: "Not readmitted"}),
        rates["inpatient_mortality_pct"],
        color=["#c0392b", "#2980b9"],
    )
    _ax.set_ylabel("Inpatient mortality (%)")
    _ax.set_title("Inpatient mortality by ICU-readmission status")
    for _i, _v in enumerate(rates["inpatient_mortality_pct"]):
        _ax.text(_i, _v + 0.5, f"{_v:.1f}%", ha="center")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Next steps

    **Run on full CLIF-MIMIC.** Populate `config/config.json` with your local
    CLIF parquet directory and re-run — the notebook will use that data instead
    of the demo.

    **Related projects.**

    - [CLIF ICU Readmission](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF_icu_readmission) — the original methodology
    - [`clifpy` docs](https://common-longitudinal-icu-data-format.github.io/clifpy/) — full API reference
    """
    )
    return


if __name__ == "__main__":
    app.run()
