"""Dev marimo: CLIF-RL training setup walkthrough + ATS submission results.

The companion to `01_clif_rl_cohort_wide_dev.py`. Walks through how the cohort
+ hourly grid becomes a Double DQN training set, shows the network and trainer
code (without executing training — 56 hospitalizations isn't enough to fit a
useful policy), and renders the ATS submission's three-site external-validation
forest plot.

Run as script:    python dev/02_clif_rl_training_dev.py
Run interactive:  uv run marimo edit dev/02_clif_rl_training_dev.py
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
    # CLIF-RL training: setup + ATS external-validation results

    **AMIA TRI08 — CLIF Meets RL Workshop**

    This notebook picks up after Saki's deep-dive on `hourly_df` + `static_df`
    (from `01_clif_rl_cohort_wide.ipynb`). It walks through how that data
    becomes a Double DQN training set, shows the production training code
    *without running it* (the 56-hospitalization demo cohort is too small to
    fit a useful policy), and renders the ATS submission's three-site
    external-validation forest plot.

    Full training + validation code lives on the
    [`ats-submission`](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF-RL/tree/ats-submission)
    branch of CLIF-RL.
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
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

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
    return (
        ClifOrchestrator, DATA_DIR, F, FILE_TYPE, TIMEZONE, nn, np, pd, plt, torch,
    )


@app.cell
def _(mo):
    mo.md(
        """
    ## 1. Recap: rebuild the hourly state grid

    Condensed recap of `01_clif_rl_cohort_wide.ipynb` so this notebook is
    runnable standalone. Same cohort (adults ≥ 18 on IMV, no trach at first
    IMV), same waterfall + imputation cascade. End state: a fully-imputed
    `hourly_df` with 20 state columns and zero NaN.
    """
    )
    return


@app.cell
def _(ClifOrchestrator, DATA_DIR, FILE_TYPE, TIMEZONE, pd):
    co = ClifOrchestrator(data_directory=DATA_DIR, filetype=FILE_TYPE, timezone=TIMEZONE)
    co.initialize(tables=["patient", "hospitalization", "adt",
                          "respiratory_support", "vitals", "labs"])

    hosp = co.hospitalization.df.copy()
    resp = co.respiratory_support.df.copy()
    imv_records = resp[resp["device_category"].str.upper() == "IMV"].copy()
    imv_records = imv_records.sort_values(["hospitalization_id", "recorded_dttm"])
    first_imv = imv_records.drop_duplicates("hospitalization_id", keep="first")[
        ["hospitalization_id", "recorded_dttm", "tracheostomy"]
    ].rename(columns={"recorded_dttm": "first_imv_dttm",
                      "tracheostomy": "trach_at_first_imv"})
    last_imv = imv_records.drop_duplicates("hospitalization_id", keep="last")[
        ["hospitalization_id", "recorded_dttm"]
    ].rename(columns={"recorded_dttm": "last_imv_dttm"})
    cohort = (
        hosp[hosp["age_at_admission"] >= 18]
        .merge(first_imv, on="hospitalization_id", how="inner")
        .merge(last_imv, on="hospitalization_id", how="inner")
    )
    cohort = cohort[
        cohort["trach_at_first_imv"].isin([False, 0]) | cohort["trach_at_first_imv"].isna()
    ].reset_index(drop=True)
    print(f"Cohort: {cohort['hospitalization_id'].nunique():,} hospitalizations.")

    cohort_df = cohort[["hospitalization_id", "first_imv_dttm", "last_imv_dttm"]].rename(
        columns={"first_imv_dttm": "start_time", "last_imv_dttm": "end_time"}
    )
    cohort_df["hospitalization_id"] = cohort_df["hospitalization_id"].astype(str)

    from clifpy import process_resp_support_waterfall
    resp_clean = process_resp_support_waterfall(co.respiratory_support.df.copy(), verbose=False)
    resp_clean["recorded_dttm"] = (
        pd.to_datetime(resp_clean["recorded_dttm"], utc=True).dt.tz_convert(co.timezone)
    )
    co.respiratory_support.df = resp_clean

    category_filters = {
        "vitals": ["heart_rate","respiratory_rate","sbp","dbp","map","spo2","temp_c"],
        "labs": ["ph_arterial","pco2_arterial","po2_arterial","bicarbonate",
                 "sodium","potassium","creatinine","lactate","hemoglobin"],
        "respiratory_support": ["device_category","mode_category",
                                 "fio2_set","peep_set","tidal_volume_set","resp_rate_set"],
    }
    co.create_wide_dataset(
        category_filters=category_filters,
        cohort_df=cohort_df[["hospitalization_id","start_time","end_time"]],
        show_progress=False,
    )
    aggregation_config = {
        "mean": ["heart_rate","respiratory_rate","sbp","dbp","map","spo2","temp_c",
                "ph_arterial","pco2_arterial","po2_arterial","bicarbonate",
                "sodium","potassium","creatinine","lactate","hemoglobin",
                "fio2_set","peep_set","tidal_volume_set","resp_rate_set"],
        "first": ["device_category","mode_category"],
    }
    hourly_raw = co.convert_wide_to_hourly(
        wide_df=co.wide_df,
        aggregation_config=aggregation_config,
        hourly_window=1,
        fill_gaps=True,
    )

    NORMAL_VALUES = {
        "bicarbonate_mean": 25.0, "creatinine_mean": 0.9, "hemoglobin_mean": 15.0,
        "lactate_mean": 1.25, "pco2_arterial_mean": 40.0, "ph_arterial_mean": 7.40,
        "po2_arterial_mean": 90.0, "potassium_mean": 4.25, "sodium_mean": 140.0,
        "heart_rate_mean": 80.0, "respiratory_rate_mean": 16.0, "sbp_mean": 120.0,
        "dbp_mean": 70.0, "map_mean": 80.0, "spo2_mean": 98.0, "temp_c_mean": 36.8,
        "fio2_set_mean": 0.21, "peep_set_mean": 5.0, "tidal_volume_set_mean": 450.0,
        "resp_rate_set_mean": 16.0,
    }
    state_cols = list(NORMAL_VALUES.keys())
    _df = hourly_raw.sort_values(["hospitalization_id","window_number"]).copy()
    _df[state_cols] = _df.groupby("hospitalization_id")[state_cols].ffill()
    _df[["temp_c_mean"]] = _df.groupby("hospitalization_id")[["temp_c_mean"]].bfill()
    for col, val in NORMAL_VALUES.items():
        _df[col] = _df[col].fillna(val)
    for cat_col, default in [("device_category_first","Room Air"),
                              ("mode_category_first","unknown")]:
        if cat_col in _df.columns:
            _df[cat_col] = _df.groupby("hospitalization_id")[cat_col].ffill()
            _df[cat_col] = _df[cat_col].fillna(default)
    hourly_df = _df
    print(f"hourly_df: {hourly_df.shape}, NaN in state: {hourly_df[state_cols].isna().sum().sum()}")
    return co, cohort, hourly_df, state_cols


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. Action: 2×2 grid

    The ATS submission discretizes the ventilator action into four codes that
    cross **mode** (controlled vs uncontrolled) with **oxygenation support**
    (low vs high PEEP/FiO₂):

    | code | mode | high PEEP/FiO₂ |
    |---|---|---|
    | 0 | controlled | low |
    | 1 | controlled | high |
    | 2 | uncontrolled | low |
    | 3 | uncontrolled | high |

    where `high = fio2_set > 0.6  OR  peep_set > 8`, and `controlled` means
    `mode_category` ∈ {assist control-volume control, pressure control,
    pressure-regulated volume control}.
    """
    )
    return


@app.cell
def _(hourly_df, np):
    _df = hourly_df.copy()
    _df["high_PEEP_or_FIO2"] = (
        (_df["fio2_set_mean"] > 0.6) | (_df["peep_set_mean"] > 8)
    ).astype(int)
    CONTROLLED_MODES = {
        "assist control-volume control",
        "pressure control",
        "pressure-regulated volume control",
    }
    _df["mode"] = np.where(
        _df["mode_category_first"].str.lower().isin(CONTROLLED_MODES),
        "controlled", "uncontrolled",
    )

    def _encode_action(row):
        if row["mode"] == "controlled":
            return 1 if row["high_PEEP_or_FIO2"] == 1 else 0
        return 3 if row["high_PEEP_or_FIO2"] == 1 else 2

    _df["ACTION"] = _df.apply(_encode_action, axis=1).astype(int)
    print("Physician ACTION distribution in the demo cohort:")
    print(_df["ACTION"].value_counts().sort_index())
    rl_df = _df
    return (rl_df,)


@app.cell
def _(mo):
    mo.md(
        """
    ## 3. Reward: terminal survival only

    The ATS submission uses a sparse terminal reward: at the *last* hourly
    window for each hospitalization, **+1 if survived to discharge, −1 if
    died**. Zero everywhere else. (Earlier CLIF-RL drafts experimented with
    dense per-hour reward from Δ pH and Δ P/F ratio; the ATS submission
    dropped those.)
    """
    )
    return


@app.cell
def _(cohort, co, rl_df):
    mortality = cohort[["hospitalization_id","patient_id"]].merge(
        co.patient.df[["patient_id","death_dttm"]],
        on="patient_id", how="left",
    ).merge(
        co.hospitalization.df[["hospitalization_id","discharge_dttm"]],
        on="hospitalization_id", how="left",
    )
    mortality["died_in_hosp"] = (
        mortality["death_dttm"].notna()
        & (mortality["death_dttm"] <= mortality["discharge_dttm"])
    ).astype(int)
    mortality["hospitalization_id"] = mortality["hospitalization_id"].astype(str)

    _df = rl_df.copy()
    _df["hospitalization_id"] = _df["hospitalization_id"].astype(str)
    _df = _df.sort_values(["hospitalization_id","window_number"]).reset_index(drop=True)
    _df["is_last_block"] = (
        _df.groupby("hospitalization_id")["window_number"].transform("max")
        == _df["window_number"]
    ).astype(int)
    _df = _df.merge(mortality[["hospitalization_id","died_in_hosp"]],
                    on="hospitalization_id", how="left")
    _df["reward"] = 0.0
    _df.loc[_df["is_last_block"] == 1, "reward"] = (
        _df.loc[_df["is_last_block"] == 1, "died_in_hosp"].map({1: -1.0, 0: 1.0})
    )
    n_survive = int((_df["reward"] == 1.0).sum())
    n_die = int((_df["reward"] == -1.0).sum())
    print(f"Terminal rewards: {n_survive} survived (+1), {n_die} died (-1)")
    print(f"Non-terminal rows (reward = 0): {(_df['reward'] == 0).sum():,}")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 4. State vector

    The ATS submission uses 44 features (demographics + vitals + ABG +
    treatments + labs). On the demo dataset we have a 20-feature subset; the
    production pipeline adds binary flags for each continuous-medication
    category which the demo doesn't carry.

    State columns are taken directly from the imputed `hourly_df`, so the state
    matrix has zero NaN.
    """
    )
    return


@app.cell
def _(state_cols):
    STATE = state_cols
    print(f"STATE features: {len(STATE)}")
    print(STATE)
    return (STATE,)


@app.cell
def _(mo):
    mo.md(
        """
    ## 5. Q-network architecture (dueling)

    Lifted from [`CLIF-RL/code/training/training.ipynb`](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF-RL/blob/ats-submission/code/training/training.ipynb).
    Backbone: `state_dim → 256 → 256` (GELU). Dueling head splits into a
    state-value stream V(s) and an advantage stream A(s, ·); the final
    Q-value is `Q = V + (A − mean(A))`. About 200 k parameters for a
    20-feature state and 4 actions.

    We instantiate it to print the parameter count but **do not train it**.
    Training on 56 hospitalizations would memorize noise. Production training
    runs on the full MIMIC-IV CLIF cohort.
    """
    )
    return


@app.cell
def _(F, STATE, nn, torch):
    class QNet(nn.Module):
        def __init__(self, state_dim, num_actions, hidden=(256, 256)):
            super().__init__()
            layers = []
            in_dim = state_dim
            for h in hidden:
                layers += [nn.Linear(in_dim, h), nn.GELU()]
                in_dim = h
            self.backbone = nn.Sequential(*layers)
            self.adv = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(),
                                     nn.Linear(in_dim, num_actions))
            self.val = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(),
                                     nn.Linear(in_dim, 1))

        def forward(self, x):
            z = self.backbone(x)
            a = self.adv(z)
            v = self.val(z)
            return v + a - a.mean(dim=1, keepdim=True)

    qnet = QNet(state_dim=len(STATE), num_actions=4)
    n_params = sum(p.numel() for p in qnet.parameters())
    print(f"QNet: state_dim={len(STATE)}, num_actions=4, parameters: {n_params:,}")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 6. Double DQN training loop (reference)

    The trainer below is exactly what the production CLIF-RL pipeline runs. We
    don't execute the loop — the cell defines the class for reference so
    attendees see the full training contract.

    Mechanics:

    - Online network picks `a* = argmax_a Q_online(s', a)` (Double DQN action selection).
    - Target network evaluates `Q_target(s', a*)` (decouples action selection from value bootstrap).
    - Bellman target: `y = r + γ · (1 − done) · Q_target(s', a*)`.
    - Huber (Smooth-L1) loss between `Q_online(s, a)` and `y`.
    - Adam optimizer, gradient clipping, soft target update τ = 1e-3.

    Real training uses a `WeightedRandomSampler` to upweight terminal-reward
    transitions, since the ±1 rewards are < 1 % of all hourly rows.
    """
    )
    return


@app.cell
def _(F, torch):
    class DDQNTrainer:
        """Reference implementation of the Double DQN trainer. Not executed in this notebook."""

        def __init__(self, q, target, opt, device, gamma=0.99, tau=1e-3, max_norm=1.0):
            self.q = q.to(device)
            self.tgt = target.to(device)
            self.tgt.load_state_dict(self.q.state_dict())
            self.opt = opt
            self.device = device
            self.gamma = gamma
            self.tau = tau
            self.max_norm = max_norm

        def _soft_update(self):
            for tp, op in zip(self.tgt.parameters(), self.q.parameters()):
                tp.data.lerp_(op.data, self.tau)

        def train_step(self, batch):
            s, a, r = batch["s"].to(self.device), batch["a"].to(self.device), batch["r"].to(self.device)
            s_next, done = batch["s_next"].to(self.device), batch["done"].to(self.device)
            q_pred = self.q(s).gather(1, a.view(-1, 1)).squeeze(1)
            with torch.no_grad():
                a_star = torch.argmax(self.q(s_next), dim=1)
                q_next = self.tgt(s_next).gather(1, a_star.view(-1, 1)).squeeze(1)
                y = r + self.gamma * (~done).float() * q_next
            loss = F.smooth_l1_loss(q_pred, y)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.max_norm)
            self.opt.step()
            self._soft_update()
            return loss.item()

    print("DDQNTrainer defined. Not executed on demo data.")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7. Concordance evaluation methodology

    The ATS submission validated the trained agent by computing, for each
    hospitalization, the fraction of hourly decisions where the physician's
    actual action matched the agent's recommendation. Then per site, fit a
    logistic regression:

    $$\mathrm{logit}\,P(\mathrm{mortality}) = \beta_0 + \beta_1 \cdot p_{\mathrm{concordance}} + \boldsymbol{\beta}^\top \mathbf{X}$$

    where the covariates **X** are age at admission plus the 6 SOFA subscores.
    The reported effect is the **adjusted OR per +10 percentage-point**
    increase in `p_concordance`. The function below is the implementation from
    [`CLIF-RL/code/training/training.ipynb`](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF-RL/blob/ats-submission/code/training/training.ipynb).
    """
    )
    return


@app.cell
def _():
    CONCORDANCE_EVAL_REFERENCE = '''
    import statsmodels.api as sm
    import math

    def concordance_evaluation(
        df, id_col="hospitalization_id",
        act_col="ACTION", pi_col="ACTION_REC",
        death_col="mortality",
        covars=("age_at_admission", "sofa_cv_97", "sofa_coag",
                "sofa_liver", "sofa_resp", "sofa_cns", "sofa_renal"),
        delta_for_or=0.10,
    ):
        df = df.copy()
        df["concordant"] = (df[act_col] == df[pi_col]).astype(int)
        agg = {"p_concordance": ("concordant", "mean"),
               death_col: (death_col, "first")}
        for c in covars:
            agg[c] = (c, "first")
        pt = df.groupby(id_col, as_index=False).agg(**agg)
        X = sm.add_constant(pt[["p_concordance", *covars]])
        y = pt[death_col].astype(int)
        glm = sm.GLM(y, X, family=sm.families.Binomial())
        res = glm.fit(disp=False, cov_type="HC3")
        # OR per +10pp concordance
        beta = res.params["p_concordance"]
        se   = res.bse["p_concordance"]
        or_delta = math.exp(beta * delta_for_or)
        ci_lo    = math.exp((beta - 1.96 * se) * delta_for_or)
        ci_hi    = math.exp((beta + 1.96 * se) * delta_for_or)
        return or_delta, ci_lo, ci_hi
    '''
    print(CONCORDANCE_EVAL_REFERENCE.strip())
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 8. ATS submission: external-validation results

    Three independent CLIF-standardized ICU datasets, each with its own
    site-specific logistic regression. The OR is the **adjusted** odds of
    in-hospital mortality per **+10 percentage-point** increase in clinician–
    agent concordance. CIs are 95 %; sites anonymized as A / B / C per
    poster convention.
    """
    )
    return


@app.cell
def _(pd):
    # From the ATS poster (Fig 1A). Sites anonymized per poster convention.
    ats_results = pd.DataFrame([
        {"site": "Site A", "patient_hours": 1_535_876,
         "pct_decrease": 3.2, "ci_lo": 0.9, "ci_hi": 5.5, "p_value": 0.006},
        {"site": "Site B", "patient_hours": 2_016_259,
         "pct_decrease": 4.8, "ci_lo": 2.6, "ci_hi": 7.0, "p_value": 0.001},
        {"site": "Site C", "patient_hours":   962_222,
         "pct_decrease": 8.4, "ci_lo": 5.7, "ci_hi": 11.0, "p_value": 0.001},
    ])
    ats_results["or"] = 1 - ats_results["pct_decrease"] / 100
    ats_results["or_lo"] = 1 - ats_results["ci_hi"] / 100
    ats_results["or_hi"] = 1 - ats_results["ci_lo"] / 100
    ats_results
    return (ats_results,)


@app.cell
def _(ats_results, plt):
    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    _y_pos = list(range(len(ats_results)))
    _ax.errorbar(
        ats_results["or"], _y_pos,
        xerr=[ats_results["or"] - ats_results["or_lo"],
              ats_results["or_hi"] - ats_results["or"]],
        fmt="s", markersize=10, capsize=6, color="#2c3e50", ecolor="#2c3e50",
    )
    _ax.axvline(1.0, linestyle="--", color="grey", linewidth=1)
    _ax.set_yticks(_y_pos)
    _ax.set_yticklabels(
        [f"{r.site}\n({r.patient_hours/1e6:.2f}M pt-hrs)"
         for r in ats_results.itertuples()]
    )
    _ax.invert_yaxis()
    _ax.set_xlabel("Adjusted OR for in-hospital mortality (per +10% concordance)")
    _ax.set_title("CLIF-RL external validation — ATS submission")
    for _i, _r in enumerate(ats_results.itertuples()):
        _p_label = "< 0.001" if _r.p_value < 0.001 else f"= {_r.p_value:.3f}"
        _ax.text(1.02 * _r.or_hi, _i,
                 f"  {_r.pct_decrease:.1f}% ↓   p {_p_label}",
                 va="center", fontsize=10)
    _ax.set_xlim(0.85, 1.05)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Headline

    > Across three CLIF-standardized ICUs, every **+10 percentage-point**
    > increase in clinician–agent concordance was associated with a **3–8 %
    > lower adjusted odds of in-hospital mortality** (all p ≤ 0.006).

    What this finding supports:

    - **Replication across heterogeneous sites** — different case mix, protocols,
      EHR systems. Supports generalization beyond the institution where the
      agent was trained.
    - **Methodological contribution of CLIF** — a single trained policy and a
      single evaluation pipeline ran at multiple ICUs without sharing
      patient-level data.

    What this finding does **not** support:

    - A causal effect of agent recommendations on mortality. The validation is
      an observational concordance analysis; residual confounding is likely.
    - Action space is coarse (2 × 2); does not represent the full ventilator
      parameter manifold.

    Full code + training: [`Common-Longitudinal-ICU-data-Format/CLIF-RL`](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF-RL) (`ats-submission` branch).
    """
    )
    return


if __name__ == "__main__":
    app.run()
