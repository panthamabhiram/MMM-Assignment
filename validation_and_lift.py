"""
validation_and_lift.py — Lift Test Loading, Ablation Benchmarks, OOS CV & Bootstrap

Handles `lift_test_results.csv` and all model testing / validation routines:
1. Parses Meta geo-holdout experiment parameters (iROAS = 1.39, 90% CI [0.94, 1.84]).
2. Runs ablation benchmarks (v0 Raw OLS, v1 Cleaned OLS, v2 Dead-End Mediator OLS).
3. Evaluates 3-fold expanding-window out-of-sample cross-validation.
4. Runs 300-draw residual + experiment-uncertainty bootstrap for 90% confidence intervals.

Run standalone (inspects lift test & baseline ablation):
    uv run --python 3.11 --with pandas --with numpy --with statsmodels --with scikit-learn python validation_and_lift.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error

from data_cleaning import CHANNELS, CONTROLS, audit_and_clean_data


def load_lift_test(path: str = "lift_test_results.csv") -> tuple[pd.DataFrame, dict]:
    """Load geo-holdout test results and extract calibration prior parameters."""
    lift_df = pd.read_csv(path)
    row = lift_df.iloc[0]
    ci_lo = float(row["iroas_90ci_low"])
    ci_hi = float(row["iroas_90ci_high"])
    meta = {
        "test_name": str(row["test_name"]),
        "channel": str(row["channel"]),
        "test_start": str(row["test_start"]),
        "test_end": str(row["test_end"]),
        "holdout_share": float(row["holdout_share_of_spend"]),
        "spend_withheld_inr": float(row["spend_withheld_inr"]),
        "incremental_revenue_inr": float(row["incremental_revenue_inr"]),
        "target_iroas": float(row["measured_iroas"]),
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "iroas_se": (ci_hi - ci_lo) / (2.0 * 1.64485),
    }
    return lift_df, meta


def run_ablation_benchmarks(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> dict:
    """Run v0 (raw), v1 (cleaned), and v2 (mediator-trap) OLS benchmarks."""
    df_b0 = df_raw.dropna().copy()
    X0 = sm.add_constant(df_b0[CHANNELS + ["festive_flag", "avg_discount_pct"]])
    m0 = sm.OLS(df_b0["revenue_inr"], X0).fit()

    X1 = sm.add_constant(df_clean[CHANNELS + ["festive_flag", "avg_discount_pct"]])
    m1 = sm.OLS(df_clean["revenue_inr"], X1).fit()

    X2 = sm.add_constant(df_clean[CHANNELS + CONTROLS + ["brand_search_index", "organic_traffic_index"]])
    m2 = sm.OLS(df_clean["revenue_inr"], X2).fit()

    return {
        "raw_ols": m0,
        "clean_ols": m1,
        "naive_mediator_ols": m2,
    }


def run_cross_validation(df: pd.DataFrame, lift_df: pd.DataFrame, fit_fn) -> pd.DataFrame:
    """3-fold expanding-window out-of-sample evaluation."""
    folds = [
        (np.arange(len(df)) < 104, (np.arange(len(df)) >= 104) & (np.arange(len(df)) < 130)),
        (np.arange(len(df)) < 120, np.arange(len(df)) >= 120),
        (np.arange(len(df)) < 130, np.arange(len(df)) >= 130),
    ]
    cv_rows = []
    y_rev = df["revenue_inr"].values
    for idx, (tr, te) in enumerate(folds, 1):
        m_cv = fit_fn(df, lift_df, train_mask=tr, calibrate_meta=True)
        r2_te = r2_score(y_rev[te], m_cv["rev_pred"][te])
        rmse_te = root_mean_squared_error(y_rev[te], m_cv["rev_pred"][te]) / 1e6
        mape_te = mean_absolute_percentage_error(y_rev[te], m_cv["rev_pred"][te]) * 100.0
        cv_rows.append({
            "fold": idx,
            "train_wks": int(tr.sum()),
            "test_wks": int(te.sum()),
            "oos_r2": r2_te,
            "oos_rmse_m": rmse_te,
            "oos_mape": mape_te,
        })
    return pd.DataFrame(cv_rows)


def run_bootstrap(df: pd.DataFrame, model: dict, n_boot: int = 300) -> dict:
    """Residual + experiment-uncertainty bootstrap for channel ROIs and Q1 forecasts."""
    rng = np.random.default_rng(42)
    n_obs = len(df)
    y_log = np.log(df["revenue_inr"].values)
    resid_log = y_log - model["log_pred"]
    X_m = model["X_m"]
    X_m_z = (X_m - model["xm_mean"]) / model["xm_std"]
    X_full = np.column_stack([np.ones(n_obs), X_m_z, model["X_ctrl_z"]])

    wt = (15.0 / n_obs) / (model["iroas_se"] ** 2)
    c_cal = model["c_cal"]
    A_boot = model["A_mat"]

    df25 = df[df["dt"].dt.year == 2025]
    rev_25_base = df25["revenue_inr"].mean()
    s25_all = df25[CHANNELS].mean().values

    boot_bm, boot_b0, boot_bc, boot_smear = [], [], [], []
    boot_avg_roi, boot_mroas_25 = [], []

    for _ in range(n_boot):
        res_draw = rng.choice(resid_log, size=n_obs, replace=True)
        y_boot_z = ((model["log_pred"] + res_draw) - model["yl_mean"]) / model["yl_std"]
        meta_iroas_draw = rng.normal(model["target_iroas"], model["iroas_se"])
        b_vec = (X_full.T @ y_boot_z) / n_obs
        b_vec[1] += wt * c_cal * meta_iroas_draw
        wb = np.linalg.solve(A_boot, b_vec)
        bm_b = np.maximum(wb[1:7] * model["yl_std"] / model["xm_std"], 0.0)
        bc_b = wb[7:] * model["yl_std"]
        b0_b = wb[0] * model["yl_std"] + model["yl_mean"] - np.sum(bm_b * model["xm_mean"])
        lp_b = b0_b + X_m @ bm_b + model["X_ctrl_z"] @ bc_b
        sm_b = float(np.mean(np.exp((model["log_pred"] + res_draw) - lp_b)))
        rp_b = np.exp(lp_b) * sm_b

        boot_bm.append(bm_b)
        boot_b0.append(b0_b)
        boot_bc.append(bc_b)
        boot_smear.append(sm_b)

        avg_r = [(rp_b * (1.0 - np.exp(-bm_b[i] * X_m[:, i]))).sum() / df[c].sum() for i, c in enumerate(CHANNELS)]
        mr_25 = [rev_25_base * bm_b[i] / (s25_all[i] + model["ks"][i]) for i in range(len(CHANNELS))]
        boot_avg_roi.append(avg_r)
        boot_mroas_25.append(mr_25)

    return {
        "bm": np.array(boot_bm),
        "b0": np.array(boot_b0),
        "bc": np.array(boot_bc),
        "smear": np.array(boot_smear),
        "avg_roi": np.array(boot_avg_roi),
        "mroas_25": np.array(boot_mroas_25),
    }


if __name__ == "__main__":
    _, lift_meta = load_lift_test()
    df_raw, df_clean, _ = audit_and_clean_data(save_clean=False)
    bench = run_ablation_benchmarks(df_raw, df_clean)
    print("Lift Test Anchor:", lift_meta)
    print("\nAblation Benchmarks (R2):")
    for k, m in bench.items():
        print(f"  {k:20s}: R2={m.rsquared:.4f}, Meta={m.params['meta_spend']:.2f}, Google={m.params['google_search_spend']:.2f}, YT={m.params['youtube_spend']:.2f}, TV={m.params['tv_spend']:.2f}")
