"""
refined_mmm.py — Main MMM Model, Q1 2026 Budget Optimization & Pipeline Runner

Imports modular components:
- `data_cleaning.py`: loads, audits, and cleans `mmm_weekly_data.csv`
- `validation_and_lift.py`: loads `lift_test_results.csv`, runs ablation, OOS CV, and bootstrap

Core modeling & optimization in this file:
1. Stage-1 mediator orthogonalization (`organic_shock`, `brand_shock`)
2. Calibrated Multiplicative (Log-Revenue) Adstock-Saturation Ridge MMM
3. Q1 2026 constrained budget optimization (₹16 Cr, 50-150% channel bounds) & risk frontier
4. Saves `fig2`, `fig3`, `fig4` and `results_summary.json` into `report_assets/`

Run:
    uv run --python 3.11 --with pandas --with numpy --with statsmodels --with scikit-learn --with scipy --with matplotlib python refined_mmm.py
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_cleaning import (
    ASSETS_DIR,
    CHANNELS,
    CH_LABELS,
    CONTROLS,
    audit_and_clean_data,
    make_bug_screenshot,
)
from validation_and_lift import (
    load_lift_test,
    run_ablation_benchmarks,
    run_bootstrap,
    run_cross_validation,
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.8,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})


# ---------------------------------------------------------------------------
# 1. Core Adstock & Mediator Orthogonalization
# ---------------------------------------------------------------------------
def adstock(x: np.ndarray, alpha: float) -> np.ndarray:
    """Geometric carryover adstock transformation."""
    out = np.empty(len(x), dtype=float)
    prev = 0.0
    for i in range(len(x)):
        prev = x[i] + alpha * prev
        out[i] = prev
    return out


def orthogonalize_mediators(df: pd.DataFrame):
    """Strip upper-funnel media influence from brand_search and organic_traffic indices."""
    upper = ["meta_spend", "youtube_spend", "influencer_spend", "tv_spend"]
    X_s1 = df[CONTROLS].copy()
    for u, al in zip(upper, [0.30, 0.60, 0.65, 0.65]):
        X_s1[u + "_ad"] = adstock(df[u].values, al)
    X_s1 = sm.add_constant(X_s1)
    m_org = sm.OLS(np.log(df["organic_traffic_index"]), X_s1).fit()
    m_brd = sm.OLS(np.log(df["brand_search_index"]), X_s1).fit()
    df["organic_shock"] = m_org.resid
    df["brand_shock"] = m_brd.resid
    return m_org, m_brd


# ---------------------------------------------------------------------------
# 2. Calibrated Multiplicative Log-Revenue MMM
# ---------------------------------------------------------------------------
def fit_refined_mmm(
    df: pd.DataFrame,
    lift_df: pd.DataFrame,
    train_mask: np.ndarray = None,
    ridge_l2: float = 0.005,
    calibrate_meta: bool = True,
) -> dict:
    """Fit Log-Revenue MMM with adstock, log1p saturation, Ridge & Meta holdout calibration."""
    if train_mask is None:
        train_mask = np.ones(len(df), dtype=bool)

    ctrl_cols = CONTROLS + ["organic_shock", "brand_shock"]
    X_ctrl = df[ctrl_cols].values
    c_mean = X_ctrl[train_mask].mean(axis=0)
    c_std = X_ctrl[train_mask].std(axis=0) + 1e-8
    X_ctrl_z = (X_ctrl - c_mean) / c_std

    y_log = np.log(df["revenue_inr"].values)
    y_rev = df["revenue_inr"].values
    yl_tr = y_log[train_mask]
    yl_mean = yl_tr.mean()
    yl_std = yl_tr.std()
    yl_tr_z = (yl_tr - yl_mean) / yl_std
    n_tr = int(train_mask.sum())

    raw_spends = [df[c].values for c in CHANNELS]
    active_means = np.array([df.loc[df[c] > 0, c].mean() for c in CHANNELS])

    # Geo-holdout indices (test window + 4 post weeks)
    t_start = lift_df["test_start"].iloc[0]
    t_end = lift_df["test_end"].iloc[0]
    hold_test = np.where((df["week_start"] >= t_start) & (df["week_start"] <= t_end))[0]
    hold_eval = np.arange(hold_test[0], hold_test[-1] + 5)
    hold_share = float(lift_df["holdout_share_of_spend"].iloc[0])
    target_iroas = float(lift_df["measured_iroas"].iloc[0])
    ci_lo = float(lift_df["iroas_90ci_low"].iloc[0])
    ci_hi = float(lift_df["iroas_90ci_high"].iloc[0])
    iroas_se = (ci_hi - ci_lo) / (2.0 * 1.64485)

    meta_cf = raw_spends[0].copy()
    meta_cf[hold_test] *= (1.0 - hold_share)
    withheld = (raw_spends[0][hold_test] * hold_share).sum()

    alpha_bounds = np.array([
        (0.10, 0.55),  # Meta
        (0.00, 0.25),  # Google Search
        (0.25, 0.75),  # YouTube
        (0.25, 0.75),  # Influencer
        (0.35, 0.80),  # TV
        (0.00, 0.25),  # Amazon Ads
    ])

    def build_X_m(alphas, ks):
        cols = []
        for i in range(len(CHANNELS)):
            ad = adstock(raw_spends[i], alphas[i]) * (1.0 - alphas[i])
            cols.append(np.log1p(ad / ks[i]))
        return np.column_stack(cols)

    def solve_inner(X_m, alphas, ks):
        xm_mean = X_m[train_mask].mean(axis=0)
        xm_std = X_m[train_mask].std(axis=0) + 1e-8
        X_m_z = (X_m - xm_mean) / xm_std
        n_m = len(CHANNELS)
        n_c = X_ctrl_z.shape[1]
        X_tr = np.column_stack([np.ones(n_tr), X_m_z[train_mask], X_ctrl_z[train_mask]])
        A = (X_tr.T @ X_tr) / n_tr
        b = (X_tr.T @ yl_tr_z) / n_tr

        reg = np.zeros(1 + n_m + n_c)
        reg[1:1 + n_m] = ridge_l2
        reg[1 + n_m:] = ridge_l2 * 0.5
        A += np.diag(reg)

        c_cal = 0.0
        if calibrate_meta:
            ad_f = adstock(raw_spends[0], alphas[0]) * (1.0 - alphas[0])
            ad_cf = adstock(meta_cf, alphas[0]) * (1.0 - alphas[0])
            d_sat = (np.log1p(ad_f / ks[0]) - np.log1p(ad_cf / ks[0]))[hold_eval]
            c_cal = (yl_std / xm_std[0]) * np.sum(y_rev[hold_eval] * d_sat) / withheld
            wt = (15.0 / n_tr) / (iroas_se ** 2)
            A[1, 1] += wt * (c_cal ** 2)
            b[1] += wt * c_cal * target_iroas

        w = np.linalg.solve(A, b)
        mse_z = np.mean((yl_tr_z - X_tr @ w) ** 2)
        neg_pen = np.sum(np.minimum(w[1:1 + n_m], 0.0) ** 2) * 100.0
        cal_loss = 0.0
        if calibrate_meta:
            pred_iroas = w[1] * c_cal
            cal_loss = (15.0 / n_tr) * ((pred_iroas - target_iroas) / iroas_se) ** 2
        obj = mse_z + np.sum(reg * (w ** 2)) + neg_pen + cal_loss
        return w, xm_mean, xm_std, A, c_cal, obj

    def unpack(theta):
        sig_a = 1.0 / (1.0 + np.exp(-np.clip(theta[:6], -10, 10)))
        alphas = alpha_bounds[:, 0] + (alpha_bounds[:, 1] - alpha_bounds[:, 0]) * sig_a
        sig_k = 1.0 / (1.0 + np.exp(-np.clip(theta[6:12], -10, 10)))
        ks = active_means * (0.5 + 2.5 * sig_k)
        return alphas, ks

    res = minimize(
        lambda th: solve_inner(build_X_m(*unpack(th)), *unpack(th))[5],
        np.zeros(12),
        method="L-BFGS-B",
        bounds=[(-4, 4)] * 12,
        options={"maxiter": 90},
    )
    alphas, ks = unpack(res.x)
    X_m = build_X_m(alphas, ks)
    w, xm_mean, xm_std, A_mat, c_cal, _ = solve_inner(X_m, alphas, ks)

    bm = w[1:7] * yl_std / xm_std
    bc = w[7:] * yl_std
    b0 = w[0] * yl_std + yl_mean - np.sum(bm * xm_mean)
    log_pred = b0 + X_m @ bm + X_ctrl_z @ bc
    smear = float(np.mean(np.exp(y_log[train_mask] - log_pred[train_mask])))
    rev_pred = np.exp(log_pred) * smear

    # Predicted Meta geo-holdout iROAS
    ad_cf = adstock(meta_cf, alphas[0]) * (1.0 - alphas[0])
    d_log_m = (X_m[:, 0] - np.log1p(ad_cf / ks[0])) * bm[0]
    rev_cf = rev_pred * np.exp(-d_log_m)
    pred_iroas = float((rev_pred[hold_eval] - rev_cf[hold_eval]).sum() / withheld)

    return {
        "alphas": alphas, "ks": ks, "b0": float(b0), "bm": bm, "bc": bc,
        "smear": smear, "X_m": X_m, "xm_mean": xm_mean, "xm_std": xm_std,
        "X_ctrl_z": X_ctrl_z, "c_mean": c_mean, "c_std": c_std,
        "ctrl_cols": ctrl_cols, "log_pred": log_pred, "rev_pred": rev_pred,
        "pred_iroas": pred_iroas, "A_mat": A_mat, "c_cal": c_cal,
        "yl_mean": float(yl_mean), "yl_std": float(yl_std),
        "target_iroas": target_iroas, "iroas_se": iroas_se,
        "active_means": active_means,
    }


# ---------------------------------------------------------------------------
# 3. Q1 2026 Constrained Budget Optimization
# ---------------------------------------------------------------------------
def optimize_q1_2026(df: pd.DataFrame, model: dict, boot: dict) -> dict:
    """Optimize ₹16 Cr Q1 2026 budget under 50-150% channel constraints & risk frontier."""
    df25 = df[df["dt"].dt.year == 2025]
    q1 = df[df["dt"].dt.month.isin([1, 2, 3])]
    q25_cr = (df25[CHANNELS].sum() / 4.0).values / 1e7
    s25_all = df25[CHANNELS].mean().values
    raw_spends = [df[c].values for c in CHANNELS]

    q1_dates = pd.date_range("2026-01-05", periods=13, freq="W-MON")
    q1_woy = q1_dates.isocalendar().week.astype(int).values
    q1_t = np.arange(len(df), len(df) + 13)
    q1_ctrl = np.zeros((13, len(model["ctrl_cols"])))
    q1_ctrl[:, 0] = q1_t
    q1_ctrl[:, 1] = np.sin(2.0 * np.pi * q1_woy / 52.0)
    q1_ctrl[:, 2] = np.cos(2.0 * np.pi * q1_woy / 52.0)
    q1_ctrl[:, 3] = np.sin(4.0 * np.pi * q1_woy / 52.0)
    q1_ctrl[:, 4] = np.cos(4.0 * np.pi * q1_woy / 52.0)
    q1_ctrl[:, 5] = q1.loc[q1["dt"].dt.year == 2025, "avg_discount_pct"].mean()
    q1_ctrl[:, 6] = df25["price_index"].iloc[-4:].mean()
    q1_ctrl[:, 7] = 0.0
    q1_ctrl[:, 8] = 0.0
    q1_ctrl[:, 9] = q1["competitor_promo_flag"].mean()
    q1_ctrl[:, 10] = 0.25
    q1_ctrl_z = (q1_ctrl - model["c_mean"]) / model["c_std"]

    alphas = model["alphas"]
    ks = model["ks"]
    init_ad = np.array([adstock(raw_spends[i], alphas[i])[-1] for i in range(len(CHANNELS))])

    def eval_q1_cr(q_cr, bm_u=model["bm"], b0_u=model["b0"], bc_u=model["bc"], sm_u=model["smear"], tv_burst_wks=None):
        w_spend = (q_cr * 1e7) / 13.0
        base_lr = b0_u + q1_ctrl_z @ bc_u
        rev_q1 = 0.0
        ad_tv = init_ad[4]
        tv_sched = None
        if tv_burst_wks is not None:
            tv_sched = np.zeros(13)
            for wk_i in tv_burst_wks:
                tv_sched[wk_i] = (q_cr[4] * 1e7) / len(tv_burst_wks)

        for wk in range(13):
            lr = base_lr[wk]
            for i in range(len(CHANNELS)):
                a = alphas[i]
                if i == 4 and tv_sched is not None:
                    ad_tv = tv_sched[wk] + a * ad_tv
                    ad_wk = ad_tv
                else:
                    ad_wk = (a ** (wk + 1)) * init_ad[i] + w_spend[i] * (1.0 - a ** (wk + 1)) / (1.0 - a)
                lr += bm_u[i] * np.log1p(ad_wk * (1.0 - a) / ks[i])
            rev_q1 += np.exp(lr) * sm_u

        end_with = np.array([
            (alphas[i] ** 13) * init_ad[i] + w_spend[i] * (1.0 - alphas[i] ** 13) / (1.0 - alphas[i])
            for i in range(len(CHANNELS))
        ])
        if tv_sched is not None:
            end_with[4] = ad_tv
        end_zero = np.array([(alphas[i] ** 13) * init_ad[i] for i in range(len(CHANNELS))])
        base_q2_lr = base_lr[-1]
        tail_gain = 0.0
        for pw in range(1, 17):
            lr_w, lr_z = base_q2_lr, base_q2_lr
            for i in range(len(CHANNELS)):
                a = alphas[i]
                ad_w = (a ** pw) * end_with[i] + s25_all[i] * (1.0 - a ** pw) / (1.0 - a)
                ad_z = (a ** pw) * end_zero[i] + s25_all[i] * (1.0 - a ** pw) / (1.0 - a)
                lr_w += bm_u[i] * np.log1p(ad_w * (1.0 - a) / ks[i])
                lr_z += bm_u[i] * np.log1p(ad_z * (1.0 - a) / ks[i])
            tail_gain += (np.exp(lr_w) - np.exp(lr_z)) * sm_u
        return rev_q1 / 1e7, (rev_q1 + tail_gain) / 1e7

    bounds_cr = [(0.5 * q25_cr[i], 1.5 * q25_cr[i]) for i in range(len(CHANNELS))]
    x0_cr = q25_cr * (16.0 / q25_cr.sum())

    def eval_dist(x_cr, n_draws=300, tv_burst_wks=None):
        q1_revs = np.empty(n_draws)
        lifts_tot = np.empty(n_draws)
        for b in range(n_draws):
            ro_q1, ro_tot = eval_q1_cr(x_cr, boot["bm"][b], boot["b0"][b], boot["bc"][b], boot["smear"][b], tv_burst_wks)
            _, rs_tot = eval_q1_cr(x0_cr, boot["bm"][b], boot["b0"][b], boot["bc"][b], boot["smear"][b], None)
            q1_revs[b] = ro_q1
            lifts_tot[b] = ro_tot - rs_tot
        return q1_revs, lifts_tot

    frontier = []
    for lam in [0.0, 0.25, 0.50, 0.75, 1.0]:
        def obj(x):
            _, lifts = eval_dist(x, n_draws=120)
            return -(np.mean(lifts) - lam * np.std(lifts))
        res = minimize(
            obj, x0_cr, bounds=bounds_cr,
            constraints={"type": "eq", "fun": lambda x: np.sum(x) - 16.0},
            method="SLSQP", options={"ftol": 1e-6, "maxiter": 100},
        )
        q1_revs, lifts_tot = eval_dist(res.x, n_draws=300)
        pt_q1, pt_tot = eval_q1_cr(res.x)
        _, sq_tot = eval_q1_cr(x0_cr)
        frontier.append({
            "lambda": lam,
            "alloc": res.x,
            "pt_q1_rev": pt_q1,
            "pt_tot_lift": pt_tot - sq_tot,
            "mean_q1_rev": float(np.mean(q1_revs)),
            "q1_ci_lo": float(np.percentile(q1_revs, 5)),
            "q1_ci_hi": float(np.percentile(q1_revs, 95)),
            "mean_tot_lift": float(np.mean(lifts_tot)),
            "sd_tot_lift": float(np.std(lifts_tot)),
            "lift_ci_lo": float(np.percentile(lifts_tot, 5)),
            "lift_ci_hi": float(np.percentile(lifts_tot, 95)),
            "prob_pos": float(np.mean(lifts_tot > 0) * 100.0),
        })

    rec = frontier[3]  # lambda = 0.75
    burst_q1, burst_tot = eval_q1_cr(rec["alloc"], tv_burst_wks=[4, 5])
    return {
        "q25_cr": q25_cr, "x0_cr": x0_cr, "bounds_cr": bounds_cr,
        "frontier": frontier, "rec": rec,
        "burst_q1_rev": burst_q1, "burst_tot_rev": burst_tot,
        "eval_q1_cr": eval_q1_cr,
    }


# ---------------------------------------------------------------------------
# 4. Visualization Builders (Figures 2, 3, 4)
# ---------------------------------------------------------------------------
def make_roi_and_validation_chart(df, bench, model, boot, cv_df):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.5, 4.3), dpi=240, facecolor="#FFFFFF",
        gridspec_kw={"width_ratios": [1.05, 1.15], "wspace": 0.22, "left": 0.115, "right": 0.98, "top": 0.86, "bottom": 0.14},
    )
    raw_roi = bench["raw_ols"].params[CHANNELS].values
    pt_avg = np.array([(model["rev_pred"] * (1.0 - np.exp(-model["bm"][i] * model["X_m"][:, i]))).sum() / df[c].sum() for i, c in enumerate(CHANNELS)])
    rev_25_base = df.loc[df["dt"].dt.year == 2025, "revenue_inr"].mean()
    s25_all = df.loc[df["dt"].dt.year == 2025, CHANNELS].mean().values
    pt_mroas = np.array([rev_25_base * model["bm"][i] / (s25_all[i] + model["ks"][i]) for i in range(len(CHANNELS))])
    m_lo = np.percentile(boot["mroas_25"], 5, axis=0)
    m_hi = np.percentile(boot["mroas_25"], 95, axis=0)

    order = np.argsort(pt_mroas)
    y_pos = np.arange(len(CHANNELS))
    h = 0.27

    ax1.barh(y_pos - h, np.clip(raw_roi[order], -1.5, 4.8), height=h, color="#94A3B8", label="Baseline v0 OLS (Google clipped: 13.72)")
    ax1.barh(y_pos, pt_avg[order], height=h, color="#40916C", label="Refined MMM: Historical Avg ROI")
    ax1.barh(y_pos + h, pt_mroas[order], height=h, color="#1B4332",
             xerr=[pt_mroas[order] - m_lo[order], m_hi[order] - pt_mroas[order]],
             error_kw={"ecolor": "#0F172A", "lw": 1.1, "capsize": 2.5},
             label="Refined MMM: 2025 mROAS (90% CI)")

    g_idx = list(order).index(CHANNELS.index("google_search_spend"))
    ax1.text(4.72, g_idx - h, "13.72 →", ha="right", va="center", fontsize=7.5, fontweight="bold", color="#DC2626",
             bbox=dict(boxstyle="round,pad=0.15", facecolor="#FEE2E2", edgecolor="none"))
    ax1.axvspan(0.94, 1.84, color="#FEF3C7", alpha=0.45, zorder=0)
    ax1.axvline(1.0, color="#DC2626", ls="--", lw=1.0, label="Break-even (₹1.00)")
    ax1.axvline(1.39, color="#D97706", ls=":", lw=1.2, label="Meta Holdout iROAS (1.39)")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([CH_LABELS[CHANNELS[i]] for i in order], fontsize=8.5, fontweight="bold", color="#1E293B")
    ax1.set_xlim(-1.2, 5.2)
    ax1.set_ylim(-0.6, 5.6)
    ax1.set_xlabel("Incremental Revenue per ₹1 Spend (₹)", fontsize=8.5, color="#334155")
    ax1.set_title("Baseline vs. Refined Channel ROI & 2025 Marginal ROAS", loc="left", fontsize=9.8, fontweight="bold", color="#0F172A")
    ax1.grid(True, axis="x", alpha=0.5)
    ax1.legend(fontsize=6.2, loc="lower right", framealpha=0.95, labelspacing=0.25, handlelength=1.4)

    dates = df["dt"]
    ax2.plot(dates, df["revenue_inr"] / 1e7, color="#0F172A", lw=1.5, label="Actual Weekly Revenue")
    ax2.plot(dates, model["rev_pred"] / 1e7, color="#2D6A4F", lw=1.4, ls="--", label=f"Refined MMM Fit (R²={r2_score(df['revenue_inr'], model['rev_pred']):.3f})")
    ax2.axvspan(df.loc[130, "dt"], df.loc[155, "dt"], color="#DCFCE7", alpha=0.45,
                label=f"OOS Holdout Fold 3 (R²={cv_df.iloc[2]['oos_r2']:.3f}, MAPE={cv_df['oos_mape'].mean():.1f}%)")
    ax2.axvspan(pd.to_datetime("2025-04-07"), pd.to_datetime("2025-06-09"), color="#FEF3C7", alpha=0.55,
                label="Meta Geo-Holdout + 4 Post Wks")
    ax2.set_title("Model Fit & Out-of-Sample Generalization (156 Weeks)", loc="left", fontsize=9.8, fontweight="bold", color="#0F172A")
    ax2.set_ylabel("Weekly Revenue (₹ Crores)", fontsize=8.5, color="#334155")
    ax2.tick_params(axis="x", labelsize=7.5, rotation=18)
    ax2.tick_params(axis="y", labelsize=8)
    ax2.grid(True, alpha=0.5)
    ax2.legend(fontsize=7.0, loc="upper left", framealpha=0.95)

    fig.savefig(ASSETS_DIR / "fig2_baseline_vs_refined_roi.png", dpi=240)
    plt.close(fig)


def make_saturation_and_adstock_chart(df, model):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.5, 4.1), dpi=240, facecolor="#FFFFFF",
        gridspec_kw={"wspace": 0.22, "left": 0.07, "right": 0.98, "top": 0.86, "bottom": 0.14},
    )
    colors = ["#1D4ED8", "#DC2626", "#D97706", "#059669", "#7C3AED", "#0F766E"]
    df25 = df[df["dt"].dt.year == 2025]
    rev_25_base = df25["revenue_inr"].mean()
    s25_all = df25[CHANNELS].mean().values

    mults = np.linspace(0.4, 1.6, 80)
    for i, c in enumerate(CHANNELS):
        s_grid = mults * s25_all[i]
        mroas_curve = rev_25_base * model["bm"][i] / (s_grid + model["ks"][i])
        ax1.plot(mults * 100, mroas_curve, color=colors[i], lw=2.0, label=CH_LABELS[c])
        pt = rev_25_base * model["bm"][i] / (s25_all[i] + model["ks"][i])
        ax1.scatter([100], [pt], color=colors[i], s=32, zorder=5, edgecolor="white", lw=0.8)

    ax1.axhline(1.0, color="#475569", ls="--", lw=1.0, label="Break-even (₹1.00)")
    ax1.axvspan(50, 150, color="#F1F5F9", alpha=0.7, zorder=0, label="Allowed Q1 Band (50–150%)")
    ax1.set_title("Diminishing Returns: Marginal ROAS vs. 2025 Spend Level", loc="left", fontsize=9.8, fontweight="bold", color="#0F172A")
    ax1.set_xlabel("% of 2025 Average Weekly Spend (100% = Status Quo)", fontsize=8.5, color="#334155")
    ax1.set_ylabel("Marginal ROAS (₹ Revenue / Next ₹1)", fontsize=8.5, color="#334155")
    ax1.set_ylim(0, 3.6)
    ax1.grid(True, alpha=0.5)
    ax1.legend(fontsize=7.0, ncol=2, loc="upper right", framealpha=0.95)

    lags = np.arange(0, 9)
    for i, c in enumerate(CHANNELS):
        a = model["alphas"][i]
        weights = (1.0 - a) * (a ** lags) * 100.0
        ax2.plot(lags, weights, color=colors[i], lw=2.0, marker="o", ms=3.5,
                 label=f"{CH_LABELS[c]} (α={a:.2f}, t½={np.log(0.5)/np.log(max(a,1e-6)):.1f}w)" if a > 0.01 else f"{CH_LABELS[c]} (α={a:.2f})")
    ax2.set_title("Adstock Memory: % of Media Effect Realized by Week Lag", loc="left", fontsize=9.8, fontweight="bold", color="#0F172A")
    ax2.set_xlabel("Weeks After Ad Exposure (0 = Same Week)", fontsize=8.5, color="#334155")
    ax2.set_ylabel("Share of Total Effect (%)", fontsize=8.5, color="#334155")
    ax2.set_xticks(lags)
    ax2.grid(True, alpha=0.5)
    ax2.legend(fontsize=7.2, loc="upper right", framealpha=0.95)

    fig.savefig(ASSETS_DIR / "fig3_saturation_and_adstock.png", dpi=240)
    plt.close(fig)


def make_budget_optimization_chart(opt):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.5, 4.1), dpi=240, facecolor="#FFFFFF",
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.24, "left": 0.07, "right": 0.98, "top": 0.86, "bottom": 0.14},
    )
    q25 = opt["q25_cr"]
    rec_risk_adj = opt["rec"]["alloc"]
    rec_neutral = opt["frontier"][0]["alloc"]
    bounds = opt["bounds_cr"]

    x = np.arange(len(CHANNELS))
    w = 0.26
    ax1.bar(x - w, q25, width=w, color="#94A3B8", label="2025 Avg Quarterly (₹16.0 Cr)")
    ax1.bar(x, rec_neutral, width=w, color="#60A5FA", label="Point-Optimal λ=0.00 (Risk-Neutral)")
    ax1.bar(x + w, rec_risk_adj, width=w, color="#1B4332", label="Recommended λ=0.75 (High-Confidence)")

    for i in range(len(CHANNELS)):
        lo, hi = bounds[i]
        ax1.plot([i - 0.35, i + 0.35], [lo, lo], color="#DC2626", ls=":", lw=1.1)
        ax1.plot([i - 0.35, i + 0.35], [hi, hi], color="#DC2626", ls=":", lw=1.1)
        pct = (rec_risk_adj[i] / q25[i] - 1.0) * 100.0
        ax1.text(i + w, rec_risk_adj[i] + 0.12, f"{pct:+.0f}%", ha="center", va="bottom",
                 fontsize=7.2, fontweight="bold", color="#1B4332")

    ax1.plot([], [], color="#DC2626", ls=":", lw=1.1, label="50%–150% Constraint Bounds")
    ax1.set_xticks(x)
    ax1.set_xticklabels([CH_LABELS[c] for c in CHANNELS], fontsize=8.0, fontweight="bold", rotation=15)
    ax1.set_ylabel("Quarterly Budget (₹ Crores)", fontsize=8.5, color="#334155")
    ax1.set_ylim(0, 8.2)
    ax1.set_title("Q1 2026 Budget Reallocation (₹16 Cr Total)", loc="left", fontsize=9.8, fontweight="bold", color="#0F172A")
    ax1.grid(True, axis="y", alpha=0.5)
    ax1.legend(fontsize=7.0, loc="upper right", framealpha=0.95)

    fr = opt["frontier"]
    sds = [f["sd_tot_lift"] for f in fr]
    means = [f["mean_tot_lift"] for f in fr]
    ci_los = [f["lift_ci_lo"] for f in fr]
    ci_his = [f["lift_ci_hi"] for f in fr]

    ax2.plot(sds, means, color="#1B4332", lw=2.0, marker="o", ms=6, label="Mean Expected Lift (incl. Q2 Tail)")
    ax2.fill_between(sds, ci_los, ci_his, color="#1B4332", alpha=0.14, label="90% Bootstrap CI Band")
    ax2.axhline(0.0, color="#DC2626", ls="--", lw=1.0, label="Break-Even vs. Status Quo")

    for f in fr:
        lbl = f"λ={f['lambda']:.2f}\n(P>0: {f['prob_pos']:.0f}%)"
        if f["lambda"] == 0.00:
            xy_off = (-6, -26)
        elif f["lambda"] == 0.25:
            xy_off = (-22, 10)
        elif f["lambda"] == 0.75:
            xy_off = (8, -24)
        else:
            xy_off = (0, 10)
        ax2.annotate(lbl, (f["sd_tot_lift"], f["mean_tot_lift"]),
                     textcoords="offset points", xytext=xy_off, ha="center", fontsize=6.8,
                     fontweight="bold" if f["lambda"] == 0.75 else "normal",
                     color="#1B4332" if f["lambda"] == 0.75 else "#334155")

    rec = opt["rec"]
    ax2.scatter([rec["sd_tot_lift"]], [rec["mean_tot_lift"]], color="#D97706", s=90, zorder=6,
                edgecolor="#0F172A", lw=1.2, label="Chosen Split (λ=0.75, 100% P>0)")
    ax2.set_title("Risk-Return Frontier Across Bootstrap Posterior Draws", loc="left", fontsize=9.8, fontweight="bold", color="#0F172A")
    ax2.set_xlabel("Lift Uncertainty (Bootstrap Std Dev, ₹ Crores)", fontsize=8.5, color="#334155")
    ax2.set_ylabel("Expected Incremental Revenue Lift (₹ Crores)", fontsize=8.5, color="#334155")
    ax2.set_xlim(0.25, 2.02)
    ax2.set_ylim(-1.0, 5.8)
    ax2.grid(True, alpha=0.5)
    ax2.legend(fontsize=7.0, loc="upper left", framealpha=0.95)

    fig.savefig(ASSETS_DIR / "fig4_budget_optimization.png", dpi=240)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. End-to-End Orchestrator
# ---------------------------------------------------------------------------
def main():
    df_raw, df, audit = audit_and_clean_data()
    lift_df, _ = load_lift_test()

    bench = run_ablation_benchmarks(df_raw, df)
    m_org, m_brd = orthogonalize_mediators(df)

    m_uncal = fit_refined_mmm(df, lift_df, calibrate_meta=False)
    m_cal = fit_refined_mmm(df, lift_df, calibrate_meta=True)
    cv_df = run_cross_validation(df, lift_df, fit_refined_mmm)
    boot = run_bootstrap(df, m_cal, n_boot=300)
    opt = optimize_q1_2026(df, m_cal, boot)

    make_bug_screenshot(df_raw, df, audit)
    make_roi_and_validation_chart(df, bench, m_cal, boot, cv_df)
    make_saturation_and_adstock_chart(df, m_cal)
    make_budget_optimization_chart(opt)

    df25 = df[df["dt"].dt.year == 2025]
    rev_25_base = df25["revenue_inr"].mean()
    s25_all = df25[CHANNELS].mean().values

    channel_rows = []
    for i, c in enumerate(CHANNELS):
        pt_avg = float((m_cal["rev_pred"] * (1.0 - np.exp(-m_cal["bm"][i] * m_cal["X_m"][:, i]))).sum() / df[c].sum())
        pt_mroas = float(rev_25_base * m_cal["bm"][i] / (s25_all[i] + m_cal["ks"][i]))
        a_lo, a_hi = np.percentile(boot["avg_roi"][:, i], [5, 95])
        m_lo, m_hi = np.percentile(boot["mroas_25"][:, i], [5, 95])
        alpha_v = float(m_cal["alphas"][i])
        hl = float(np.log(0.5) / np.log(alpha_v)) if alpha_v > 0.01 else 0.0
        channel_rows.append({
            "channel": c, "label": CH_LABELS[c],
            "raw_ols_roi": float(bench["raw_ols"].params[c]),
            "clean_ols_roi": float(bench["clean_ols"].params[c]),
            "naive_med_ols_roi": float(bench["naive_mediator_ols"].params[c]),
            "alpha": alpha_v, "half_life_wks": hl,
            "k_over_active_mean": float(m_cal["ks"][i] / m_cal["active_means"][i]),
            "avg_roi": pt_avg, "avg_roi_ci": [float(a_lo), float(a_hi)],
            "mroas_2025": pt_mroas, "mroas_2025_ci": [float(m_lo), float(m_hi)],
            "q25_avg_cr": float(opt["q25_cr"][i]),
            "rec_q1_cr": float(opt["rec"]["alloc"][i]),
            "rec_chg_pct": float((opt["rec"]["alloc"][i] / opt["q25_cr"][i] - 1.0) * 100.0),
            "pt_opt_cr": float(opt["frontier"][0]["alloc"][i]),
        })

    summary = {
        "audit": {
            "raw_rows": audit["raw_rows"], "clean_rows": audit["clean_rows"],
            "duplicate_date": "2025-03-10", "meta_lakhs_count": len(audit["meta_lakhs_rows"]),
            "yt_imputed_count": len(audit["yt_na_rows"]), "yt_median_cpm": round(audit["yt_median_cpm"], 2),
            "tv_neg_val": -1250000.0,
        },
        "benchmarks": {
            "raw_ols_r2": float(bench["raw_ols"].rsquared),
            "raw_ols_dw": float(sm.stats.durbin_watson(bench["raw_ols"].resid)),
            "clean_ols_r2": float(bench["clean_ols"].rsquared),
            "naive_med_ols_r2": float(bench["naive_mediator_ols"].rsquared),
            "stage1_org_r2": float(m_org.rsquared), "stage1_brd_r2": float(m_brd.rsquared),
            "uncal_full_r2_rev": float(r2_score(df["revenue_inr"], m_uncal["rev_pred"])),
            "uncal_meta_holdout_iroas": float(m_uncal["pred_iroas"]),
            "cal_full_r2_rev": float(r2_score(df["revenue_inr"], m_cal["rev_pred"])),
            "cal_full_r2_log": float(r2_score(np.log(df["revenue_inr"]), m_cal["log_pred"])),
            "cal_dw": float(sm.stats.durbin_watson(np.log(df["revenue_inr"]) - m_cal["log_pred"])),
            "cal_meta_holdout_iroas": float(m_cal["pred_iroas"]),
            "cv_mean_r2": float(cv_df["oos_r2"].mean()),
            "cv_mean_rmse_m": float(cv_df["oos_rmse_m"].mean()),
            "cv_mean_mape": float(cv_df["oos_mape"].mean()),
            "cv_folds": cv_df.to_dict("records"),
        },
        "controls": {
            col: {"raw_coef": float(m_cal["bc"][j] / m_cal["c_std"][j]), "std_coef": float(m_cal["bc"][j])}
            for j, col in enumerate(m_cal["ctrl_cols"])
        },
        "channels": channel_rows,
        "q1_optimization": {
            "status_quo_q1_cr": float(opt["eval_q1_cr"](opt["x0_cr"])[0]),
            "status_quo_tot_cr": float(opt["eval_q1_cr"](opt["x0_cr"])[1]),
            "rec_lambda": 0.75,
            "rec_q1_rev_cr": opt["rec"]["mean_q1_rev"],
            "rec_q1_ci": [opt["rec"]["q1_ci_lo"], opt["rec"]["q1_ci_hi"]],
            "rec_tot_lift_cr": opt["rec"]["mean_tot_lift"],
            "rec_tot_lift_ci": [opt["rec"]["lift_ci_lo"], opt["rec"]["lift_ci_hi"]],
            "rec_prob_pos": opt["rec"]["prob_pos"],
            "burst_tv_q1_rev_cr": opt["burst_q1_rev"],
            "frontier": [
                {k: v for k, v in f.items() if k != "alloc"} | {"alloc": [float(x) for x in f["alloc"]]}
                for f in opt["frontier"]
            ],
        },
    }
    (ASSETS_DIR / "results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== REFINED MMM COMPLETE ===")
    print(f"Full Fit R2 (Revenue): {summary['benchmarks']['cal_full_r2_rev']:.4f} | DW: {summary['benchmarks']['cal_dw']:.3f}")
    print(f"3-Fold OOS Mean R2: {summary['benchmarks']['cv_mean_r2']:.4f} | Fold 3 R2: {cv_df.iloc[2]['oos_r2']:.4f} | Mean MAPE: {summary['benchmarks']['cv_mean_mape']:.2f}%")
    print(f"Meta Holdout iROAS -> Uncalibrated: {summary['benchmarks']['uncal_meta_holdout_iroas']:.3f} | Calibrated: {summary['benchmarks']['cal_meta_holdout_iroas']:.3f} (Target: 1.39)")
    print("\nChannel Summary:")
    for r in channel_rows:
        print(f"  {r['label']:14s} | RawOLS={r['raw_ols_roi']:6.2f} | AvgROI={r['avg_roi']:4.2f} [{r['avg_roi_ci'][0]:.2f},{r['avg_roi_ci'][1]:.2f}] | 2025 mROAS={r['mroas_2025']:4.2f} [{r['mroas_2025_ci'][0]:.2f},{r['mroas_2025_ci'][1]:.2f}] | Q1 Rec=₹{r['rec_q1_cr']:.2f} Cr ({r['rec_chg_pct']:+5.1f}%)")
    print(f"\nQ1 2026 Expected Rev (λ=0.75): ₹{summary['q1_optimization']['rec_q1_rev_cr']:.2f} Cr [90% CI: {summary['q1_optimization']['rec_q1_ci'][0]:.2f}–{summary['q1_optimization']['rec_q1_ci'][1]:.2f}]")
    print(f"Q1 + Carryover Lift: +₹{summary['q1_optimization']['rec_tot_lift_cr']:.2f} Cr [90% CI: +{summary['q1_optimization']['rec_tot_lift_ci'][0]:.2f} to +{summary['q1_optimization']['rec_tot_lift_ci'][1]:.2f}] (P>0 = {summary['q1_optimization']['rec_prob_pos']:.1f}%)")
    print("Charts & JSON saved to report_assets/")


if __name__ == "__main__":
    main()
