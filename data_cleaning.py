"""
data_cleaning.py — Raw CSV Audit, Bug Detection, Cleaning & Feature Engineering

Handles `mmm_weekly_data.csv`:
1. Detects and removes exact duplicate week row (`2025-03-10`).
2. Fixes Q4 2023 Meta spend unit-scale error (reported in Lakhs -> multiplied by 100,000).
3. Imputes 4 missing `youtube_spend` rows via median historical YouTube CPM.
4. Clips negative `tv_spend` accounting credit (`-₹12.5L` on `2024-11-11`) to `₹0`.
5. Adds time trend and annual/semi-annual harmonic seasonality columns.
6. Generates `report_assets/cleaned_mmm_weekly_data.csv` and `fig1_data_bugs_screenshot.png`.

Run standalone:
    uv run --python 3.11 --with pandas --with numpy --with matplotlib python data_cleaning.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ASSETS_DIR = Path("report_assets")
ASSETS_DIR.mkdir(exist_ok=True)

CHANNELS = [
    "meta_spend",
    "google_search_spend",
    "youtube_spend",
    "influencer_spend",
    "tv_spend",
    "amazon_ads_spend",
]

CH_LABELS = {
    "meta_spend": "Meta",
    "google_search_spend": "Google Search",
    "youtube_spend": "YouTube",
    "influencer_spend": "Influencer",
    "tv_spend": "TV",
    "amazon_ads_spend": "Amazon Ads",
}

CONTROLS = [
    "t", "sin1", "cos1", "sin2", "cos2",
    "avg_discount_pct", "price_index",
    "festive_flag", "marketplace_sale_flag",
    "competitor_promo_flag", "new_launch_flag",
]


def audit_and_clean_data(raw_path: str = "mmm_weekly_data.csv", save_clean: bool = True):
    """Audit raw weekly export for data issues and return (df_raw, df_clean, audit_dict)."""
    df_raw = pd.read_csv(raw_path)
    audit = {"raw_rows": len(df_raw)}

    # Capture raw bug evidence before fixing
    dup_mask = df_raw.duplicated(subset=["week_start"], keep=False)
    audit["duplicate_rows"] = df_raw.loc[dup_mask, ["week_start", "revenue_inr", "meta_spend", "tv_spend"]].copy()
    audit["duplicate_indices"] = df_raw.index[dup_mask].tolist()

    raw_meta_cpm = df_raw["meta_spend"] / df_raw["meta_impressions"] * 1000.0
    lakhs_mask = raw_meta_cpm < 1.0
    audit["meta_lakhs_rows"] = df_raw.loc[lakhs_mask, ["week_start", "meta_spend", "meta_impressions"]].copy()
    audit["meta_lakhs_rows"]["raw_cpm"] = raw_meta_cpm[lakhs_mask]

    yt_na_mask = df_raw["youtube_spend"].isna()
    audit["yt_na_rows"] = df_raw.loc[yt_na_mask, ["week_start", "youtube_spend", "youtube_impressions"]].copy()

    tv_neg_mask = df_raw["tv_spend"] < 0
    audit["tv_neg_rows"] = df_raw.loc[tv_neg_mask, ["week_start", "tv_spend", "festive_flag", "revenue_inr"]].copy()

    # Apply deterministic fixes
    df = df_raw.drop_duplicates(subset=["week_start"], keep="first").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["week_start"])

    # Fix Meta Q4 2023 Lakhs -> INR
    meta_cpm = df["meta_spend"] / df["meta_impressions"] * 1000.0
    fix_meta = meta_cpm < 1.0
    df.loc[fix_meta, "meta_spend"] = df.loc[fix_meta, "meta_spend"] * 100000.0

    # Impute YouTube missing spend via median historical CPM
    yt_valid = df["youtube_spend"].notna() & (df["youtube_impressions"] > 0)
    yt_median_cpm = (df.loc[yt_valid, "youtube_spend"] / df.loc[yt_valid, "youtube_impressions"] * 1000.0).median()
    audit["yt_median_cpm"] = float(yt_median_cpm)
    fix_yt = df["youtube_spend"].isna()
    df.loc[fix_yt, "youtube_spend"] = df.loc[fix_yt, "youtube_impressions"] / 1000.0 * yt_median_cpm

    # Clip negative TV accounting rebate to 0 for media exposure
    df.loc[df["tv_spend"] < 0, "tv_spend"] = 0.0

    # Time & seasonal harmonic features
    df["t"] = np.arange(len(df))
    woy = df["dt"].dt.isocalendar().week.astype(int)
    df["sin1"] = np.sin(2.0 * np.pi * woy / 52.0)
    df["cos1"] = np.cos(2.0 * np.pi * woy / 52.0)
    df["sin2"] = np.sin(4.0 * np.pi * woy / 52.0)
    df["cos2"] = np.cos(4.0 * np.pi * woy / 52.0)

    audit["clean_rows"] = len(df)
    if save_clean:
        # 1. Save modeling dataset with engineered features into report_assets/
        df.to_csv(ASSETS_DIR / "cleaned_mmm_weekly_data.csv", index=False)

        # 2. Save clean 19-column CSV in workspace root matching raw schema 1-to-1
        orig_cols = list(df_raw.columns)
        df_root = df[orig_cols].copy()
        df_root["meta_spend"] = df_root["meta_spend"].round(2)
        df_root["youtube_spend"] = df_root["youtube_spend"].round(2)
        df_root.to_csv("cleaned_mmm_weekly_data.csv", index=False)

        # 3. Save a clear row-by-row changelog CSV explaining every fix made
        log_rows = []
        dup_indices = audit["duplicate_indices"]
        if len(dup_indices) > 1:
            for d_idx in dup_indices[1:]:
                log_rows.append({
                    "raw_row_idx": int(d_idx),
                    "week_start": str(df_raw.loc[d_idx, "week_start"]),
                    "bug_type": "Duplicate Week Row",
                    "column_changed": "ALL (row dropped)",
                    "raw_value": f"Duplicate of row {dup_indices[0]}",
                    "cleaned_value": "Removed",
                    "fix_explanation": "Exact duplicate of 2025-03-10; dropped second occurrence (157 -> 156 rows).",
                })
        for r_idx, row in audit["meta_lakhs_rows"].iterrows():
            old_v = float(row["meta_spend"])
            new_v = round(old_v * 100000.0, 2)
            log_rows.append({
                "raw_row_idx": int(r_idx),
                "week_start": str(row["week_start"]),
                "bug_type": "Unit Scale Error (Lakhs -> INR)",
                "column_changed": "meta_spend",
                "raw_value": old_v,
                "cleaned_value": new_v,
                "fix_explanation": f"Raw CPM was ₹{row['raw_cpm']:.4f} vs ~₹139 normal; multiplied by 100,000.",
            })
        for r_idx, row in audit["yt_na_rows"].iterrows():
            imp = float(row["youtube_impressions"])
            new_v = round(imp / 1000.0 * yt_median_cpm, 2)
            log_rows.append({
                "raw_row_idx": int(r_idx),
                "week_start": str(row["week_start"]),
                "bug_type": "Missing Spend (NaN)",
                "column_changed": "youtube_spend",
                "raw_value": "NaN",
                "cleaned_value": new_v,
                "fix_explanation": f"Imputed from {int(imp):,} impressions using median YT CPM (₹{yt_median_cpm:.2f}).",
            })
        for r_idx, row in audit["tv_neg_rows"].iterrows():
            log_rows.append({
                "raw_row_idx": int(r_idx),
                "week_start": str(row["week_start"]),
                "bug_type": "Negative Accounting Rebate",
                "column_changed": "tv_spend",
                "raw_value": float(row["tv_spend"]),
                "cleaned_value": 0.0,
                "fix_explanation": "Post-Diwali agency credit note; clipped to ₹0 for media exposure modeling.",
            })
        pd.DataFrame(log_rows).to_csv("data_cleaning_log.csv", index=False)
    return df_raw, df, audit


def make_bug_screenshot(df_raw: pd.DataFrame, df_clean: pd.DataFrame, audit: dict):
    """Render visual screenshot of raw export bugs and before/after validation."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.8,
        "grid.color": "#E2E8F0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
    })

    fig = plt.figure(figsize=(11.5, 6.2), dpi=240, facecolor="#FFFFFF")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.18, 1.0], hspace=0.35, wspace=0.18,
                          left=0.04, right=0.97, top=0.89, bottom=0.08)

    fig.suptitle("RAW DATA AUDIT SCREENSHOTS: 4 Export Bugs Caught & Fixed (mmm_weekly_data.csv)",
                 fontsize=13, fontweight="bold", color="#0F172A", y=0.96)

    # Panel A: Raw CSV Snippets
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#0F172A")
    ax1.set_xticks([])
    ax1.set_yticks([])
    for sp in ax1.spines.values():
        sp.set_edgecolor("#334155")
    ax1.set_title("1. Raw CSV Snippets (Highlighted Anomalies)", loc="left", fontsize=10, fontweight="bold", color="#1E293B", pad=6)

    lines = [
        ("Row  week_start   meta_spend   meta_impr   tv_spend   yt_spend", "#94A3B8", False),
        ("-----------------------------------------------------------------", "#475569", False),
        (" 38  2023-09-25  2,656,591.0  17,986,575          0    992,286", "#E2E8F0", False),
        (" 39  2023-10-02        27.23  22,494,164          0          0  [BUG 2: Lakhs]", "#FCA5A5", True),
        (" 46  2023-11-20        16.59  12,068,269          0    781,967  [CPM ₹0.0014]", "#FCA5A5", True),
        (" 51  2023-12-25        33.94  25,725,012          0    792,202  [13 wks x1e-5]", "#FCA5A5", True),
        (" 58  2024-02-12  3,709,325.0  27,992,431          0        NaN  [BUG 3: NaN]", "#FCD34D", True),
        (" 97  2024-11-11  4,432,683.0  31,991,203  -1,250,000  1,055,103 [BUG 4: Neg]", "#FCA5A5", True),
        ("114  2025-03-10  3,658,451.0  28,636,923          0    625,949", "#E2E8F0", False),
        ("115  2025-03-10  3,658,451.0  28,636,923          0    625,949  [BUG 1: Dup!]", "#F87171", True),
    ]
    y_pos = 0.90
    for text, color, hl in lines:
        if hl:
            ax1.add_patch(patches.Rectangle((0.015, y_pos - 0.03), 0.97, 0.082,
                                            transform=ax1.transAxes, facecolor="#7F1D1D" if color != "#FCD34D" else "#78350F",
                                            alpha=0.55, edgecolor="none"))
        ax1.text(0.025, y_pos, text, transform=ax1.transAxes, fontsize=6.8,
                 fontfamily="monospace", color=color, va="center", fontweight="bold" if hl else "normal")
        y_pos -= 0.092

    # Panel B: Resolution Table
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    ax2.set_title("2. How Each Bug Was Verified & Resolved", loc="left", fontsize=10, fontweight="bold", color="#1E293B", pad=6)

    table_data = [
        ["Bug / Scope", "Symptom in Raw Export", "Root Cause & Fix"],
        ["1. Duplicate Row\n(1 wk: 2025-03-10)", "157 rows instead of 156;\nrows 114 & 115 identical", "Drop duplicate date -> 156\nunique weeks (2023–2025)"],
        ["2. Meta Unit Scale\n(13 wks: Q4 2023)", "Spend ₹16.6–42.4 for\n12–30M impr (CPM ₹0.001)", "Exported in ₹ Lakhs! ×1e5\n-> CPM restored (₹121–156)"],
        ["3. Missing YT Spend\n(4 wks: NaN spend)", "youtube_spend = NaN\ndespite 7.8–12.4M impr", "Impute via median YT CPM\n(₹90.54) instead of dropna"],
        ["4. Negative TV Spend\n(1 wk: 2024-11-11)", "tv_spend = -₹12,50,000\nright after Diwali flight", "Agency rebate credit; clip\nto ₹0 for media exposure"],
    ]
    tbl = ax2.table(cellText=table_data, cellLoc="left", loc="center", bbox=[0.0, 0.02, 1.0, 0.96])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.2)
    col_widths = [0.27, 0.35, 0.38]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_width(col_widths[c])
        cell.set_edgecolor("#CBD5E1")
        if r == 0:
            cell.set_facecolor("#1B4332")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F8FAFC" if r % 2 == 1 else "#FFFFFF")
            cell.set_text_props(color="#0F172A")

    # Panel C: Before vs After Meta Spend Q4 2023
    ax3 = fig.add_subplot(gs[1, 0])
    w_raw = df_raw.drop_duplicates(subset=["week_start"]).reset_index(drop=True)
    idx_slice = np.arange(32, 60)
    x_dates = pd.to_datetime(w_raw.loc[idx_slice, "week_start"])
    ax3.plot(x_dates, w_raw.loc[idx_slice, "meta_spend"] / 1e5, color="#DC2626", lw=2.0, ls="--", marker="o", ms=3.5, label="Raw Export (Collapsed to ~₹0)")
    ax3.plot(x_dates, df_clean.loc[idx_slice, "meta_spend"] / 1e5, color="#1B4332", lw=2.2, marker="s", ms=3.5, label="Fixed (×100,000 Lakhs → INR)")
    ax3.axvspan(pd.to_datetime("2023-10-02"), pd.to_datetime("2023-12-25"), color="#FEE2E2", alpha=0.45, label="Q4 2023 Bug Window (13 wks)")
    ax3.set_title("3. Visual Proof: Meta Spend Q4 2023 Before vs. After Fix", loc="left", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax3.set_ylabel("Weekly Spend (₹ Lakhs)", fontsize=8.5, color="#334155")
    ax3.tick_params(axis="x", labelsize=7.5, rotation=20)
    ax3.tick_params(axis="y", labelsize=8)
    ax3.grid(True, alpha=0.5)
    ax3.legend(fontsize=7.5, loc="lower left", framealpha=0.95)

    # Panel D: Meta CPM Stability Check
    ax4 = fig.add_subplot(gs[1, 1])
    meta_cpm_raw = w_raw["meta_spend"] / w_raw["meta_impressions"] * 1000.0
    meta_cpm_fix = df_clean["meta_spend"] / df_clean["meta_impressions"] * 1000.0
    ax4.plot(pd.to_datetime(df_clean["week_start"]), meta_cpm_raw, color="#DC2626", alpha=0.65, lw=1.4, label="Raw Meta CPM (Drops to ₹0.0014)")
    ax4.plot(pd.to_datetime(df_clean["week_start"]), meta_cpm_fix, color="#2D6A4F", lw=1.6, label="Fixed Meta CPM (Stable ₹110–168)")
    ax4.axhline(meta_cpm_fix.median(), color="#0F172A", ls=":", lw=1.1, label=f"Median CPM = ₹{meta_cpm_fix.median():.1f}")
    ax4.set_title("4. Sanity Check: Meta CPM Stability Across All 156 Weeks", loc="left", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax4.set_ylabel("Effective CPM (₹ per 1,000 Impr)", fontsize=8.5, color="#334155")
    ax4.tick_params(axis="x", labelsize=7.5, rotation=20)
    ax4.tick_params(axis="y", labelsize=8)
    ax4.grid(True, alpha=0.5)
    ax4.legend(fontsize=7.5, loc="lower right", framealpha=0.95)

    fig.savefig(ASSETS_DIR / "fig1_data_bugs_screenshot.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    df_raw, df_clean, audit = audit_and_clean_data()
    make_bug_screenshot(df_raw, df_clean, audit)
    print(f"Data audit complete: {audit['raw_rows']} raw rows -> {audit['clean_rows']} clean rows.")
    print(f"  - Duplicate row fixed: {audit['duplicate_rows']['week_start'].unique().tolist()}")
    print(f"  - Meta Lakhs->INR rows fixed: {len(audit['meta_lakhs_rows'])}")
    print(f"  - YouTube NaN rows imputed (median CPM ₹{audit['yt_median_cpm']:.2f}): {len(audit['yt_na_rows'])}")
    print(f"  - Negative TV spend rows clipped: {len(audit['tv_neg_rows'])}")
    print("Saved cleaned_mmm_weekly_data.csv, data_cleaning_log.csv, and report_assets/fig1_data_bugs_screenshot.png")
