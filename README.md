# Verdant Leaf — Marketing Mix Modeling (MMM) & Q1 2026 Budget Optimization

Refinement of Verdant Leaf's agency baseline Marketing Mix Model (`baseline_mmm.py`), fixing raw data export bugs, modeling adstock carryover and saturation on a multiplicative scale, validating against out-of-sample folds and a Meta geo-holdout experiment, and optimizing the **₹16 Cr Q1 2026** media budget.

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `baseline_mmm.py` | Original v0 agency OLS model (for diagnostic comparison). |
| `data_cleaning.py` | Audits `mmm_weekly_data.csv`, fixes 4 export bugs, and exports cleaned CSVs + Figure 1. |
| `validation_and_lift.py` | Loads `lift_test_results.csv` and runs ablation benchmarks, 3-fold expanding OOS CV, and 300-draw bootstrap. |
| `refined_mmm.py` | **Main pipeline:** Stage-1 mediator orthogonalization, calibrated Log-Revenue Adstock-Saturation Ridge MMM, Q1 2026 budget optimization, charts, and JSON summary. |
| `mmm_weekly_data.csv` | Raw weekly dataset export (`157` rows, Jan 2023 – Dec 2025). |
| `cleaned_mmm_weekly_data.csv` | Cleaned weekly dataset (`156` unique weeks, same 19-column schema). |
| `data_cleaning_log.csv` | Row-by-row audit changelog documenting every data fix applied. |
| `lift_test_results.csv` | Meta 6-week 20% geo-holdout experiment result (`iROAS = 1.39`, 90% CI `[0.94, 1.84]`). |

---

## Setup & Installation (Python 3.11 Recommended)

> **Note for Windows users:** If your default `python` command points to Python 3.14+, target **Python 3.11** (`py -3.11` or `uv --python 3.11`) so pre-built binary wheels for `statsmodels`, `scipy`, and `scikit-learn` install cleanly.

### Option 1: Using a Virtual Environment (`pip`)
```powershell
# 1. Create and activate a Python 3.11 virtual environment
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install required dependencies
pip install pandas numpy statsmodels scikit-learn scipy matplotlib
```

### Option 2: Using `uv` (No manual venv required)
You can run any script directly with `uv run --python 3.11 --with ...` as shown below.

---

## How to Run

### 1. Run the Main Refined Model & Budget Optimizer (Full Pipeline)
Runs data cleaning, benchmarks, model fitting, 3-fold CV, 300-draw bootstrap, Q1 2026 optimization, and saves charts/JSON into `report_assets/`:

```powershell
# With activated venv:
python refined_mmm.py

# Or directly via uv:
uv run --python 3.11 --with pandas --with numpy --with statsmodels --with scikit-learn --with scipy --with matplotlib python refined_mmm.py
```

### 2. Run Individual Modules (Optional)
```powershell
# A) Run the original broken agency baseline (v0):
uv run --python 3.11 --with pandas --with statsmodels python baseline_mmm.py

# B) Run only the data audit & cleaning step (creates cleaned_mmm_weekly_data.csv & data_cleaning_log.csv):
uv run --python 3.11 --with pandas --with numpy --with matplotlib python data_cleaning.py

# C) Inspect lift-test calibration targets & baseline ablation benchmarks:
uv run --python 3.11 --with pandas --with numpy --with statsmodels --with scikit-learn python validation_and_lift.py
```

---

## Generated Outputs (`refined_mmm.py`)
Running `refined_mmm.py` creates/updates:
- `cleaned_mmm_weekly_data.csv` & `data_cleaning_log.csv`
- `report_assets/cleaned_mmm_weekly_data.csv` (includes trend & seasonal harmonic features)
- `report_assets/results_summary.json` (model metrics, control elasticities, channel ROIs/mROAS with 90% CIs, Q1 2026 frontier)
- `report_assets/fig1_data_bugs_screenshot.png` — Raw data bug snippets & CPM sanity checks
- `report_assets/fig2_baseline_vs_refined_roi.png` — v0 vs. Refined channel ROI/mROAS & 156-week fit
- `report_assets/fig3_saturation_and_adstock.png` — 2025 marginal ROAS curves & adstock carryover decay
- `report_assets/fig4_budget_optimization.png` — Q1 2026 budget allocation & risk-return frontier

---

## Key Summary Results
* **Data Fixes (`157` $\rightarrow$ `156` rows):** Removed duplicate `2025-03-10`, scaled 13 Q4 2023 Meta rows from Lakhs to INR (`×100,000`), imputed 4 missing YouTube spend rows via median CPM (`₹90.54`), and clipped `-₹12.5L` TV credit note to `₹0`.
* **Model Performance:** Full-sample `R² = 0.957` (`DW = 2.32`), 3-fold expanding OOS mean `R² = 0.787` (Fold 3 `R² = 0.874`, mean MAPE `3.40%`), predicting Meta geo-holdout iROAS = **`1.47`** uncalibrated and **`1.39`** calibrated.
* **Q1 2026 Budget Plan (`₹16.00 Cr`, 50–150% constraints, $\lambda = 0.75$ risk-adjusted):**
  * **Meta:** `₹5.05 Cr` (`+2.5%`) | **Amazon Ads:** `₹4.11 Cr` (`-1.1%`) | **Google Search:** `₹2.63 Cr` (`+13.8%`)
  * **Influencer:** `₹1.79 Cr` (`+50.0%`) | **TV:** `₹1.23 Cr` (`-50.0%`, 2-week burst) | **YouTube:** `₹1.20 Cr` (`+24.5%`)
  * Expected Q1 Revenue: **`₹80.36 Cr`** (`90% CI: ₹78.61–82.21 Cr`), **`+₹1.71 Cr`** incremental lift incl. Q2 carryover (`100%` positive rate across 300 bootstrap draws).
