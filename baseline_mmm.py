"""
baseline_mmm.py  —  "v0" marketing mix model currently used by the growth team.

Written quickly last year by an agency analyst. Leadership has started questioning
its outputs (e.g. why does TV look useless? is Google Search really our best channel?).
Your job is to refine it — see ASSIGNMENT.md.

Run:  python baseline_mmm.py
"""
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("mmm_weekly_data.csv")
df = df.dropna()

channels = ["meta_spend", "google_search_spend", "youtube_spend",
            "influencer_spend", "tv_spend", "amazon_ads_spend"]
controls = ["festive_flag", "avg_discount_pct"]

X = sm.add_constant(df[channels + controls])
y = df["revenue_inr"]
model = sm.OLS(y, X).fit()
print(model.summary())

print("\nChannel ROI (coefficient = revenue per ₹1 spent):")
roi = model.params[channels].sort_values(ascending=False)
for ch, v in roi.items():
    print(f"  {ch:<22} {v:6.2f}")

print("\nIn-sample R²: %.3f" % model.rsquared)
print("Recommendation: shift budget from lowest-ROI to highest-ROI channels.")
