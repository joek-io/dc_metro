
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon, shapiro, friedmanchisquare
import statsmodels.formula.api as smf
import os

IN = "data_clean/station_monthly_recovery.csv"
OUT_DIR = "results/tables"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN)

# Weekend vs Weekday (station-level means, 2023+)
df_curr = df[df["Year"]>=2023].copy()
pivot = df_curr.pivot_table(index="Station", columns="Day_Type", values="Recovery_Ratio", aggfunc="mean").dropna()

results = {}

if {"Weekend","Weekday"}.issubset(pivot.columns):
    weekend = pivot["Weekend"]
    weekday = pivot["Weekday"]
    delta = weekend - weekday

    # Normality
    if len(delta) >= 3:
        W, p_normal = shapiro(delta)
    else:
        p_normal = 0  # force nonparam for tiny sample

    if p_normal > 0.05:
        stat, p_val = ttest_rel(weekend, weekday, nan_policy="omit")
        test_used = "Paired t-test"
        # Cohen's d (paired) = mean(diff)/std(diff)
        d = np.nanmean(delta) / np.nanstd(delta, ddof=1) if np.nanstd(delta, ddof=1)>0 else np.nan
    else:
        stat, p_val = wilcoxon(weekend, weekday, zero_method="wilcox", correction=False)
        test_used = "Wilcoxon signed-rank"
        d = np.nan  # not directly computed here

    results["weekend_vs_weekday"] = {
        "test": test_used,
        "stations_n": int(pivot.shape[0]),
        "weekend_mean": float(np.nanmean(weekend)),
        "weekday_mean": float(np.nanmean(weekday)),
        "mean_diff": float(np.nanmean(delta)),
        "p_value": float(p_val),
        "cohens_d": None if np.isnan(d) else float(d)
    }
else:
    results["weekend_vs_weekday"] = {"error": "Missing Weekend/Weekday columns after pivot."}

# Time-of-day differences (Weekday only), Friedman nonparametric
tod = (df_curr[df_curr["Day_Type"]=="Weekday"]
       .pivot_table(index="Station", columns="Time_Period", values="Recovery_Ratio", aggfunc="mean")
       .dropna(axis=0, how="any"))
if tod.shape[0] >= 3 and tod.shape[1] >= 3:
    try:
        stat, p_friedman = friedmanchisquare(*[tod[c] for c in tod.columns])
        results["time_of_day"] = {"test": "Friedman", "stations_n": int(tod.shape[0]), "p_value": float(p_friedman)}
    except Exception as e:
        results["time_of_day"] = {"error": str(e)}
else:
    results["time_of_day"] = {"error": "Insufficient stations or periods for Friedman test."}

# Mixed-effects model on log ratio
df_curr = df_curr.copy()
df_curr["Log_Ratio"] = np.where(df_curr["Recovery_Ratio"]>0, np.log(df_curr["Recovery_Ratio"]), np.nan)
# Encode Day_Type and Time_Period as categorical
df_curr["Day_Type"] = df_curr["Day_Type"].astype("category")
df_curr["Time_Period"] = df_curr["Time_Period"].astype("category")

# Simple mixed model: random intercept for Station
try:
    model = smf.mixedlm("Log_Ratio ~ Day_Type + Time_Period + Year", df_curr, groups=df_curr["Station"], missing="drop")
    fit = model.fit()
    with open(os.path.join(OUT_DIR, "mixedlm_summary.txt"), "w") as f:
        f.write(str(fit.summary()))
    results["mixedlm"] = {"converged": bool(fit.converged)}
except Exception as e:
    results["mixedlm"] = {"error": str(e)}

# Save JSON summary
import json
with open(os.path.join(OUT_DIR, "stats_summary.json"), "w") as f:
    json.dump(results, f, indent=2)

print("✅ Inference finished. See results/tables/ for outputs.")
