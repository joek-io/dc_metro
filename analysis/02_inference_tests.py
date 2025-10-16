import os
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, shapiro, friedmanchisquare
import statsmodels.formula.api as smf

IN = "data_clean/station_monthly_recovery.csv"
OUT_DIR = "results/tables"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN)

# FILTERS
df = df.copy()
# keep only valid ratios with a 2019 denominator present and positive
df = df[(~df["Denom_Flag"]) & np.isfinite(df["Recovery_Ratio"])].copy()

# winsorize/clamp ratios to tame outliers from tiny baselines
LOW, HIGH = 0.05, 5.0
df["Recovery_Ratio_Clamped"] = df["Recovery_Ratio"].clip(lower=LOW, upper=HIGH)

# unify labels expected by tests
df["Day_Type"] = df["Day_Type"].astype(str).str.title()          # 'Weekday'/'Weekend'
df["Time_Period"] = df["Time_Period"].astype(str).str.upper()     # AM/MIDDAY/PM/EVENING

# require enough data per station (≥ 6 months after 2022) for stability
df_curr = df[df["Year"] >= 2023].copy()
counts_by_station = df_curr.groupby("Station")["Recovery_Ratio_Clamped"].count()
eligible = counts_by_station[counts_by_station >= 6].index
df_curr = df_curr[df_curr["Station"].isin(eligible)].copy()

results = {}

# WEEKEND vs WEEKDAY
pivot = (
    df_curr.pivot_table(
        index="Station",
        columns="Day_Type",
        values="Recovery_Ratio_Clamped",
        aggfunc="mean"
    )
    .dropna(subset=["Weekend", "Weekday"], how="any")
)

if {"Weekend", "Weekday"}.issubset(pivot.columns) and len(pivot) >= 3:
    weekend = pivot["Weekend"]
    weekday = pivot["Weekday"]
    delta = weekend - weekday

    # normality of paired deltas
    p_normal = shapiro(delta)[1] if len(delta) >= 3 else 0.0
    if p_normal > 0.05:
        stat, p_val = ttest_rel(weekend, weekday, nan_policy="omit")
        test_used = "Paired t-test"
        sd = np.nanstd(delta, ddof=1)
        d = (np.nanmean(delta) / sd) if sd > 0 else np.nan
    else:
        stat, p_val = wilcoxon(weekend, weekday, zero_method="wilcox", correction=False)
        test_used = "Wilcoxon signed-rank"
        d = np.nan  # not defined for Wilcoxon directly

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
    results["weekend_vs_weekday"] = {"error": "Insufficient paired stations after filtering."}

# TIME-OF-DAY (Friedman over AM/MIDDAY/PM/EVENING, Weekday only) 
tod = (
    df_curr[df_curr["Day_Type"] == "Weekday"]
    .pivot_table(index="Station", columns="Time_Period", values="Recovery_Ratio_Clamped", aggfunc="mean")
    .reindex(columns=["AM", "MIDDAY", "PM", "EVENING"])
    .dropna(how="any")
)

if tod.shape[0] >= 3 and tod.shape[1] == 4:
    try:
        stat, p_friedman = friedmanchisquare(tod["AM"], tod["MIDDAY"], tod["PM"], tod["EVENING"])
        results["time_of_day"] = {
            "test": "Friedman",
            "stations_n": int(tod.shape[0]),
            "periods": ["AM", "MIDDAY", "PM", "EVENING"],
            "p_value": float(p_friedman)
        }
    except Exception as e:
        results["time_of_day"] = {"error": str(e)}
else:
    results["time_of_day"] = {"error": "Insufficient stations or missing one or more periods."}

# MIXED EFFECTS on log scale
df_mx = df_curr.copy()
df_mx["Log_Ratio"] = np.where(df_mx["Recovery_Ratio_Clamped"] > 0, np.log(df_mx["Recovery_Ratio_Clamped"]), np.nan)
df_mx = df_mx.dropna(subset=["Log_Ratio"])

# categorical encodings
df_mx["Day_Type"] = df_mx["Day_Type"].astype("category")
df_mx["Time_Period"] = pd.Categorical(df_mx["Time_Period"], categories=["AM", "MIDDAY", "PM", "EVENING"], ordered=True)

try:
    model = smf.mixedlm("Log_Ratio ~ Day_Type + Time_Period + Year", df_mx, groups=df_mx["Station"], missing="drop")
    fit = model.fit()
    with open(os.path.join(OUT_DIR, "mixedlm_summary.txt"), "w") as f:
        f.write(str(fit.summary()))
    results["mixedlm"] = {"converged": bool(fit.converged), "n_obs": int(fit.nobs)}
except Exception as e:
    results["mixedlm"] = {"error": str(e)}

with open(os.path.join(OUT_DIR, "stats_summary.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Inference finished. See results/tables/ for outputs.")