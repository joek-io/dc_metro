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

results = {}

# choose ratio column
RATIO_COL = "Recovery_Ratio_Final" if "Recovery_Ratio_Final" in df.columns else (
    "Recovery_Ratio" if "Recovery_Ratio" in df.columns else None
)
if RATIO_COL is None:
    raise ValueError("No recovery ratio column found (expected 'Recovery_Ratio_Final' or 'Recovery_Ratio').")

# normalize labels
# Day_Type to upper, Time_Period already standardized by cleaner
df["Day_Type"] = df["Day_Type"].astype(str).str.upper()
df["Time_Period"] = df["Time_Period"].astype(str).str.upper()

# restrict to analysis window and valid ratios
df_curr = df[(df["Year"] >= 2023) & (df[RATIO_COL].notna())].copy()

# Weekend vs Weekday test
# station-level average recovery by day type
pivot = (
    df_curr.pivot_table(
        index="Station",
        columns="Day_Type",
        values=RATIO_COL,
        aggfunc="mean"
    )
    .rename(columns={"WEEKDAY": "Weekday", "WEEKEND": "Weekend"})
)

if {"Weekend", "Weekday"}.issubset(pivot.columns) and pivot.dropna(subset=["Weekend","Weekday"]).shape[0] >= 3:
    pivot_nonan = pivot.dropna(subset=["Weekend", "Weekday"])
    weekend = pivot_nonan["Weekend"]
    weekday = pivot_nonan["Weekday"]
    delta = weekend - weekday

    # normality of paired differences
    if len(delta) >= 3:
        try:
            W, p_normal = shapiro(delta)
        except Exception:
            p_normal = 0.0
    else:
        p_normal = 0.0

    if p_normal > 0.05:
        stat, p_val = ttest_rel(weekend, weekday, nan_policy="omit")
        test_used = "Paired t-test"
        sd = np.nanstd(delta, ddof=1)
        d = float(np.nanmean(delta) / sd) if sd > 0 else np.nan
    else:
        # Wilcoxon requires no zeros; fallback handles scipy behavior
        try:
            stat, p_val = wilcoxon(weekend, weekday, zero_method="wilcox", correction=False)
        except ValueError:
            # If too many zeros, try Pratt method (includes zeros)
            stat, p_val = wilcoxon(weekend, weekday, zero_method="pratt", correction=False)
        test_used = "Wilcoxon signed-rank"
        d = np.nan  # not defined for Wilcoxon here

    results["weekend_vs_weekday"] = {
        "test": test_used,
        "stations_n": int(pivot_nonan.shape[0]),
        "weekend_mean": float(np.nanmean(weekend)),
        "weekday_mean": float(np.nanmean(weekday)),
        "mean_diff": float(np.nanmean(delta)),
        "p_value": float(p_val),
        "cohens_d": None if np.isnan(d) else float(d)
    }
else:
    results["weekend_vs_weekday"] = {
        "error": "Missing Weekend/Weekday columns after pivot or too few paired stations.",
        "available_columns": list(pivot.columns.astype(str)),
        "stations_with_both": int(pivot.dropna(subset=["Weekend"], how="any").join(
            pivot.dropna(subset=["Weekday"], how="any"), how="inner", lsuffix="_Wknd", rsuffix="_Wkdy"
        ).shape[0]) if {"Weekend","Weekday"}.issubset(pivot.columns) else 0
    }

# Time-of-day differences (Weekday only) via Friedman
PERIODS_ORDER = ["AM", "MIDDAY", "PM", "EVENING"]

tod = (
    df_curr[df_curr["Day_Type"] == "WEEKDAY"]
    .pivot_table(index="Station", columns="Time_Period", values=RATIO_COL, aggfunc="mean")
)

# keep only the canonical periods, in fixed order
tod = tod.reindex(columns=PERIODS_ORDER)

# drop stations missing any period (Friedman needs complete blocks)
tod_complete = tod.dropna(axis=0, how="any")

if tod_complete.shape[0] >= 3 and tod_complete.shape[1] >= 3:
    try:
        arrays = [tod_complete[c] for c in tod_complete.columns]
        stat, p_friedman = friedmanchisquare(*arrays)
        results["time_of_day"] = {
            "test": "Friedman",
            "stations_n": int(tod_complete.shape[0]),
            "periods": list(tod_complete.columns),
            "p_value": float(p_friedman)
        }
    except Exception as e:
        results["time_of_day"] = {"error": str(e)}
else:
    results["time_of_day"] = {
        "error": "Insufficient stations or periods for Friedman test.",
        "stations_with_all_periods": int(tod_complete.shape[0]),
        "periods_present": [c for c in PERIODS_ORDER if c in tod.columns],
        "periods_required": PERIODS_ORDER
    }

# Mixed-effects model on log ratio
df_curr = df_curr.copy()
df_curr["Log_Ratio"] = np.where(df_curr[RATIO_COL] > 0, np.log(df_curr[RATIO_COL]), np.nan)
df_curr["Day_Type"] = df_curr["Day_Type"].astype("category")     # "WEEKDAY"/"WEEKEND"
df_curr["Time_Period"] = df_curr["Time_Period"].astype("category")  # "AM"/"MIDDAY"/"PM"/"EVENING"

try:
    model = smf.mixedlm("Log_Ratio ~ Day_Type + Time_Period + Year", df_curr, groups=df_curr["Station"], missing="drop")
    fit = model.fit()
    with open(os.path.join(OUT_DIR, "mixedlm_summary.txt"), "w") as f:
        f.write(str(fit.summary()))
    results["mixedlm"] = {
        "converged": bool(getattr(fit, "converged", False)),
        "n_obs": int(fit.nobs) if hasattr(fit, "nobs") else None
    }
except Exception as e:
    results["mixedlm"] = {"error": str(e)}

# Save JSON summary
with open(os.path.join(OUT_DIR, "stats_summary.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Inference script finished. See results/tables/ for outputs.")