import os, json, math
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, shapiro, friedmanchisquare
import statsmodels.formula.api as smf

IN = "data_clean/station_monthly_recovery.csv"
OUT_DIR = "results/tables"
os.makedirs(OUT_DIR, exist_ok=True)

def write_summary(payload):
    with open(os.path.join(OUT_DIR, "stats_summary.json"), "w") as f:
        json.dump(payload, f, indent=2)

# Load
if not os.path.exists(IN):
    write_summary({"status": "error", "error": f"missing file: {IN}"})
    raise SystemExit(0)

df = pd.read_csv(IN)

required_base = {"Station","Year","Time_Period","Day_Type","Entries"}
missing_base = required_base - set(df.columns)
if missing_base:
    write_summary({"status":"error","error":f"missing columns: {missing_base}"})
    raise SystemExit(0)

# Backfill Recovery_Ratio if missing 
if "Recovery_Ratio" not in df.columns:
    # Need Entries_2019 to compute; if missing, report and exit gracefully
    if "Entries_2019" not in df.columns:
        write_summary({"status":"error","error":"recovery backfill failed: 'Entries_2019'"})
        raise SystemExit(0)
    e = pd.to_numeric(df.get("Entries"), errors="coerce")
    b = pd.to_numeric(df.get("Entries_2019"), errors="coerce")
    rr = np.where((b.notna()) & (b != 0), e / b, np.nan)
    df["Recovery_Ratio"] = rr

# Coerce/clean types
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Recovery_Ratio"] = pd.to_numeric(df["Recovery_Ratio"], errors="coerce")

# Normalize Day_Type for pivots: WEEKDAY/WEEKEND -> Weekday/Weekend
def norm_daytype(x):
    if isinstance(x,str):
        u = x.strip().upper()
        if u == "WEEKDAY": return "Weekday"
        if u == "WEEKEND": return "Weekend"
        if u in {"SATURDAY","SUNDAY"}: return "Weekend"
        if u in {"MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY"}: return "Weekday"
    return np.nan

df["Day_Type"] = df["Day_Type"].map(norm_daytype)

# Keep 2023+ for “recovery” period tests
df_curr = df[df["Year"] >= 2023].copy()

results = {}

#  Weekend vs Weekday (paired across stations)
try:
    pivot = df_curr.pivot_table(
        index="Station",
        columns="Day_Type",
        values="Recovery_Ratio",
        aggfunc="mean"
    ).dropna(subset=["Weekday","Weekend"], how="any")

    if {"Weekend","Weekday"}.issubset(pivot.columns) and len(pivot) >= 3:
        weekend = pivot["Weekend"]
        weekday = pivot["Weekday"]
        delta = weekend - weekday

        # Normality on paired differences
        p_normal = np.nan
        if len(delta.dropna()) >= 3:
            try:
                _, p_normal = shapiro(delta.dropna())
            except Exception:
                p_normal = np.nan

        if not np.isnan(p_normal) and p_normal > 0.05:
            stat, p_val = ttest_rel(weekend, weekday, nan_policy="omit")
            test_used = "Paired t-test"
            sd = np.nanstd(delta, ddof=1)
            d = (np.nanmean(delta) / sd) if sd > 0 else np.nan
        else:
            # Wilcoxon
            stat, p_val = wilcoxon(weekend, weekday, zero_method="wilcox", correction=False)
            test_used = "Wilcoxon signed-rank"
            d = np.nan

        results["weekend_vs_weekday"] = {
            "test": test_used,
            "stations_n": int(pivot.shape[0]),
            "weekend_mean": float(np.nanmean(weekend)),
            "weekday_mean": float(np.nanmean(weekday)),
            "mean_diff": float(np.nanmean(delta)),
            "p_value": float(p_val) if (p_val is not None and not np.isnan(p_val)) else None,
            "cohens_d": None if (np.isnan(d) or d is None) else float(d)
        }
    else:
        results["weekend_vs_weekday"] = {"p_value": None, "note": "insufficient paired stations"}
except Exception as e:
    results["weekend_vs_weekday"] = {"p_value": None, "error": str(e)}

# Time-of-day differences (Friedman, Weekday only)
try:
    tod = (
        df_curr[df_curr["Day_Type"] == "Weekday"]
        .pivot_table(index="Station", columns="Time_Period", values="Recovery_Ratio", aggfunc="mean")
        .reindex(columns=["AM","MIDDAY","PM","EVENING"])
        .dropna(axis=0, how="any")
    )
    if tod.shape[0] >= 3 and tod.shape[1] >= 3:
        stat, p_friedman = friedmanchisquare(*[tod[c] for c in tod.columns])
        results["time_of_day"] = {
            "test": "Friedman",
            "stations_n": int(tod.shape[0]),
            "periods": [c for c in tod.columns],
            "p_value": float(p_friedman) if not np.isnan(p_friedman) else None
        }
    else:
        results["time_of_day"] = {"p_value": None, "note": "insufficient stations/periods"}
except Exception as e:
    results["time_of_day"] = {"p_value": None, "error": str(e)}

# Mixed-effects model on log ratio
try:
    m = df_curr.copy()
    m["Log_Ratio"] = np.where(m["Recovery_Ratio"] > 0, np.log(m["Recovery_Ratio"]), np.nan)
    m = m.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Log_Ratio", "Station", "Year", "Time_Period", "Day_Type"]
    )

    # Center Year for interpretability
    year_center = float(np.nanmedian(m["Year"]))
    m["Year_centered"] = m["Year"] - year_center

    # Categorical encodings
    m["Day_Type"] = m["Day_Type"].astype("category")        # Weekday/Weekend
    m["Time_Period"] = m["Time_Period"].astype("category")  # AM/MIDDAY/PM/EVENING

    if len(m) >= 100:  # minimal size guard
        # Weekend × Year_centered
        # Time_Period × Year_centered
        formula = "Log_Ratio ~ Day_Type * Time_Period + Day_Type * Year_centered + Time_Period * Year_centered"
        model = smf.mixedlm(formula, m, groups=m["Station"], missing="drop")
        fit = model.fit()

        # Write summary for review
        with open(os.path.join(OUT_DIR, "mixedlm_summary.txt"), "w") as f:
            f.write(str(fit.summary()))

        # Collect key results
        params = getattr(fit, "params", pd.Series(dtype=float))

        # Pull slopes of interest
        weekend_slope = None
        tod_slopes = {}
        for k, v in params.items():
            if "Day_Type[T.Weekend]:Year_centered" in k:
                weekend_slope = float(v)
            elif "Time_Period" in k and "Year_centered" in k:
                tod_slopes[k] = float(v)

        results["mixedlm"] = {
            "formula": formula,
            "converged": bool(getattr(fit, "converged", True)),
            "n_obs": int(getattr(fit, "nobs", len(m))),
            "fixed_effects": {k: float(v) for k, v in params.items()},
            "year_center_reference": year_center,
            "weekend_year_interaction": weekend_slope,
            "time_period_year_interactions": tod_slopes
        }
    else:
        results["mixedlm"] = {"note": "too few observations for mixed model"}
except Exception as e:
    results["mixedlm"] = {"error": str(e)}

# Final status 
results["status"] = "ok" if ("weekend_vs_weekday" in results and "time_of_day" in results) else "warning"
write_summary(results)
print("Inference finished; see results/tables/")