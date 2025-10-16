import os, json, math
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, shapiro, friedmanchisquare
import statsmodels.formula.api as smf

IN = "data_clean/station_monthly_recovery.csv"
OUT_DIR = "results/tables"
os.makedirs(OUT_DIR, exist_ok=True)

def ensure_recovery_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Recovery_Ratio exists; if missing, compute vs 2019 or sensible fallbacks."""
    needed = {"Station","Year","Time_Period","Entries"}
    if not needed.issubset(df.columns):
        raise AssertionError(f"missing columns for backfill: {needed - set(df.columns)}")

    if "Recovery_Ratio" in df.columns and df["Recovery_Ratio"].notna().any():
        df["Recovery_Ratio"] = pd.to_numeric(df["Recovery_Ratio"], errors="coerce")
        return df

    # Primary baseline: 2019 mean by Station×Time_Period
    base19 = (
        df[df["Year"] == 2019]
        .groupby(["Station","Time_Period"], as_index=False)["Entries"]
        .mean()
        .rename(columns={"Entries": "Entries_2019"})
    )

    # Fallback A: mean across ALL years by Station×Time_Period
    base_any = (
        df.groupby(["Station","Time_Period"], as_index=False)["Entries"]
          .mean()
          .rename(columns={"Entries": "Entries_any"})
    )

    # Fallback B: period-wide mean (collapses stations)
    base_period = (
        df.groupby(["Time_Period"], as_index=False)["Entries"]
          .mean()
          .rename(columns={"Entries": "Entries_period"})
    )

    out = df.merge(base19, on=["Station","Time_Period"], how="left")
    out = out.merge(base_any, on=["Station","Time_Period"], how="left")
    out = out.merge(base_period, on=["Time_Period"], how="left")

    # If column missing for any reason, create it
    if "Entries_2019" not in out.columns:
        out["Entries_2019"] = np.nan

    # Compose denominator preference: 2019 -> any-year -> period mean
    denom = out["Entries_2019"]
    denom = denom.where(~denom.isna(), out["Entries_any"])
    denom = denom.where(~denom.isna(), out["Entries_period"])

    # Compute ratio with guards
    bad = denom.isna() | (denom <= 0)
    out["Recovery_Ratio"] = np.where(bad, np.nan, out["Entries"] / denom)

    return out

def write_results(results: dict):
    with open(os.path.join(OUT_DIR, "stats_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

# Load & normalize
results = {
    "weekend_vs_weekday": {"p_value": None},
    "time_of_day": {},
    "mixedlm": {}
}

try:
    df = pd.read_csv(IN)
except Exception as e:
    results["error"] = f"failed to read input: {e}"
    write_results(results)
    raise

core_needed = {"Station","Year","Time_Period","Day_Type","Entries"}
missing_core = core_needed - set(df.columns)
if missing_core:
    results["error"] = f"missing columns: {missing_core}"
    write_results(results)
    raise AssertionError(results["error"])

# Normalize text/numerics
df["Day_Type"] = df["Day_Type"].astype(str).str.upper()
df["Time_Period"] = df["Time_Period"].astype(str).str.upper()
df["Entries"] = pd.to_numeric(df["Entries"], errors="coerce").fillna(0)

# Backfill Recovery_Ratio if needed (tolerant)
try:
    df = ensure_recovery_ratio(df)
except Exception as e:
    results["error"] = f"recovery backfill failed: {e}"
    write_results(results)
    raise

# Restrict to 2023+ for inferential tests
df_curr = df[df["Year"] >= 2023].copy()
df_curr["Recovery_Ratio"] = pd.to_numeric(df_curr["Recovery_Ratio"], errors="coerce")

# Weekend vs Weekday
try:
    piv = (
        df_curr.pivot_table(index="Station", columns="Day_Type", values="Recovery_Ratio", aggfunc="mean")
          .rename(columns={"WEEKEND":"Weekend","WEEKDAY":"Weekday"})
          .dropna(subset=["Weekend","Weekday"], how="any")
    )
    if {"Weekend","Weekday"}.issubset(piv.columns) and len(piv) >= 3:
        weekend = piv["Weekend"]
        weekday = piv["Weekday"]
        delta = weekend - weekday
        # Normality of differences
        p_normal = shapiro(delta.dropna()).pvalue if len(delta.dropna()) >= 3 else 0.0
        if p_normal > 0.05:
            stat, p_val = ttest_rel(weekend, weekday, nan_policy="omit")
            test_used = "Paired t-test"
            sd = np.nanstd(delta, ddof=1)
            d = (np.nanmean(delta) / sd) if sd > 0 else np.nan
        else:
            stat, p_val = wilcoxon(weekend, weekday, zero_method="wilcox", correction=False)
            test_used = "Wilcoxon signed-rank"
            d = np.nan
        results["weekend_vs_weekday"] = {
            "test": test_used,
            "stations_n": int(piv.shape[0]),
            "weekend_mean": float(np.nanmean(weekend)),
            "weekday_mean": float(np.nanmean(weekday)),
            "mean_diff": float(np.nanmean(delta)),
            "p_value": float(p_val),
            "cohens_d": None if np.isnan(d) else float(d),
        }
    else:
        results["weekend_vs_weekday"] = {"p_value": None, "error": "insufficient paired stations"}
except Exception as e:
    results["weekend_vs_weekday"] = {"p_value": None, "error": str(e)}

# Time-of-day (Friedman, Weekday only)
try:
    tod = (
        df_curr[df_curr["Day_Type"] == "WEEKDAY"]
        .pivot_table(index="Station", columns="Time_Period", values="Recovery_Ratio", aggfunc="mean")
        .reindex(columns=["AM","MIDDAY","PM","EVENING"])
        .dropna(how="any")
    )
    if tod.shape[0] >= 3 and tod.shape[1] >= 3:
        stat, p_friedman = friedmanchisquare(*[tod[c] for c in tod.columns])
        results["time_of_day"] = {
            "test": "Friedman",
            "stations_n": int(tod.shape[0]),
            "periods": [c for c in tod.columns],
            "p_value": float(p_friedman),
        }
    else:
        results["time_of_day"] = {"error": "insufficient stations/periods for Friedman"}
except Exception as e:
    results["time_of_day"] = {"error": str(e)}

# Mixed-effects model
try:
    dfm = df_curr.copy()
    dfm["Log_Ratio"] = np.where(dfm["Recovery_Ratio"] > 0, np.log(dfm["Recovery_Ratio"]), np.nan)
    dfm["Day_Type"] = dfm["Day_Type"].astype("category")
    dfm["Time_Period"] = dfm["Time_Period"].astype("category")
    import warnings
    warnings.filterwarnings("ignore")
    model = smf.mixedlm("Log_Ratio ~ Day_Type + Time_Period + Year", dfm, groups=dfm["Station"], missing="drop")
    fit = model.fit()
    with open(os.path.join(OUT_DIR, "mixedlm_summary.txt"), "w") as f:
        f.write(str(fit.summary()))
    results["mixedlm"] = {"converged": bool(getattr(fit, "converged", True)), "n_obs": int(getattr(fit, "nobs", len(dfm.dropna(subset=['Log_Ratio']))))}
except Exception as e:
    results["mixedlm"] = {"error": str(e)}

write_results(results)
print("Inference complete. Wrote results/tables/stats_summary.json")