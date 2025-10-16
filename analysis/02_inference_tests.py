import os, json
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, shapiro, friedmanchisquare
import statsmodels.formula.api as smf

IN = "data_clean/station_monthly_recovery.csv"
OUT_DIR = "results/tables"
os.makedirs(OUT_DIR, exist_ok=True)

def ensure_recovery_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Recovery_Ratio exists; if missing, compute it vs 2019 baseline."""
    # Minimal columns required for backfill
    needed = {"Station","Year","Time_Period","Entries"}
    if not needed.issubset(df.columns):
        raise AssertionError(f"missing columns for backfill: {needed - set(df.columns)}")

    if "Recovery_Ratio" in df.columns and df["Recovery_Ratio"].notna().any():
        # Column already present with values: standardize dtype and return
        df["Recovery_Ratio"] = pd.to_numeric(df["Recovery_Ratio"], errors="coerce")
        return df

    # Build 2019 baseline by Station × Time_Period (mean of Entries in 2019)
    base = (
        df[df["Year"] == 2019]
        .groupby(["Station","Time_Period"], as_index=False)["Entries"]
        .mean()
        .rename(columns={"Entries": "Entries_2019"})
    )

    # Merge baseline
    out = df.merge(base, on=["Station","Time_Period"], how="left")
    out["Entries_2019"] = pd.to_numeric(out["Entries_2019"], errors="coerce")

    # Compute ratio; guard denom
    denom_bad = out["Entries_2019"].isna() | (out["Entries_2019"] == 0)
    out["Recovery_Ratio"] = np.where(denom_bad, np.nan, out["Entries"] / out["Entries_2019"])
    return out

# Load data
results = {
    "weekend_vs_weekday": {"p_value": None},
    "time_of_day": {},
    "mixedlm": {}
}

try:
    df = pd.read_csv(IN)
except Exception as e:
    results["error"] = f"failed to read input: {e}"
    with open(os.path.join(OUT_DIR, "stats_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    raise

# Core schema check (but we can backfill Recovery_Ratio)
core_needed = {"Station","Year","Time_Period","Day_Type","Entries"}
missing_core = core_needed - set(df.columns)
if missing_core:
    results["error"] = f"missing columns: {missing_core}"
    with open(os.path.join(OUT_DIR, "stats_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    raise AssertionError(results["error"])

# Normalize
df["Day_Type"] = df["Day_Type"].astype(str).str.upper()
df["Time_Period"] = df["Time_Period"].astype(str).str.upper()
df["Entries"] = pd.to_numeric(df["Entries"], errors="coerce").fillna(0)

# Backfill Recovery_Ratio if needed
try:
    df = ensure_recovery_ratio(df)
except Exception as e:
    results["error"] = f"recovery backfill failed: {e}"
    with open(os.path.join(OUT_DIR, "stats_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    raise

# Restrict to current years for tests
df_curr = df[df["Year"] >= 2023].copy()
df_curr["Recovery_Ratio"] = pd.to_numeric(df_curr["Recovery_Ratio"], errors="coerce")

# Weekend vs Weekday (paired) 
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

# Time-of-day (Friedman on Weekday)
try:
    tod = (
        df_curr[df_curr["Day_Type"] == "WEEKDAY"]
        .pivot_table(index="Station", columns="Time_Period", values="Recovery_Ratio", aggfunc="mean")
        .reindex(columns=["AM","MIDDAY","PM","EVENING"])
        .dropna(how="any")
    )
    if tod.shape[0] >= 3 and tod.shape[1] >= 3:
        stat, p_friedman = friedmanchisquare(*[tod[c] for c in tod.columns if c in tod.columns])
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
    results["mixedlm"] = {"converged": bool(getattr(fit, "converged", True)), "n_obs": int(fit.nobs)}
except Exception as e:
    results["mixedlm"] = {"error": str(e)}

with open(os.path.join(OUT_DIR, "stats_summary.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Inference complete. Wrote results/tables/stats_summary.json")