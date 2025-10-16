import os
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_rel, wilcoxon, shapiro, friedmanchisquare
import statsmodels.formula.api as smf

IN_CSV = Path("data_clean/station_monthly_recovery.csv")
OUT_DIR = Path("results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "stats_summary.json"
OUT_MIXEDLM = OUT_DIR / "mixedlm_summary.txt"

results = {}

def _ok_cols(df, needed):
    missing = [c for c in needed if c not in df.columns]
    return (len(missing) == 0, missing)

def _paired_test(weekend, weekday):
    """
    Robust paired test: Shapiro on diffs -> paired t if ~normal else Wilcoxon.
    Returns (test_name, p_value, d_effect or None)
    """
    delta = weekend - weekday
    # Drop NaNs pairwise
    mask = np.isfinite(weekend) & np.isfinite(weekday)
    x = weekend[mask]
    y = weekday[mask]
    d = (x - y)
    if len(d) < 3:
        return ("Insufficient pairs", None, None)
    try:
        W, p_norm = shapiro(d)
    except Exception:
        # If Shapiro fails (large N, ties), fall back to nonparam
        p_norm = 0.0

    if (p_norm is not None) and (p_norm > 0.05):
        # Paired t
        stat, p_val = ttest_rel(x, y, nan_policy="omit")
        denom = np.nanstd(d, ddof=1)
        d_eff = None if (denom is None or denom == 0 or math.isnan(denom)) else float(np.nanmean(d) / denom)
        return ("Paired t-test", float(p_val), d_eff)
    else:
        # Wilcoxon
        try:
            stat, p_val = wilcoxon(x, y, zero_method="wilcox", correction=False)
        except Exception:
            return ("Wilcoxon signed-rank (failed)", None, None)
        return ("Wilcoxon signed-rank", float(p_val), None)

def _friedman_by_daytype(df, day_type):
    """
    Friedman test across time periods within a given Day_Type.
    Requires each station to have AM/MIDDAY/PM/EVENING means.
    """
    sub = df[(df["Day_Type"] == day_type) & df["Time_Period"].isin(["AM", "MIDDAY", "PM", "EVENING"])].copy()
    if sub.empty:
        return {"error": f"No rows for {day_type}."}
    pivot = sub.pivot_table(index="Station", columns="Time_Period", values="Recovery_Ratio", aggfunc="mean")
    # Keep only stations with all four periods
    needed = ["AM", "MIDDAY", "PM", "EVENING"]
    pivot = pivot.dropna(axis=0, how="any")
    if any(c not in pivot.columns for c in needed):
        return {"error": f"Missing periods in {day_type} pivot."}
    pivot = pivot[needed]
    if pivot.shape[0] < 3:
        return {"error": f"Insufficient stations for {day_type} Friedman."}
    try:
        stat, p_val = friedmanchisquare(*[pivot[c] for c in pivot.columns])
        return {
            "test": "Friedman",
            "day_type": day_type,
            "stations_n": int(pivot.shape[0]),
            "periods": needed,
            "p_value": float(p_val),
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    if not IN_CSV.exists():
        with open(OUT_JSON, "w") as f:
            json.dump({"error": f"missing file: {str(IN_CSV)}"}, f, indent=2)
        return

    df = pd.read_csv(IN_CSV)
    ok, miss = _ok_cols(df, ["Station", "Year", "Time_Period", "Day_Type", "Recovery_Ratio", "Entries_2019", "Denom_Flag"])
    if not ok:
        with open(OUT_JSON, "w") as f:
            json.dump({"error": f"missing columns: {set(miss)}"}, f, indent=2)
        return

    # Normalize Day_Type casing (defensive)
    df["Day_Type"] = df["Day_Type"].astype(str).str.upper().str.strip()
    df["Time_Period"] = df["Time_Period"].astype(str).str.upper().str.strip()
    # Keep only valid time periods
    df = df[df["Time_Period"].isin(["AM", "MIDDAY", "PM", "EVENING"])].copy()

    # Core analysis subset: post-2019 data with finite ratios
    df_curr = df[(df["Year"] >= 2023) & np.isfinite(df["Recovery_Ratio"])].copy()

    # Sensitivity subset: exclude weak baselines (Denom_Flag True or NaN Entries_2019)
    df_curr_strong = df_curr[~df_curr["Denom_Flag"] & np.isfinite(df_curr["Entries_2019"])].copy()

    results_local = {}

    # 1) Weekend vs Weekday (station-level means)
    for label, frame in [("all", df_curr), ("strong_baseline", df_curr_strong)]:
        try:
            pivot = frame.pivot_table(index="Station", columns="Day_Type", values="Recovery_Ratio", aggfunc="mean")
            # Normalize expected columns
            col_map = {c: c.upper() for c in pivot.columns}
            pivot.rename(columns=col_map, inplace=True)
            if {"WEEKEND", "WEEKDAY"}.issubset(set(pivot.columns)):
                weekend = pivot["WEEKEND"]
                weekday = pivot["WEEKDAY"]
                test_used, p_val, d_eff = _paired_test(weekend, weekday)
                results_local[f"weekend_vs_weekday__{label}"] = {
                    "test": test_used,
                    "stations_n": int(pivot.dropna().shape[0]),
                    "weekend_mean": float(np.nanmean(weekend)),
                    "weekday_mean": float(np.nanmean(weekday)),
                    "mean_diff": float(np.nanmean(weekend - weekday)),
                    "p_value": None if p_val is None else float(p_val),
                    "cohens_d": None if d_eff is None else float(d_eff),
                }
            else:
                results_local[f"weekend_vs_weekday__{label}"] = {"error": "Weekend/Weekday columns missing after pivot."}
        except Exception as e:
            results_local[f"weekend_vs_weekday__{label}"] = {"error": str(e)}

    # 2) Time-of-day differences (weekday and weekend separately)
    results_local["time_of_day_weekday"] = _friedman_by_daytype(df_curr, "WEEKDAY")
    results_local["time_of_day_weekend"] = _friedman_by_daytype(df_curr, "WEEKEND")

    # 3) Mixed-effects model on log ratio, with Year centered and interaction
    #    Use strong-baseline subset to avoid extreme ratios from weak denominators.
    dfm = df_curr_strong.copy()
    dfm = dfm[np.isfinite(dfm["Recovery_Ratio"]) & (dfm["Recovery_Ratio"] > 0)].copy()
    if len(dfm) >= 50 and dfm["Station"].nunique() >= 5:
        dfm["Log_Ratio"] = np.log(dfm["Recovery_Ratio"])
        dfm["Year_centered"] = dfm["Year"] - 2019
        dfm["Day_Type"] = dfm["Day_Type"].astype("category")
        dfm["Time_Period"] = dfm["Time_Period"].astype("category")

        # Primary model: Day_Type * Time_Period + Year_centered, random intercept for Station
        try:
            model = smf.mixedlm("Log_Ratio ~ Day_Type * Time_Period + Year_centered",
                                dfm, groups=dfm["Station"], missing="drop")
            fit = model.fit()
            with open(OUT_MIXEDLM, "w") as f:
                f.write(str(fit.summary()))
            results_local["mixedlm"] = {
                "converged": bool(getattr(fit, "converged", False)),
                "n_obs": int(fit.nobs),
                "aic": float(getattr(fit, "aic", np.nan)),
                "bic": float(getattr(fit, "bic", np.nan)),
            }
        except Exception as e:
            # Fallback to additive (no interaction)
            try:
                model = smf.mixedlm("Log_Ratio ~ Day_Type + Time_Period + Year_centered",
                                    dfm, groups=dfm["Station"], missing="drop")
                fit = model.fit()
                with open(OUT_MIXEDLM, "w") as f:
                    f.write("Fallback (no interaction)\n")
                    f.write(str(fit.summary()))
                results_local["mixedlm"] = {
                    "converged": bool(getattr(fit, "converged", False)),
                    "n_obs": int(fit.nobs),
                    "aic": float(getattr(fit, "aic", np.nan)),
                    "bic": float(getattr(fit, "bic", np.nan)),
                    "fallback": True,
                    "error_primary": str(e),
                }
            except Exception as e2:
                results_local["mixedlm"] = {"error": f"both primary and fallback failed: {e}; {e2}"}
    else:
        results_local["mixedlm"] = {"error": "insufficient rows or stations for mixed model."}

    # Write JSON
    with open(OUT_JSON, "w") as f:
        json.dump(results_local, f, indent=2)

    print("Inference complete. Wrote:", OUT_JSON)

if __name__ == "__main__":
    main()