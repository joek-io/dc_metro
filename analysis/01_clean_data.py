
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = "data_raw"
OUT = "data_clean/station_monthly_recovery.csv"
YEARS = [2019, 2023, 2024, 2025]

COL_ALIASES = {
    "station": "Station",
    "station_name": "Station",
    "name": "Station",
    "line": "Line",
    "year": "Year",
    "yr": "Year",
    "month": "Month",
    "mo": "Month",
    "time_period": "Time_Period",
    "timeperiod": "Time_Period",
    "time_period_name": "Time_Period",
    "day_type": "Day_Type",
    "daytype": "Day_Type",
    "entries": "Entries",
    "tapped_entries": "Average_Daily_Tapped_Entries",
    "non_tapped_entries": "Average_Daily_Non_Tapped_Entries",
    "average_daily_tapped_entries": "Average_Daily_Tapped_Entries",
    "average_daily_non_tapped_entries": "Average_Daily_Non_Tapped_Entries",
    "avg_daily_tapped_entries": "Average_Daily_Tapped_Entries",
    "avg_daily_non_tapped_entries": "Average_Daily_Non_Tapped_Entries",
    "date": "Date",
}

def standardize_columns(df):
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", c.strip()).lower() for c in df.columns]
    rename_map = {}
    for c in df.columns:
        if c in COL_ALIASES:
            rename_map[c] = COL_ALIASES[c]
    df.rename(columns=rename_map, inplace=True)
    return df

def load_single_csv(path, default_year=None):
    df = pd.read_csv(path, encoding="utf-16le", sep="\t", engine="python")
    df = standardize_columns(df)

    # Get Year/Month from Date if not found
    if ("Year" not in df.columns) or ("Month" not in df.columns):
        if "Date" in df.columns:
            dt = pd.to_datetime(df["Date"], errors="coerce")
            if "Year" not in df.columns:
                df["Year"] = dt.dt.year
            if "Month" not in df.columns:
                df["Month"] = dt.dt.month
        elif (default_year is not None) and ("Year" not in df.columns):
            df["Year"] = default_year

    # Ensure columns exist
    for col in ["Station","Year","Month","Time_Period","Day_Type"]:
        if col not in df.columns:
            if col in ("Time_Period","Day_Type"):
                df[col] = np.nan
            elif col == "Station":
                for alt in ["STOP_NAME","NAME"]:
                    if alt in df.columns:
                        df["Station"] = df[alt]
                        break
                if "Station" not in df.columns:
                    df["Station"] = "UNKNOWN"
            elif col == "Month":
                df["Month"] = 1
            elif col == "Year":
                df["Year"] = default_year if default_year else 2019

    # Combine Entries from tapped/non-tapped for post 2019 data
    tapped = df["Average_Daily_Tapped_Entries"] if "Average_Daily_Tapped_Entries" in df.columns else None
    nontap = df["Average_Daily_Non_Tapped_Entries"] if "Average_Daily_Non_Tapped_Entries" in df.columns else None

    if (tapped is not None) or (nontap is not None):
        df["Entries"] = (tapped.fillna(0) if tapped is not None else 0) + (nontap.fillna(0) if nontap is not None else 0)
    else:
        if "Entries" not in df.columns:
            for alt in ["entry","ridership","avg_entries","average_entries","total_entries"]:
                if alt in df.columns:
                    df["Entries"] = df[alt]
                    break

    keep = ["Station","Year","Month","Time_Period","Day_Type","Entries"]
    for k in keep:
        if k not in df.columns:
            df[k] = np.nan
    return df[keep]

def load_year(y):
    base = Path(RAW_DIR)
    direct = base / f"ridership_{y}.csv"
    tap = base / f"ridership_{y}_tapped.csv"
    non = base / f"ridership_{y}_nontapped.csv"
    non2 = base / f"ridership_{y}_non_tapped.csv"

    if direct.exists():
        df = load_single_csv(direct, default_year=y)
        df["Year"] = y
        return df

    parts = []
    if tap.exists():
        parts.append(("tap", load_single_csv(tap, default_year=y).rename(columns={"Entries":"Entries_Tapped"})))
    if non.exists():
        parts.append(("non", load_single_csv(non, default_year=y).rename(columns={"Entries":"Entries_NonTapped"})))
    if non2.exists():
        parts.append(("non", load_single_csv(non2, default_year=y).rename(columns={"Entries":"Entries_NonTapped"})))

    if len(parts) == 0:
        raise FileNotFoundError(f"No CSV found for year {y} in {RAW_DIR}. Provide ridership_{y}.csv or ridership_{y}_tapped.csv and ridership_{y}_nontapped.csv")

    keys = ["Station","Year","Month","Time_Period","Day_Type"]
    merged = None
    for tag, part in parts:
        if merged is None:
            merged = part
        else:
            merged = merged.merge(part, on=keys, how="outer")

    if "Entries" not in merged.columns:
        merged["Entries"] = merged.get("Entries_Tapped", pd.Series(0, index=merged.index)).fillna(0) + merged.get("Entries_NonTapped", pd.Series(0, index=merged.index)).fillna(0)

    out = merged[["Station","Year","Month","Time_Period","Day_Type","Entries"]].copy()
    return out

def main():
    all_years = []
    for y in YEARS:
        dfy = load_year(y)
        dfy["Station"] = dfy["Station"].astype(str).str.upper().str.strip()
        for col in ["Time_Period","Day_Type"]:
            dfy[col] = dfy[col].astype(str).str.strip()
        dfy["Year"] = pd.to_numeric(dfy["Year"], errors="coerce").astype("Int64")
        dfy["Month"] = pd.to_numeric(dfy["Month"], errors="coerce").astype("Int64")
        all_years.append(dfy)

    df = pd.concat(all_years, ignore_index=True)
    df["Month"] = df["Month"].clip(lower=1, upper=12)

    agg = (df
           .groupby(["Station","Year","Month","Time_Period","Day_Type"], as_index=False)
           .agg(Entries=("Entries","sum"))
    )

    base = (agg[agg["Year"]==2019]
            .groupby(["Station","Time_Period"], as_index=False)["Entries"].mean()
            .rename(columns={"Entries":"Entries_2019"}))

    merged = agg.merge(base, on=["Station","Time_Period"], how="left")

    merged["Recovery_Ratio"] = merged["Entries"] / merged["Entries_2019"]
    merged["Denom_Flag"] = merged["Entries_2019"].isna() | (merged["Entries_2019"]==0)

    os.makedirs("data_clean", exist_ok=True)
    merged.to_csv(OUT, index=False)
    print(f"Saved {OUT} with {len(merged):,} rows.")

if __name__ == "__main__":
    main()
