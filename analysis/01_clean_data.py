
import pandas as pd
import os

RAW_DIR = "data_raw"
OUT = "data_clean/station_monthly_recovery.csv"

YEARS = [2019, 2023, 2024, 2025]

def load_year(y):
    path = os.path.join(RAW_DIR, f"ridership_{y}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Export from WMATA portal and place in data_raw/.")
    df = pd.read_csv(path)
    # Try to standardize column names; adjust if your export uses different labels
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    # Expect columns like: Station, Year, Month, Entries, Time_Period, Day_Type
    rename_map = {
        "station": "Station",
        "year": "Year",
        "month": "Month",
        "entries": "Entries",
        "time_period": "Time_Period",
        "day_type": "Day_Type"
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]
    # If Year not present, add
    if "Year" not in df.columns:
        df["Year"] = y
    return df[["Station","Year","Month","Entries","Time_Period","Day_Type"]]

def main():
    dfs = [load_year(y) for y in YEARS]
    ridership = pd.concat(dfs, ignore_index=True)

    # Clean strings
    for col in ["Station","Time_Period","Day_Type"]:
        ridership[col] = ridership[col].astype(str).str.strip()

    # Uppercase station for stable joins
    ridership["Station"] = ridership["Station"].str.upper()

    # Aggregate to month
    agg = (ridership
        .groupby(["Station","Year","Month","Time_Period","Day_Type"], as_index=False)
        .agg(Entries=("Entries","sum"))
    )

    # Compute 2019 baseline by Station x Time_Period
    base = (agg[agg["Year"]==2019]
            .groupby(["Station","Time_Period"], as_index=False)["Entries"].mean()
            .rename(columns={"Entries":"Entries_2019"}))

    merged = agg.merge(base, on=["Station","Time_Period"], how="left")

    # Recovery ratio
    merged["Recovery_Ratio"] = merged["Entries"] / merged["Entries_2019"]
    # Flag divisions where denominator missing/zero
    merged["Denom_Flag"] = merged["Entries_2019"].isna() | (merged["Entries_2019"]==0)

    os.makedirs("data_clean", exist_ok=True)
    merged.to_csv(OUT, index=False)
    print(f"✅ Saved {OUT} with {len(merged):,} rows.")

if __name__ == "__main__":
    main()
