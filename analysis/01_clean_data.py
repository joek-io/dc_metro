import os, re, json
from pathlib import Path
import numpy as np
import pandas as pd

# Folder paths
RAW_DIR = Path("data_raw")
OUT_DIR = Path("data_clean")
LOG_DIR = Path("results/tables")

# Output files
OUT_CSV = OUT_DIR / "station_monthly_recovery.csv"
LOG_JSON = LOG_DIR / "clean_log.json"

# Years to process
YEARS = [2019, 2023, 2024, 2025]

# FILE READING HELPERS

def _looks_utf16(p: Path) -> bool:
    """Return True if file likely UTF-16LE (BOM or NULs in first bytes)."""
    with open(p, "rb") as fb:
        head = fb.read(4)
    return head.startswith(b"\xff\xfe") or b"\x00" in head

def _sniff_sep(sample: str) -> str:
    """Choose a delimiter from sample text, preferring tab if present."""
    tabs = sample.count("\t")
    commas = sample.count(",")
    semis = sample.count(";")
    if tabs >= max(commas, semis):
        return "\t"
    if semis > commas:
        return ";"
    return ","  # default

def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Strip BOMs, trim, and collapse spaces in column names."""
    def _fix(c):
        c = str(c).replace("\ufeff", "")  # BOM
        c = re.sub(r"\s+", " ", c).strip()
        return c
    df.columns = [_fix(c) for c in df.columns]
    return df

def read_table(p: Path) -> pd.DataFrame:
    """Read CSV/Excel and sniff encoding + delimiter per file."""
    # Excel
    if p.suffix.lower() in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(p, engine="openpyxl")
        except Exception:
            return pd.read_excel(p)

    # UTF-16LE CSV
    if _looks_utf16(p):
        with open(p, "r", encoding="utf-16le", errors="replace", newline="") as f:
            sample = "".join([f.readline() for _ in range(25)])
        sep = _sniff_sep(sample)
        df = pd.read_csv(p, encoding="utf-16le", sep=sep, engine="python")
        return _clean_headers(df)

    # UTF-8 CSV
    import csv
    with open(p, "r", encoding="utf-8", errors="replace", newline="") as f:
        sample = "".join([f.readline() for _ in range(25)])
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]).delimiter
    except Exception:
        sep = _sniff_sep(sample)
    df = pd.read_csv(p, encoding="utf-8", sep=sep, engine="python")
    return _clean_headers(df)

# DATA CLEANING HELPERS

def normalize_station(x: str) -> str:
    """Clean and standardize station names for consistent matching."""
    if pd.isna(x):
        return "UNKNOWN"
    s = str(x).upper().strip()
    s = re.sub(r"[.]", "", s)          # remove periods
    s = re.sub(r"[–—]+", "-", s)       # replace long dashes with hyphen
    s = re.sub(r"\s+", " ", s).strip() # collapse spaces
    return s or "UNKNOWN"

# Acceptable final time-period buckets
VALID_TIME_PERIODS = {"AM", "MIDDAY", "PM", "EVENING"}
VALID_DAY_TYPES = {"WEEKDAY", "WEEKEND"}

def normalize_time_period(x: str):
    """
    Map many WMATA variants and explicit time ranges into {AM, MIDDAY, PM, EVENING}.
    Handles:
      - 'AM Peak', 'Midday', 'PM Peak', 'Evening'
      - Prefixes like 'Weekday - AM Peak'
      - Ranges like '4:00 AM - 9:59 AM', '10:00 AM - 2:59 PM', etc.
    Unknowns -> np.nan.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return np.nan

    t = str(x)
    t = t.replace("\u00A0", " ")               # non-breaking space
    t = re.sub(r"[–—]+", "-", t)               # long dash -> hyphen
    t = re.sub(r"\s+", " ", t).strip().upper() # collapse spaces

    # Remove leading day-type prefixes like 'WEEKDAY - ' or 'SATURDAY - '
    t = re.sub(r'^(WEEKDAY|SATURDAY|SUNDAY|WEEKEND|HOLIDAY)\s*-\s*', '', t)

    # Exact matches
    if t in {"AM", "AM PEAK"}:
        return "AM"
    if t in {"MIDDAY", "MID DAY", "MID-DAY"}:
        return "MIDDAY"
    if t in {"PM", "PM PEAK"}:
        return "PM"
    if t in {"EVENING", "EVE", "LATE EVENING", "NIGHT"}:
        return "EVENING"

    # Time range pattern: 'H:MM AM - H:MM PM'
    m = re.search(r'(\d{1,2}):\d{2}\s*(AM|PM)\s*-\s*(\d{1,2}):\d{2}\s*(AM|PM)', t)
    if m:
        sh, sap = int(m.group(1)), m.group(2)  # start hour and AM/PM
        # Convert start hour to 24h
        if sap == "AM":
            start_24 = 0 if sh == 12 else sh
        else:
            start_24 = 12 if sh == 12 else sh + 12

        # Bucket by start time
        # AM   : 05:00–09:59
        # MID  : 10:00–14:59
        # PM   : 15:00–18:59
        # EVE  : 19:00–23:59 and 00:00–04:59
        if   5 <= start_24 <= 9:   return "AM"
        elif 10 <= start_24 <= 14: return "MIDDAY"
        elif 15 <= start_24 <= 18: return "PM"
        else:                      return "EVENING"

    # Keyword rules
    if "EARLY AM" in t or ("AM" in t and "PEAK" in t):
        return "AM"
    if "MID" in t:
        return "MIDDAY"
    if "PM" in t and "PEAK" in t:
        return "PM"
    if any(k in t for k in ["EVEN", "NIGHT", "LATE"]):
        return "EVENING"

    # Non-analytical labels -> drop
    if any(k in t for k in ["ALL DAY", "OFF-PEAK", "OFF PEAK", "WEEKEND", "HOLIDAY"]):
        return np.nan

    return np.nan

def derive_day_type(service_type: str, day_of_week: str) -> str:
    """Use 'Service Type' or 'Day of Week' to classify as WEEKDAY or WEEKEND."""
    if isinstance(service_type, str) and service_type.strip():
        st = service_type.strip().upper()
        if st == "WEEKDAY": return "WEEKDAY"
        if st in {"SATURDAY", "SUNDAY"}: return "WEEKEND"
        if st == "HOLIDAY": return "WEEKDAY"  # adjust if you prefer holidays as WEEKEND
    if isinstance(day_of_week, str) and day_of_week.strip():
        d = day_of_week.strip().upper()
        return "WEEKEND" if d in {"SATURDAY", "SUNDAY"} else "WEEKDAY"
    return np.nan

# COLUMN NAMES EXPECTED

EXPECTED_COLS = [
    "Year of Date", "Date", "Day of Week", "Holiday", "Service Type",
    "Station Name", "Time Period", "Avg Daily Tapped Entries",
    "Entries", "NonTapped Entries", "SUM([NonTapped Entries])/COUNTD([Date])",
    "Tap Entries"
]

# LOAD WMATA TABLEAU FILE

def load_wmata_tableau(p: Path, default_year: int, log: dict) -> pd.DataFrame:
    """Load and clean a single WMATA CSV exported from the Ridership Portal."""
    df = read_table(p)
    log.setdefault("files_read", []).append({"path": str(p), "rows": int(len(df)), "cols": df.columns.tolist()})

    # Ensure all expected columns exist (some views may omit a few)
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Create Year and Month
    df["Year"] = pd.to_numeric(df["Year of Date"], errors="coerce").astype("Int64")
    date_parsed = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Year"].fillna(date_parsed.dt.year).fillna(default_year).astype("Int64")
    df["Month"] = date_parsed.dt.month.astype("Int64")

    # Clean station names
    df["Station"] = df["Station Name"].map(normalize_station)

    # Map time periods and log distinct raw vs. mapped values
    df["Time_Period"] = df["Time Period"].map(normalize_time_period)
    orig_unique = sorted(pd.Series(df["Time Period"].astype(str).str.strip().str.upper().unique()).tolist())
    mapped_unique = sorted(pd.Series(df["Time_Period"].astype(str).unique()).tolist())
    log.setdefault("time_period_observed", {})[str(p)] = {
        "original_unique": orig_unique[:200],
        "mapped_unique": mapped_unique[:200],
    }

    # Derive weekday/weekend type
    df["Day_Type"] = [derive_day_type(st, dow) for st, dow in zip(df["Service Type"], df["Day of Week"])]

    # Combine tapped + non-tapped counts
    tapped = pd.to_numeric(df["Avg Daily Tapped Entries"], errors="coerce")
    nontap = pd.to_numeric(df["NonTapped Entries"], errors="coerce")
    alt_entries = pd.to_numeric(df["Entries"], errors="coerce")
    entries = tapped.fillna(0) + nontap.fillna(0)
    entries = np.where(entries > 0, entries, alt_entries.fillna(0))
    df["Entries"] = entries

    # Keep only needed columns
    keep = ["Station", "Year", "Month", "Time_Period", "Day_Type", "Entries"]
    df = df[keep].copy()

    # Drop rows without a mapped period to keep analysis clean
    before = len(df)
    df = df[df["Time_Period"].isin(VALID_TIME_PERIODS) | df["Time_Period"].isna()].copy()
    dropped = before - len(df)
    if dropped > 0:
        log.setdefault("warnings", []).append(f"Dropped {dropped} rows with unmapped Time_Period in {p.name}")

    # Warn if any mapped but not valid
    tp_bad = df["Time_Period"].notna() & (~df["Time_Period"].isin(VALID_TIME_PERIODS))
    dt_bad = df["Day_Type"].notna() & (~df["Day_Type"].isin(VALID_DAY_TYPES))
    if int(tp_bad.sum()) > 0:
        log.setdefault("warnings", []).append(
            f"Unrecognized Time_Period values after mapping: {int(tp_bad.sum())} in {p.name}"
        )
    if int(dt_bad.sum()) > 0:
        log.setdefault("warnings", []).append(f"Unrecognized Day_Type values: {int(dt_bad.sum())}")

    # Convert Entries to numeric
    df["Entries"] = pd.to_numeric(df["Entries"], errors="coerce").fillna(0)
    return df

# LOAD EACH YEAR

def load_year(year: int, log: dict) -> pd.DataFrame:
    """Load one year's data, supports single or split tapped/non-tapped files."""
    single = RAW_DIR / f"ridership_{year}.csv"
    tapped = RAW_DIR / f"ridership_{year}_tapped.csv"
    nontapped = RAW_DIR / f"ridership_{year}_nontapped.csv"
    non_tapped_alt = RAW_DIR / f"ridership_{year}_non_tapped.csv"

    if single.exists():
        return load_wmata_tableau(single, default_year=year, log=log)

    parts = []
    for f in [tapped, nontapped, non_tapped_alt]:
        if f.exists():
            parts.append(load_wmata_tableau(f, default_year=year, log=log))

    if not parts:
        raise FileNotFoundError(f"No data found for {year}")

    # Combine if split between tapped/non-tapped
    df = pd.concat(parts, ignore_index=True)
    df = df.groupby(["Station", "Year", "Month", "Time_Period", "Day_Type"], as_index=False)["Entries"].sum()
    return df

# MAIN PIPELINE

def main():
    """Combine yearly data, compute recovery ratios vs. 2019 baseline, save CSV + log."""
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    log = {"notes": [], "files_read": [], "warnings": []}

    # Load and clean all years
    frames = []
    for y in YEARS:
        dfy = load_year(y, log)
        before = len(dfy)
        dfy = dfy.dropna(subset=["Station", "Year", "Month"]).copy()
        if len(dfy) < before:
            log["warnings"].append(f"Dropped {before - len(dfy)} rows missing Station/Year/Month for {y}")
        frames.append(dfy)

    # Combine all years into one dataset
    df = pd.concat(frames, ignore_index=True)

    # Group to remove duplicates
    df = df.groupby(["Station", "Year", "Month", "Time_Period", "Day_Type"], as_index=False)["Entries"].sum()

    # Compute 2019 baseline (average entries per station/time period)
    base = (
        df[df["Year"] == 2019]
        .groupby(["Station", "Time_Period"], as_index=False)["Entries"]
        .mean()
        .rename(columns={"Entries": "Entries_2019"})
    )

    # Join baseline and compute Recovery Ratio
    merged = df.merge(base, on=["Station", "Time_Period"], how="left")
    merged["Denom_Flag"] = merged["Entries_2019"].isna() | (merged["Entries_2019"] == 0)
    merged["Recovery_Ratio"] = np.where(
        merged["Denom_Flag"], np.nan, merged["Entries"] / merged["Entries_2019"]
    )

    # Save final dataset
    merged = merged.sort_values(["Station", "Year", "Month", "Time_Period", "Day_Type"]).reset_index(drop=True)
    merged.to_csv(OUT_CSV, index=False)

    # Write log file
    summary = {
        "rows_written": int(len(merged)),
        "unique_stations": int(merged["Station"].nunique()),
        "denom_missing": int(merged["Denom_Flag"].sum()),
        "recovery_nan": int(merged["Recovery_Ratio"].isna().sum()),
    }
    with open(LOG_JSON, "w") as f:
        json.dump({"summary": summary, **log}, f, indent=2)

    print(f"Wrote {OUT_CSV} with {len(merged)} rows")

if __name__ == "__main__":
    main()