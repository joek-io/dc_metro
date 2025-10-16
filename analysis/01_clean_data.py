import os, re, json
from pathlib import Path
import numpy as np
import pandas as pd

# Paths
RAW_DIR = Path("data_raw")
OUT_DIR = Path("data_clean")
LOG_DIR = Path("results/tables")

# Outputs
OUT_CSV = OUT_DIR / "station_monthly_recovery.csv"
LOG_JSON = LOG_DIR / "clean_log.json"

# Years
YEARS = [2019, 2023, 2024, 2025]

# File reading

def _looks_utf16(p: Path) -> bool:
    with open(p, "rb") as fb:
        head = fb.read(4)
    return head.startswith(b"\xff\xfe") or b"\x00" in head

def _sniff_sep(sample: str) -> str:
    tabs, commas, semis = sample.count("\t"), sample.count(","), sample.count(";")
    if tabs >= max(commas, semis): return "\t"
    if semis > commas: return ";"
    return ","

def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    def _fix(c):
        c = str(c).replace("\ufeff", "")
        c = re.sub(r"\s+", " ", c).strip()
        return c
    df.columns = [_fix(c) for c in df.columns]
    return df

def read_table(p: Path) -> pd.DataFrame:
    if p.suffix.lower() in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(p, engine="openpyxl")
        except Exception:
            return pd.read_excel(p)
    if _looks_utf16(p):
        with open(p, "r", encoding="utf-16le", errors="replace", newline="") as f:
            sample = "".join([f.readline() for _ in range(25)])
        sep = _sniff_sep(sample)
        df = pd.read_csv(p, encoding="utf-16le", sep=sep, engine="python")
        return _clean_headers(df)
    import csv
    with open(p, "r", encoding="utf-8", errors="replace", newline="") as f:
        sample = "".join([f.readline() for _ in range(25)])
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]).delimiter
    except Exception:
        sep = _sniff_sep(sample)
    df = pd.read_csv(p, encoding="utf-8", sep=sep, engine="python")
    return _clean_headers(df)

# Cleaning helpers

def normalize_station(x: str) -> str:
    if pd.isna(x): return "UNKNOWN"
    s = str(x).upper().strip()
    s = re.sub(r"[.]", "", s)
    s = re.sub(r"[–—]+", "-", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or "UNKNOWN"

VALID_TIME_PERIODS = {"AM", "MIDDAY", "PM", "EVENING"}
VALID_DAY_TYPES = {"WEEKDAY", "WEEKEND"}

def normalize_time_period(x: str):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return np.nan
    t = str(x).replace("\u00A0", " ")
    t = re.sub(r"[–—]+", "-", t)
    t = re.sub(r"\s+", " ", t).strip().upper()
    t = re.sub(r'^(WEEKDAY|SATURDAY|SUNDAY|WEEKEND|HOLIDAY)\s*-\s*', '', t)
    if t in {"AM", "AM PEAK"}: return "AM"
    if t in {"MIDDAY", "MID DAY", "MID-DAY"}: return "MIDDAY"
    if t in {"PM", "PM PEAK"}: return "PM"
    if t in {"EVENING", "EVE", "LATE EVENING", "NIGHT"}: return "EVENING"
    m = re.search(r'(\d{1,2}):\d{2}\s*(AM|PM)\s*-\s*(\d{1,2}):\d{2}\s*(AM|PM)', t)
    if m:
        sh, sap = int(m.group(1)), m.group(2)
        start_24 = (0 if sh == 12 else sh) if sap == "AM" else (12 if sh == 12 else sh + 12)
        if   5 <= start_24 <= 9:  return "AM"
        elif 10 <= start_24 <= 14:return "MIDDAY"
        elif 15 <= start_24 <= 18:return "PM"
        else:                     return "EVENING"
    if "EARLY AM" in t or ("AM" in t and "PEAK" in t): return "AM"
    if "MID" in t: return "MIDDAY"
    if "PM" in t and "PEAK" in t: return "PM"
    if any(k in t for k in ["EVEN", "NIGHT", "LATE"]): return "EVENING"
    if any(k in t for k in ["ALL DAY", "OFF-PEAK", "OFF PEAK", "WEEKEND", "HOLIDAY"]): return np.nan
    return np.nan

def derive_day_type(service_type: str, day_of_week: str) -> str:
    # WEEKDAY/WEEKEND derivation
    if isinstance(service_type, str) and service_type.strip():
        st = service_type.strip().upper().replace("\u00A0", " ")
        st = re.sub(r"\s+", " ", st)
        if st in {"WEEKDAY", "WEEKDAYS", "ALL WEEKDAYS"}: return "WEEKDAY"
        if st in {"WEEKEND", "WEEKENDS"}: return "WEEKEND"
        if st in {"SATURDAY", "SUNDAY"}: return "WEEKEND"
        if "WEEKDAY" in st and "WEEKEND" not in st: return "WEEKDAY"
        if "WEEKEND" in st: return "WEEKEND"
        if "HOLIDAY" in st: return "WEEKDAY"
    if isinstance(day_of_week, str) and day_of_week.strip():
        d = day_of_week.strip().upper().replace("\u00A0", " ")
        d = re.sub(r"\s+", " ", d)
        if d in {"MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY"}: return "WEEKDAY"
        if d in {"SATURDAY","SUNDAY"}: return "WEEKEND"
        if "SAT" in d or "SUN" in d: return "WEEKEND"
        if any(w in d for w in ["MON","TUE","WED","THU","FRI"]): return "WEEKDAY"
    return np.nan

def _parse_date_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, origin="1899-12-30", unit="D", errors="coerce")
    probe = next((str(v) for v in s.dropna().head(50).tolist() if str(v).strip()), "").strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", probe):
        return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", probe):
        return pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    return pd.to_datetime(s, errors="coerce")

# Expected columns

EXPECTED_COLS = [
    "Year of Date", "Date", "Day of Week", "Holiday", "Service Type",
    "Station Name", "Time Period", "Avg Daily Tapped Entries",
    "Entries", "NonTapped Entries", "SUM([NonTapped Entries])/COUNTD([Date])",
    "Tap Entries"
]

# Load file

def load_wmata_tableau(p: Path, default_year: int, log: dict) -> pd.DataFrame:
    df = read_table(p)
    log.setdefault("files_read", []).append({"path": str(p), "rows": int(len(df)), "cols": df.columns.tolist()})
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = np.nan

    df["Year"] = pd.to_numeric(df["Year of Date"], errors="coerce").astype("Int64")
    date_parsed = _parse_date_series(df["Date"])
    df["Year"] = df["Year"].fillna(date_parsed.dt.year).fillna(default_year).astype("Int64")
    df["Month"] = date_parsed.dt.month.astype("Int64")

    df["Station"] = df["Station Name"].map(normalize_station)
    df["Time_Period"] = df["Time Period"].map(normalize_time_period)

    orig_unique = sorted(pd.Series(df["Time Period"].astype(str).str.strip().str.upper().unique()).tolist())
    mapped_unique = sorted(pd.Series(df["Time_Period"].astype(str).unique()).tolist())
    log.setdefault("time_period_observed", {})[str(p)] = {
        "original_unique": orig_unique[:200],
        "mapped_unique": mapped_unique[:200],
    }

    df["Day_Type"] = [derive_day_type(st, dow) for st, dow in zip(df["Service Type"], df["Day of Week"])]

    tapped = pd.to_numeric(df["Avg Daily Tapped Entries"], errors="coerce")
    nontap = pd.to_numeric(df["NonTapped Entries"], errors="coerce")
    alt_entries = pd.to_numeric(df["Entries"], errors="coerce")
    entries = tapped.fillna(0) + nontap.fillna(0)
    entries = np.where(entries > 0, entries, alt_entries.fillna(0))
    df["Entries"] = entries

    keep = ["Station", "Year", "Month", "Time_Period", "Day_Type", "Entries"]
    df = df[keep].copy()

    before = len(df)
    df = df[df["Time_Period"].isin(VALID_TIME_PERIODS) | df["Time_Period"].isna()].copy()
    dropped = before - len(df)
    if dropped > 0:
        log.setdefault("warnings", []).append(f"Dropped {dropped} rows with unmapped Time_Period in {p.name}")

    # Log day and time
    dt_counts = df["Day_Type"].value_counts(dropna=False).to_dict()
    tp_counts = df["Time_Period"].value_counts(dropna=False).to_dict()
    log.setdefault("coverage", {})[str(p)] = {
        "day_type_counts": {str(k): int(v) for k, v in dt_counts.items()},
        "time_period_counts": {str(k): int(v) for k, v in tp_counts.items()},
    }

    tp_bad = df["Time_Period"].notna() & (~df["Time_Period"].isin(VALID_TIME_PERIODS))
    dt_bad = df["Day_Type"].notna() & (~df["Day_Type"].isin(VALID_DAY_TYPES))
    if int(tp_bad.sum()) > 0:
        log.setdefault("warnings", []).append(
            f"Unrecognized Time_Period values after mapping: {int(tp_bad.sum())} in {p.name}"
        )
    if int(dt_bad.sum()) > 0:
        log.setdefault("warnings", []).append(f"Unrecognized Day_Type values: {int(dt_bad.sum())}")

    df["Entries"] = pd.to_numeric(df["Entries"], errors="coerce").fillna(0)
    return df

# Load year

def load_year(year: int, log: dict) -> pd.DataFrame:
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

    df = pd.concat(parts, ignore_index=True)
    df = df.groupby(["Station", "Year", "Month", "Time_Period", "Day_Type"], as_index=False)["Entries"].sum()
    return df

# Main

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    log = {"notes": [], "files_read": [], "warnings": []}

    frames = []
    for y in YEARS:
        dfy = load_year(y, log)
        before = len(dfy)
        dfy = dfy.dropna(subset=["Station", "Year", "Month"]).copy()
        if len(dfy) < before:
            log["warnings"].append(f"Dropped {before - len(dfy)} rows missing Station/Year/Month for {y}")
        frames.append(dfy)

    df = pd.concat(frames, ignore_index=True)

    # Global coverage diagnostics
    tp_global = df["Time_Period"].value_counts(dropna=False).to_dict()
    dt_global = df["Day_Type"].value_counts(dropna=False).to_dict()
    log.setdefault("global_coverage", {})["Time_Period"] = {str(k): int(v) for k, v in tp_global.items()}
    log.setdefault("global_coverage", {})["Day_Type"] = {str(k): int(v) for k, v in dt_global.items()}

    df = df.groupby(["Station", "Year", "Month", "Time_Period", "Day_Type"], as_index=False)["Entries"].sum()

    # Dual baselines (period + station) with fallback
    MIN_BASELINE_COUNT = 1  # keep at 1 per your test

    # Period baseline (Station × Time_Period)
    base_raw = df[df["Year"] == 2019].groupby(["Station", "Time_Period"])["Entries"]
    base_period = base_raw.agg(Entries_2019="mean", n_2019="count").reset_index()
    base_period.loc[base_period["n_2019"] < MIN_BASELINE_COUNT, "Entries_2019"] = np.nan
    base_period = base_period.drop(columns=["n_2019"])

    # Station baseline (Station)
    base_station_raw = df[df["Year"] == 2019].groupby(["Station"])["Entries"]
    base_station = base_station_raw.agg(Entries_2019_station="mean", n_2019_station="count").reset_index()
    base_station.loc[base_station["n_2019_station"] < MIN_BASELINE_COUNT, "Entries_2019_station"] = np.nan
    base_station = base_station.drop(columns=["n_2019_station"])

    # Join baselines
    merged = df.merge(base_period, on=["Station", "Time_Period"], how="left")
    merged = merged.merge(base_station, on="Station", how="left")

    # Preferred period baseline; fallback to station baseline when missing
    merged["Entries_2019_final"] = merged["Entries_2019"]
    need_fallback = merged["Entries_2019_final"].isna() | (merged["Entries_2019_final"] == 0)
    merged.loc[need_fallback, "Entries_2019_final"] = merged.loc[need_fallback, "Entries_2019_station"]

    # Flags and ratios (keep original + station + final)
    merged["Denom_Flag_Period"] = merged["Entries_2019"].isna() | (merged["Entries_2019"] == 0)
    merged["Denom_Flag_Station"] = merged["Entries_2019_station"].isna() | (merged["Entries_2019_station"] == 0)
    merged["Denom_Flag"] = merged["Entries_2019_final"].isna() | (merged["Entries_2019_final"] == 0)

    merged["Recovery_Ratio_Period"] = np.where(
        merged["Denom_Flag_Period"], np.nan, merged["Entries"] / merged["Entries_2019"]
    )
    merged["Recovery_Ratio_Station"] = np.where(
        merged["Denom_Flag_Station"], np.nan, merged["Entries"] / merged["Entries_2019_station"]
    )
    merged["Recovery_Ratio_Final"] = np.where(
        merged["Denom_Flag"], np.nan, merged["Entries"] / merged["Entries_2019_final"]
    )

    # Log how many rows used fallback
    used_fallback = int(need_fallback.sum())
    log.setdefault("notes", []).append(f"Applied station baseline fallback to {used_fallback} rows.")

    # Post-merge coverage on usable Recovery_Ratio_Final 
    valid_rr = merged[~merged["Recovery_Ratio_Final"].isna()]
    rr_counts_by_daytype = valid_rr["Day_Type"].value_counts(dropna=False).to_dict()
    rr_counts_by_period = valid_rr["Time_Period"].value_counts(dropna=False).to_dict()
    log.setdefault("postmerge_recovery_coverage", {})["by_day_type"] = {str(k): int(v) for k, v in rr_counts_by_daytype.items()}
    log.setdefault("postmerge_recovery_coverage", {})["by_time_period"] = {str(k): int(v) for k, v in rr_counts_by_period.items()}

    merged = merged.sort_values(["Station", "Year", "Month", "Time_Period", "Day_Type"]).reset_index(drop=True)
    merged.to_csv(OUT_CSV, index=False)

    # Export missing baselines for review (using final denominator flag)
    missing_base = merged[merged["Denom_Flag"]].copy()
    if not missing_base.empty:
        diag_dir = LOG_DIR / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        missing_base.sort_values(["Station", "Time_Period", "Year", "Month"]).to_csv(
            diag_dir / "missing_2019_baseline.csv", index=False
        )

    summary = {
        "rows_written": int(len(merged)),
        "unique_stations": int(merged["Station"].nunique()),
        "denom_missing": int(merged["Denom_Flag"].sum()),
        "recovery_nan": int(merged["Recovery_Ratio_Final"].isna().sum()),
    }
    with open(LOG_JSON, "w") as f:
        json.dump({"summary": summary, **log}, f, indent=2)

    print(f"Wrote {OUT_CSV} with {len(merged)} rows")

if __name__ == "__main__":
    main()