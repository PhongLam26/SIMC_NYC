import os
import gc
import time
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
from tqdm import tqdm


# =========================
# CONFIG
# =========================

# =========================
# NOAA TOKEN
# =========================
# Dán token NOAA của bạn vào đây
# Lưu ý: không up file này lên GitHub hoặc gửi public vì token nằm trong code.

NOAA_TOKEN = "kezRKPCxfVZIpBrttuQCGNPFnGdySQQG"

if NOAA_TOKEN == "DAN_TOKEN_CUA_BAN_VAO_DAY" or not NOAA_TOKEN.strip():
    raise RuntimeError(
        "Bạn chưa dán NOAA token vào biến NOAA_TOKEN trong file .py"
    )

BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"

OUT_DIR = Path("data/raw/noaa_weather")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fit với data 311 của mình
START_YEAR = 2015
END_YEAR = 2025

# Central Park station
# Đây là trạm ổn nhất để làm weather đại diện cho NYC.
STATIONS = {
    "central_park": "GHCND:USW00094728"
}

# Các biến thời tiết cần cho bài smart city
DATATYPES = [
    "TMAX",  # max temperature
    "TMIN",  # min temperature
    "PRCP",  # precipitation
    "SNOW",  # snowfall
    "SNWD",  # snow depth
    "AWND"   # average wind speed, nếu trạm có
]

LIMIT = 1000
SLEEP_SECONDS = 0.3


# =========================
# API FUNCTIONS
# =========================

def request_with_retry(params, max_retries=5):
    headers = {
        "token": NOAA_TOKEN
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(
                BASE_URL,
                headers=headers,
                params=params,
                timeout=90
            )

            if response.status_code == 200:
                return response.json()

            print(f"HTTP {response.status_code}: {response.text[:300]}")

        except requests.RequestException as e:
            print(f"Request error: {e}")

        wait = 2 ** attempt
        print(f"Retry after {wait}s...")
        time.sleep(wait)

    raise RuntimeError("NOAA request failed after max retries.")


def fetch_station_year_long(station_name, station_id, year):
    """
    Pull NOAA GHCND daily data dạng long:
    date | datatype | value | station | attributes
    """
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    all_rows = []
    offset = 1

    while True:
        params = {
            "datasetid": "GHCND",
            "stationid": station_id,
            "startdate": start_date,
            "enddate": end_date,
            "datatypeid": DATATYPES,
            "units": "metric",
            "limit": LIMIT,
            "offset": offset,
            "includemetadata": "false"
        }

        data = request_with_retry(params)

        results = data.get("results", [])

        if not results:
            break

        all_rows.extend(results)

        print(
            f"{station_name} {year} | "
            f"offset={offset:,} | rows={len(results):,} | total={len(all_rows):,}"
        )

        if len(results) < LIMIT:
            break

        offset += LIMIT
        time.sleep(SLEEP_SECONDS)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Thêm station info
    df["station_name"] = station_name
    df["station_id"] = station_id
    df["year"] = year

    return df


def convert_long_to_wide(df_long):
    """
    NOAA trả dạng long:
    date | datatype | value

    Mình pivot thành wide:
    date | TMAX | TMIN | PRCP | SNOW | SNWD | AWND
    """
    if df_long.empty:
        return df_long

    keep_cols = [
        "date",
        "datatype",
        "value",
        "station",
        "attributes",
        "station_name",
        "station_id",
        "year"
    ]

    keep_cols = [c for c in keep_cols if c in df_long.columns]
    df_long = df_long[keep_cols].copy()

    # Parse date
    df_long["date"] = pd.to_datetime(df_long["date"]).dt.date

    # Ép value numeric
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")

    # Pivot weather variables
    df_wide = (
        df_long
        .pivot_table(
            index=["date", "station_name", "station_id", "year"],
            columns="datatype",
            values="value",
            aggfunc="first"
        )
        .reset_index()
    )

    df_wide.columns.name = None

    # Đảm bảo cột nào thiếu thì vẫn có
    for col in DATATYPES:
        if col not in df_wide.columns:
            df_wide[col] = pd.NA

    # Feature đơn giản dùng cho model sau này
    if "TMAX" in df_wide.columns and "TMIN" in df_wide.columns:
        df_wide["TAVG_est"] = (df_wide["TMAX"] + df_wide["TMIN"]) / 2

    if "TMAX" in df_wide.columns:
        df_wide["hot_day_30c"] = (df_wide["TMAX"] >= 30).astype(int)
        df_wide["hot_day_35c"] = (df_wide["TMAX"] >= 35).astype(int)

    if "TMIN" in df_wide.columns:
        df_wide["cold_day_0c"] = (df_wide["TMIN"] <= 0).astype(int)

    if "PRCP" in df_wide.columns:
        df_wide["rain_day"] = (df_wide["PRCP"] > 0).astype(int)
        df_wide["heavy_rain_day_25mm"] = (df_wide["PRCP"] >= 25).astype(int)

    if "SNOW" in df_wide.columns:
        df_wide["snow_day"] = (df_wide["SNOW"] > 0).astype(int)

    return df_wide


def save_station_year(station_name, station_id, year):
    long_out = OUT_DIR / f"noaa_{station_name}_{year}_long.csv.gz"
    wide_out = OUT_DIR / f"noaa_{station_name}_{year}_wide.csv.gz"

    if wide_out.exists() and long_out.exists():
        print(f"Skip existing: {station_name} {year}")
        return

    print(f"\n========== Downloading NOAA {station_name} {year} ==========")

    df_long = fetch_station_year_long(station_name, station_id, year)

    if df_long.empty:
        print(f"No data found: {station_name} {year}")
        return

    df_wide = convert_long_to_wide(df_long)

    df_long.to_csv(
        long_out,
        index=False,
        encoding="utf-8",
        compression="gzip"
    )

    df_wide.to_csv(
        wide_out,
        index=False,
        encoding="utf-8",
        compression="gzip"
    )

    print(f"Saved long -> {long_out}")
    print(f"Saved wide -> {wide_out}")
    print(f"Wide shape: {df_wide.shape}")

    del df_long
    del df_wide
    gc.collect()


def combine_all_wide():
    files = sorted(OUT_DIR.glob("noaa_*_wide.csv.gz"))

    if not files:
        print("No wide files found.")
        return

    dfs = []

    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["station_name", "date"])

    out_file = OUT_DIR / "noaa_nyc_daily_weather_2015_2025.csv.gz"
    combined.to_csv(
        out_file,
        index=False,
        encoding="utf-8",
        compression="gzip"
    )

    print(f"\nCombined weather saved -> {out_file}")
    print(f"Combined shape: {combined.shape}")
    print(combined.head())

    del dfs
    del combined
    gc.collect()


# =========================
# MAIN
# =========================

def main():
    for station_name, station_id in STATIONS.items():
        for year in tqdm(range(START_YEAR, END_YEAR + 1), desc=f"{station_name}"):
            save_station_year(station_name, station_id, year)
            time.sleep(SLEEP_SECONDS)

    combine_all_wide()

    print("\nDONE. NOAA weather data is ready.")


if __name__ == "__main__":
    main()