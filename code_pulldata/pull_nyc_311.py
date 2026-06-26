import os
import gc
import time
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

import requests
import pandas as pd


# =========================
# CONFIG
# =========================

DATASET_ID = "erm2-nwe9"  # NYC 311 Service Requests 2010-present
BASE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"

START_DATE = "2021-01-01"
END_DATE = "2025-12-31"

OUT_DIR = Path("data/raw/nyc_311")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LIMIT = 50000
SLEEP_SECONDS = 0.5

# Socrata token không bắt buộc, nhưng có thì đỡ bị limit hơn
APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN", "")

COLUMNS = [
    "unique_key",
    "created_date",
    "closed_date",
    "complaint_type",
    "descriptor",
    "agency",
    "agency_name",
    "borough",
    "incident_zip",
    "status",
    "latitude",
    "longitude",
    "location_type",
    "community_board",
    "open_data_channel_type"
]


# =========================
# DATE UTILS
# =========================

def to_datetime(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d")


def generate_month_ranges(start_date, end_date):
    """
    Tạo từng khoảng tháng: [month_start, next_month_start)
    """
    start = to_datetime(start_date)
    end = to_datetime(end_date) + relativedelta(days=1)

    current = start

    while current < end:
        next_month = current.replace(day=1) + relativedelta(months=1)
        month_end = min(next_month, end)

        yield current, month_end

        current = month_end


# =========================
# API REQUEST
# =========================

def request_data(session, params, max_retries=5):
    for attempt in range(max_retries):
        try:
            res = session.get(BASE_URL, params=params, timeout=90)

            if res.status_code == 200:
                return res.json()

            print(f"HTTP {res.status_code}: {res.text[:300]}")

        except Exception as e:
            print(f"Request error: {e}")

        wait_time = 2 ** attempt
        print(f"Retry sau {wait_time}s...")
        time.sleep(wait_time)

    raise RuntimeError("Request failed after max retries.")


def build_where(start_dt, end_dt):
    start_s = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
    end_s = end_dt.strftime("%Y-%m-%dT%H:%M:%S")

    where = (
        f"created_date >= '{start_s}' "
        f"AND created_date < '{end_s}' "
        f"AND latitude IS NOT NULL "
        f"AND longitude IS NOT NULL"
    )

    return where


# =========================
# DOWNLOAD ONE MONTH
# =========================

def download_one_month(start_dt, end_dt):
    month_label = start_dt.strftime("%Y_%m")
    print(f"\n========== Downloading {month_label} ==========")

    headers = {}
    if APP_TOKEN:
        headers["X-App-Token"] = APP_TOKEN

    session = requests.Session()
    session.headers.update(headers)

    offset = 0
    part = 0
    total_rows = 0

    where_clause = build_where(start_dt, end_dt)

    while True:
        out_file = OUT_DIR / f"nyc_311_{month_label}_part{part:03d}.csv.gz"

        # Nếu chạy bị dừng giữa chừng, file nào có rồi thì bỏ qua
        if out_file.exists():
            print(f"Skip existing file: {out_file.name}")
            offset += LIMIT
            part += 1
            continue

        params = {
            "$select": ",".join(COLUMNS),
            "$where": where_clause,
            "$order": "created_date, unique_key",
            "$limit": LIMIT,
            "$offset": offset
        }

        data = request_data(session, params)

        # Hết data tháng đó
        if not data:
            print(f"Done {month_label}. Total rows: {total_rows:,}")
            break

        df = pd.DataFrame(data)

        df.to_csv(
            out_file,
            index=False,
            encoding="utf-8",
            compression="gzip"
        )

        rows = len(df)
        total_rows += rows

        print(
            f"Saved {out_file.name} | rows: {rows:,} | "
            f"offset: {offset:,} | total month rows: {total_rows:,}"
        )

        # =========================
        # CLEAR RAM SAU MỖI PART
        # =========================
        del df
        del data
        gc.collect()

        # Nếu rows < LIMIT nghĩa là part cuối của tháng
        if rows < LIMIT:
            print(f"Finished month {month_label}. Total rows: {total_rows:,}")
            break

        offset += LIMIT
        part += 1

        time.sleep(SLEEP_SECONDS)

    session.close()

    return total_rows


# =========================
# MAIN
# =========================

def main():
    summary = []

    for start_dt, end_dt in generate_month_ranges(START_DATE, END_DATE):
        total_rows = download_one_month(start_dt, end_dt)

        summary.append({
            "month": start_dt.strftime("%Y-%m"),
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_exclusive": end_dt.strftime("%Y-%m-%d"),
            "rows": total_rows
        })

        # Clear thêm sau mỗi tháng
        gc.collect()

    summary_df = pd.DataFrame(summary)
    summary_path = OUT_DIR / "download_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    print("\n========== ALL DONE ==========")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()