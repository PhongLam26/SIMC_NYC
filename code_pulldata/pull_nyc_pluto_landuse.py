import gc
import time
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

DATASET_ID = "64uk-42ks"
BASE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"

OUT_DIR = Path("data/raw/nyc_pluto")
PART_DIR = OUT_DIR / "parts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PART_DIR.mkdir(parents=True, exist_ok=True)

RAW_COMBINED_OUT = OUT_DIR / "nyc_pluto_raw.csv.gz"
CLEAN_OUT = OUT_DIR / "nyc_pluto_clean.csv.gz"

# Nếu đã pull NTA rồi thì code sẽ tự gắn PLUTO vào NTA
NTA_GEOJSON_PATH = Path("data/raw/nyc_nta/nyc_nta_2020_clean.geojson")
WITH_NTA_OUT = OUT_DIR / "nyc_pluto_with_nta.csv.gz"
NTA_FEATURE_OUT = OUT_DIR / "nyc_pluto_nta_landuse_features.csv"

LIMIT = 50000
SLEEP_SECONDS = 0.5
MAX_RETRIES = 5

# Các cột cần cho bài. Nếu API thiếu cột nào, code sẽ tự bỏ qua.
REQUESTED_COLUMNS = [
    "bbl",
    "borough",
    "block",
    "lot",
    "address",
    "zipcode",

    "landuse",
    "bldgclass",
    "zonedist1",
    "zonedist2",
    "overlay1",
    "overlay2",
    "spdist1",

    "lotarea",
    "bldgarea",
    "resarea",
    "comarea",
    "officearea",
    "retailarea",
    "garagearea",
    "strgearea",
    "factryarea",
    "otherarea",

    "numbldgs",
    "numfloors",
    "unitsres",
    "unitstotal",

    "yearbuilt",
    "yearalter1",
    "yearalter2",

    "builtfar",
    "residfar",
    "commfar",
    "facilfar",

    "latitude",
    "longitude",

    "version"
]


# ============================================================
# API UTILS
# ============================================================

def request_with_retry(params):
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(BASE_URL, params=params, timeout=120)

            if response.status_code == 200:
                return response.json()

            print(f"HTTP {response.status_code}: {response.text[:500]}")

        except requests.RequestException as e:
            print(f"Request error: {e}")

        wait = 2 ** attempt
        print(f"Retry after {wait}s...")
        time.sleep(wait)

    raise RuntimeError("Request failed after max retries.")


def get_available_columns():
    """
    Lấy 1 dòng sample để biết API hiện có cột nào.
    """
    params = {
        "$limit": 1
    }

    data = request_with_retry(params)

    if not data:
        raise RuntimeError("No data returned from PLUTO API.")

    available = list(data[0].keys())
    available = [c.lower() for c in available]

    selected = [c for c in REQUESTED_COLUMNS if c in available]

    print("Available selected columns:")
    print(selected)

    missing = [c for c in REQUESTED_COLUMNS if c not in available]
    if missing:
        print("\nMissing columns ignored:")
        print(missing)

    return selected


# ============================================================
# DOWNLOAD PARTS
# ============================================================

def download_parts(selected_columns):
    offset = 0
    part = 0
    total_rows = 0

    select_clause = ",".join(selected_columns)

    while True:
        out_file = PART_DIR / f"nyc_pluto_part{part:03d}.csv.gz"

        if out_file.exists():
            print(f"Skip existing: {out_file.name}")
            try:
                existing_rows = len(pd.read_csv(out_file))
            except Exception:
                existing_rows = LIMIT

            total_rows += existing_rows
            offset += LIMIT
            part += 1
            continue

        params = {
            "$select": select_clause,
            "$limit": LIMIT,
            "$offset": offset,
            "$order": "bbl"
        }

        print(f"\nDownloading PLUTO part {part:03d} | offset={offset:,}")

        data = request_with_retry(params)

        if not data:
            print("No more data.")
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
            f"Saved {out_file.name} | rows={rows:,} | total={total_rows:,}"
        )

        del df
        del data
        gc.collect()

        if rows < LIMIT:
            break

        offset += LIMIT
        part += 1

        time.sleep(SLEEP_SECONDS)

    print(f"\nFinished downloading PLUTO. Total rows: {total_rows:,}")


# ============================================================
# CLEANING
# ============================================================

def map_borough(value):
    if pd.isna(value):
        return "unknown"

    value = str(value).strip().upper()

    mapping = {
        "1": "Manhattan",
        "2": "Bronx",
        "3": "Brooklyn",
        "4": "Queens",
        "5": "Staten Island",
        "MN": "Manhattan",
        "BX": "Bronx",
        "BK": "Brooklyn",
        "QN": "Queens",
        "SI": "Staten Island",
        "MANHATTAN": "Manhattan",
        "BRONX": "Bronx",
        "BROOKLYN": "Brooklyn",
        "QUEENS": "Queens",
        "STATEN ISLAND": "Staten Island",
    }

    return mapping.get(value, value.title())


def normalize_landuse_code(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    # Có thể là "1", "01", "1.0"
    try:
        value_float = float(value)
        value_int = int(value_float)
        return str(value_int)
    except Exception:
        return value


def map_land_use_type(value):
    code = normalize_landuse_code(value)

    mapping = {
        "1": "one_two_family_residential",
        "2": "multi_family_walkup_residential",
        "3": "multi_family_elevator_residential",
        "4": "mixed_residential_commercial",
        "5": "commercial_office",
        "6": "industrial_manufacturing",
        "7": "transportation_utility",
        "8": "public_facility_institution",
        "9": "open_space_recreation",
        "10": "parking",
        "11": "vacant_land",
    }

    return mapping.get(code, "unknown")


def map_land_use_group(value):
    code = normalize_landuse_code(value)

    if code in {"1", "2", "3"}:
        return "residential"
    if code == "4":
        return "mixed_use"
    if code == "5":
        return "commercial"
    if code == "6":
        return "industrial"
    if code == "7":
        return "transportation_utility"
    if code == "8":
        return "public_facility"
    if code == "9":
        return "open_space"
    if code == "10":
        return "parking"
    if code == "11":
        return "vacant"

    return "unknown"


def combine_parts():
    files = sorted(PART_DIR.glob("nyc_pluto_part*.csv.gz"))

    if not files:
        raise RuntimeError("No PLUTO part files found.")

    dfs = []

    for f in tqdm(files, desc="Combining PLUTO parts"):
        df = pd.read_csv(f, low_memory=False)
        if not df.empty:
            dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    combined.to_csv(
        RAW_COMBINED_OUT,
        index=False,
        encoding="utf-8",
        compression="gzip"
    )

    print(f"\nSaved raw combined -> {RAW_COMBINED_OUT}")
    print(f"Raw shape: {combined.shape}")

    del dfs
    gc.collect()

    return combined


def clean_pluto(df):
    df.columns = [c.lower() for c in df.columns]

    # Numeric columns
    numeric_cols = [
        "lotarea",
        "bldgarea",
        "resarea",
        "comarea",
        "officearea",
        "retailarea",
        "garagearea",
        "strgearea",
        "factryarea",
        "otherarea",
        "numbldgs",
        "numfloors",
        "unitsres",
        "unitstotal",
        "yearbuilt",
        "yearalter1",
        "yearalter2",
        "builtfar",
        "residfar",
        "commfar",
        "facilfar",
        "latitude",
        "longitude",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "borough" in df.columns:
        df["borough_name"] = df["borough"].apply(map_borough)

    if "landuse" in df.columns:
        df["landuse_code"] = df["landuse"].apply(normalize_landuse_code)
        df["land_use_type"] = df["landuse"].apply(map_land_use_type)
        df["land_use_group"] = df["landuse"].apply(map_land_use_group)
    else:
        df["landuse_code"] = pd.NA
        df["land_use_type"] = "unknown"
        df["land_use_group"] = "unknown"

    # Một số tỷ lệ hữu ích
    if "bldgarea" in df.columns and "lotarea" in df.columns:
        df["building_to_lot_area_ratio"] = df["bldgarea"] / df["lotarea"].replace(0, pd.NA)

    if "resarea" in df.columns and "bldgarea" in df.columns:
        df["residential_area_ratio"] = df["resarea"] / df["bldgarea"].replace(0, pd.NA)

    if "comarea" in df.columns and "bldgarea" in df.columns:
        df["commercial_area_ratio"] = df["comarea"] / df["bldgarea"].replace(0, pd.NA)

    if "officearea" in df.columns and "bldgarea" in df.columns:
        df["office_area_ratio"] = df["officearea"] / df["bldgarea"].replace(0, pd.NA)

    if "retailarea" in df.columns and "bldgarea" in df.columns:
        df["retail_area_ratio"] = df["retailarea"] / df["bldgarea"].replace(0, pd.NA)

    # Lọc tọa độ lỗi nhưng không bắt buộc bỏ hết dòng thiếu tọa độ
    if "latitude" in df.columns and "longitude" in df.columns:
        valid_coord = (
            df["latitude"].between(40.45, 40.95) &
            df["longitude"].between(-74.30, -73.65)
        )

        df["has_valid_coord"] = valid_coord.astype(int)

    df.to_csv(
        CLEAN_OUT,
        index=False,
        encoding="utf-8",
        compression="gzip"
    )

    print(f"\nSaved clean PLUTO -> {CLEAN_OUT}")
    print(f"Clean shape: {df.shape}")

    print("\nLand use group count:")
    print(df["land_use_group"].value_counts(dropna=False))

    print("\nBorough count:")
    if "borough_name" in df.columns:
        print(df["borough_name"].value_counts(dropna=False))

    return df


# ============================================================
# OPTIONAL: JOIN PLUTO TO NTA
# ============================================================

def assign_nta_to_pluto(df):
    if not NTA_GEOJSON_PATH.exists():
        print(f"\nNTA file not found, skip NTA join: {NTA_GEOJSON_PATH}")
        print("Bạn có thể chạy pull_nyc_nta.py trước rồi chạy lại file này.")
        return None

    if "latitude" not in df.columns or "longitude" not in df.columns:
        print("No latitude/longitude columns, skip NTA join.")
        return None

    print("\nJoining PLUTO lots to NYC NTA...")

    import geopandas as gpd

    valid = df[
        df["latitude"].between(40.45, 40.95) &
        df["longitude"].between(-74.30, -73.65)
    ].copy()

    nta = gpd.read_file(NTA_GEOJSON_PATH)
    nta.columns = [c.lower() for c in nta.columns]

    if nta.crs is None:
        nta = nta.set_crs("EPSG:4326")
    else:
        nta = nta.to_crs("EPSG:4326")

    keep_cols = [
        "nta2020",
        "ntaname",
        "boroname",
        "borocode",
        "cdta2020",
        "cdtaname",
        "ntatype",
        "geometry",
    ]

    keep_cols = [c for c in keep_cols if c in nta.columns]
    nta = nta[keep_cols].copy()

    gdf = gpd.GeoDataFrame(
        valid,
        geometry=gpd.points_from_xy(valid["longitude"], valid["latitude"]),
        crs="EPSG:4326"
    )

    joined = gpd.sjoin(
        gdf,
        nta,
        how="left",
        predicate="within"
    )

    joined = joined.drop(columns=["geometry", "index_right"], errors="ignore")

    joined.to_csv(
        WITH_NTA_OUT,
        index=False,
        encoding="utf-8",
        compression="gzip"
    )

    matched_rate = joined["nta2020"].notna().mean() if "nta2020" in joined.columns else 0

    print(f"Saved PLUTO with NTA -> {WITH_NTA_OUT}")
    print(f"NTA matched rate: {matched_rate:.2%}")

    del nta
    del gdf
    gc.collect()

    return joined


# ============================================================
# CREATE NTA LAND USE FEATURES
# ============================================================

def create_nta_landuse_features(pluto_nta):
    if pluto_nta is None or pluto_nta.empty:
        print("No PLUTO-NTA data, skip feature creation.")
        return

    if "nta2020" not in pluto_nta.columns:
        print("No nta2020 column, skip feature creation.")
        return

    print("\nCreating NTA-level land-use features...")

    df = pluto_nta.copy()

    numeric_cols = [
        "lotarea",
        "bldgarea",
        "resarea",
        "comarea",
        "officearea",
        "retailarea",
        "garagearea",
        "strgearea",
        "factryarea",
        "otherarea",
        "numbldgs",
        "numfloors",
        "unitsres",
        "unitstotal",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    group_keys = ["nta2020"]

    meta_cols = [c for c in ["ntaname", "boroname"] if c in df.columns]

    # Base aggregates
    agg_dict = {
        "bbl": "count",
    }

    sum_cols = [
        "lotarea",
        "bldgarea",
        "resarea",
        "comarea",
        "officearea",
        "retailarea",
        "garagearea",
        "strgearea",
        "factryarea",
        "otherarea",
        "numbldgs",
        "unitsres",
        "unitstotal",
    ]

    for col in sum_cols:
        if col in df.columns:
            agg_dict[col] = "sum"

    if "numfloors" in df.columns:
        agg_dict["numfloors"] = "mean"

    features = (
        df
        .groupby(group_keys)
        .agg(agg_dict)
        .reset_index()
    )

    rename_map = {
        "bbl": "pluto_lot_count",
        "lotarea": "total_lot_area",
        "bldgarea": "total_building_area",
        "resarea": "total_residential_area",
        "comarea": "total_commercial_area",
        "officearea": "total_office_area",
        "retailarea": "total_retail_area",
        "garagearea": "total_garage_area",
        "strgearea": "total_storage_area",
        "factryarea": "total_factory_area",
        "otherarea": "total_other_area",
        "numbldgs": "total_buildings",
        "unitsres": "total_residential_units",
        "unitstotal": "total_units",
        "numfloors": "avg_num_floors",
    }

    features = features.rename(columns=rename_map)

    # Add metadata
    if meta_cols:
        meta = df[["nta2020"] + meta_cols].drop_duplicates(subset=["nta2020"])
        features = features.merge(meta, on="nta2020", how="left")

    # Ratios
    if "total_building_area" in features.columns:
        denom = features["total_building_area"].replace(0, pd.NA)

        if "total_residential_area" in features.columns:
            features["residential_area_ratio"] = features["total_residential_area"] / denom

        if "total_commercial_area" in features.columns:
            features["commercial_area_ratio"] = features["total_commercial_area"] / denom

        if "total_office_area" in features.columns:
            features["office_area_ratio"] = features["total_office_area"] / denom

        if "total_retail_area" in features.columns:
            features["retail_area_ratio"] = features["total_retail_area"] / denom

    if "total_lot_area" in features.columns:
        denom_lot = features["total_lot_area"].replace(0, pd.NA)

        if "total_building_area" in features.columns:
            features["building_area_to_lot_area_ratio"] = features["total_building_area"] / denom_lot

    # Pivot land-use group count
    if "land_use_group" in df.columns:
        count_pivot = (
            df
            .pivot_table(
                index="nta2020",
                columns="land_use_group",
                values="bbl",
                aggfunc="count",
                fill_value=0
            )
            .reset_index()
        )

        count_pivot.columns = [
            "nta2020" if c == "nta2020" else f"lot_count_{c}"
            for c in count_pivot.columns
        ]

        features = features.merge(count_pivot, on="nta2020", how="left")

    # Pivot land-use lot area
    if "land_use_group" in df.columns and "lotarea" in df.columns:
        area_pivot = (
            df
            .pivot_table(
                index="nta2020",
                columns="land_use_group",
                values="lotarea",
                aggfunc="sum",
                fill_value=0
            )
            .reset_index()
        )

        area_pivot.columns = [
            "nta2020" if c == "nta2020" else f"lot_area_{c}"
            for c in area_pivot.columns
        ]

        features = features.merge(area_pivot, on="nta2020", how="left")

        # area ratios by land-use group
        if "total_lot_area" in features.columns:
            for col in features.columns:
                if col.startswith("lot_area_"):
                    features[col.replace("lot_area_", "lot_area_ratio_")] = (
                        features[col] / features["total_lot_area"].replace(0, pd.NA)
                    )

    features.to_csv(
        NTA_FEATURE_OUT,
        index=False,
        encoding="utf-8"
    )

    print(f"Saved NTA land-use features -> {NTA_FEATURE_OUT}")
    print(f"Feature shape: {features.shape}")

    print("\nSample:")
    print(features.head())


# ============================================================
# MAIN
# ============================================================

def main():
    print("========== PULL NYC PLUTO / LAND USE ==========")
    print(f"Dataset ID: {DATASET_ID}")
    print(f"Output folder: {OUT_DIR}")

    selected_columns = get_available_columns()

    download_parts(selected_columns)

    raw_df = combine_parts()

    clean_df = clean_pluto(raw_df)

    del raw_df
    gc.collect()

    pluto_nta = assign_nta_to_pluto(clean_df)

    create_nta_landuse_features(pluto_nta)

    print("\nDONE. NYC PLUTO / Land Use data is ready.")
    print(f"Raw combined : {RAW_COMBINED_OUT}")
    print(f"Clean file   : {CLEAN_OUT}")
    print(f"With NTA     : {WITH_NTA_OUT}")
    print(f"NTA features : {NTA_FEATURE_OUT}")


if __name__ == "__main__":
    main()