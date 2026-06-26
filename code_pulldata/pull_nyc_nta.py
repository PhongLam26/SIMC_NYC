from pathlib import Path
import requests
import pandas as pd
import geopandas as gpd


# =========================
# CONFIG
# =========================

OUT_DIR = Path("data/raw/nyc_nta")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID = "9nt8-h7nd"

# Official NYC Open Data geospatial export
GEOJSON_URL = (
    f"https://data.cityofnewyork.us/api/geospatial/{DATASET_ID}"
    f"?method=export&format=GeoJSON"
)

RAW_GEOJSON_PATH = OUT_DIR / "nyc_nta_2020_raw.geojson"
CLEAN_GEOJSON_PATH = OUT_DIR / "nyc_nta_2020_clean.geojson"
METADATA_CSV_PATH = OUT_DIR / "nyc_nta_2020_metadata.csv"


# =========================
# DOWNLOAD
# =========================

def download_nta():
    if RAW_GEOJSON_PATH.exists():
        print(f"Already exists: {RAW_GEOJSON_PATH}")
        return

    print("Downloading NYC NTA 2020 boundaries...")
    print(GEOJSON_URL)

    response = requests.get(GEOJSON_URL, timeout=180)
    response.raise_for_status()

    RAW_GEOJSON_PATH.write_bytes(response.content)

    print(f"Saved raw GeoJSON -> {RAW_GEOJSON_PATH}")


# =========================
# CLEAN + CHECK
# =========================

def clean_nta():
    print("Reading NTA GeoJSON...")

    nta = gpd.read_file(RAW_GEOJSON_PATH)

    # Chuẩn hóa tên cột
    nta.columns = [c.lower() for c in nta.columns]

    # Chuẩn CRS về EPSG:4326 để khớp lat/lon của 311
    if nta.crs is None:
        nta = nta.set_crs("EPSG:4326")
    else:
        nta = nta.to_crs("EPSG:4326")

    print("\nColumns:")
    print(nta.columns.tolist())

    print("\nCRS:")
    print(nta.crs)

    print("\nShape:")
    print(nta.shape)

    # Các cột cần giữ để join với 311 sau này
    keep_cols = [
        "nta2020",
        "ntaname",
        "boroname",
        "borocode",
        "cdta2020",
        "cdtaname",
        "ntatype",
        "geometry"
    ]

    existing_cols = [c for c in keep_cols if c in nta.columns]
    nta_clean = nta[existing_cols].copy()

    # Lưu clean GeoJSON
    nta_clean.to_file(CLEAN_GEOJSON_PATH, driver="GeoJSON")

    # Lưu metadata không geometry để dễ xem bằng Excel
    meta_cols = [c for c in nta_clean.columns if c != "geometry"]
    nta_clean[meta_cols].drop_duplicates().to_csv(
        METADATA_CSV_PATH,
        index=False,
        encoding="utf-8"
    )

    print(f"\nSaved clean GeoJSON -> {CLEAN_GEOJSON_PATH}")
    print(f"Saved metadata CSV -> {METADATA_CSV_PATH}")

    print("\nSample metadata:")
    print(nta_clean[meta_cols].head())

    print("\nBorough count:")
    if "boroname" in nta_clean.columns:
        print(nta_clean["boroname"].value_counts())


# =========================
# MAIN
# =========================

def main():
    download_nta()
    clean_nta()

    print("\nDONE. NTA boundary is ready.")
    print("Sau này dùng file này để spatial join với NYC 311:")
    print(CLEAN_GEOJSON_PATH)


if __name__ == "__main__":
    main()