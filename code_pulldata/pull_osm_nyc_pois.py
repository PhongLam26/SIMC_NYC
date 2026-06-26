import re
import gc
import json
import time
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

OUT_DIR = Path("data/raw/osm_poi")
PART_DIR = OUT_DIR / "parts"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PART_DIR.mkdir(parents=True, exist_ok=True)

# Nếu đã pull NTA rồi thì code sẽ tự join POI vào NTA.
NTA_GEOJSON_PATH = Path("data/raw/nyc_nta/nyc_nta_2020_clean.geojson")

COMBINED_OUT = OUT_DIR / "osm_nyc_pois_raw.csv.gz"
WITH_NTA_OUT = OUT_DIR / "osm_nyc_pois_with_nta.csv.gz"

# User-Agent quan trọng để tránh lỗi 429 của Overpass.
USER_AGENT = (
    "SIMC-NYC-Urban-Service-Demand-Research/1.0 "
    "(student research; contact: phonglam2599@gmail.com)"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
}

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
]

# Bounding box từng borough để query nhỏ hơn, tránh timeout.
# Format: south, west, north, east
NYC_BBOXES = {
    "manhattan": (40.6800, -74.0300, 40.8800, -73.9000),
    "bronx": (40.7800, -73.9400, 40.9200, -73.7500),
    "brooklyn": (40.5600, -74.0500, 40.7400, -73.8300),
    "queens": (40.4800, -73.9600, 40.8200, -73.7000),
    "staten_island": (40.4700, -74.2600, 40.6600, -74.0300),
}

# Các nhóm POI cần cho bài smart city.
QUERY_CONFIGS = [
    {
        "tag_key": "amenity",
        "values": [
            "restaurant", "cafe", "bar", "pub", "fast_food",
            "school", "university", "college", "kindergarten",
            "hospital", "clinic", "doctors", "pharmacy",
            "parking", "parking_entrance",
            "bank", "atm",
            "police", "fire_station",
            "library", "community_centre",
            "marketplace",
        ],
    },
    {
        "tag_key": "leisure",
        "values": [
            "park", "playground", "sports_centre",
            "fitness_centre", "garden", "recreation_ground",
        ],
    },
    {
        "tag_key": "shop",
        "values": [
            "supermarket", "convenience", "mall",
            "department_store", "bakery", "clothes",
            "laundry", "hardware", "furniture",
        ],
    },
    {
        "tag_key": "tourism",
        "values": [
            "hotel", "museum", "attraction", "gallery",
        ],
    },
    {
        "tag_key": "railway",
        "values": [
            "station", "subway_entrance", "tram_stop",
        ],
    },
    {
        "tag_key": "public_transport",
        "values": [
            "station", "platform", "stop_position",
        ],
    },
    {
        "tag_key": "highway",
        "values": [
            "bus_stop",
        ],
    },
]

# Overpass dễ rate-limit, nên không để quá thấp.
REQUEST_SLEEP = 15.0
MAX_RETRIES = 6
TIMEOUT = 240

POI_COLUMNS = [
    "osm_type",
    "osm_id",
    "lat",
    "lon",
    "source_bbox",
    "matched_tag_key",
    "matched_tag_value",
    "poi_group",
    "name",
    "amenity",
    "leisure",
    "shop",
    "tourism",
    "railway",
    "public_transport",
    "highway",
    "operator",
    "brand",
    "opening_hours",
    "addr_housenumber",
    "addr_street",
    "addr_city",
    "addr_postcode",
    "all_tags",
]


# ============================================================
# SEMANTIC GROUPING
# ============================================================

def infer_poi_group(tag_key, tag_value):
    """
    Gom POI thành nhóm ngữ nghĩa để fit với hướng semantic của bài.
    """
    if pd.isna(tag_value):
        return "unknown"

    tag_value = str(tag_value).lower()

    food = {
        "restaurant", "cafe", "bar", "pub", "fast_food",
        "marketplace", "bakery",
    }
    education = {
        "school", "university", "college", "kindergarten", "library",
    }
    health = {
        "hospital", "clinic", "doctors", "pharmacy",
    }
    transport = {
        "parking", "parking_entrance", "station", "subway_entrance",
        "tram_stop", "platform", "stop_position", "bus_stop",
    }
    public_service = {
        "police", "fire_station", "community_centre",
    }
    recreation = {
        "park", "playground", "sports_centre", "fitness_centre",
        "garden", "recreation_ground", "museum", "attraction", "gallery",
    }
    retail = {
        "supermarket", "convenience", "mall", "department_store",
        "clothes", "laundry", "hardware", "furniture",
    }
    finance = {
        "bank", "atm",
    }
    accommodation = {
        "hotel",
    }

    if tag_value in food:
        return "food_nightlife"
    if tag_value in education:
        return "education"
    if tag_value in health:
        return "healthcare"
    if tag_value in transport:
        return "transport_parking"
    if tag_value in public_service:
        return "public_service"
    if tag_value in recreation:
        return "recreation"
    if tag_value in retail:
        return "retail"
    if tag_value in finance:
        return "finance"
    if tag_value in accommodation:
        return "accommodation"

    return "other"


# ============================================================
# OVERPASS QUERY
# ============================================================

def build_regex(values):
    escaped = [re.escape(v) for v in values]
    return "^(" + "|".join(escaped) + ")$"


def build_overpass_query(tag_key, values, bbox):
    south, west, north, east = bbox
    value_regex = build_regex(values)

    query = f"""
[out:json][timeout:{TIMEOUT}];
(
  node["{tag_key}"~"{value_regex}"]({south},{west},{north},{east});
  way["{tag_key}"~"{value_regex}"]({south},{west},{north},{east});
  relation["{tag_key}"~"{value_regex}"]({south},{west},{north},{east});
);
out center tags;
"""
    return query.strip()


def call_overpass(query):
    last_error = None

    for attempt in range(MAX_RETRIES):
        endpoint = OVERPASS_ENDPOINTS[attempt % len(OVERPASS_ENDPOINTS)]

        try:
            response = requests.post(
                endpoint,
                data={"data": query},
                headers=HEADERS,
                timeout=TIMEOUT + 60,
            )

            if response.status_code == 200:
                return response.json()

            last_error = f"HTTP {response.status_code}: {response.text[:700]}"
            print(last_error)

        except requests.RequestException as e:
            last_error = str(e)
            print(f"Request error: {last_error}")

        wait = min(180, (2 ** attempt) * 10)
        print(f"Retry after {wait}s...")
        time.sleep(wait)

    raise RuntimeError(f"Overpass request failed. Last error: {last_error}")


def parse_osm_elements(data, source_bbox_name, tag_key):
    rows = []

    for el in data.get("elements", []):
        tags = el.get("tags", {})

        lat = el.get("lat")
        lon = el.get("lon")

        # Way/relation thường không có lat/lon trực tiếp, nhưng có center.
        if lat is None or lon is None:
            center = el.get("center", {})
            lat = center.get("lat")
            lon = center.get("lon")

        if lat is None or lon is None:
            continue

        matched_tag_value = tags.get(tag_key)

        row = {
            "osm_type": el.get("type"),
            "osm_id": el.get("id"),
            "lat": lat,
            "lon": lon,
            "source_bbox": source_bbox_name,
            "matched_tag_key": tag_key,
            "matched_tag_value": matched_tag_value,
            "poi_group": infer_poi_group(tag_key, matched_tag_value),
            "name": tags.get("name"),
            "amenity": tags.get("amenity"),
            "leisure": tags.get("leisure"),
            "shop": tags.get("shop"),
            "tourism": tags.get("tourism"),
            "railway": tags.get("railway"),
            "public_transport": tags.get("public_transport"),
            "highway": tags.get("highway"),
            "operator": tags.get("operator"),
            "brand": tags.get("brand"),
            "opening_hours": tags.get("opening_hours"),
            "addr_housenumber": tags.get("addr:housenumber"),
            "addr_street": tags.get("addr:street"),
            "addr_city": tags.get("addr:city"),
            "addr_postcode": tags.get("addr:postcode"),
            "all_tags": json.dumps(tags, ensure_ascii=False),
        }

        rows.append(row)

    return rows


# ============================================================
# DOWNLOAD PARTS
# ============================================================

def download_one_query(bbox_name, bbox, tag_key, values):
    out_file = PART_DIR / f"osm_{bbox_name}_{tag_key}.csv.gz"

    if out_file.exists():
        print(f"Skip existing: {out_file.name}")
        return

    print(f"\nDownloading OSM POI | bbox={bbox_name} | tag={tag_key}")

    query = build_overpass_query(tag_key, values, bbox)
    data = call_overpass(query)
    rows = parse_osm_elements(data, bbox_name, tag_key)

    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=POI_COLUMNS)

    # Đảm bảo đủ cột dù query không ra data.
    for col in POI_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[POI_COLUMNS].copy()

    df.to_csv(
        out_file,
        index=False,
        compression="gzip",
        encoding="utf-8"
    )

    print(f"Saved {out_file.name} | rows={len(df):,}")

    del df
    del rows
    del data
    gc.collect()


def download_all_parts():
    tasks = []

    for bbox_name, bbox in NYC_BBOXES.items():
        for config in QUERY_CONFIGS:
            tasks.append((bbox_name, bbox, config["tag_key"], config["values"]))

    print(f"Total Overpass tasks: {len(tasks)}")
    print(f"User-Agent: {USER_AGENT}")

    for bbox_name, bbox, tag_key, values in tqdm(tasks, desc="Pulling OSM POIs"):
        download_one_query(bbox_name, bbox, tag_key, values)
        time.sleep(REQUEST_SLEEP)


# ============================================================
# COMBINE + DEDUP
# ============================================================

def combine_parts():
    files = sorted(PART_DIR.glob("osm_*.csv.gz"))

    if not files:
        raise RuntimeError("No OSM part files found.")

    dfs = []

    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except pd.errors.EmptyDataError:
            print(f"Skip empty file: {f.name}")
            continue

        if df.empty:
            continue

        dfs.append(df)

    if not dfs:
        raise RuntimeError("All OSM part files are empty.")

    combined = pd.concat(dfs, ignore_index=True)

    combined["lat"] = pd.to_numeric(combined["lat"], errors="coerce")
    combined["lon"] = pd.to_numeric(combined["lon"], errors="coerce")

    combined = combined.dropna(subset=["lat", "lon"])

    # Lọc bbox chung của NYC để loại điểm lỗi.
    combined = combined[
        (combined["lat"].between(40.45, 40.95)) &
        (combined["lon"].between(-74.30, -73.65))
    ].copy()

    before = len(combined)

    combined = combined.drop_duplicates(
        subset=["osm_type", "osm_id"],
        keep="first"
    ).copy()

    after = len(combined)

    combined = combined.sort_values(
        ["poi_group", "matched_tag_key", "matched_tag_value", "osm_type", "osm_id"]
    )

    combined.to_csv(
        COMBINED_OUT,
        index=False,
        compression="gzip",
        encoding="utf-8"
    )

    print(f"\nCombined saved -> {COMBINED_OUT}")
    print(f"Rows before dedup: {before:,}")
    print(f"Rows after dedup : {after:,}")

    print("\nTop POI values:")
    print(combined["matched_tag_value"].value_counts().head(30))

    print("\nTop POI groups:")
    print(combined["poi_group"].value_counts())

    del dfs
    gc.collect()

    return combined


# ============================================================
# OPTIONAL: JOIN POI TO NTA
# ============================================================

def assign_nta_to_pois(pois_df):
    if not NTA_GEOJSON_PATH.exists():
        print(f"\nNTA file not found, skip NTA join: {NTA_GEOJSON_PATH}")
        print("Bạn có thể chạy pull_nyc_nta.py trước, rồi chạy lại file này.")
        return

    print("\nJoining OSM POIs to NYC NTA...")

    import geopandas as gpd

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
        pois_df.copy(),
        geometry=gpd.points_from_xy(pois_df["lon"], pois_df["lat"]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(
        gdf,
        nta,
        how="left",
        predicate="within"
    )

    joined = joined.drop(columns=["geometry", "index_right"], errors="ignore")

    matched_rate = joined["nta2020"].notna().mean() if "nta2020" in joined.columns else 0

    joined.to_csv(
        WITH_NTA_OUT,
        index=False,
        compression="gzip",
        encoding="utf-8"
    )

    print(f"Saved POI with NTA -> {WITH_NTA_OUT}")
    print(f"NTA matched rate: {matched_rate:.2%}")

    print("\nTop NTA by POI count:")
    if "ntaname" in joined.columns:
        print(joined["ntaname"].value_counts().head(20))

    print("\nTop POI group by NTA sample:")
    if "ntaname" in joined.columns and "poi_group" in joined.columns:
        print(
            joined.groupby(["ntaname", "poi_group"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(30)
        )

    del nta
    del gdf
    del joined
    gc.collect()


# ============================================================
# MAIN
# ============================================================

def main():
    print("========== PULL OSM NYC POIS ==========")
    print("Do not run multiple Overpass download scripts at the same time.")
    print(f"Output folder: {OUT_DIR}")

    download_all_parts()

    pois_df = combine_parts()

    assign_nta_to_pois(pois_df)

    print("\nDONE. OSM NYC POI data is ready.")
    print(f"Raw combined file: {COMBINED_OUT}")
    print(f"With NTA file    : {WITH_NTA_OUT}")


if __name__ == "__main__":
    main()