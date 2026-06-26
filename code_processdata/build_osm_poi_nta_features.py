#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_osm_poi_nta_features.py

Step 7 for the SIMC NYC pipeline.

Purpose
-------
Build static NTA-level urban-context features from OpenStreetMap POIs that have
already been assigned to NYC 2020 NTA boundaries.

Input
-----
Default:
    data/raw/osm_poi/osm_nyc_pois_with_nta.csv.gz

Expected minimum column:
    nta2020

Common optional columns supported automatically:
    ntaname, boroname, borocode, cdta2020, cdtaname
    lat/lon or latitude/longitude
    name
    amenity, shop, tourism, leisure, office, healthcare, public_transport,
    railway, highway, natural, historic, craft, emergency, sport, landuse, etc.

Output
------
Default:
    data/processed/feature_tables/osm_poi_nta_features.csv

Merge key for later final dataset:
    nta2020

Design notes
------------
1. The output is one row per NTA.
2. POIs are mapped into paper-friendly semantic categories such as food/drink,
   retail, education, healthcare, transport, public service, recreation/green,
   culture/tourism, etc.
3. The script creates counts, shares, diversity/mixed-use metrics, tag-family
   counts, composite urban-context indices, and optional area-normalized density
   metrics if the NTA GeoJSON can be read with geopandas.
4. The script writes multiple diagnostic files for reproducibility and paper
   reporting.

Run from project root
---------------------
    .\\.venv\\Scripts\\python.exe .\\code_processdata\\build_osm_poi_nta_features.py --overwrite

If geopandas is unavailable, the script still runs, but area/density features
will be missing/NaN unless an external area table is provided in later steps.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_NAME = "build_osm_poi_nta_features"

# Paper-friendly semantic POI categories. Keep these stable after Step 7.
SEMANTIC_CATEGORY_LABELS: Dict[str, str] = {
    "food_drink": "Food, Drink & Nightlife Support",
    "retail": "Retail & Consumer Services",
    "education": "Education & Childcare",
    "healthcare": "Healthcare & Social Care",
    "transport": "Transport & Mobility Infrastructure",
    "recreation_green": "Recreation, Parks & Green Space",
    "culture_tourism": "Culture, Tourism & Visitor Attractions",
    "public_service": "Public Services & Civic Facilities",
    "emergency_safety": "Emergency & Safety Services",
    "finance": "Finance & Business Services",
    "accommodation": "Accommodation & Lodging",
    "religion": "Religious & Community Worship",
    "office_work": "Office & Employment Activity",
    "industrial_logistics": "Industrial, Utility & Logistics",
    "other": "Other / Unclassified POI",
}

SEMANTIC_CATEGORIES: List[str] = list(SEMANTIC_CATEGORY_LABELS.keys())

# Ordered rules. First match wins. Patterns are applied to normalized tag values
# only, not to POI names, to reduce false semantic precision.
CATEGORY_RULES: List[Tuple[str, str]] = [
    (
        "healthcare",
        r"(?:hospital|clinic|doctors|doctor|dentist|pharmacy|healthcare|nursing_home|social_facility|veterinary|optometrist|physiotherapist)",
    ),
    (
        "education",
        r"(?:school|university|college|kindergarten|childcare|daycare|language_school|music_school|driving_school|library)",
    ),
    (
        "transport",
        r"(?:bus_station|bus_stop|subway|subway_entrance|train_station|railway|station|ferry_terminal|taxi|parking|bicycle_parking|bicycle_rental|car_rental|car_sharing|fuel|charging_station|public_transport|aeroway|airport|tram_stop|pier)",
    ),
    (
        "emergency_safety",
        r"(?:police|fire_station|emergency|ambulance_station|rescue_station|siren|fire_hydrant|lifeguard)",
    ),
    (
        "public_service",
        r"(?:courthouse|townhall|city_hall|post_office|government|public_building|community_centre|community_center|shelter|toilets|waste_basket|recycling|waste_disposal|drinking_water|bench|prison|ranger_station)",
    ),
    (
        "food_drink",
        r"(?:restaurant|cafe|coffee|fast_food|food_court|bar|pub|biergarten|ice_cream|juice_bar|bakery|deli)",
    ),
    (
        "retail",
        r"(?:supermarket|convenience|mall|department_store|clothes|shoes|jewelry|jewellery|florist|hardware|furniture|electronics|mobile_phone|computer|books|bookshop|butcher|greengrocer|marketplace|chemist|beauty|hairdresser|laundry|dry_cleaning|car_repair|bicycle_shop|pet|gift|variety_store|retail|shop)",
    ),
    (
        "finance",
        r"(?:bank|atm|money_transfer|bureau_de_change|financial|insurance)",
    ),
    (
        "accommodation",
        r"(?:hotel|motel|hostel|guest_house|apartment|apartments|chalet|camp_site|caravan_site)",
    ),
    (
        "recreation_green",
        r"(?:park|garden|playground|sports_centre|sport|pitch|swimming_pool|fitness_centre|fitness_station|stadium|track|recreation_ground|dog_park|nature_reserve|picnic_site|leisure)",
    ),
    (
        "culture_tourism",
        r"(?:museum|gallery|artwork|attraction|viewpoint|zoo|aquarium|theme_park|theatre|theater|cinema|arts_centre|tourism|historic|monument|memorial|information)",
    ),
    (
        "religion",
        r"(?:place_of_worship|church|mosque|synagogue|temple|shrine|religion|chapel)",
    ),
    (
        "office_work",
        r"(?:office|coworking|company|commercial|business|employment_agency|estate_agent|lawyer|accountant|consulting)",
    ),
    (
        "industrial_logistics",
        r"(?:industrial|warehouse|storage|logistics|factory|works|utility|power|substation|water_works|wastewater_plant|depot)",
    ),
]

# Candidate columns are normalized before matching. These are OSM tag columns or
# broad type columns commonly produced by OSMnx / Overpass / custom scripts.
TAG_COLUMN_CANDIDATES: List[str] = [
    "amenity",
    "shop",
    "tourism",
    "leisure",
    "office",
    "healthcare",
    "public_transport",
    "railway",
    "highway",
    "natural",
    "historic",
    "craft",
    "emergency",
    "sport",
    "landuse",
    "building",
    "man_made",
    "aeroway",
    "waterway",
    "religion",
    "category",
    "subcategory",
    "type",
    "poi_type",
    "osm_type",
    "osm_category",
    "fclass",
    "class",
]

TAG_FAMILIES_FOR_COUNTS: List[str] = [
    "amenity",
    "shop",
    "tourism",
    "leisure",
    "office",
    "healthcare",
    "public_transport",
    "railway",
    "highway",
    "natural",
    "historic",
    "craft",
    "emergency",
    "sport",
    "landuse",
    "building",
    "man_made",
    "aeroway",
    "waterway",
]

METADATA_CANDIDATES: Dict[str, List[str]] = {
    "nta2020": ["nta2020", "nta_2020", "nta", "nta_code", "ntacode"],
    "ntaname": ["ntaname", "nta_name", "name_nta"],
    "boroname": ["boroname", "boro_name", "borough", "boro"],
    "borocode": ["borocode", "boro_code"],
    "cdta2020": ["cdta2020", "cdta_2020"],
    "cdtaname": ["cdtaname", "cdta_name"],
    "name": ["name", "poi_name", "osm_name"],
    "latitude": ["latitude", "lat", "y"],
    "longitude": ["longitude", "lon", "lng", "x"],
    "osm_id": ["osm_id", "osmid", "id", "element_id"],
}

BASE_OUTPUT_COLUMNS: List[str] = [
    "nta2020",
    "ntaname",
    "boroname",
    "borocode",
    "cdta2020",
    "cdtaname",
    "nta_area_km2",
    "poi_total_count",
    "poi_density_per_km2",
    "poi_log1p_total_count",
    "poi_log1p_density_per_km2",
    "poi_named_count",
    "poi_named_share",
    "poi_with_any_detected_tag_count",
    "poi_with_any_detected_tag_share",
    "poi_with_valid_coordinate_count",
    "poi_with_valid_coordinate_share",
    "poi_unique_raw_type_count",
    "poi_unique_semantic_category_count",
    "poi_semantic_entropy",
    "poi_semantic_entropy_norm",
    "poi_semantic_hhi",
    "poi_mixed_use_index",
    "poi_dominant_semantic_category",
    "poi_dominant_semantic_category_label",
    "poi_dominant_semantic_category_share",
]

COMPOSITE_GROUPS: Dict[str, List[str]] = {
    "commercial_activity": [
        "food_drink",
        "retail",
        "finance",
        "accommodation",
        "office_work",
    ],
    "daily_activity": [
        "food_drink",
        "retail",
        "education",
        "healthcare",
        "transport",
    ],
    "social_infrastructure": [
        "education",
        "healthcare",
        "public_service",
        "religion",
        "culture_tourism",
    ],
    "mobility_access": ["transport"],
    "recreation_environment": ["recreation_green", "culture_tourism"],
    "safety_public_service": ["emergency_safety", "public_service"],
    "employment_industrial": ["office_work", "industrial_logistics"],
}


# ----------------------------- utility functions -----------------------------


def clean_column_name(col: object) -> str:
    s = str(col).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def normalize_text_series(s: pd.Series) -> pd.Series:
    out = s.astype("string").fillna("").str.strip().str.lower()
    out = out.str.replace(r"[^a-z0-9]+", "_", regex=True)
    out = out.str.replace(r"_+", "_", regex=True).str.strip("_")
    out = out.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA})
    return out


def normalize_key_series(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def safe_divide(num: pd.Series, den: pd.Series | float | int) -> pd.Series:
    den_series = den if isinstance(den, pd.Series) else pd.Series(den, index=num.index)
    den_series = den_series.replace(0, np.nan)
    return (num / den_series).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: dict) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # If duplicate normalized names appear, keep the first and suffix later ones.
    new_cols: List[str] = []
    seen: Dict[str, int] = defaultdict(int)
    for c in df.columns:
        base = clean_column_name(c)
        if not base:
            base = "unnamed"
        seen[base] += 1
        if seen[base] == 1:
            new_cols.append(base)
        else:
            new_cols.append(f"{base}_{seen[base]}")
    out = df.copy()
    out.columns = new_cols
    return out


def detect_first_existing(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    colset = set(columns)
    for c in candidates:
        cc = clean_column_name(c)
        if cc in colset:
            return cc
    return None


def detect_columns(columns: Sequence[str]) -> Dict[str, object]:
    columns = list(columns)
    detected: Dict[str, object] = {}
    for canonical, candidates in METADATA_CANDIDATES.items():
        detected[canonical] = detect_first_existing(columns, candidates)
    tag_cols = [c for c in TAG_COLUMN_CANDIDATES if c in columns]
    tag_family_cols = [c for c in TAG_FAMILIES_FOR_COUNTS if c in columns]
    detected["tag_columns"] = tag_cols
    detected["tag_family_columns"] = tag_family_cols
    return detected


def build_combined_tag_text(df: pd.DataFrame, tag_cols: Sequence[str]) -> pd.Series:
    if not tag_cols:
        return pd.Series([""] * len(df), index=df.index, dtype="string")
    # Use values only, not column names, to avoid false matches.
    tmp = df[list(tag_cols)].copy()
    for c in tmp.columns:
        tmp[c] = normalize_text_series(tmp[c]).fillna("")
    combined = tmp.agg(" ".join, axis=1).astype("string")
    combined = combined.str.replace(r"\s+", " ", regex=True).str.strip()
    return combined


def choose_primary_raw_type(df: pd.DataFrame, tag_cols: Sequence[str]) -> pd.Series:
    if not tag_cols:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    tmp = df[list(tag_cols)].copy()
    for c in tmp.columns:
        tmp[c] = normalize_text_series(tmp[c])
    primary = tmp.bfill(axis=1).iloc[:, 0]
    return primary.astype("string")


def classify_poi_semantics(combined_tag_text: pd.Series) -> pd.Series:
    category = pd.Series("other", index=combined_tag_text.index, dtype="string")
    remaining = category.eq("other")
    text = combined_tag_text.fillna("").astype("string")
    for cat, pattern in CATEGORY_RULES:
        mask = remaining & text.str.contains(pattern, regex=True, na=False)
        category.loc[mask] = cat
        remaining = category.eq("other")
    return category


def first_non_null(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return pd.NA
    # Use mode when available for consistency across chunks / duplicate tags.
    mode = s.astype("string").mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return s.iloc[0]


def detect_project_root() -> Path:
    # This file should live in SIMC_PROJECT/code_processdata/.
    return Path(__file__).resolve().parents[1]


def load_nta_reference(path: Path, columns_needed: Sequence[str]) -> Tuple[pd.DataFrame, dict]:
    """Load NTA universe from weekly 311/temporal file if available.

    Returns a dataframe with at least nta2020 and optional metadata columns.
    """
    info = {
        "nta_reference_file": str(path),
        "nta_reference_exists": path.exists(),
        "nta_reference_rows_raw": None,
        "nta_reference_unique_nta2020": None,
        "nta_reference_status": "not_loaded",
    }
    if not path.exists():
        return pd.DataFrame(columns=list(columns_needed)), info

    try:
        # Try to read only metadata columns. If usecols fails due to schema drift,
        # fall back to reading the full file and selecting later.
        try:
            ref = pd.read_csv(path, usecols=lambda c: clean_column_name(c) in set(columns_needed), low_memory=False)
            ref = canonicalize_columns(ref)
        except Exception:
            ref = canonicalize_columns(pd.read_csv(path, low_memory=False))
            keep = [c for c in columns_needed if c in ref.columns]
            ref = ref[keep]

        if "nta2020" not in ref.columns:
            info["nta_reference_status"] = "missing_nta2020"
            return pd.DataFrame(columns=list(columns_needed)), info

        ref["nta2020"] = normalize_key_series(ref["nta2020"])
        ref = ref.dropna(subset=["nta2020"])
        agg_dict = {c: first_non_null for c in ref.columns if c != "nta2020"}
        if agg_dict:
            ref = ref.groupby("nta2020", as_index=False).agg(agg_dict)
        else:
            ref = ref[["nta2020"]].drop_duplicates()

        info["nta_reference_rows_raw"] = int(len(ref))
        info["nta_reference_unique_nta2020"] = int(ref["nta2020"].nunique())
        info["nta_reference_status"] = "loaded"
        return ref, info
    except Exception as e:
        info["nta_reference_status"] = f"error: {type(e).__name__}: {e}"
        return pd.DataFrame(columns=list(columns_needed)), info


def load_nta_area_from_geojson(path: Path) -> Tuple[pd.DataFrame, dict]:
    """Load NTA area in km^2 from GeoJSON.

    Preferred path:
        geopandas + EPSG:32618 projection, suitable for NYC.

    Fallback path:
        If geopandas is unavailable, parse GeoJSON directly and estimate area
        with a local equirectangular projection centered on NYC. This fallback
        is sufficient to avoid all-NaN density features, but for final paper
        numbers geopandas is still preferred.
    """
    info = {
        "nta_geojson_file": str(path),
        "nta_geojson_exists": path.exists(),
        "area_status": "not_loaded",
        "area_unique_nta2020": 0,
        "area_crs_used": "EPSG:32618",
    }
    if not path.exists():
        return pd.DataFrame(columns=["nta2020", "nta_area_km2"]), info

    # Primary: accurate projected area with geopandas.
    try:
        import geopandas as gpd  # type: ignore

        gdf = gpd.read_file(path)
        gdf = canonicalize_columns(pd.DataFrame(gdf.drop(columns="geometry"))).join(gdf.geometry)
        if "nta2020" not in gdf.columns:
            info["area_status"] = "missing_nta2020"
            return pd.DataFrame(columns=["nta2020", "nta_area_km2"]), info

        gdf["nta2020"] = normalize_key_series(gdf["nta2020"])
        gdf = gdf.dropna(subset=["nta2020"])
        gdf = gdf.set_geometry("geometry")

        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
            info["area_status"] = "loaded_with_geopandas_assumed_epsg4326"
        else:
            info["area_status"] = "loaded_with_geopandas"

        gdf_m = gdf.to_crs(epsg=32618)
        area = pd.DataFrame({
            "nta2020": gdf_m["nta2020"].astype("string"),
            "nta_area_km2": gdf_m.geometry.area.astype(float) / 1_000_000.0,
        })
        area = area.groupby("nta2020", as_index=False)["nta_area_km2"].sum()
        info["area_unique_nta2020"] = int(area["nta2020"].nunique())
        return area, info

    except ModuleNotFoundError as e:
        # Fall through to pure-Python approximation if geopandas is unavailable.
        info["area_status"] = f"geopandas_unavailable_using_geojson_fallback: {e}"
    except Exception as e:
        # For final paper, this should be investigated. The fallback is still
        # useful for continuing the pipeline and producing non-missing density
        # features.
        info["area_status"] = f"geopandas_error_using_geojson_fallback: {type(e).__name__}: {e}"

    # Fallback: pure Python GeoJSON area approximation.
    try:
        import json
        import math

        with path.open("r", encoding="utf-8") as f:
            gj = json.load(f)

        earth_radius_m = 6_371_008.8
        lat0_rad = math.radians(40.7128)  # NYC centroid approximation

        def project_lonlat(lon, lat):
            x = earth_radius_m * math.radians(float(lon)) * math.cos(lat0_rad)
            y = earth_radius_m * math.radians(float(lat))
            return x, y

        def ring_area_m2(ring):
            if not ring or len(ring) < 4:
                return 0.0
            pts = []
            for pt in ring:
                if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                    continue
                lon, lat = pt[0], pt[1]
                pts.append(project_lonlat(lon, lat))
            if len(pts) < 4:
                return 0.0
            total = 0.0
            for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
                total += x1 * y2 - x2 * y1
            return abs(total) / 2.0

        def polygon_area_m2(poly):
            if not poly:
                return 0.0
            outer = ring_area_m2(poly[0])
            holes = sum(ring_area_m2(r) for r in poly[1:])
            return max(outer - holes, 0.0)

        def geometry_area_m2(geom):
            if not geom:
                return 0.0
            gtype = geom.get("type")
            coords = geom.get("coordinates", [])
            if gtype == "Polygon":
                return polygon_area_m2(coords)
            if gtype == "MultiPolygon":
                return sum(polygon_area_m2(poly) for poly in coords)
            if gtype == "GeometryCollection":
                return sum(geometry_area_m2(g) for g in geom.get("geometries", []))
            return 0.0

        records = []
        for feat in gj.get("features", []):
            props = canonicalize_columns(pd.DataFrame([feat.get("properties", {})]))
            if "nta2020" not in props.columns:
                continue
            nta = normalize_key_series(props["nta2020"]).iloc[0]
            if pd.isna(nta) or str(nta).strip() == "":
                continue
            area_m2 = geometry_area_m2(feat.get("geometry", {}))
            records.append({
                "nta2020": str(nta),
                "nta_area_km2": area_m2 / 1_000_000.0,
            })

        area = pd.DataFrame(records, columns=["nta2020", "nta_area_km2"])
        if area.empty:
            info["area_status"] = f"{info['area_status']} | fallback_empty"
            return pd.DataFrame(columns=["nta2020", "nta_area_km2"]), info

        area = area.groupby("nta2020", as_index=False)["nta_area_km2"].sum()
        area.loc[area["nta_area_km2"] <= 0, "nta_area_km2"] = np.nan
        info["area_unique_nta2020"] = int(area["nta2020"].nunique())
        info["area_crs_used"] = "fallback_local_equirectangular_approx"
        info["area_status"] = f"{info['area_status']} | fallback_loaded"
        return area, info

    except Exception as e:
        info["area_status"] = f"{info['area_status']} | fallback_error: {type(e).__name__}: {e}"
        return pd.DataFrame(columns=["nta2020", "nta_area_km2"]), info

def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in df.columns:
        miss = int(df[c].isna().sum())
        rows.append({
            "column": c,
            "missing_rows": miss,
            "missing_share": float(miss / n) if n else 0.0,
            "dtype": str(df[c].dtype),
        })
    return pd.DataFrame(rows).sort_values(["missing_rows", "column"], ascending=[False, True])


def profile_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    if not numeric_cols:
        return pd.DataFrame()
    prof = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    prof = prof.reset_index().rename(columns={"index": "feature"})
    return prof


def parse_args() -> argparse.Namespace:
    project_root = detect_project_root()
    parser = argparse.ArgumentParser(
        description="Build NTA-level OpenStreetMap POI features for the SIMC NYC pipeline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "raw" / "osm_poi" / "osm_nyc_pois_with_nta.csv.gz",
        help="Input OSM POI file already joined to NTA. Default: data/raw/osm_poi/osm_nyc_pois_with_nta.csv.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "processed" / "feature_tables" / "osm_poi_nta_features.csv",
        help="Output NTA-level POI feature table. Default: data/processed/feature_tables/osm_poi_nta_features.csv",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=project_root / "data" / "processed" / "_feature_summaries",
        help="Directory for diagnostic summary files.",
    )
    parser.add_argument(
        "--nta-reference",
        type=Path,
        default=project_root / "data" / "processed" / "weekly_311" / "nyc_311_weekly_by_nta_category.csv.gz",
        help="Reference file for the full NTA universe. Default: weekly 311 dense panel.",
    )
    parser.add_argument(
        "--nta-geojson",
        type=Path,
        default=project_root / "data" / "raw" / "nyc_nta" / "nyc_nta_2020_clean.geojson",
        help="NTA GeoJSON for area/density calculation. Optional if geopandas is unavailable.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Rows per read_csv chunk. Default: 250000.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    parser.add_argument(
        "--no-area",
        action="store_true",
        help="Skip loading NTA GeoJSON and do not compute area-normalized density.",
    )
    return parser.parse_args()


# ----------------------------- main processing ------------------------------


def main() -> int:
    start = time.time()
    args = parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 2
    if args.output.exists() and not args.overwrite:
        print(f"ERROR: output already exists. Use --overwrite to replace: {args.output}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.summary_dir / f"{SCRIPT_NAME}_summary.json"
    column_detection_path = args.summary_dir / f"{SCRIPT_NAME}_column_detection.csv"
    semantic_category_counts_path = args.summary_dir / f"{SCRIPT_NAME}_semantic_category_counts.csv"
    tag_family_counts_path = args.summary_dir / f"{SCRIPT_NAME}_tag_family_counts.csv"
    raw_type_top_path = args.summary_dir / f"{SCRIPT_NAME}_raw_type_top200.csv"
    nta_summary_path = args.summary_dir / f"{SCRIPT_NAME}_nta_summary.csv"
    feature_profile_path = args.summary_dir / f"{SCRIPT_NAME}_feature_profile.csv"
    missing_values_path = args.summary_dir / f"{SCRIPT_NAME}_missing_values.csv"
    area_coverage_path = args.summary_dir / f"{SCRIPT_NAME}_area_coverage.csv"
    mapping_path = args.summary_dir / f"{SCRIPT_NAME}_semantic_mapping_reference.csv"

    summary: Dict[str, object] = {
        "script": SCRIPT_NAME,
        "status": "started",
        "input_file": str(args.input),
        "output_file": str(args.output),
        "summary_dir": str(args.summary_dir),
        "chunksize": args.chunksize,
    }

    # Reference NTA universe from weekly 311. This ensures one row per final NTA,
    # even if a particular NTA has zero OSM POIs in the raw file.
    ref_cols = ["nta2020", "ntaname", "boroname", "borocode", "cdta2020", "cdtaname"]
    nta_ref, ref_info = load_nta_reference(args.nta_reference, ref_cols)
    summary.update(ref_info)

    area_df = pd.DataFrame(columns=["nta2020", "nta_area_km2"])
    area_info = {
        "nta_geojson_file": str(args.nta_geojson),
        "nta_geojson_exists": args.nta_geojson.exists(),
        "area_status": "skipped_by_no_area_flag" if args.no_area else "not_loaded",
        "area_unique_nta2020": 0,
    }
    if not args.no_area:
        area_df, area_info = load_nta_area_from_geojson(args.nta_geojson)
    summary.update(area_info)

    total_frames: List[pd.DataFrame] = []
    semantic_frames: List[pd.DataFrame] = []
    tag_family_frames: List[pd.DataFrame] = []
    raw_type_frames: List[pd.DataFrame] = []
    raw_type_by_nta_frames: List[pd.DataFrame] = []
    metadata_frames: List[pd.DataFrame] = []

    city_semantic_counter = defaultdict(int)
    city_tag_family_counter = defaultdict(int)
    city_raw_type_counter = defaultdict(int)

    input_rows_raw = 0
    rows_after_missing_nta_drop = 0
    missing_nta_rows = 0
    duplicate_osm_id_rows = 0
    detected_columns_first_chunk: Optional[Dict[str, object]] = None
    input_columns_raw: Optional[List[str]] = None

    reader = pd.read_csv(args.input, chunksize=args.chunksize, low_memory=False)

    for chunk_i, chunk in enumerate(reader, start=1):
        input_rows_raw += len(chunk)
        if input_columns_raw is None:
            input_columns_raw = list(chunk.columns)
        chunk = canonicalize_columns(chunk)
        detected = detect_columns(chunk.columns)
        if detected_columns_first_chunk is None:
            detected_columns_first_chunk = detected

        nta_col = detected.get("nta2020")
        if not nta_col:
            raise ValueError(
                "Input file does not contain an NTA key column. Expected one of: "
                + ", ".join(METADATA_CANDIDATES["nta2020"])
            )

        chunk["nta2020"] = normalize_key_series(chunk[str(nta_col)])
        missing_nta = int(chunk["nta2020"].isna().sum())
        missing_nta_rows += missing_nta
        chunk = chunk.dropna(subset=["nta2020"]).copy()
        rows_after_missing_nta_drop += len(chunk)
        if chunk.empty:
            continue

        # Metadata by NTA.
        meta_cols = ["nta2020"]
        for c in ["ntaname", "boroname", "borocode", "cdta2020", "cdtaname"]:
            detected_col = detected.get(c)
            if detected_col and detected_col in chunk.columns:
                chunk[c] = normalize_key_series(chunk[str(detected_col)])
                meta_cols.append(c)
        metadata_frames.append(chunk[meta_cols].copy())

        tag_cols = [c for c in detected.get("tag_columns", []) if c in chunk.columns]
        tag_family_cols = [c for c in detected.get("tag_family_columns", []) if c in chunk.columns]

        combined_text = build_combined_tag_text(chunk, tag_cols)
        semantic_category = classify_poi_semantics(combined_text)
        primary_raw_type = choose_primary_raw_type(chunk, tag_cols)

        chunk["poi_semantic_category"] = semantic_category
        chunk["poi_primary_raw_type"] = primary_raw_type
        chunk["poi_has_any_detected_tag"] = combined_text.ne("")

        # Basic quality counts.
        name_col = detected.get("name")
        if name_col and name_col in chunk.columns:
            name_norm = chunk[str(name_col)].astype("string").str.strip()
            named_flag = name_norm.notna() & name_norm.ne("")
        else:
            named_flag = pd.Series(False, index=chunk.index)

        lat_col = detected.get("latitude")
        lon_col = detected.get("longitude")
        if lat_col and lon_col and lat_col in chunk.columns and lon_col in chunk.columns:
            lat = pd.to_numeric(chunk[str(lat_col)], errors="coerce")
            lon = pd.to_numeric(chunk[str(lon_col)], errors="coerce")
            valid_coord = lat.between(-90, 90) & lon.between(-180, 180)
        else:
            valid_coord = pd.Series(False, index=chunk.index)

        osm_id_col = detected.get("osm_id")
        if osm_id_col and osm_id_col in chunk.columns:
            duplicate_osm_id_rows += int(chunk[str(osm_id_col)].duplicated(keep=False).sum())

        # NTA totals.
        tmp_total = pd.DataFrame({
            "nta2020": chunk["nta2020"],
            "poi_total_count": 1,
            "poi_named_count": named_flag.astype(int),
            "poi_with_any_detected_tag_count": chunk["poi_has_any_detected_tag"].astype(int),
            "poi_with_valid_coordinate_count": valid_coord.astype(int),
        })
        total_frames.append(tmp_total.groupby("nta2020", as_index=False).sum(numeric_only=True))

        # Semantic category counts by NTA.
        sem = (
            chunk.groupby(["nta2020", "poi_semantic_category"], observed=True)
            .size()
            .rename("count")
            .reset_index()
        )
        semantic_frames.append(sem)
        for row in sem.itertuples(index=False):
            city_semantic_counter[str(row.poi_semantic_category)] += int(row.count)

        # Tag-family counts: a row can contribute to multiple OSM tag families.
        fam_records = []
        for fam in tag_family_cols:
            flag = normalize_text_series(chunk[fam]).notna()
            if flag.any():
                fam_count = chunk.loc[flag].groupby("nta2020").size().rename("count").reset_index()
                fam_count["tag_family"] = fam
                fam_records.append(fam_count)
                city_tag_family_counter[fam] += int(flag.sum())
        if fam_records:
            tag_family_frames.append(pd.concat(fam_records, ignore_index=True))

        # Raw type counts for interpretability and diagnostic checking.
        raw_valid = chunk["poi_primary_raw_type"].notna()
        if raw_valid.any():
            raw_city = chunk.loc[raw_valid].groupby("poi_primary_raw_type").size().rename("count").reset_index()
            raw_type_frames.append(raw_city)
            for row in raw_city.itertuples(index=False):
                city_raw_type_counter[str(row.poi_primary_raw_type)] += int(row.count)

            raw_nta = (
                chunk.loc[raw_valid]
                .groupby(["nta2020", "poi_primary_raw_type"], observed=True)
                .size()
                .rename("count")
                .reset_index()
            )
            raw_type_by_nta_frames.append(raw_nta)

    if detected_columns_first_chunk is None:
        raise ValueError("Input file appears to be empty.")

    summary["input_rows_raw"] = int(input_rows_raw)
    summary["input_columns_raw"] = input_columns_raw or []
    summary["missing_nta_rows"] = int(missing_nta_rows)
    summary["rows_after_missing_nta_drop"] = int(rows_after_missing_nta_drop)
    summary["duplicate_osm_id_rows_involved_approx"] = int(duplicate_osm_id_rows)

    # Column detection summary.
    detected_records = []
    for k, v in detected_columns_first_chunk.items():
        if isinstance(v, list):
            detected_records.append({"field": k, "detected_columns": "|".join(v), "n_detected": len(v)})
        else:
            detected_records.append({"field": k, "detected_columns": "" if v is None else str(v), "n_detected": 0 if v is None else 1})
    pd.DataFrame(detected_records).to_csv(column_detection_path, index=False, encoding="utf-8-sig")

    if not total_frames:
        raise ValueError("No valid OSM POI rows remained after dropping missing NTA rows.")

    total_by_nta = pd.concat(total_frames, ignore_index=True).groupby("nta2020", as_index=False).sum(numeric_only=True)

    # Metadata from OSM file.
    if metadata_frames:
        meta = pd.concat(metadata_frames, ignore_index=True)
        meta_agg_dict = {c: first_non_null for c in meta.columns if c != "nta2020"}
        meta_by_nta = meta.groupby("nta2020", as_index=False).agg(meta_agg_dict) if meta_agg_dict else meta[["nta2020"]].drop_duplicates()
    else:
        meta_by_nta = pd.DataFrame(columns=["nta2020"])

    # Semantic category pivot.
    semantic_all = pd.concat(semantic_frames, ignore_index=True) if semantic_frames else pd.DataFrame(columns=["nta2020", "poi_semantic_category", "count"])
    semantic_city = semantic_all.groupby("poi_semantic_category", as_index=False)["count"].sum()
    semantic_city["share"] = semantic_city["count"] / semantic_city["count"].sum() if len(semantic_city) else 0
    semantic_city["category_label"] = semantic_city["poi_semantic_category"].map(SEMANTIC_CATEGORY_LABELS).fillna("Unknown")
    semantic_city = semantic_city.sort_values("count", ascending=False)
    semantic_city.to_csv(semantic_category_counts_path, index=False, encoding="utf-8-sig")

    semantic_pivot = (
        semantic_all.pivot_table(
            index="nta2020",
            columns="poi_semantic_category",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    for cat in SEMANTIC_CATEGORIES:
        if cat not in semantic_pivot.columns:
            semantic_pivot[cat] = 0
    semantic_pivot = semantic_pivot[["nta2020"] + SEMANTIC_CATEGORIES]
    semantic_pivot = semantic_pivot.rename(columns={cat: f"poi_{cat}_count" for cat in SEMANTIC_CATEGORIES})

    # Tag family pivot.
    if tag_family_frames:
        tag_all = pd.concat(tag_family_frames, ignore_index=True)
        tag_city = tag_all.groupby("tag_family", as_index=False)["count"].sum().sort_values("count", ascending=False)
        tag_city["share_of_all_poi_rows"] = tag_city["count"] / max(rows_after_missing_nta_drop, 1)
        tag_city.to_csv(tag_family_counts_path, index=False, encoding="utf-8-sig")

        tag_pivot = (
            tag_all.pivot_table(index="nta2020", columns="tag_family", values="count", aggfunc="sum", fill_value=0)
            .reset_index()
        )
        tag_pivot = tag_pivot.rename(columns={c: f"poi_osm_tag_{c}_count" for c in tag_pivot.columns if c != "nta2020"})
    else:
        pd.DataFrame(columns=["tag_family", "count", "share_of_all_poi_rows"]).to_csv(tag_family_counts_path, index=False, encoding="utf-8-sig")
        tag_pivot = pd.DataFrame(columns=["nta2020"])

    # Raw type summary.
    if raw_type_frames:
        raw_all = pd.concat(raw_type_frames, ignore_index=True)
        raw_city = raw_all.groupby("poi_primary_raw_type", as_index=False)["count"].sum().sort_values("count", ascending=False)
        raw_city["share"] = raw_city["count"] / raw_city["count"].sum()
        raw_city.head(200).to_csv(raw_type_top_path, index=False, encoding="utf-8-sig")
    else:
        raw_city = pd.DataFrame(columns=["poi_primary_raw_type", "count", "share"])
        raw_city.to_csv(raw_type_top_path, index=False, encoding="utf-8-sig")

    if raw_type_by_nta_frames:
        raw_nta_all = pd.concat(raw_type_by_nta_frames, ignore_index=True)
        unique_raw_by_nta = (
            raw_nta_all.groupby("nta2020", as_index=False)["poi_primary_raw_type"].nunique()
            .rename(columns={"poi_primary_raw_type": "poi_unique_raw_type_count"})
        )
    else:
        unique_raw_by_nta = pd.DataFrame(columns=["nta2020", "poi_unique_raw_type_count"])

    # Build final NTA universe.
    if not nta_ref.empty and "nta2020" in nta_ref.columns:
        final = nta_ref.copy()
    else:
        final = total_by_nta[["nta2020"]].drop_duplicates().copy()

    # Merge metadata. Prefer reference metadata, then OSM metadata fill.
    final = final.merge(meta_by_nta, on="nta2020", how="outer", suffixes=("", "_osm"))
    for c in ["ntaname", "boroname", "borocode", "cdta2020", "cdtaname"]:
        osm_c = f"{c}_osm"
        if c in final.columns and osm_c in final.columns:
            final[c] = final[c].where(final[c].notna(), final[osm_c])
            final = final.drop(columns=[osm_c])
        elif osm_c in final.columns:
            final = final.rename(columns={osm_c: c})

    final = final.merge(area_df, on="nta2020", how="left")
    final = final.merge(total_by_nta, on="nta2020", how="left")
    final = final.merge(semantic_pivot, on="nta2020", how="left")
    final = final.merge(tag_pivot, on="nta2020", how="left")
    final = final.merge(unique_raw_by_nta, on="nta2020", how="left")

    count_cols = [c for c in final.columns if c.startswith("poi_") and c.endswith("_count")]
    for c in count_cols:
        final[c] = pd.to_numeric(final[c], errors="coerce").fillna(0).astype("int64")
    final["poi_total_count"] = pd.to_numeric(final.get("poi_total_count", 0), errors="coerce").fillna(0).astype("int64")

    # Core rates.
    final["poi_named_share"] = safe_divide(final["poi_named_count"], final["poi_total_count"])
    final["poi_with_any_detected_tag_share"] = safe_divide(final["poi_with_any_detected_tag_count"], final["poi_total_count"])
    final["poi_with_valid_coordinate_share"] = safe_divide(final["poi_with_valid_coordinate_count"], final["poi_total_count"])

    # Category shares and densities.
    semantic_count_cols = [f"poi_{cat}_count" for cat in SEMANTIC_CATEGORIES]
    for cat, col in zip(SEMANTIC_CATEGORIES, semantic_count_cols):
        share_col = f"poi_{cat}_share"
        final[share_col] = safe_divide(final[col], final["poi_total_count"])

    # Diversity metrics.
    p_mat = final[[f"poi_{cat}_share" for cat in SEMANTIC_CATEGORIES]].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -(p_mat * np.where(p_mat > 0, np.log(p_mat), 0)).sum(axis=1)
    k = len(SEMANTIC_CATEGORIES)
    final["poi_semantic_entropy"] = entropy
    final["poi_semantic_entropy_norm"] = entropy / math.log(k) if k > 1 else 0.0
    final["poi_semantic_hhi"] = (p_mat ** 2).sum(axis=1)
    final["poi_mixed_use_index"] = 1.0 - final["poi_semantic_hhi"]
    final.loc[final["poi_total_count"].eq(0), ["poi_semantic_entropy", "poi_semantic_entropy_norm", "poi_semantic_hhi", "poi_mixed_use_index"]] = 0.0
    final["poi_unique_semantic_category_count"] = (final[semantic_count_cols] > 0).sum(axis=1).astype("int64")

    count_values = final[semantic_count_cols].to_numpy(dtype=float)
    dominant_idx = count_values.argmax(axis=1)
    dominant_cats = np.array(SEMANTIC_CATEGORIES, dtype=object)[dominant_idx]
    final["poi_dominant_semantic_category"] = dominant_cats
    final.loc[final["poi_total_count"].eq(0), "poi_dominant_semantic_category"] = pd.NA
    final["poi_dominant_semantic_category_label"] = final["poi_dominant_semantic_category"].map(SEMANTIC_CATEGORY_LABELS)
    dominant_counts = count_values.max(axis=1)
    final["poi_dominant_semantic_category_share"] = safe_divide(pd.Series(dominant_counts, index=final.index), final["poi_total_count"])

    # Composite indices for model interpretation / paper narrative.
    for group_name, cats in COMPOSITE_GROUPS.items():
        cols = [f"poi_{cat}_count" for cat in cats]
        final[f"poi_{group_name}_count"] = final[cols].sum(axis=1).astype("int64")
        final[f"poi_{group_name}_share"] = safe_divide(final[f"poi_{group_name}_count"], final["poi_total_count"])

    # Area-normalized densities. If area is unavailable, values stay NaN.
    final["nta_area_km2"] = pd.to_numeric(final.get("nta_area_km2", np.nan), errors="coerce")
    if "nta_area_km2" in final.columns:
        area_den = final["nta_area_km2"].replace(0, np.nan)
        final["poi_density_per_km2"] = (final["poi_total_count"] / area_den).replace([np.inf, -np.inf], np.nan)
        for cat in SEMANTIC_CATEGORIES:
            final[f"poi_{cat}_density_per_km2"] = (final[f"poi_{cat}_count"] / area_den).replace([np.inf, -np.inf], np.nan)
        for group_name in COMPOSITE_GROUPS:
            final[f"poi_{group_name}_density_per_km2"] = (final[f"poi_{group_name}_count"] / area_den).replace([np.inf, -np.inf], np.nan)
    else:
        final["poi_density_per_km2"] = np.nan

    final["poi_log1p_total_count"] = np.log1p(final["poi_total_count"].astype(float))
    final["poi_log1p_density_per_km2"] = np.log1p(final["poi_density_per_km2"].fillna(0.0))

    # OSM tag family shares.
    tag_count_cols = [c for c in final.columns if c.startswith("poi_osm_tag_") and c.endswith("_count")]
    for c in tag_count_cols:
        share_col = c.replace("_count", "_share")
        final[share_col] = safe_divide(final[c], final["poi_total_count"])

    # Stable column ordering.
    for c in BASE_OUTPUT_COLUMNS:
        if c not in final.columns:
            final[c] = pd.NA
    semantic_cols_order = []
    for cat in SEMANTIC_CATEGORIES:
        semantic_cols_order.extend([
            f"poi_{cat}_count",
            f"poi_{cat}_share",
            f"poi_{cat}_density_per_km2",
        ])
    composite_cols_order = []
    for group_name in COMPOSITE_GROUPS:
        composite_cols_order.extend([
            f"poi_{group_name}_count",
            f"poi_{group_name}_share",
            f"poi_{group_name}_density_per_km2",
        ])
    tag_cols_order = sorted([c for c in final.columns if c.startswith("poi_osm_tag_")])
    remaining_cols = [c for c in final.columns if c not in BASE_OUTPUT_COLUMNS + semantic_cols_order + composite_cols_order + tag_cols_order]
    final = final[BASE_OUTPUT_COLUMNS + semantic_cols_order + composite_cols_order + tag_cols_order + remaining_cols]

    final = final.sort_values("nta2020").reset_index(drop=True)
    final.to_csv(args.output, index=False, encoding="utf-8-sig")

    # Diagnostics.
    summarize_missing(final).to_csv(missing_values_path, index=False, encoding="utf-8-sig")
    profile_numeric_features(final).to_csv(feature_profile_path, index=False, encoding="utf-8-sig")

    nta_summ_cols = [
        "nta2020",
        "ntaname",
        "boroname",
        "nta_area_km2",
        "poi_total_count",
        "poi_density_per_km2",
        "poi_unique_semantic_category_count",
        "poi_semantic_entropy_norm",
        "poi_mixed_use_index",
        "poi_dominant_semantic_category",
        "poi_dominant_semantic_category_share",
        "poi_commercial_activity_count",
        "poi_daily_activity_count",
        "poi_social_infrastructure_count",
        "poi_mobility_access_count",
        "poi_recreation_environment_count",
        "poi_safety_public_service_count",
    ]
    nta_summ_cols = [c for c in nta_summ_cols if c in final.columns]
    final[nta_summ_cols].sort_values("poi_total_count", ascending=False).to_csv(nta_summary_path, index=False, encoding="utf-8-sig")

    # Area coverage diagnostic.
    area_cov = pd.DataFrame({
        "metric": [
            "output_rows",
            "rows_with_area",
            "rows_missing_area",
            "rows_with_positive_area",
            "rows_with_density",
        ],
        "value": [
            len(final),
            int(final["nta_area_km2"].notna().sum()),
            int(final["nta_area_km2"].isna().sum()),
            int((final["nta_area_km2"] > 0).sum()),
            int(final["poi_density_per_km2"].notna().sum()),
        ],
    })
    area_cov.to_csv(area_coverage_path, index=False, encoding="utf-8-sig")

    mapping_ref = pd.DataFrame([
        {"semantic_category": cat, "display_label": label}
        for cat, label in SEMANTIC_CATEGORY_LABELS.items()
    ])
    mapping_ref.to_csv(mapping_path, index=False, encoding="utf-8-sig")

    elapsed = time.time() - start
    summary.update({
        "status": "done",
        "elapsed_seconds": round(elapsed, 3),
        "elapsed_minutes": round(elapsed / 60, 3),
        "output_rows": int(len(final)),
        "output_columns": list(final.columns),
        "output_unique_nta2020": int(final["nta2020"].nunique()),
        "output_duplicate_nta2020_rows": int(final["nta2020"].duplicated().sum()),
        "output_total_poi_count": int(final["poi_total_count"].sum()),
        "output_zero_poi_nta_count": int((final["poi_total_count"] == 0).sum()),
        "output_rows_with_area": int(final["nta_area_km2"].notna().sum()),
        "output_rows_missing_area": int(final["nta_area_km2"].isna().sum()),
        "semantic_categories": SEMANTIC_CATEGORIES,
        "semantic_category_labels": SEMANTIC_CATEGORY_LABELS,
        "composite_groups": COMPOSITE_GROUPS,
        "citywide_semantic_category_counts": {
            str(row.poi_semantic_category): int(row.count)
            for row in semantic_city.itertuples(index=False)
        },
        "detected_columns": detected_columns_first_chunk,
        "summary_files": {
            "summary_json": str(summary_path),
            "column_detection": str(column_detection_path),
            "semantic_category_counts": str(semantic_category_counts_path),
            "tag_family_counts": str(tag_family_counts_path),
            "raw_type_top200": str(raw_type_top_path),
            "nta_summary": str(nta_summary_path),
            "feature_profile": str(feature_profile_path),
            "missing_values": str(missing_values_path),
            "area_coverage": str(area_coverage_path),
            "semantic_mapping_reference": str(mapping_path),
        },
    })
    write_json(summary_path, summary)

    print(json.dumps({
        "status": "done",
        "output_file": str(args.output),
        "output_rows": len(final),
        "total_poi_count": int(final["poi_total_count"].sum()),
        "summary_json": str(summary_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
