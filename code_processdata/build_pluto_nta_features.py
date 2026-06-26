#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 8 — Verify/build PLUTO NTA land-use and built-environment features.

Project: SIMC NYC smart-city data pipeline
Purpose:
    Build a clean NTA-level PLUTO feature table for final modeling.

Default input:
    data/raw/nyc_pluto/nyc_pluto_nta_landuse_features.csv

Default output:
    data/processed/feature_tables/pluto_nta_landuse_features.csv

Design goals:
    1. Support both already-aggregated NTA-level PLUTO files and parcel-level PLUTO files.
    2. Preserve one row per NTA and align to the 262 NTA reference set from the weekly 311 panel.
    3. Create paper-friendly land-use and built-environment features:
       - land-use composition by parcels and lot area
       - residential/commercial/industrial/public/open-space shares
       - building area and floor-area intensity
       - residential/total units density
       - building age profile
       - built-form diversity / land-use entropy
    4. Export summary/check files for reproducibility and paper reporting.

Run from project root:
    .\.venv\Scripts\python.exe .\code_processdata\build_pluto_nta_features.py --overwrite

Notes:
    - NYC PLUTO area variables such as LotArea and BldgArea are typically in square feet.
    - NTA polygon area is computed in square kilometers, preferably using GeoPandas.
    - If GeoPandas is unavailable, a lightweight GeoJSON fallback computes approximate geodesic area.
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
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_NAME = "build_pluto_nta_features"
SQFT_TO_KM2 = 9.290304e-8
CURRENT_YEAR_FOR_AGE = 2026

# Stable PLUTO land-use grouping. Raw PLUTO LandUse values are usually 1-11.
LANDUSE_CODE_TO_CATEGORY = {
    "01": "residential_low_mid",
    "1": "residential_low_mid",
    "02": "residential_multifamily_walkup",
    "2": "residential_multifamily_walkup",
    "03": "residential_multifamily_elevator",
    "3": "residential_multifamily_elevator",
    "04": "mixed_residential_commercial",
    "4": "mixed_residential_commercial",
    "05": "commercial_office",
    "5": "commercial_office",
    "06": "industrial_manufacturing",
    "6": "industrial_manufacturing",
    "07": "transportation_utility",
    "7": "transportation_utility",
    "08": "public_facility_institution",
    "8": "public_facility_institution",
    "09": "open_space_recreation",
    "9": "open_space_recreation",
    "10": "parking_facility",
    "11": "vacant_land",
}

LANDUSE_CATEGORY_LABELS = {
    "residential_low_mid": "One-/Two-Family and Low-Rise Residential",
    "residential_multifamily_walkup": "Multi-Family Walk-Up Residential",
    "residential_multifamily_elevator": "Multi-Family Elevator Residential",
    "mixed_residential_commercial": "Mixed Residential and Commercial",
    "commercial_office": "Commercial and Office",
    "industrial_manufacturing": "Industrial and Manufacturing",
    "transportation_utility": "Transportation and Utility",
    "public_facility_institution": "Public Facilities and Institutions",
    "open_space_recreation": "Open Space and Outdoor Recreation",
    "parking_facility": "Parking Facilities",
    "vacant_land": "Vacant Land",
    "unknown": "Unknown / Missing Land Use",
}

LANDUSE_CATEGORIES = list(LANDUSE_CATEGORY_LABELS.keys())

# Higher-level groups useful for paper claims and ablation/explainability.
LANDUSE_COMPOSITE_GROUPS = {
    "residential": [
        "residential_low_mid",
        "residential_multifamily_walkup",
        "residential_multifamily_elevator",
    ],
    "mixed_use": ["mixed_residential_commercial"],
    "commercial_activity": ["mixed_residential_commercial", "commercial_office"],
    "industrial_utility": ["industrial_manufacturing", "transportation_utility"],
    "civic_institutional": ["public_facility_institution"],
    "open_recreation": ["open_space_recreation"],
    "parking_auto": ["parking_facility"],
    "vacant_underused": ["vacant_land"],
}

# Candidate raw columns in common PLUTO exports. The detector normalizes names first.
COLUMN_CANDIDATES = {
    "nta2020": ["nta2020", "nta", "ntacode", "nta_code", "nta2020code"],
    "ntaname": ["ntaname", "nta_name", "ntaname2020", "nta2020name"],
    "boroname": ["boroname", "borough", "boro", "boroughname"],
    "borocode": ["borocode", "borocode1", "borocode2020", "boro_code"],
    "cdta2020": ["cdta2020", "cdtacode", "cdta_code"],
    "cdtaname": ["cdtaname", "cdta_name"],
    "bbl": ["bbl", "bbl10", "boroughblocklot"],
    "landuse": ["landuse", "land_use", "plutolanduse"],
    "bldgclass": ["bldgclass", "buildingclass", "bldg_class"],
    "lotarea": ["lotarea", "lot_area", "lotsqft", "lotareasqft"],
    "bldgarea": ["bldgarea", "bldg_area", "buildingarea", "building_area", "bldgareafeet"],
    "comarea": ["comarea", "commercialarea", "commercial_area", "com_area"],
    "resarea": ["resarea", "residentialarea", "residential_area", "res_area"],
    "officearea": ["officearea", "office_area"],
    "retailarea": ["retailarea", "retail_area"],
    "garagearea": ["garagearea", "garage_area"],
    "strgearea": ["strgearea", "storagearea", "storage_area", "strgarea"],
    "factryarea": ["factryarea", "factoryarea", "factory_area", "industrialarea", "industrial_area"],
    "otherarea": ["otherarea", "other_area"],
    "numbldgs": ["numbldgs", "num_bldgs", "numberofbuildings", "number_buildings"],
    "numfloors": ["numfloors", "num_floors", "numberoffloors", "floors"],
    "unitsres": ["unitsres", "units_res", "residentialunits", "res_units"],
    "unitstotal": ["unitstotal", "units_total", "totalunits", "total_units"],
    "yearbuilt": ["yearbuilt", "year_built", "yrbuilt"],
    "yearalter1": ["yearalter1", "year_alter1", "yearaltered1"],
    "yearalter2": ["yearalter2", "year_alter2", "yearaltered2"],
    "builtfar": ["builtfar", "built_far", "builtfloorarearatio"],
    "residfar": ["residfar", "resid_far", "residentialfar"],
    "commfar": ["commfar", "comm_far", "commercialfar"],
    "facilfar": ["facilfar", "facil_far", "facilityfar"],
    "assesstot": ["assesstot", "assess_total", "assessedtotal", "assess_tot"],
    "assessland": ["assessland", "assess_land", "assessedland"],
    "zonedist1": ["zonedist1", "zone_dist1", "zoningdistrict1"],
    "latitude": ["latitude", "lat", "ycoord", "ycoordinate"],
    "longitude": ["longitude", "lon", "lng", "xcoord", "xcoordinate"],
}

META_OUTPUT_COLS = ["nta2020", "ntaname", "boroname", "borocode", "cdta2020", "cdtaname"]


def normalize_name(name: str) -> str:
    """Normalize a column name for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def make_safe_col(name: str) -> str:
    """Create a conservative snake_case column name."""
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"


def detect_columns(columns: Iterable[str]) -> Dict[str, Optional[str]]:
    """Detect canonical PLUTO columns from an input schema."""
    original_cols = list(columns)
    norm_to_original: Dict[str, str] = {}
    for col in original_cols:
        norm_to_original.setdefault(normalize_name(col), col)

    detected: Dict[str, Optional[str]] = {}
    for canonical, candidates in COLUMN_CANDIDATES.items():
        detected[canonical] = None
        for cand in candidates:
            norm = normalize_name(cand)
            if norm in norm_to_original:
                detected[canonical] = norm_to_original[norm]
                break
    return detected


def read_csv_safely(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    return pd.read_csv(path, low_memory=False)


def ensure_output_path(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists. Use --overwrite to replace it: {path}")


def load_nta_reference(path: Path, detected_meta: Optional[Dict[str, Optional[str]]] = None) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load 262-NTA reference from weekly 311 if available, else return empty reference."""
    info: Dict[str, object] = {
        "nta_reference_file": str(path),
        "nta_reference_exists": path.exists(),
        "nta_reference_status": "not_loaded",
        "nta_reference_rows_raw": 0,
        "nta_reference_unique_nta2020": 0,
    }
    if not path.exists():
        return pd.DataFrame(columns=META_OUTPUT_COLS), info

    try:
        header = pd.read_csv(path, nrows=0)
        available = list(header.columns)
        cols = [c for c in META_OUTPUT_COLS if c in available]
        if "nta2020" not in cols:
            info["nta_reference_status"] = "missing_nta2020"
            return pd.DataFrame(columns=META_OUTPUT_COLS), info
        ref = pd.read_csv(path, usecols=cols, low_memory=False)
        ref = ref.drop_duplicates(subset=["nta2020"]).copy()
        for col in META_OUTPUT_COLS:
            if col not in ref.columns:
                ref[col] = np.nan
        ref = ref[META_OUTPUT_COLS]
        info.update(
            {
                "nta_reference_status": "loaded",
                "nta_reference_rows_raw": int(len(ref)),
                "nta_reference_unique_nta2020": int(ref["nta2020"].nunique(dropna=True)),
            }
        )
        return ref, info
    except Exception as exc:  # pragma: no cover - runtime diagnostics only
        info["nta_reference_status"] = f"error: {type(exc).__name__}: {exc}"
        return pd.DataFrame(columns=META_OUTPUT_COLS), info


def polygon_area_m2_lonlat(coords: List[List[float]]) -> float:
    """Approximate geodesic polygon area using an equirectangular projection around polygon centroid.

    This fallback is less exact than GeoPandas with a projected CRS but is adequate as a backup.
    """
    if not coords or len(coords) < 4:
        return 0.0
    lon = np.array([p[0] for p in coords], dtype=float)
    lat = np.array([p[1] for p in coords], dtype=float)
    lat0 = np.deg2rad(np.nanmean(lat))
    radius = 6371008.8
    x = np.deg2rad(lon) * radius * np.cos(lat0)
    y = np.deg2rad(lat) * radius
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area)


def geometry_area_m2_fallback(geometry: dict) -> float:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon":
        if not coords:
            return 0.0
        outer = polygon_area_m2_lonlat(coords[0])
        holes = sum(polygon_area_m2_lonlat(ring) for ring in coords[1:])
        return max(outer - holes, 0.0)
    if gtype == "MultiPolygon":
        total = 0.0
        for poly in coords:
            if not poly:
                continue
            outer = polygon_area_m2_lonlat(poly[0])
            holes = sum(polygon_area_m2_lonlat(ring) for ring in poly[1:])
            total += max(outer - holes, 0.0)
        return float(total)
    return 0.0


def load_nta_area(geojson_path: Path, target_crs: str = "EPSG:32618") -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load NTA polygon area in km² using GeoPandas when possible, with GeoJSON fallback."""
    info: Dict[str, object] = {
        "nta_geojson_file": str(geojson_path),
        "nta_geojson_exists": geojson_path.exists(),
        "area_status": "not_loaded",
        "area_unique_nta2020": 0,
        "area_crs_used": target_crs,
    }
    if not geojson_path.exists():
        return pd.DataFrame(columns=["nta2020", "nta_area_km2"]), info

    try:
        import geopandas as gpd  # type: ignore

        gdf = gpd.read_file(geojson_path)
        detected = detect_columns(gdf.columns)
        nta_col = detected.get("nta2020") or ("nta2020" if "nta2020" in gdf.columns else None)
        if nta_col is None:
            info["area_status"] = "missing_nta2020_in_geojson"
            return pd.DataFrame(columns=["nta2020", "nta_area_km2"]), info
        gdf = gdf[[nta_col, "geometry"]].rename(columns={nta_col: "nta2020"})
        if gdf.crs is None:
            # NYC open data usually ships lon/lat; default to WGS84 if CRS is missing.
            gdf = gdf.set_crs("EPSG:4326")
        gdf = gdf.to_crs(target_crs)
        out = pd.DataFrame({"nta2020": gdf["nta2020"].astype(str), "nta_area_km2": gdf.geometry.area / 1_000_000.0})
        out = out.groupby("nta2020", as_index=False)["nta_area_km2"].sum()
        info.update({"area_status": "loaded_with_geopandas", "area_unique_nta2020": int(out["nta2020"].nunique())})
        return out, info
    except Exception as exc:
        geopandas_error = f"{type(exc).__name__}: {exc}"

    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            gj = json.load(f)
        rows = []
        for feat in gj.get("features", []):
            props = feat.get("properties", {}) or {}
            # Try exact and normalized property matching.
            nta = None
            for k, v in props.items():
                if normalize_name(k) in {"nta2020", "ntacode", "nta"}:
                    nta = v
                    break
            if nta is None:
                continue
            area_m2 = geometry_area_m2_fallback(feat.get("geometry", {}) or {})
            rows.append({"nta2020": str(nta), "nta_area_km2": area_m2 / 1_000_000.0})
        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.groupby("nta2020", as_index=False)["nta_area_km2"].sum()
        info.update(
            {
                "area_status": f"loaded_with_geojson_fallback_after_geopandas_error: {geopandas_error}",
                "area_unique_nta2020": int(out["nta2020"].nunique()) if not out.empty else 0,
            }
        )
        return out, info
    except Exception as exc2:  # pragma: no cover - runtime diagnostics only
        info["area_status"] = f"error: geopandas failed ({geopandas_error}); fallback failed ({type(exc2).__name__}: {exc2})"
        return pd.DataFrame(columns=["nta2020", "nta_area_km2"]), info


def to_numeric_series(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def clean_landuse_code(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    s = str(value).strip()
    if not s:
        return "unknown"
    # Handle values like 1.0, 01, '01 One & Two Family Buildings'.
    m = re.match(r"^(\d+(?:\.0)?)", s)
    if m:
        num = m.group(1).replace(".0", "")
        if num.isdigit() and int(num) < 10:
            return f"0{int(num)}"
        return num
    return s


def map_landuse_category(value: object) -> str:
    code = clean_landuse_code(value)
    return LANDUSE_CODE_TO_CATEGORY.get(code, "unknown")


def entropy_from_counts(counts: np.ndarray) -> Tuple[float, float, float]:
    counts = counts.astype(float)
    total = counts.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0
    p = counts[counts > 0] / total
    entropy = float(-(p * np.log(p)).sum())
    max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
    entropy_norm = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    hhi = float((p**2).sum())
    return entropy, entropy_norm, hhi


def safe_divide(num: pd.Series | float, den: pd.Series | float) -> pd.Series | float:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = num / den
    if isinstance(result, pd.Series):
        return result.replace([np.inf, -np.inf], np.nan)
    if not np.isfinite(result):
        return np.nan
    return result


def detect_if_preaggregated(df: pd.DataFrame, nta_col: str) -> bool:
    """Heuristic: one row per NTA means the input is already an NTA-level feature file."""
    if nta_col not in df.columns:
        return False
    non_missing = df[nta_col].notna().sum()
    unique = df[nta_col].nunique(dropna=True)
    return non_missing == unique


def standardize_preaggregated(
    df: pd.DataFrame,
    detected: Dict[str, Optional[str]],
    nta_ref: pd.DataFrame,
    area_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Standardize an already-aggregated NTA-level PLUTO file."""
    info: Dict[str, object] = {"input_level": "preaggregated_nta"}
    nta_col = detected.get("nta2020")
    if nta_col is None:
        raise ValueError("Cannot standardize preaggregated PLUTO file without an NTA column.")

    out = df.copy()
    rename = {}
    for canonical in META_OUTPUT_COLS:
        col = detected.get(canonical)
        if col and col in out.columns:
            rename[col] = canonical
    out = out.rename(columns=rename)
    out["nta2020"] = out["nta2020"].astype(str)

    # Remove duplicate NTA rows if any slipped through.
    out = out.drop_duplicates(subset=["nta2020"], keep="first")

    # Prefix non-metadata feature columns for clear final merge.
    new_cols = {}
    for col in out.columns:
        if col in META_OUTPUT_COLS:
            continue
        safe = make_safe_col(col)
        if safe == "nta_area_km2" or safe.startswith("pluto_"):
            new_cols[col] = safe
        else:
            new_cols[col] = f"pluto_{safe}"
    out = out.rename(columns=new_cols)

    # Align with 262-NTA reference if available.
    if not nta_ref.empty:
        out = nta_ref.merge(out, on="nta2020", how="left", suffixes=("", "_pluto"))
        for meta in ["ntaname", "boroname", "borocode", "cdta2020", "cdtaname"]:
            alt = f"{meta}_pluto"
            if alt in out.columns:
                out[meta] = out[meta].combine_first(out[alt])
                out = out.drop(columns=[alt])

    # Ensure area exists and compute generic density if possible.
    if "nta_area_km2" not in out.columns or out["nta_area_km2"].isna().all():
        if "nta_area_km2" in out.columns:
            out = out.drop(columns=["nta_area_km2"])
        out = out.merge(area_df, on="nta2020", how="left")

    if "pluto_parcel_count" not in out.columns:
        # Try common count columns from preaggregated sources.
        count_candidates = [c for c in out.columns if c in {"pluto_total_lots", "pluto_lot_count", "pluto_count", "pluto_n"}]
        if count_candidates:
            out["pluto_parcel_count"] = pd.to_numeric(out[count_candidates[0]], errors="coerce")

    if "pluto_parcel_count" in out.columns:
        out["pluto_parcel_density_per_km2"] = safe_divide(pd.to_numeric(out["pluto_parcel_count"], errors="coerce"), out["nta_area_km2"])

    for col in META_OUTPUT_COLS:
        if col not in out.columns:
            out[col] = np.nan

    # Metadata first, then area, then PLUTO features.
    feature_cols = [c for c in out.columns if c not in META_OUTPUT_COLS]
    ordered = META_OUTPUT_COLS + sorted(feature_cols, key=lambda x: (x != "nta_area_km2", x))
    out = out[ordered]
    info["preaggregated_output_rows"] = int(len(out))
    info["preaggregated_output_columns"] = list(out.columns)
    return out, info


def aggregate_parcel_level(
    df: pd.DataFrame,
    detected: Dict[str, Optional[str]],
    nta_ref: pd.DataFrame,
    area_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:
    """Aggregate parcel-level PLUTO rows to one row per NTA."""
    info: Dict[str, object] = {"input_level": "parcel_level_or_duplicate_nta"}
    nta_col = detected.get("nta2020")
    if nta_col is None:
        raise ValueError("Cannot aggregate parcel-level PLUTO without an NTA column.")

    work = df.copy()
    work = work.rename(columns={nta_col: "nta2020"})
    work["nta2020"] = work["nta2020"].astype(str)
    info["missing_nta_rows"] = int(work["nta2020"].isna().sum() + (work["nta2020"].astype(str).str.strip().isin(["", "nan", "None"]).sum()))
    work = work[~work["nta2020"].astype(str).str.strip().isin(["", "nan", "None"])]
    info["rows_after_missing_nta_drop"] = int(len(work))

    # Attach meta columns if present. Use first non-null per NTA.
    meta_maps = {}
    for meta in ["ntaname", "boroname", "borocode", "cdta2020", "cdtaname"]:
        src = detected.get(meta)
        if src and src in work.columns:
            meta_maps[meta] = src

    # Numeric raw columns.
    num = {}
    for key in [
        "lotarea", "bldgarea", "comarea", "resarea", "officearea", "retailarea",
        "garagearea", "strgearea", "factryarea", "otherarea", "numbldgs", "numfloors",
        "unitsres", "unitstotal", "yearbuilt", "yearalter1", "yearalter2", "builtfar",
        "residfar", "commfar", "facilfar", "assesstot", "assessland",
    ]:
        num[key] = to_numeric_series(work, detected.get(key))
        work[f"__{key}"] = num[key]

    landuse_col = detected.get("landuse")
    if landuse_col and landuse_col in work.columns:
        work["__landuse_category"] = work[landuse_col].map(map_landuse_category)
        work["__landuse_code"] = work[landuse_col].map(clean_landuse_code)
    else:
        work["__landuse_category"] = "unknown"
        work["__landuse_code"] = "unknown"

    # Valid year flags.
    yearbuilt = work["__yearbuilt"]
    valid_year = yearbuilt.between(1600, CURRENT_YEAR_FOR_AGE)
    work["__yearbuilt_valid"] = valid_year
    work["__building_age"] = np.where(valid_year, CURRENT_YEAR_FOR_AGE - yearbuilt, np.nan)
    work["__built_pre_1940"] = np.where(valid_year & (yearbuilt < 1940), 1, 0)
    work["__built_1940_1979"] = np.where(valid_year & yearbuilt.between(1940, 1979), 1, 0)
    work["__built_1980_1999"] = np.where(valid_year & yearbuilt.between(1980, 1999), 1, 0)
    work["__built_2000_plus"] = np.where(valid_year & (yearbuilt >= 2000), 1, 0)

    # Base aggregation.
    gb = work.groupby("nta2020", dropna=False)
    out = pd.DataFrame(index=gb.size().index)
    out["pluto_parcel_count"] = gb.size().astype(float)

    bbl_col = detected.get("bbl")
    if bbl_col and bbl_col in work.columns:
        out["pluto_unique_bbl_count"] = gb[bbl_col].nunique(dropna=True).astype(float)
    else:
        out["pluto_unique_bbl_count"] = out["pluto_parcel_count"]

    # Metadata first non-null.
    for meta, src in meta_maps.items():
        out[meta] = gb[src].agg(lambda s: s.dropna().iloc[0] if s.dropna().shape[0] else np.nan)

    # Numeric sums.
    sum_specs = {
        "lotarea": "pluto_lot_area_sqft_sum",
        "bldgarea": "pluto_building_area_sqft_sum",
        "comarea": "pluto_commercial_area_sqft_sum",
        "resarea": "pluto_residential_area_sqft_sum",
        "officearea": "pluto_office_area_sqft_sum",
        "retailarea": "pluto_retail_area_sqft_sum",
        "garagearea": "pluto_garage_area_sqft_sum",
        "strgearea": "pluto_storage_area_sqft_sum",
        "factryarea": "pluto_factory_area_sqft_sum",
        "otherarea": "pluto_other_area_sqft_sum",
        "numbldgs": "pluto_building_count_sum",
        "unitsres": "pluto_residential_units_sum",
        "unitstotal": "pluto_total_units_sum",
        "assesstot": "pluto_assessed_total_value_sum",
        "assessland": "pluto_assessed_land_value_sum",
    }
    for raw, out_col in sum_specs.items():
        out[out_col] = gb[f"__{raw}"].sum(min_count=1)

    # Numeric distribution stats.
    stat_specs = {
        "lotarea": "pluto_lot_area_sqft",
        "bldgarea": "pluto_building_area_sqft",
        "numfloors": "pluto_num_floors",
        "builtfar": "pluto_built_far",
        "residfar": "pluto_residential_far_allowed",
        "commfar": "pluto_commercial_far_allowed",
        "facilfar": "pluto_facility_far_allowed",
        "yearbuilt": "pluto_year_built",
    }
    for raw, prefix in stat_specs.items():
        series_name = f"__{raw}"
        out[f"{prefix}_mean"] = gb[series_name].mean()
        out[f"{prefix}_median"] = gb[series_name].median()
        out[f"{prefix}_max"] = gb[series_name].max()

    # Building age profile.
    out["pluto_building_age_mean"] = gb["__building_age"].mean()
    out["pluto_building_age_median"] = gb["__building_age"].median()
    out["pluto_yearbuilt_valid_count"] = gb["__yearbuilt_valid"].sum().astype(float)
    out["pluto_yearbuilt_missing_or_invalid_share"] = 1.0 - safe_divide(out["pluto_yearbuilt_valid_count"], out["pluto_parcel_count"])
    for flag in ["built_pre_1940", "built_1940_1979", "built_1980_1999", "built_2000_plus"]:
        out[f"pluto_{flag}_parcel_share"] = safe_divide(gb[f"__{flag}"].sum(), out["pluto_parcel_count"])

    # Land-use counts and lot-area shares.
    landuse_counts = pd.crosstab(work["nta2020"], work["__landuse_category"]).astype(float)
    for cat in LANDUSE_CATEGORIES:
        if cat not in landuse_counts.columns:
            landuse_counts[cat] = 0.0
    landuse_counts = landuse_counts[LANDUSE_CATEGORIES]

    lotarea_by_landuse = (
        work.pivot_table(index="nta2020", columns="__landuse_category", values="__lotarea", aggfunc="sum", fill_value=0.0)
        if "__lotarea" in work.columns
        else pd.DataFrame(index=out.index)
    )
    for cat in LANDUSE_CATEGORIES:
        if cat not in lotarea_by_landuse.columns:
            lotarea_by_landuse[cat] = 0.0
    lotarea_by_landuse = lotarea_by_landuse[LANDUSE_CATEGORIES]

    for cat in LANDUSE_CATEGORIES:
        out[f"pluto_landuse_{cat}_parcel_count"] = landuse_counts[cat].reindex(out.index).fillna(0.0)
        out[f"pluto_landuse_{cat}_parcel_share"] = safe_divide(out[f"pluto_landuse_{cat}_parcel_count"], out["pluto_parcel_count"])
        out[f"pluto_landuse_{cat}_lot_area_sqft_sum"] = lotarea_by_landuse[cat].reindex(out.index).fillna(0.0)
        out[f"pluto_landuse_{cat}_lot_area_share"] = safe_divide(
            out[f"pluto_landuse_{cat}_lot_area_sqft_sum"], out["pluto_lot_area_sqft_sum"]
        )

    # Composite land-use groups.
    for comp, cats in LANDUSE_COMPOSITE_GROUPS.items():
        parcel_cols = [f"pluto_landuse_{cat}_parcel_count" for cat in cats]
        area_cols = [f"pluto_landuse_{cat}_lot_area_sqft_sum" for cat in cats]
        out[f"pluto_{comp}_parcel_count"] = out[parcel_cols].sum(axis=1)
        out[f"pluto_{comp}_parcel_share"] = safe_divide(out[f"pluto_{comp}_parcel_count"], out["pluto_parcel_count"])
        out[f"pluto_{comp}_lot_area_sqft_sum"] = out[area_cols].sum(axis=1)
        out[f"pluto_{comp}_lot_area_share"] = safe_divide(out[f"pluto_{comp}_lot_area_sqft_sum"], out["pluto_lot_area_sqft_sum"])

    # Diversity and dominant land use.
    entropy_rows = []
    dominant_cat = []
    dominant_share = []
    for nta, row in landuse_counts.reindex(out.index).fillna(0.0).iterrows():
        arr = row[LANDUSE_CATEGORIES].values.astype(float)
        ent, ent_norm, hhi = entropy_from_counts(arr)
        entropy_rows.append((ent, ent_norm, hhi))
        total = arr.sum()
        if total > 0:
            idx = int(np.argmax(arr))
            dominant_cat.append(LANDUSE_CATEGORIES[idx])
            dominant_share.append(float(arr[idx] / total))
        else:
            dominant_cat.append(np.nan)
            dominant_share.append(np.nan)
    out["pluto_landuse_entropy"] = [x[0] for x in entropy_rows]
    out["pluto_landuse_entropy_norm"] = [x[1] for x in entropy_rows]
    out["pluto_landuse_hhi"] = [x[2] for x in entropy_rows]
    out["pluto_landuse_mixed_use_index"] = 1.0 - out["pluto_landuse_hhi"]
    out["pluto_dominant_landuse_category"] = dominant_cat
    out["pluto_dominant_landuse_category_label"] = out["pluto_dominant_landuse_category"].map(LANDUSE_CATEGORY_LABELS)
    out["pluto_dominant_landuse_parcel_share"] = dominant_share

    # Built-environment ratios and intensities.
    out["pluto_observed_built_far"] = safe_divide(out["pluto_building_area_sqft_sum"], out["pluto_lot_area_sqft_sum"])
    out["pluto_commercial_area_share_of_bldgarea"] = safe_divide(out["pluto_commercial_area_sqft_sum"], out["pluto_building_area_sqft_sum"])
    out["pluto_residential_area_share_of_bldgarea"] = safe_divide(out["pluto_residential_area_sqft_sum"], out["pluto_building_area_sqft_sum"])
    out["pluto_office_area_share_of_bldgarea"] = safe_divide(out["pluto_office_area_sqft_sum"], out["pluto_building_area_sqft_sum"])
    out["pluto_retail_area_share_of_bldgarea"] = safe_divide(out["pluto_retail_area_sqft_sum"], out["pluto_building_area_sqft_sum"])
    out["pluto_factory_area_share_of_bldgarea"] = safe_divide(out["pluto_factory_area_sqft_sum"], out["pluto_building_area_sqft_sum"])
    out["pluto_units_res_share_of_total_units"] = safe_divide(out["pluto_residential_units_sum"], out["pluto_total_units_sum"])

    # Add NTA area and density features.
    out = out.reset_index().rename(columns={"index": "nta2020"})
    out = out.merge(area_df, on="nta2020", how="left")
    out["pluto_parcel_density_per_km2"] = safe_divide(out["pluto_parcel_count"], out["nta_area_km2"])
    out["pluto_building_density_per_km2"] = safe_divide(out["pluto_building_count_sum"], out["nta_area_km2"])
    out["pluto_units_total_density_per_km2"] = safe_divide(out["pluto_total_units_sum"], out["nta_area_km2"])
    out["pluto_units_residential_density_per_km2"] = safe_divide(out["pluto_residential_units_sum"], out["nta_area_km2"])
    out["pluto_building_area_sqft_density_per_km2"] = safe_divide(out["pluto_building_area_sqft_sum"], out["nta_area_km2"])
    out["pluto_lot_area_sqft_density_per_km2"] = safe_divide(out["pluto_lot_area_sqft_sum"], out["nta_area_km2"])

    for comp in LANDUSE_COMPOSITE_GROUPS:
        out[f"pluto_{comp}_parcel_density_per_km2"] = safe_divide(out[f"pluto_{comp}_parcel_count"], out["nta_area_km2"])
        out[f"pluto_{comp}_lot_area_sqft_density_per_km2"] = safe_divide(out[f"pluto_{comp}_lot_area_sqft_sum"], out["nta_area_km2"])

    # Align with 262-NTA reference if available.
    if not nta_ref.empty:
        out = nta_ref.merge(out, on="nta2020", how="left", suffixes=("", "_pluto"))
        for meta in ["ntaname", "boroname", "borocode", "cdta2020", "cdtaname"]:
            alt = f"{meta}_pluto"
            if alt in out.columns:
                out[meta] = out[meta].combine_first(out[alt])
                out = out.drop(columns=[alt])
        # Fill zero for counts and sums when an NTA is absent from PLUTO, but not for shares/ratios.
        zero_like_patterns = ("_count", "_sum", "_density_per_km2")
        for col in out.columns:
            if col.startswith("pluto_") and col.endswith(zero_like_patterns):
                out[col] = out[col].fillna(0.0)

    # Metadata first.
    for col in META_OUTPUT_COLS:
        if col not in out.columns:
            out[col] = np.nan
    feature_cols = [c for c in out.columns if c not in META_OUTPUT_COLS]
    ordered = META_OUTPUT_COLS + ["nta_area_km2"] + sorted([c for c in feature_cols if c != "nta_area_km2"])
    out = out[ordered]

    landuse_distribution = landuse_counts.sum(axis=0).reset_index()
    landuse_distribution.columns = ["landuse_category", "parcel_count"]
    landuse_distribution["landuse_label"] = landuse_distribution["landuse_category"].map(LANDUSE_CATEGORY_LABELS)
    total_count = landuse_distribution["parcel_count"].sum()
    landuse_distribution["parcel_share"] = landuse_distribution["parcel_count"] / total_count if total_count else np.nan
    landuse_distribution = landuse_distribution.sort_values("parcel_count", ascending=False)

    info.update(
        {
            "parcel_output_rows": int(len(out)),
            "parcel_output_columns": list(out.columns),
            "input_landuse_col": landuse_col,
            "citywide_landuse_category_counts": {
                str(k): int(v) for k, v in landuse_distribution.set_index("landuse_category")["parcel_count"].to_dict().items()
            },
        }
    )
    return out, info, landuse_distribution


def write_summary_files(
    out: pd.DataFrame,
    raw_df: pd.DataFrame,
    summary: Dict[str, object],
    detected: Dict[str, Optional[str]],
    landuse_distribution: Optional[pd.DataFrame],
    output_file: Path,
    summary_dir: Path,
) -> Dict[str, str]:
    summary_dir.mkdir(parents=True, exist_ok=True)
    files = {}

    summary_path = summary_dir / f"{SCRIPT_NAME}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    files["summary_json"] = str(summary_path)

    col_det = pd.DataFrame(
        [{"canonical_column": k, "detected_input_column": v, "detected": v is not None} for k, v in detected.items()]
    )
    p = summary_dir / f"{SCRIPT_NAME}_column_detection.csv"
    col_det.to_csv(p, index=False, encoding="utf-8-sig")
    files["column_detection"] = str(p)

    missing = pd.DataFrame(
        {
            "column": out.columns,
            "missing_count": [int(out[c].isna().sum()) for c in out.columns],
            "missing_share": [float(out[c].isna().mean()) for c in out.columns],
        }
    ).sort_values(["missing_count", "column"], ascending=[False, True])
    p = summary_dir / f"{SCRIPT_NAME}_missing_values.csv"
    missing.to_csv(p, index=False, encoding="utf-8-sig")
    files["missing_values"] = str(p)

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    profile_rows = []
    for c in numeric_cols:
        s = out[c]
        profile_rows.append(
            {
                "feature": c,
                "non_missing": int(s.notna().sum()),
                "missing": int(s.isna().sum()),
                "mean": float(s.mean()) if s.notna().any() else np.nan,
                "std": float(s.std()) if s.notna().sum() > 1 else np.nan,
                "min": float(s.min()) if s.notna().any() else np.nan,
                "p25": float(s.quantile(0.25)) if s.notna().any() else np.nan,
                "median": float(s.median()) if s.notna().any() else np.nan,
                "p75": float(s.quantile(0.75)) if s.notna().any() else np.nan,
                "max": float(s.max()) if s.notna().any() else np.nan,
            }
        )
    p = summary_dir / f"{SCRIPT_NAME}_feature_profile.csv"
    pd.DataFrame(profile_rows).to_csv(p, index=False, encoding="utf-8-sig")
    files["feature_profile"] = str(p)

    nta_coverage = pd.DataFrame(
        [
            {
                "metric": "output_rows",
                "value": int(len(out)),
            },
            {
                "metric": "output_unique_nta2020",
                "value": int(out["nta2020"].nunique(dropna=True)) if "nta2020" in out.columns else 0,
            },
            {
                "metric": "duplicate_nta2020_rows",
                "value": int(out.duplicated("nta2020").sum()) if "nta2020" in out.columns else 0,
            },
            {
                "metric": "rows_with_area",
                "value": int(out["nta_area_km2"].notna().sum()) if "nta_area_km2" in out.columns else 0,
            },
            {
                "metric": "rows_missing_area",
                "value": int(out["nta_area_km2"].isna().sum()) if "nta_area_km2" in out.columns else int(len(out)),
            },
            {
                "metric": "rows_with_positive_area",
                "value": int((out["nta_area_km2"] > 0).sum()) if "nta_area_km2" in out.columns else 0,
            },
        ]
    )
    p = summary_dir / f"{SCRIPT_NAME}_nta_coverage.csv"
    nta_coverage.to_csv(p, index=False, encoding="utf-8-sig")
    files["nta_coverage"] = str(p)

    if landuse_distribution is not None and not landuse_distribution.empty:
        p = summary_dir / f"{SCRIPT_NAME}_landuse_distribution.csv"
        landuse_distribution.to_csv(p, index=False, encoding="utf-8-sig")
        files["landuse_distribution"] = str(p)

    mapping_ref = pd.DataFrame(
        [
            {
                "landuse_category": k,
                "landuse_label": LANDUSE_CATEGORY_LABELS[k],
                "raw_pluto_codes": ", ".join([code for code, cat in LANDUSE_CODE_TO_CATEGORY.items() if cat == k]),
            }
            for k in LANDUSE_CATEGORIES
        ]
    )
    p = summary_dir / f"{SCRIPT_NAME}_landuse_mapping_reference.csv"
    mapping_ref.to_csv(p, index=False, encoding="utf-8-sig")
    files["landuse_mapping_reference"] = str(p)

    # Top NTAs by selected built-environment features.
    top_rows = []
    for feature in [
        "pluto_parcel_count",
        "pluto_parcel_density_per_km2",
        "pluto_total_units_sum",
        "pluto_units_total_density_per_km2",
        "pluto_observed_built_far",
        "pluto_commercial_activity_lot_area_share",
        "pluto_residential_lot_area_share",
        "pluto_landuse_mixed_use_index",
    ]:
        if feature in out.columns:
            cols = [c for c in ["nta2020", "ntaname", "boroname", feature] if c in out.columns]
            tmp = out[cols].sort_values(feature, ascending=False).head(20).copy()
            tmp.insert(0, "rank_feature", feature)
            top_rows.append(tmp)
    if top_rows:
        p = summary_dir / f"{SCRIPT_NAME}_top_nta_examples.csv"
        pd.concat(top_rows, ignore_index=True, sort=False).to_csv(p, index=False, encoding="utf-8-sig")
        files["top_nta_examples"] = str(p)

    return files


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build/verify NTA-level PLUTO land-use features for SIMC NYC.")
    default_project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=default_project_root, help="Project root. Default: parent of code_processdata/.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input PLUTO CSV. Default: data/raw/nyc_pluto/nyc_pluto_nta_landuse_features.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output feature CSV. Default: data/processed/feature_tables/pluto_nta_landuse_features.csv",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=None,
        help="Summary/check output directory. Default: data/processed/_feature_summaries",
    )
    parser.add_argument(
        "--nta-reference",
        type=Path,
        default=None,
        help="NTA reference table. Default: data/processed/weekly_311/nyc_311_weekly_by_nta_category.csv.gz",
    )
    parser.add_argument(
        "--nta-geojson",
        type=Path,
        default=None,
        help="NTA GeoJSON for area. Default: data/raw/nyc_nta/nyc_nta_2020_clean.geojson",
    )
    parser.add_argument("--area-crs", default="EPSG:32618", help="Projected CRS for NTA area computation. Default: EPSG:32618")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    parser.add_argument("--strict", action="store_true", help="Fail if duplicate NTA output rows or missing all area values.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    start = time.time()
    args = build_arg_parser().parse_args(argv)

    project_root: Path = args.project_root.resolve()
    input_file = args.input or (project_root / "data" / "raw" / "nyc_pluto" / "nyc_pluto_nta_landuse_features.csv")
    output_file = args.output or (project_root / "data" / "processed" / "feature_tables" / "pluto_nta_landuse_features.csv")
    summary_dir = args.summary_dir or (project_root / "data" / "processed" / "_feature_summaries")
    nta_reference_file = args.nta_reference or (project_root / "data" / "processed" / "weekly_311" / "nyc_311_weekly_by_nta_category.csv.gz")
    nta_geojson_file = args.nta_geojson or (project_root / "data" / "raw" / "nyc_nta" / "nyc_nta_2020_clean.geojson")

    ensure_output_path(output_file, args.overwrite)

    summary: Dict[str, object] = {
        "script": SCRIPT_NAME,
        "status": "started",
        "input_file": str(input_file),
        "output_file": str(output_file),
        "summary_dir": str(summary_dir),
        "project_root": str(project_root),
        "current_year_for_age": CURRENT_YEAR_FOR_AGE,
    }

    try:
        df = read_csv_safely(input_file)
        detected = detect_columns(df.columns)
        summary.update(
            {
                "input_rows_raw": int(len(df)),
                "input_columns_raw": list(df.columns),
                "detected_columns": detected,
            }
        )

        nta_ref, ref_info = load_nta_reference(nta_reference_file)
        area_df, area_info = load_nta_area(nta_geojson_file, target_crs=args.area_crs)
        summary.update(ref_info)
        summary.update(area_info)

        nta_col = detected.get("nta2020")
        if nta_col is None:
            raise ValueError(
                "Could not detect an NTA column. Expected one of: nta2020, nta, nta_code, ntacode, nta2020code."
            )

        landuse_distribution = None
        if detect_if_preaggregated(df, nta_col):
            out, mode_info = standardize_preaggregated(df, detected, nta_ref, area_df)
        else:
            out, mode_info, landuse_distribution = aggregate_parcel_level(df, detected, nta_ref, area_df)
        summary.update(mode_info)

        # Final quality metrics.
        summary.update(
            {
                "output_rows": int(len(out)),
                "output_columns": list(out.columns),
                "output_unique_nta2020": int(out["nta2020"].nunique(dropna=True)) if "nta2020" in out.columns else 0,
                "output_duplicate_nta2020_rows": int(out.duplicated("nta2020").sum()) if "nta2020" in out.columns else None,
                "output_rows_with_area": int(out["nta_area_km2"].notna().sum()) if "nta_area_km2" in out.columns else 0,
                "output_rows_missing_area": int(out["nta_area_km2"].isna().sum()) if "nta_area_km2" in out.columns else int(len(out)),
                "output_rows_with_positive_area": int((out["nta_area_km2"] > 0).sum()) if "nta_area_km2" in out.columns else 0,
                "landuse_categories": LANDUSE_CATEGORIES,
                "landuse_category_labels": LANDUSE_CATEGORY_LABELS,
                "landuse_composite_groups": LANDUSE_COMPOSITE_GROUPS,
            }
        )

        if args.strict:
            if summary["output_duplicate_nta2020_rows"] != 0:
                raise ValueError("Duplicate NTA rows found in output under --strict.")
            if summary["output_rows_with_area"] == 0:
                raise ValueError("No NTA area values were computed under --strict.")

        # Write main output.
        out.to_csv(output_file, index=False, encoding="utf-8-sig")
        summary["status"] = "done"
        summary["elapsed_seconds"] = round(time.time() - start, 3)
        summary["elapsed_minutes"] = round((time.time() - start) / 60.0, 3)

        summary_files = write_summary_files(out, df, summary, detected, landuse_distribution, output_file, summary_dir)
        summary["summary_files"] = summary_files
        # Rewrite summary once paths are known.
        with open(summary_dir / f"{SCRIPT_NAME}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        return 0
    except Exception as exc:
        summary["status"] = "error"
        summary["error_type"] = type(exc).__name__
        summary["error_message"] = str(exc)
        summary["elapsed_seconds"] = round(time.time() - start, 3)
        summary_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_dir / f"{SCRIPT_NAME}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
