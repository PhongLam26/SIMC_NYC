from pathlib import Path
import pandas as pd
import json
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "_schema_checks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = {
    "nyc_311": [
        "unique_key",
        "created_date",
        "closed_date",
        "complaint_type",
        "borough",
        "latitude",
        "longitude",
    ],
    "noaa_weather": [
        "date",
        "TMAX",
        "TMIN",
        "PRCP",
        "SNOW",
    ],
    "nyc_pluto": [
        "latitude",
        "longitude",
        "landuse",
        "lotarea",
        "bldgarea",
    ],
    "osm_poi": [
        "latitude",
        "longitude",
    ],
}


def list_files(folder: Path):
    patterns = ["*.csv", "*.csv.gz", "*.geojson", "*.json", "*.parquet"]
    files = []
    for p in patterns:
        files.extend(folder.rglob(p))
    return sorted(files)


def read_sample(file_path: Path, nrows=5000):
    suffixes = "".join(file_path.suffixes).lower()

    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(file_path, nrows=nrows, low_memory=False)

    if suffixes.endswith(".parquet"):
        return pd.read_parquet(file_path)

    if suffixes.endswith(".geojson") or suffixes.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if "features" in obj:
            rows = []
            for feat in obj["features"][:nrows]:
                props = feat.get("properties", {})
                props["has_geometry"] = feat.get("geometry") is not None
                rows.append(props)
            return pd.DataFrame(rows)

        return pd.json_normalize(obj)

    raise ValueError(f"Unsupported file type: {file_path}")


def normalize_colname(c):
    return str(c).strip()


def detect_date_range(df: pd.DataFrame):
    candidate_cols = [
        "created_date",
        "closed_date",
        "date",
        "version",
    ]

    result = {}

    for col in candidate_cols:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce")
            if s.notna().sum() > 0:
                result[col] = {
                    "min": str(s.min()),
                    "max": str(s.max()),
                    "non_null_sample_rows": int(s.notna().sum()),
                }

    return result


def check_numeric_range(df: pd.DataFrame):
    result = {}

    for col in ["latitude", "longitude", "TMAX", "TMIN", "PRCP", "SNOW", "AWND"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            result[col] = {
                "non_null": int(s.notna().sum()),
                "min": float(s.min()) if s.notna().sum() else None,
                "max": float(s.max()) if s.notna().sum() else None,
                "mean": float(s.mean()) if s.notna().sum() else None,
            }

    return result


def infer_dataset_name(file_path: Path):
    text = str(file_path).lower()

    if "nyc_311" in text:
        return "nyc_311"
    if "noaa_weather" in text or "weather" in text:
        return "noaa_weather"
    if "nyc_nta" in text or "nta" in text:
        return "nyc_nta"
    if "osm_poi" in text or "osm" in text:
        return "osm_poi"
    if "nyc_pluto" in text or "pluto" in text:
        return "nyc_pluto"

    return "unknown"


def check_required_columns(dataset_name, columns):
    required = REQUIRED_COLUMNS.get(dataset_name, [])
    lower_map = {c.lower(): c for c in columns}

    missing = []
    found = []

    for req in required:
        if req.lower() in lower_map:
            found.append(lower_map[req.lower()])
        else:
            missing.append(req)

    return found, missing


def summarize_file(file_path: Path):
    dataset_name = infer_dataset_name(file_path)

    try:
        df = read_sample(file_path)
        df.columns = [normalize_colname(c) for c in df.columns]

        found_required, missing_required = check_required_columns(dataset_name, df.columns)

        summary = {
            "dataset": dataset_name,
            "file": str(file_path.relative_to(PROJECT_ROOT)),
            "status": "OK",
            "sample_rows_read": int(len(df)),
            "num_columns": int(len(df.columns)),
            "columns": list(df.columns),
            "required_found": found_required,
            "required_missing": missing_required,
            "date_ranges_detected": detect_date_range(df),
            "numeric_ranges_detected": check_numeric_range(df),
        }

        if "complaint_type" in df.columns:
            summary["top_complaint_type_sample"] = (
                df["complaint_type"]
                .astype(str)
                .value_counts()
                .head(20)
                .to_dict()
            )

        if "borough" in df.columns:
            summary["borough_sample_counts"] = (
                df["borough"]
                .astype(str)
                .value_counts()
                .head(20)
                .to_dict()
            )

        if "nta2020" in df.columns:
            summary["nta2020_non_null_sample"] = int(df["nta2020"].notna().sum())

        if "landuse" in df.columns:
            summary["landuse_sample_counts"] = (
                df["landuse"]
                .astype(str)
                .value_counts()
                .head(20)
                .to_dict()
            )

        return summary

    except Exception as e:
        return {
            "dataset": dataset_name,
            "file": str(file_path.relative_to(PROJECT_ROOT)),
            "status": "ERROR",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def main():
    print("=" * 80)
    print("RAW DATA SCHEMA CHECK")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw dir:      {RAW_DIR}")
    print(f"Output dir:   {OUTPUT_DIR}")
    print()

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR does not exist: {RAW_DIR}")

    all_summaries = []

    raw_subfolders = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])

    if not raw_subfolders:
        print("No raw subfolders found.")
        return

    for folder in raw_subfolders:
        print("\n" + "=" * 80)
        print(f"Checking folder: {folder.relative_to(PROJECT_ROOT)}")
        print("=" * 80)

        files = list_files(folder)

        if not files:
            print("No data files found.")
            continue

        print(f"Found {len(files)} files.")

        # Với 311 có thể có rất nhiều part, chỉ sample 10 file đầu + 10 file cuối.
        if "nyc_311" in str(folder).lower() and len(files) > 20:
            files_to_check = files[:10] + files[-10:]
            print(f"NYC 311 has many files. Checking sample {len(files_to_check)} files only.")
        else:
            files_to_check = files

        for fp in files_to_check:
            print(f"\n- {fp.relative_to(PROJECT_ROOT)}")
            summary = summarize_file(fp)
            all_summaries.append(summary)

            print(f"  Status: {summary['status']}")

            if summary["status"] == "OK":
                print(f"  Rows sampled: {summary['sample_rows_read']}")
                print(f"  Num columns:   {summary['num_columns']}")

                if summary["required_missing"]:
                    print(f"  Missing required columns: {summary['required_missing']}")
                else:
                    if summary["dataset"] in REQUIRED_COLUMNS:
                        print("  Required columns: OK")

                if summary["date_ranges_detected"]:
                    print(f"  Date ranges: {summary['date_ranges_detected']}")

                if summary["numeric_ranges_detected"]:
                    print(f"  Numeric ranges: {summary['numeric_ranges_detected']}")

            else:
                print(f"  ERROR: {summary['error']}")

    # Save full JSON report
    report_path = OUTPUT_DIR / "raw_schema_check_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    # Save compact CSV report
    compact_rows = []
    for s in all_summaries:
        compact_rows.append({
            "dataset": s.get("dataset"),
            "file": s.get("file"),
            "status": s.get("status"),
            "sample_rows_read": s.get("sample_rows_read"),
            "num_columns": s.get("num_columns"),
            "required_missing": ", ".join(s.get("required_missing", [])) if s.get("required_missing") else "",
            "columns": " | ".join(s.get("columns", [])) if s.get("columns") else "",
            "error": s.get("error", ""),
        })

    compact_df = pd.DataFrame(compact_rows)
    compact_path = OUTPUT_DIR / "raw_schema_check_report.csv"
    compact_df.to_csv(compact_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Saved JSON report: {report_path}")
    print(f"Saved CSV report:  {compact_path}")


if __name__ == "__main__":
    main()