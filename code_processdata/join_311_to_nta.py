from pathlib import Path
import argparse
import gc
import json
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_311_DIR = PROJECT_ROOT / "data" / "raw" / "nyc_311"
NTA_PATH = PROJECT_ROOT / "data" / "raw" / "nyc_nta" / "nyc_nta_2020_clean.geojson"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "nyc_311_with_nta"
SUMMARY_DIR = PROJECT_ROOT / "data" / "processed" / "_join_summaries"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# Bbox NYC để loại tọa độ lỗi trước khi spatial join.
NYC_LAT_MIN = 40.45
NYC_LAT_MAX = 40.95
NYC_LON_MIN = -74.30
NYC_LON_MAX = -73.65

# Chỉ giữ các cột cần thiết để giảm RAM và giảm dung lượng output.
# Nếu file nào thiếu cột thì script tự bỏ qua.
KEEP_311_COLUMNS = [
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
    "open_data_channel_type",
]

NTA_COLUMNS = [
    "nta2020",
    "ntaname",
    "boroname",
    "borocode",
    "cdta2020",
    "cdtaname",
    "ntatype",
]


def log(msg: str):
    print(msg, flush=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa tên cột về lowercase, bỏ khoảng trắng, đổi space thành underscore.
    """
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in df.columns
    ]
    return df


def load_nta():
    """
    Load NTA boundary bằng geopandas.
    """
    try:
        import geopandas as gpd
    except ImportError as e:
        raise ImportError(
            "Bạn chưa cài geopandas. Chạy:\n"
            "pip install geopandas pyogrio shapely rtree"
        ) from e

    if not NTA_PATH.exists():
        raise FileNotFoundError(f"Không thấy file NTA: {NTA_PATH}")

    log(f"Loading NTA: {NTA_PATH}")
    nta = gpd.read_file(NTA_PATH)
    nta = normalize_columns(nta)

    # Đảm bảo CRS là WGS84.
    if nta.crs is None:
        nta = nta.set_crs("EPSG:4326")
    else:
        nta = nta.to_crs("EPSG:4326")

    needed = [c for c in NTA_COLUMNS if c in nta.columns] + ["geometry"]
    nta = nta[needed].copy()

    missing = [c for c in ["nta2020", "ntaname", "boroname", "geometry"] if c not in nta.columns]
    if missing:
        raise ValueError(f"NTA thiếu cột quan trọng: {missing}")

    log(f"NTA polygons: {len(nta):,}")
    log(f"NTA columns: {list(nta.columns)}")
    return nta


def list_311_files():
    files = sorted(RAW_311_DIR.glob("*.csv.gz"))
    files = [
        fp for fp in files
        if "download_summary" not in fp.name.lower()
        and "summary" not in fp.name.lower()
    ]
    return files


def make_output_path(input_path: Path) -> Path:
    name = input_path.name

    if name.endswith(".csv.gz"):
        base = name[:-7]
    elif name.endswith(".csv"):
        base = name[:-4]
    else:
        base = input_path.stem

    return OUTPUT_DIR / f"{base}_with_nta.csv.gz"


def get_input_columns(file_path: Path):
    sample = pd.read_csv(file_path, nrows=5, low_memory=False)
    sample = normalize_columns(sample)
    return list(sample.columns)


def select_usecols(input_columns):
    selected = [c for c in KEEP_311_COLUMNS if c in input_columns]

    # Bắt buộc phải có tọa độ.
    if "latitude" not in selected or "longitude" not in selected:
        raise ValueError(
            "File 311 thiếu latitude/longitude sau khi normalize columns. "
            f"Columns hiện có: {input_columns}"
        )

    # Các cột cốt lõi nên có.
    core_missing = [
        c for c in ["unique_key", "created_date", "complaint_type"]
        if c not in selected
    ]
    if core_missing:
        log(f"  WARNING: thiếu cột core {core_missing}")

    return selected


def clean_311_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = normalize_columns(chunk)

    # Ép tọa độ về numeric.
    chunk["latitude"] = pd.to_numeric(chunk["latitude"], errors="coerce")
    chunk["longitude"] = pd.to_numeric(chunk["longitude"], errors="coerce")

    before = len(chunk)

    # Bỏ dòng thiếu tọa độ.
    chunk = chunk.dropna(subset=["latitude", "longitude"])

    # Lọc bbox NYC.
    chunk = chunk[
        (chunk["latitude"] >= NYC_LAT_MIN)
        & (chunk["latitude"] <= NYC_LAT_MAX)
        & (chunk["longitude"] >= NYC_LON_MIN)
        & (chunk["longitude"] <= NYC_LON_MAX)
    ].copy()

    removed = before - len(chunk)

    return chunk, removed


def spatial_join_chunk(chunk: pd.DataFrame, nta):
    """
    Spatial join một chunk 311 với NTA.
    """
    import geopandas as gpd

    if chunk.empty:
        for c in NTA_COLUMNS:
            if c not in chunk.columns:
                chunk[c] = pd.NA
        return chunk

    gdf = gpd.GeoDataFrame(
        chunk,
        geometry=gpd.points_from_xy(chunk["longitude"], chunk["latitude"]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(
        gdf,
        nta,
        how="left",
        predicate="within",
    )

    # Bỏ các cột geometry/index phụ để output nhẹ.
    drop_cols = ["geometry", "index_right"]
    joined = joined.drop(columns=[c for c in drop_cols if c in joined.columns])

    # Đảm bảo đủ cột NTA dù có file nào thiếu metadata.
    for c in NTA_COLUMNS:
        if c not in joined.columns:
            joined[c] = pd.NA

    return pd.DataFrame(joined)


def append_csv_gz(df: pd.DataFrame, output_path: Path, write_header: bool):
    """
    Ghi append từng chunk ra gzip.
    Pandas có thể append gzip thành concatenated gzip stream, đọc lại vẫn được.
    """
    df.to_csv(
        output_path,
        mode="a",
        index=False,
        header=write_header,
        compression={"method": "gzip", "compresslevel": 1},
        encoding="utf-8",
    )


def process_one_file(file_path: Path, nta, chunksize: int, overwrite: bool):
    output_path = make_output_path(file_path)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

    if output_path.exists() and not overwrite:
        log(f"SKIP existing: {output_path.relative_to(PROJECT_ROOT)}")
        return {
            "file": str(file_path.relative_to(PROJECT_ROOT)),
            "output": str(output_path.relative_to(PROJECT_ROOT)),
            "status": "skipped_existing",
        }

    if tmp_output_path.exists():
        tmp_output_path.unlink()

    log("")
    log("=" * 90)
    log(f"Processing: {file_path.relative_to(PROJECT_ROOT)}")
    log(f"Output:     {output_path.relative_to(PROJECT_ROOT)}")
    log("=" * 90)

    input_columns = get_input_columns(file_path)
    usecols = select_usecols(input_columns)

    log(f"Using columns: {usecols}")

    total_raw_rows = 0
    total_after_clean = 0
    total_removed_bad_coord = 0
    total_matched_nta = 0
    total_unmatched_nta = 0
    chunk_count = 0
    start_time = time.time()

    reader = pd.read_csv(
        file_path,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    )

    write_header = True

    for chunk in reader:
        chunk_count += 1
        raw_rows = len(chunk)
        total_raw_rows += raw_rows

        chunk, removed = clean_311_chunk(chunk)
        total_removed_bad_coord += removed
        total_after_clean += len(chunk)

        joined = spatial_join_chunk(chunk, nta)

        matched = int(joined["nta2020"].notna().sum()) if "nta2020" in joined.columns else 0
        unmatched = int(len(joined) - matched)
        total_matched_nta += matched
        total_unmatched_nta += unmatched

        append_csv_gz(joined, tmp_output_path, write_header=write_header)
        write_header = False

        elapsed = time.time() - start_time
        rate = total_raw_rows / elapsed if elapsed > 0 else 0

        log(
            f"  chunk {chunk_count:04d} | "
            f"raw {raw_rows:,} | "
            f"clean {len(chunk):,} | "
            f"matched {matched:,} | "
            f"unmatched {unmatched:,} | "
            f"speed {rate:,.0f} rows/s"
        )

        del chunk, joined
        gc.collect()

    if output_path.exists():
        output_path.unlink()

    tmp_output_path.rename(output_path)

    elapsed = time.time() - start_time

    match_rate = (
        total_matched_nta / total_after_clean
        if total_after_clean > 0
        else 0
    )

    summary = {
        "file": str(file_path.relative_to(PROJECT_ROOT)),
        "output": str(output_path.relative_to(PROJECT_ROOT)),
        "status": "done",
        "chunksize": chunksize,
        "chunk_count": chunk_count,
        "total_raw_rows": total_raw_rows,
        "total_after_clean": total_after_clean,
        "total_removed_bad_coord": total_removed_bad_coord,
        "total_matched_nta": total_matched_nta,
        "total_unmatched_nta": total_unmatched_nta,
        "match_rate_after_clean": match_rate,
        "elapsed_seconds": elapsed,
    }

    log("-" * 90)
    log(f"DONE file: {file_path.name}")
    log(f"Raw rows:           {total_raw_rows:,}")
    log(f"After coord clean:  {total_after_clean:,}")
    log(f"Removed bad coords: {total_removed_bad_coord:,}")
    log(f"Matched NTA:        {total_matched_nta:,}")
    log(f"Unmatched NTA:      {total_unmatched_nta:,}")
    log(f"Match rate:         {match_rate:.4%}")
    log(f"Elapsed:            {elapsed / 60:.2f} minutes")
    log("-" * 90)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Số dòng đọc mỗi chunk. Laptop RAM thấp nên dùng 50k-100k.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Chỉ xử lý N file đầu để test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ghi đè output nếu đã tồn tại.",
    )
    args = parser.parse_args()

    log("=" * 90)
    log("JOIN NYC 311 WITH NTA")
    log("=" * 90)
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Raw 311 dir:  {RAW_311_DIR}")
    log(f"NTA path:     {NTA_PATH}")
    log(f"Output dir:   {OUTPUT_DIR}")
    log(f"Chunksize:    {args.chunksize:,}")
    log("=" * 90)

    nta = load_nta()
    files = list_311_files()

    if not files:
        raise FileNotFoundError(f"Không tìm thấy file 311 .csv.gz trong {RAW_311_DIR}")

    if args.limit_files is not None:
        files = files[:args.limit_files]

    log(f"311 files to process: {len(files):,}")

    all_summaries = []
    global_start = time.time()

    for i, fp in enumerate(files, start=1):
        log("")
        log(f"[{i}/{len(files)}]")

        try:
            summary = process_one_file(
                file_path=fp,
                nta=nta,
                chunksize=args.chunksize,
                overwrite=args.overwrite,
            )
        except Exception as e:
            summary = {
                "file": str(fp.relative_to(PROJECT_ROOT)),
                "status": "error",
                "error": str(e),
            }
            log(f"ERROR processing {fp.name}: {e}")

        all_summaries.append(summary)

        # Lưu summary sau mỗi file để nếu crash vẫn có log.
        summary_path = SUMMARY_DIR / "join_311_to_nta_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)

        summary_csv_path = SUMMARY_DIR / "join_311_to_nta_summary.csv"
        pd.DataFrame(all_summaries).to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

        gc.collect()

    total_elapsed = time.time() - global_start

    log("")
    log("=" * 90)
    log("ALL DONE")
    log("=" * 90)
    log(f"Processed files: {len(files):,}")
    log(f"Total elapsed: {total_elapsed / 60:.2f} minutes")
    log(f"Summary JSON: {SUMMARY_DIR / 'join_311_to_nta_summary.json'}")
    log(f"Summary CSV:  {SUMMARY_DIR / 'join_311_to_nta_summary.csv'}")


if __name__ == "__main__":
    main()