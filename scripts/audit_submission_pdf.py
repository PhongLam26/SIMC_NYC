#!/usr/bin/env python
"""Audit the SIMC review-manuscript PDF.

The SIMC submission PDF must keep URLs/DOIs/emails as visible text while
removing PDF navigation structures, link annotations, page numbers, headers,
and footers.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader


EXPECTED_STRINGS = [
    "Explainable Early Warning for Next-Week",
    "Abnormal Reported 311 Demand",
    "Tran Dai Phong Lam",
    "Thu Le",
    "Nguyen Quoc Hung",
    "Nguyen Trung Trinh",
    "phonglam2599@gmail.com",
    "thulvm@fpt.edu.vn",
    "hungtvt2222@gmail.com",
    "trinhnguyen112355@gmail.com",
    "https://github.com/PhongLam26/SIMC_NYC",
    "10.1371/journal.pone.0186314",
    "76ig-c548",
    "erm2-nwe9",
    "9nt8-h7nd",
    "GHCND:USW00094728",
    "Figure 1",
    "[1]",
    "PR-AUC 0.3165",
    "F1 0.3613",
    "Precision 0.2635",
    "Recall 0.5744",
    "Precision@1% 0.5697",
    "Precision@5% 0.4180",
    "Brier score 0.0869",
    "Five-seed PR-AUC 0.3185",
    "PR-AUC difference 0.1637",
    "precision@5% difference 0.2425",
]

FORBIDDEN_EDGE_TEXT = [
    "Explainable Early Warning for Next-Week Abnormal Reported 311 Demand",
    "Tran Dai Phong Lam",
    "Thu Le",
    "Nguyen Quoc Hung",
    "Nguyen Trung Trinh",
    "Springer",
    "SIMC",
    ".pdf",
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def check_pdf_structure(reader: PdfReader, failures: list[str], report: list[str]) -> None:
    root = reader.trailer["/Root"]
    for key in ["/Outlines", "/OpenAction", "/Names", "/Dests", "/PageLabels"]:
        if key in root:
            failures.append(f"PDF catalog contains forbidden key {key}")
        else:
            report.append(f"PASS catalog has no {key}")

    try:
        outline = reader.outline
    except Exception as exc:  # pragma: no cover - defensive for malformed PDFs
        failures.append(f"Could not inspect PDF outline: {exc}")
        return

    outline_count = len(outline) if isinstance(outline, list) else 0
    if outline_count:
        failures.append(f"Bookmark/outline count is {outline_count}, expected 0")
    else:
        report.append("PASS bookmark/outline count = 0")


def check_links(reader: PdfReader, failures: list[str], report: list[str]) -> None:
    link_count = 0
    annot_count = 0
    link_pages: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        for annotation_ref in page.get("/Annots", []) or []:
            annot_count += 1
            annotation = annotation_ref.get_object()
            subtype = annotation.get("/Subtype")
            if subtype == "/Link":
                link_count += 1
                link_pages.append(str(page_number))

    if link_count:
        failures.append(
            f"Found {link_count} /Subtype /Link annotations on pages "
            f"{', '.join(sorted(set(link_pages), key=int))}"
        )
    else:
        report.append("PASS link annotation count = 0")
    report.append(f"INFO total annotation count = {annot_count}")


def check_pages(reader: PdfReader, expected_pages: int, failures: list[str], report: list[str]) -> None:
    page_count = len(reader.pages)
    if page_count != expected_pages:
        failures.append(f"Page count is {page_count}, expected {expected_pages}")
    else:
        report.append(f"PASS page count = {expected_pages}")

    for page_number, page in enumerate(reader.pages, start=1):
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        if abs(width - 595.276) > 2 or abs(height - 841.89) > 2:
            failures.append(
                f"Page {page_number} is not A4: {width:.3f} x {height:.3f} pt"
            )
    if not any("not A4" in failure for failure in failures):
        report.append("PASS all pages are A4")


def check_visible_text(reader: PdfReader, failures: list[str], report: list[str]) -> None:
    page_texts = [(page.extract_text() or "") for page in reader.pages]
    full_text = "\n".join(page_texts)
    normalized = normalize_text(full_text)
    compacted = compact_text(full_text)

    missing = []
    for expected in EXPECTED_STRINGS:
        if expected in normalized:
            continue
        if compact_text(expected) in compacted:
            continue
        missing.append(expected)

    if missing:
        failures.append("Missing required visible text: " + "; ".join(missing))
    else:
        report.append("PASS required title/authors/emails/URLs/DOIs/citations/metrics remain visible")

    page_number_failures: list[str] = []
    edge_failures: list[str] = []
    for page_number, text in enumerate(page_texts, start=1):
        lines = [normalize_text(line) for line in text.splitlines() if normalize_text(line)]
        if lines and lines[-1] == str(page_number):
            page_number_failures.append(str(page_number))

        if page_number > 1:
            edge_lines = lines[:3] + lines[-3:]
            edge_text = " | ".join(edge_lines)
            for forbidden in FORBIDDEN_EDGE_TEXT:
                if forbidden in edge_text:
                    edge_failures.append(f"page {page_number}: {forbidden}")

    if page_number_failures:
        failures.append(
            "Standalone page-number footer detected on pages "
            + ", ".join(page_number_failures)
        )
    else:
        report.append("PASS no standalone page-number footer detected in extracted page text")

    if edge_failures:
        failures.append("Possible header/footer running text detected: " + "; ".join(edge_failures))
    else:
        report.append("PASS no known running-header/footer strings detected at page edges")


def check_fonts(pdf_path: Path, failures: list[str], report: list[str]) -> None:
    pdffonts = shutil.which("pdffonts")
    if not pdffonts:
        report.append("WARN pdffonts not found; embedded-font audit skipped")
        return

    result = subprocess.run(
        [pdffonts, str(pdf_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    unembedded = []
    for line in result.stdout.splitlines():
        match = re.search(r"\s(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", line)
        if match and match.group(1) != "yes":
            unembedded.append(line.strip())

    if unembedded:
        failures.append("Unembedded fonts detected: " + " | ".join(unembedded))
    else:
        report.append("PASS all fonts reported by pdffonts are embedded")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--expected-pages", type=int, default=13)
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"FAIL PDF does not exist: {args.pdf}", file=sys.stderr)
        return 2

    reader = PdfReader(str(args.pdf))
    failures: list[str] = []
    report: list[str] = []

    check_pdf_structure(reader, failures, report)
    check_links(reader, failures, report)
    check_pages(reader, args.expected_pages, failures, report)
    check_visible_text(reader, failures, report)
    check_fonts(args.pdf, failures, report)

    for line in report:
        print(line)

    if failures:
        print("FAIL SIMC submission PDF audit failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("PASS automated SIMC submission PDF audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
