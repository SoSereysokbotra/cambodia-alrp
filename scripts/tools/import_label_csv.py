#!/usr/bin/env python3
"""
scripts/tools/import_label_csv.py
=================================
Merge a downloaded label CSV (from make_label_sheet.py's browser page) into
data/crnn_crops/real_labels.csv, de-duplicating by image_path (newest wins).

Run:
    python scripts/tools/import_label_csv.py "C:/Users/TUF/Downloads/label_sheet_test.csv"
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
OUT_CSV = PROJECT_ROOT / "data" / "crnn_crops" / "real_labels.csv"
ALLOWED = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ- ")


def read_csv(path: Path) -> dict[str, str]:
    rows = {}
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            p = (r.get("image_path") or "").strip()
            t = (r.get("plate_text") or "").strip().upper()
            t = "".join(c for c in t if c in ALLOWED).strip()
            if p and t:
                rows[p] = t
    return rows


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/tools/import_label_csv.py <downloaded.csv>")
        sys.exit(1)
    incoming = Path(sys.argv[1])
    if not incoming.exists():
        print(f"[X] file not found: {incoming}")
        sys.exit(1)

    existing = read_csv(OUT_CSV)
    new = read_csv(incoming)
    before = len(existing)
    added = updated = 0
    for p, t in new.items():
        if p in existing:
            if existing[p] != t:
                updated += 1
        else:
            added += 1
        existing[p] = t

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "plate_text"])
        for p in sorted(existing):
            w.writerow([p, existing[p]])

    print(f"imported {len(new)} rows from {incoming.name}")
    print(f"  added {added}, updated {updated}  (was {before}, now {len(existing)})")
    print(f"  -> {OUT_CSV}")
    print("\nNext: python scripts/recognition/evaluate_crnn_on_real.py --split test")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] import failed: {exc}")
        sys.exit(1)
