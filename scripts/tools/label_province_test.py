#!/usr/bin/env python3
"""
scripts/tools/label_province_test.py
====================================
Human-in-the-loop province labeller for the 149 real TEST frames — unblocks the
MEASURED composed-plate accuracy (roadmap 1.1) and full validation of 2.2.

Why this tool: the test frames have NUMBER ground truth but no PROVINCE ground
truth. Labelling province from scratch is slow, so this PRE-FILLS each frame with
the province classifier's prediction (~97% accurate) and builds readable contact
sheets. You then just CORRECT the few wrong rows in the CSV — verifying against
the **English province name printed at the bottom of each plate**.

Test-label integrity: the pre-fill is only a starting point; the CSV is meant to
be HUMAN-verified (the classifier must not grade itself). Correct anything wrong.

Workflow
--------
1. Build the pre-filled CSV + contact sheets:
       python scripts/tools/label_province_test.py --build
   -> data/crnn_crops/province_test_labels.csv   (columns: image,province_class,...)
   -> results/province_sheets/sheet_00.png ...    (thumbnails + predicted province)

2. Open the sheets, and in the CSV fix `province_class` for any plate whose
   printed English province differs from `province_latin`. (Legend printed below
   and saved to results/province_sheets/LEGEND.txt.)

3. Measure composed accuracy for real:
       python scripts/system/benchmark_composed.py \
           --province-gt data/crnn_crops/province_test_labels.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CONFIG = PROJECT_ROOT / "configs" / "system_config.yaml"
REAL_CSV = PROJECT_ROOT / "data" / "crnn_crops" / "real_labels.csv"
TEST_IMAGES = PROJECT_ROOT / "data" / "annotated" / "test" / "images"
OUT_CSV = PROJECT_ROOT / "data" / "crnn_crops" / "province_test_labels.csv"
SHEETS_DIR = PROJECT_ROOT / "results" / "province_sheets"
PER_SHEET = 15          # 3 cols x 5 rows
THUMB_W = 360


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else PROJECT_ROOT / q


def test_rows() -> list[dict]:
    rows = []
    with open(REAL_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if "crnn_crops/test/" not in r["image_path"].replace("\\", "/"):
                continue
            fn = Path(r["image_path"]).name
            img = TEST_IMAGES / fn
            if img.exists():
                rows.append({"file": fn, "image": img,
                             "gt_number": r["plate_text"].strip()})
    rows.sort(key=lambda x: x["file"])
    return rows


def build() -> None:
    import cv2
    import numpy as np
    import yaml
    from detection.detector import PlateDetector
    from recognition.province_classifier import ProvinceClassifier
    from recognition.province_map import PROVINCE_LATIN, province_latin

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    detector = PlateDetector(_resolve(cfg["yolo_weights"]),
                             conf=cfg.get("gate", {}).get("yolo_confidence_threshold", 0.5))
    clf = ProvinceClassifier(_resolve("models/recognition/province_classifier_best.pth"))

    rows = test_rows()
    if not rows:
        print("[X] no test rows found."); sys.exit(1)
    print(f"[build] {len(rows)} test frames — predicting province (pre-fill)...")

    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    records, cells = [], []
    for i, row in enumerate(rows):
        frame = cv2.imread(str(row["image"]))
        if frame is None:
            continue
        dets = detector.detect(frame)
        crop = dets[0]["crop"] if dets else frame     # fallback: whole frame
        pid, conf = clf.predict(crop)
        latin = province_latin(pid) if pid < 25 else "Other"
        records.append({
            "image": row["file"],
            "province_class": pid,
            "province_latin": latin,
            "pred_conf": round(float(conf), 3),
            "gt_number": row["gt_number"],
            "note": "",
        })
        # thumbnail for the contact sheet
        h, w = frame.shape[:2]
        th = int(THUMB_W * h / max(1, w))
        thumb = cv2.resize(frame, (THUMB_W, min(th, 260)))
        cells.append((i, thumb, latin, pid, conf, row["gt_number"]))

    # write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "province_class", "province_latin",
                                          "pred_conf", "gt_number", "note"])
        w.writeheader()
        w.writerows(records)

    # contact sheets (Latin renders in OpenCV; Khmer does not, hence English names)
    cols = 3
    pad, header = 8, 34
    n_sheets = 0
    for start in range(0, len(cells), PER_SHEET):
        page = cells[start:start + PER_SHEET]
        rows_n = (len(page) + cols - 1) // cols
        cell_h = max(c[1].shape[0] for c in page) + header
        cell_w = THUMB_W
        sheet = np.full((rows_n * (cell_h + pad) + pad,
                         cols * (cell_w + pad) + pad, 3), 40, np.uint8)
        for k, (idx, thumb, latin, pid, conf, gtnum) in enumerate(page):
            r, c = divmod(k, cols)
            y0 = pad + r * (cell_h + pad)
            x0 = pad + c * (cell_w + pad)
            cv2.putText(sheet, f"#{idx}  {latin} ({pid})  {conf:.2f}",
                        (x0 + 2, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1)
            cv2.putText(sheet, f"num={gtnum}", (x0 + 2, y0 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 180), 1)
            th_h, th_w = thumb.shape[:2]
            sheet[y0 + header:y0 + header + th_h, x0:x0 + th_w] = thumb
        out = SHEETS_DIR / f"sheet_{n_sheets:02d}.png"
        cv2.imwrite(str(out), sheet)
        n_sheets += 1

    # legend
    legend = ["Province class legend (class_id: Latin name)", "-" * 40]
    legend += [f"{i:>2}: {name}" for i, name in enumerate(PROVINCE_LATIN)]
    legend.append("25: Other (Cambodia/Police/RCAF/State -> no province prefix)")
    (SHEETS_DIR / "LEGEND.txt").write_text("\n".join(legend), encoding="utf-8")

    print(f"[ok] wrote {OUT_CSV.relative_to(PROJECT_ROOT)} ({len(records)} rows, pre-filled)")
    print(f"[ok] wrote {n_sheets} contact sheets -> {SHEETS_DIR.relative_to(PROJECT_ROOT)}")
    print("\n" + "\n".join(legend))
    print("\nNext: correct wrong province_class rows in the CSV (check each plate's")
    print("English bottom line against province_latin), then run:")
    print("  python scripts/system/benchmark_composed.py "
          "--province-gt data/crnn_crops/province_test_labels.csv")


def stats() -> None:
    """Quick summary of the current label CSV (how many verified/changed)."""
    if not OUT_CSV.exists():
        print("(no province_test_labels.csv yet — run --build first)"); return
    rows = list(csv.DictReader(open(OUT_CSV, encoding="utf-8")))
    noted = sum(1 for r in rows if r.get("note"))
    low = sum(1 for r in rows if float(r.get("pred_conf", 1)) < 0.7)
    print(f"{len(rows)} rows | {noted} with notes | "
          f"{low} low-confidence pre-fills to double-check first")


def main() -> None:
    ap = argparse.ArgumentParser(description="province labeller for the test set (roadmap 1.1)")
    ap.add_argument("--build", action="store_true",
                    help="predict + write pre-filled CSV and contact sheets")
    ap.add_argument("--stats", action="store_true", help="summarise the label CSV")
    args = ap.parse_args()
    if args.build:
        build()
    elif args.stats:
        stats()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
