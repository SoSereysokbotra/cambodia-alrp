#!/usr/bin/env python3
"""
scripts/recognition/build_province_dataset.py
=============================================
Build a province-classification dataset by cropping plates from the ORIGINAL
Plate_v4 download (which still has the 29-class province labels) and mapping
them to this project's 26-class scheme (25 provinces + 'other').

Output:
    data/province_crops/{train,val,test}/{0..25}/*.jpg   (ImageFolder layout)
    data/province_crops/manifest.json                    (counts + split ratios)

Split: stratified per class (default 70/15/15).

Run:
    python scripts/recognition/build_province_dataset.py
    python scripts/recognition/build_province_dataset.py --src "C:/path/Plate_v4.v3i.yolov8"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src" / "recognition"))

from province_map import plate_v4_to_class, N_CLASSES, province_latin  # noqa: E402

DEFAULT_SRC = Path(r"C:/Users/TUF/OneDrive/Pictures/Downloads/Plate_v4.v3i.yolov8")
OUT_ROOT = PROJECT_ROOT / "data" / "province_crops"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CROP_SIZE = 128     # saved square crop size


def collect_boxes(src: Path) -> list[tuple[Path, tuple, int]]:
    """Return [(image_path, (cx,cy,w,h), class_id), ...] across all src splits."""
    items = []
    for split in ("train", "valid", "test"):
        img_dir = src / split / "images"
        lbl_dir = src / split / "labels"
        if not img_dir.is_dir():
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                continue
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                v4_id = int(float(parts[0]))
                cx, cy, w, h = (float(x) for x in parts[1:5])
                cls = plate_v4_to_class(v4_id)
                items.append((img, (cx, cy, w, h), cls))
    return items


def stratified_split(items, ratios=(0.70, 0.15, 0.15), seed=42):
    """Group by class, shuffle, split each class -> {train,val,test} index lists."""
    rng = random.Random(seed)
    by_cls: dict[int, list] = {}
    for it in items:
        by_cls.setdefault(it[2], []).append(it)
    assign = {"train": [], "val": [], "test": []}
    for cls, group in by_cls.items():
        rng.shuffle(group)
        n = len(group)
        n_tr = int(n * ratios[0])
        n_va = int(n * ratios[1])
        assign["train"] += group[:n_tr]
        assign["val"] += group[n_tr:n_tr + n_va]
        assign["test"] += group[n_tr + n_va:]
    return assign


def crop_and_save(item, split: str, idx: int) -> bool:
    import cv2
    img_path, (cx, cy, w, h), cls = item
    image = cv2.imread(str(img_path))
    if image is None:
        return False
    H, W = image.shape[:2]
    x1 = int((cx - w / 2) * W); x2 = int((cx + w / 2) * W)
    y1 = int((cy - h / 2) * H); y2 = int((cy + h / 2) * H)
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return False
    crop = cv2.resize(image[y1:y2, x1:x2], (CROP_SIZE, CROP_SIZE))
    out_dir = OUT_ROOT / split / str(cls)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{img_path.stem}_{idx}.jpg"), crop)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.src.is_dir():
        print(f"[X] Plate_v4 source not found: {args.src}")
        print("    Pass the correct path with --src.")
        sys.exit(1)

    print("=" * 60)
    print(" BUILD PROVINCE DATASET (26 classes: 25 provinces + other)")
    print("=" * 60)
    print(f"Source: {args.src}")

    items = collect_boxes(args.src)
    if not items:
        print("[X] no labelled boxes found in source.")
        sys.exit(1)
    print(f"Collected {len(items)} labelled plate boxes")

    assign = stratified_split(items, seed=args.seed)

    # per-class/per-split counts for the manifest
    counts = {s: {c: 0 for c in range(N_CLASSES)} for s in assign}
    saved = 0
    for split, group in assign.items():
        for i, item in enumerate(group):
            if crop_and_save(item, split, i):
                counts[split][item[2]] += 1
                saved += 1
        print(f"  [{split}] {sum(counts[split].values())} crops")

    manifest = {
        "n_classes": N_CLASSES,
        "crop_size": CROP_SIZE,
        "total_crops": saved,
        "splits": {s: sum(counts[s].values()) for s in counts},
        "per_class": {s: {str(c): counts[s][c] for c in range(N_CLASSES)}
                      for s in counts},
        "class_names": {str(c): province_latin(c) for c in range(N_CLASSES)},
    }
    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("-" * 60)
    print(f"DONE. {saved} crops -> {OUT_ROOT}")
    print(f"manifest -> {OUT_ROOT / 'manifest.json'}")
    # warn about empty classes (provinces with no samples)
    empty = [c for c in range(N_CLASSES)
             if sum(counts[s][c] for s in counts) == 0]
    if empty:
        print(f"WARNING: {len(empty)} class(es) have NO samples: "
              f"{[province_latin(c) for c in empty]}")
    print("\nNext: python scripts/recognition/train_province_classifier.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] build failed: {exc}")
        sys.exit(1)
