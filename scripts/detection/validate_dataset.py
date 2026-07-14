#!/usr/bin/env python3
"""
prepare_training_week2.py
=========================
STEP 1 of Week 2 — verify the dataset is correct BEFORE training.

Checks:
  * data/annotated/data.yaml exists and its train/val/test paths resolve
  * counts images + label files in each split
  * every image has a matching .txt label (and flags mismatches)
  * counts total boxes
Creates output dirs (runs/detect/, results/, metrics/) and prints
"READY TO TRAIN" if all checks pass, otherwise the specific errors.

Run:
    python prepare_training_week2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make Unicode (checkmarks) safe on the Windows cp1252 console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), Path(__file__).resolve().parents[2])
DATA_YAML = PROJECT_ROOT / "data" / "annotated" / "data.yaml"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Output dirs required by the rest of Week 2.
OUTPUT_DIRS = [
    PROJECT_ROOT / "runs" / "detect",
    PROJECT_ROOT / "results",
    PROJECT_ROOT / "metrics",
]


def parse_data_yaml(path: Path) -> dict:
    """Read data.yaml. Uses PyYAML if available, else a minimal parser."""
    try:
        import yaml
        return yaml.safe_load(path.read_text())
    except Exception:
        # Minimal fallback for the simple key: value layout we generate.
        data = {}
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if ":" in line:
                k, v = line.split(":", 1)
                data[k.strip()] = v.strip()
        return data


def resolve_split_dir(base: Path, rel: str) -> Path:
    """Resolve a split path (may be relative to data.yaml's 'path')."""
    p = Path(rel)
    return p if p.is_absolute() else (base / rel)


def count_split(images_dir: Path) -> tuple[int, int, int, list[str]]:
    """Return (n_images, n_labels, n_boxes, list_of_problem_messages)."""
    labels_dir = images_dir.parent / "labels"
    problems: list[str] = []

    if not images_dir.is_dir():
        problems.append(f"images folder missing: {images_dir}")
        return 0, 0, 0, problems
    if not labels_dir.is_dir():
        problems.append(f"labels folder missing: {labels_dir}")
        return 0, 0, 0, problems

    images = [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    n_labels = 0
    n_boxes = 0
    missing = 0

    for img in images:
        lbl = labels_dir / f"{img.stem}.txt"
        if not lbl.exists():
            missing += 1
            if missing <= 5:
                problems.append(f"no label for image: {img.name}")
            continue
        n_labels += 1
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                n_boxes += 1

    if missing > 5:
        problems.append(f"...and {missing - 5} more images with no label file")

    return len(images), n_labels, n_boxes, problems


def main() -> None:
    print("=" * 60)
    print(" WEEK 2 — STEP 1: DATASET VERIFICATION")
    print("=" * 60)

    errors: list[str] = []

    # 1. data.yaml exists
    if not DATA_YAML.exists():
        print(f"[X] data.yaml not found: {DATA_YAML}")
        print("    Run scripts/prepare_detection_dataset.py first.")
        sys.exit(1)
    print(f"[OK] data.yaml found: {DATA_YAML}")

    cfg = parse_data_yaml(DATA_YAML)
    base = Path(cfg.get("path", DATA_YAML.parent))
    if not base.is_absolute():
        base = (DATA_YAML.parent / base).resolve()

    print(f"[OK] dataset root : {base}")
    print(f"[OK] classes      : nc={cfg.get('nc')} names={cfg.get('names')}")
    print("-" * 60)

    # 2. per-split counts + label matching
    totals = {"images": 0, "labels": 0, "boxes": 0}
    split_keys = {"train": "train", "valid": "val", "test": "test"}
    counts = {}

    for split, yaml_key in split_keys.items():
        rel = cfg.get(yaml_key)
        if not rel:
            errors.append(f"'{yaml_key}' missing from data.yaml")
            continue
        images_dir = resolve_split_dir(base, str(rel))
        n_img, n_lbl, n_box, problems = count_split(images_dir)
        counts[split] = (n_img, n_lbl, n_box)
        totals["images"] += n_img
        totals["labels"] += n_lbl
        totals["boxes"] += n_box
        status = "OK" if (n_img > 0 and not problems) else "X"
        print(f"[{status}] {split:<5} : {n_img:>5} images | {n_lbl:>5} labels | {n_box:>5} boxes")
        for msg in problems:
            errors.append(f"[{split}] {msg}")

    # 3. summary
    print("-" * 60)
    print(" DATASET SUMMARY")
    print(f"   Train images : {counts.get('train', (0,))[0]}")
    print(f"   Valid images : {counts.get('valid', (0,))[0]}")
    print(f"   Test images  : {counts.get('test', (0,))[0]}")
    print(f"   Total images : {totals['images']}")
    print(f"   Total boxes  : {totals['boxes']}")
    print("-" * 60)

    # 4. create output dirs
    for d in OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[OK] output dirs ready: {', '.join(d.name for d in OUTPUT_DIRS)}")
    print("-" * 60)

    # 5. verdict
    if errors:
        print(f"[ERROR] {len(errors)} problem(s) found:")
        for e in errors[:20]:
            print(f"    - {e}")
        if len(errors) > 20:
            print(f"    ...and {len(errors) - 20} more")
        print("\nStatus: NOT READY — fix the above before training.")
        sys.exit(1)

    print("Status: ✓ READY TO TRAIN")
    print("\nNext: python train_yolov10_week2.py")


if __name__ == "__main__":
    main()
