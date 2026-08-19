#!/usr/bin/env python3
"""
Create hard-condition CRNN training crops from labelled real crops.

This is a practical substitute for part of Phase 6.3 fieldwork: it makes the
existing labelled train crops look like night, motion blur, steep angle, glare,
compression, and dirty/worn plate cases while keeping the same plate_text label.

Important guard:
  - source rows from data/crnn_crops/test are refused, so evaluation remains clean.
  - the generated CSV is meant to be passed to finetune_crnn.py as extra train
    data, not used as the validation split.

Run:
    python scripts/recognition/augment_real_crops.py
    python scripts/recognition/finetune_crnn.py --extra-real-csv data/crnn_crops_augmented/augmented_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Missing dependency: {exc}. Install opencv-python and numpy.")


PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
    Path(__file__).resolve().parents[2],
)

DEFAULT_LABELS = PROJECT_ROOT / "data" / "crnn_crops" / "real_labels.csv"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "crnn_crops_augmented"
FORBIDDEN_SPLIT = "/test/"

CONDITIONS = ("night", "motion", "perspective", "dirty", "glare", "compression")


def rel_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def load_rows(csv_path: Path, match: str) -> list[tuple[str, str]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            image_path = (row.get("image_path") or "").strip().replace("\\", "/")
            plate_text = (row.get("plate_text") or "").strip().upper()
            if not image_path or not plate_text:
                continue
            if FORBIDDEN_SPLIT in image_path:
                continue
            if match and match not in image_path:
                continue
            full = Path(image_path)
            if not full.is_absolute():
                full = PROJECT_ROOT / image_path
            if full.exists():
                rows.append((image_path, plate_text))
    return rows


def resize_model_input(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (320, 64), interpolation=cv2.INTER_CUBIC)


def add_noise(img: np.ndarray, sigma_min: float, sigma_max: float) -> np.ndarray:
    sigma = random.uniform(sigma_min, sigma_max)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def jpeg_roundtrip(img: np.ndarray, quality_min: int, quality_max: int) -> np.ndarray:
    quality = random.randint(quality_min, quality_max)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img


def apply_night(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    gamma = random.uniform(1.5, 2.6)
    img = 255.0 * ((img / 255.0) ** gamma)
    img *= random.uniform(0.45, 0.75)
    img += random.uniform(-8, 8)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = add_noise(img, 8, 24)
    return jpeg_roundtrip(img, 45, 82)


def motion_kernel(size: int, angle_degrees: float) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle_degrees, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (size, size))
    total = kernel.sum()
    return kernel / total if total else kernel


def apply_motion(img: np.ndarray) -> np.ndarray:
    k = random.choice([7, 9, 11, 13])
    angle = random.choice([0, 8, -8, 15, -15])
    img = cv2.filter2D(img, -1, motion_kernel(k, angle))
    if random.random() < 0.7:
        img = add_noise(img, 3, 12)
    return img


def apply_perspective(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    squeeze = random.uniform(0.08, 0.22)
    lift = random.uniform(0.02, 0.10)
    side = random.choice([-1, 1])
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    if side < 0:
        dst = np.float32([
            [w * squeeze, h * lift],
            [w - 1, 0],
            [w - 1, h - 1],
            [w * squeeze, h * (1 - lift)],
        ])
    else:
        dst = np.float32([
            [0, 0],
            [w * (1 - squeeze), h * lift],
            [w * (1 - squeeze), h * (1 - lift)],
            [0, h - 1],
        ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if random.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def apply_dirty(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = img.copy()
    for _ in range(random.randint(18, 42)):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        radius = random.randint(1, max(2, h // 12))
        shade = random.randint(35, 120)
        cv2.circle(overlay, (x, y), radius, (shade, shade, shade), -1, lineType=cv2.LINE_AA)
    for _ in range(random.randint(3, 8)):
        x1 = random.randint(0, w - 1)
        y1 = random.randint(0, h - 1)
        x2 = min(w - 1, max(0, x1 + random.randint(-w // 3, w // 3)))
        y2 = min(h - 1, max(0, y1 + random.randint(-h // 3, h // 3)))
        cv2.line(overlay, (x1, y1), (x2, y2), (random.randint(30, 120),) * 3,
                 random.choice([1, 1, 2]), lineType=cv2.LINE_AA)
    alpha = random.uniform(0.18, 0.38)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def apply_glare(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = np.zeros_like(img, dtype=np.uint8)
    for _ in range(random.randint(1, 3)):
        center = (random.randint(w // 5, w * 4 // 5), random.randint(0, h - 1))
        axes = (random.randint(w // 10, w // 4), random.randint(max(2, h // 12), max(3, h // 4)))
        angle = random.randint(-20, 20)
        cv2.ellipse(overlay, center, axes, angle, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=random.uniform(4, 10))
    strength = random.uniform(0.25, 0.55)
    img = cv2.addWeighted(img, 1.0, overlay, strength, 0)
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_compression(img: np.ndarray) -> np.ndarray:
    img = jpeg_roundtrip(img, 18, 48)
    if random.random() < 0.5:
        img = add_noise(img, 2, 8)
    return img


def augment(img: np.ndarray, condition: str) -> np.ndarray:
    img = resize_model_input(img)
    if condition == "night":
        out = apply_night(img)
    elif condition == "motion":
        out = apply_motion(img)
    elif condition == "perspective":
        out = apply_perspective(img)
    elif condition == "dirty":
        out = apply_dirty(img)
    elif condition == "glare":
        out = apply_glare(img)
    elif condition == "compression":
        out = apply_compression(img)
    else:
        raise ValueError(f"unknown condition: {condition}")

    if random.random() < 0.35 and condition not in {"night", "compression"}:
        out = jpeg_roundtrip(out, 50, 88)
    return resize_model_input(out)


def parse_conditions(raw: str) -> list[str]:
    if raw.lower() == "all":
        return list(CONDITIONS)
    selected = [c.strip().lower() for c in raw.split(",") if c.strip()]
    bad = [c for c in selected if c not in CONDITIONS]
    if bad:
        raise SystemExit(f"Unknown condition(s): {', '.join(bad)}. Choose from: {', '.join(CONDITIONS)}")
    return selected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--match", default="/train/", help="Only augment rows whose image_path contains this substring.")
    ap.add_argument("--limit", type=int, default=None, help="Only use the first N source rows.")
    ap.add_argument("--variants-per-image", type=int, default=4)
    ap.add_argument("--conditions", default="all",
                    help="Comma list from night,motion,perspective,dirty,glare,compression or 'all'.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not args.labels.exists():
        print(f"[X] labels CSV not found: {args.labels}")
        sys.exit(1)
    if args.variants_per_image < 1:
        print("[X] --variants-per-image must be >= 1")
        sys.exit(1)

    conditions = parse_conditions(args.conditions)
    rows = load_rows(args.labels, args.match)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print(f"[X] no source rows found in {args.labels} with match={args.match!r}")
        print("    Label train crops first, or adjust --match.")
        sys.exit(1)

    out_dir = args.out_root / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_root / "augmented_labels.csv"

    generated = []
    skipped = 0
    print("=" * 64)
    print(" SYNTHETIC HARD-CONDITION AUGMENTATION")
    print(f" source rows          : {len(rows)}")
    print(f" variants per source  : {args.variants_per_image}")
    print(f" conditions           : {', '.join(conditions)}")
    print(f" output               : {rel_path(args.out_root)}")
    print("=" * 64)

    for source_idx, (image_path, plate_text) in enumerate(rows, 1):
        full = Path(image_path)
        if not full.is_absolute():
            full = PROJECT_ROOT / image_path
        img = cv2.imread(str(full), cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            continue

        stem = full.stem[:96]
        for variant_idx in range(args.variants_per_image):
            condition = conditions[(source_idx + variant_idx - 1) % len(conditions)]
            out = augment(img, condition)
            out_path = out_dir / f"{stem}__{condition}__{variant_idx:02d}.jpg"
            cv2.imwrite(str(out_path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            generated.append({
                "image_path": rel_path(out_path),
                "plate_text": plate_text,
                "source_image_path": image_path,
                "condition": condition,
            })

        if source_idx % 100 == 0:
            print(f"  processed {source_idx}/{len(rows)} sources...")

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "plate_text", "source_image_path", "condition"],
        )
        writer.writeheader()
        writer.writerows(generated)

    print("-" * 64)
    print(f"[OK] generated {len(generated)} augmented crops")
    if skipped:
        print(f"[warn] skipped {skipped} unreadable source images")
    print(f"[OK] labels -> {rel_path(out_csv)}")
    print("\nFine-tune with:")
    print(f"  python scripts/recognition/finetune_crnn.py --extra-real-csv {rel_path(out_csv)}")


if __name__ == "__main__":
    main()
