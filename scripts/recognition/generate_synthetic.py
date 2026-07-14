#!/usr/bin/env python3
"""
scripts/generate_synthetic_plates.py
=====================================
Generate SYNTHETIC Cambodian-style plate-number images with KNOWN text, so the
CRNN has labelled (image -> text) training data without any manual labelling.

Each image shows a plate number like "1AB-2345" on a plate-like background,
with random augmentation (rotation, blur, noise, colour) for robustness.
Labels are written to CSV files the CRNN dataset reads directly.

Output
------
    data/synthetic/train/*.jpg   + data/synthetic/train_labels.csv
    data/synthetic/valid/*.jpg   + data/synthetic/valid_labels.csv
    data/synthetic/test/*.jpg    + data/synthetic/test_labels.csv

The 8 whitelisted demo numbers are always included in the TEST split so the
Week-5 pipeline can show real ENTRY_ALLOWED matches.

Run
---
    python scripts/generate_synthetic_plates.py               # defaults
    python scripts/generate_synthetic_plates.py --train 6000 --valid 1000 --test 800
"""

from __future__ import annotations

import argparse
import csv
import random
import string
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Missing dependency: {exc}. pip install pillow numpy")

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), Path(__file__).resolve().parents[2])
OUT_ROOT = PROJECT_ROOT / "data" / "synthetic"

DIGITS = string.digits
LETTERS = string.ascii_uppercase

# The 8 demo plates that are registered in the whitelist (number part only).
WHITELIST_NUMBERS = [
    "1AB-2345", "2CD-6789", "3EF-0123", "4GH-4567",
    "5IJ-8901", "6KL-2345", "7MN-6789", "8OP-0123",
]

# Candidate fonts (first that exists wins). Latin/digits only -> any TTF works.
FONT_CANDIDATES = [
    r"C:/Windows/Fonts/arialbd.ttf",
    r"C:/Windows/Fonts/arial.ttf",
    r"C:/Windows/Fonts/consolab.ttf",
    r"C:/Windows/Fonts/segoeui.ttf",
]


def find_font(size: int) -> ImageFont.FreeTypeFont:
    for cand in FONT_CANDIDATES:
        if Path(cand).exists():
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                continue
    # Fallback: try matplotlib's bundled DejaVuSans, else PIL default.
    try:
        import matplotlib
        dj = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
        if dj.exists():
            return ImageFont.truetype(str(dj), size)
    except Exception:
        pass
    return ImageFont.load_default()


def random_plate_number() -> str:
    """Generate a Cambodian-style plate number string."""
    pattern = random.choice(["DL-DDDD", "DLL-DDDD", "DDL-DDDD", "DL-DDD"])
    out = []
    for ch in pattern:
        if ch == "D":
            out.append(random.choice(DIGITS))
        elif ch == "L":
            out.append(random.choice(LETTERS))
        else:
            out.append("-")
    return "".join(out)


def render_plate(text: str, font: ImageFont.FreeTypeFont,
                 augment: bool = True) -> Image.Image:
    """Render one plate-number image (BGR-agnostic grayscale-friendly)."""
    W, H = 320, 96
    # plate background: white or pale yellow
    bg = random.choice([(255, 255, 255), (250, 240, 200), (235, 235, 235)])
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    # optional border
    if random.random() < 0.7:
        draw.rectangle([2, 2, W - 3, H - 3], outline=(0, 0, 0), width=2)

    # centre the text
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx, ty = (W - tw) // 2 - bbox[0], (H - th) // 2 - bbox[1]
    except Exception:
        tw, th = draw.textlength(text, font=font), 40
        tx, ty = (W - tw) // 2, (H - th) // 2
    ink = random.choice([(0, 0, 0), (10, 10, 40), (20, 20, 20)])
    draw.text((tx, ty), text, fill=ink, font=font)

    if augment:
        # slight rotation
        if random.random() < 0.6:
            img = img.rotate(random.uniform(-6, 6), expand=False,
                             fillcolor=bg, resample=Image.BILINEAR)
        # blur
        if random.random() < 0.4:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
        # gaussian noise
        if random.random() < 0.6:
            arr = np.asarray(img).astype("int16")
            noise = np.random.normal(0, random.uniform(4, 18), arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype("uint8")
            img = Image.fromarray(arr)
    return img


def build_split(name: str, count: int, font, extra: list[str] | None = None) -> int:
    """Generate `count` images for a split; returns number written."""
    split_dir = OUT_ROOT / name
    split_dir.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_ROOT / f"{name}_labels.csv"

    # Build the list of texts: forced 'extra' (whitelist) FIRST and in order
    # (so demos hit them deterministically), then shuffled random plates.
    extra = list(extra or [])
    n_random = max(0, count - len(extra))
    randoms = [random_plate_number() for _ in range(n_random)]
    random.shuffle(randoms)
    texts = extra + randoms

    written = 0
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "plate_text"])
        for i, text in enumerate(texts):
            try:
                img = render_plate(text, font, augment=(name == "train"))
                img_path = split_dir / f"{name}_{i:06d}.jpg"
                img.save(img_path, quality=92)
                # store a path relative to project root (portable, forward slashes)
                rel = img_path.relative_to(PROJECT_ROOT).as_posix()
                writer.writerow([rel, text])
                written += 1
            except Exception as exc:
                print(f"  [warn] failed to render '{text}': {exc}")
            if (i + 1) % 500 == 0:
                print(f"  {name}: {i + 1}/{len(texts)}")
    print(f"[{name}] wrote {written} images + {csv_path.name}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--valid", type=int, default=1000)
    ap.add_argument("--test", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--font-size", type=int, default=48)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print(" SYNTHETIC PLATE GENERATION")
    print("=" * 60)
    font = find_font(args.font_size)
    print(f"Font: {getattr(font, 'path', 'PIL-default')}")
    print(f"Output: {OUT_ROOT}\n")

    total = 0
    total += build_split("train", args.train, font)
    total += build_split("valid", args.valid, font)
    # ensure the whitelisted numbers exist in TEST so ENTRY_ALLOWED is showable
    total += build_split("test", args.test, font, extra=WHITELIST_NUMBERS)

    print("-" * 60)
    print(f"DONE. {total} synthetic images generated.")
    print(f"Whitelisted demo numbers embedded in test split: "
          f"{', '.join(WHITELIST_NUMBERS)}")
    print("\nNext: python scripts/train_crnn_week5.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] generation failed: {exc}")
        sys.exit(1)
