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
# Windows paths first (local runs), then Linux paths so this also works on Colab.
FONT_CANDIDATES = [
    r"C:/Windows/Fonts/arialbd.ttf",
    r"C:/Windows/Fonts/arial.ttf",
    r"C:/Windows/Fonts/consolab.ttf",
    r"C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",     # Colab / Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
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


# Real Cambodian plates: near-white background, dark BLUE number ink (IMPROVEMENT #1).
PLATE_BG = [(252, 252, 252), (246, 246, 242), (238, 240, 245), (250, 248, 240)]
PLATE_INK = [(18, 28, 110), (12, 20, 90), (25, 30, 120), (15, 15, 30)]


def _realistic_augment(img: Image.Image, bg) -> Image.Image:
    """IMPROVEMENT #1: make a synthetic crop look like a real gate PHOTO — mild
    perspective, uneven lighting, blur, sensor noise, JPEG artefacts — so the model
    (and the STN straightening layer) transfer from synthetic to real plates."""
    # perspective (plates are photographed at an angle) — needs cv2, skip if absent
    if random.random() < 0.6:
        try:
            import cv2
            arr = np.array(img)
            h, w = arr.shape[:2]
            m = 0.12
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = np.float32([
                [random.uniform(0, w * m), random.uniform(0, h * m)],
                [w - random.uniform(0, w * m), random.uniform(0, h * m)],
                [w - random.uniform(0, w * m), h - random.uniform(0, h * m)],
                [random.uniform(0, w * m), h - random.uniform(0, h * m)]])
            M = cv2.getPerspectiveTransform(src, dst)
            arr = cv2.warpPerspective(arr, M, (w, h), borderValue=bg)
            img = Image.fromarray(arr)
        except Exception:
            pass
    # slight rotation
    if random.random() < 0.6:
        img = img.rotate(random.uniform(-7, 7), expand=False,
                         fillcolor=bg, resample=Image.BILINEAR)
    # uneven lighting (a bright-to-dark gradient across the plate)
    if random.random() < 0.5:
        arr = np.array(img).astype("float32")
        w = arr.shape[1]
        grad = np.linspace(random.uniform(0.65, 1.0),
                           random.uniform(1.0, 1.35), w)[None, :, None]
        img = Image.fromarray(np.clip(arr * grad, 0, 255).astype("uint8"))
    # blur
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.6)))
    # sensor noise
    if random.random() < 0.6:
        arr = np.asarray(img).astype("int16")
        arr = np.clip(arr + np.random.normal(0, random.uniform(4, 22), arr.shape),
                      0, 255).astype("uint8")
        img = Image.fromarray(arr)
    # JPEG compression artefacts (real photos are JPEGs)
    if random.random() < 0.5:
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(35, 80))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def render_plate(text: str, font: ImageFont.FreeTypeFont,
                 augment: bool = True) -> Image.Image:
    """Render one realistic plate-number image (IMPROVEMENT #1)."""
    W, H = 320, 96
    bg = random.choice(PLATE_BG)
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    ink = random.choice(PLATE_INK)
    # blue-ish border like a real plate
    if random.random() < 0.8:
        draw.rectangle([2, 2, W - 3, H - 3], outline=ink, width=2)

    # centre the text
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx, ty = (W - tw) // 2 - bbox[0], (H - th) // 2 - bbox[1]
    except Exception:
        tw, th = draw.textlength(text, font=font), 40
        tx, ty = (W - tw) // 2, (H - th) // 2
    draw.text((tx, ty), text, fill=ink, font=font)
    # underline under the number, like real Cambodian plates
    if random.random() < 0.5:
        draw.line([(tx, ty + th + 5), (tx + tw, ty + th + 5)], fill=ink, width=2)

    if augment:
        img = _realistic_augment(img, bg)
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
