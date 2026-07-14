#!/usr/bin/env python3
"""
scripts/test_one_image.py
=========================
Test the full ALPR pipeline on ANY single image and print a step-by-step trace:
    detect -> crop -> read -> whitelist lookup -> gate decision.

Also saves an annotated copy so you can see the box + text.

Usage
-----
    # a real photo (runs full YOLO detection):
    python scripts/test_one_image.py path/to/photo.jpg

    # a pre-cropped plate image (skips YOLO, reads the whole image):
    python scripts/test_one_image.py data/synthetic/test/test_000000.jpg --crop
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "recognition"))

WEIGHTS = PROJECT_ROOT / "models" / "detection" / "best.pt"
CRNN_WEIGHTS = PROJECT_ROOT / "models" / "recognition" / "crnn_best.pth"
CHARSET_TXT = PROJECT_ROOT / "models" / "recognition" / "charset.txt"
DB_PATH = PROJECT_ROOT / "plates.db"
OUT_DIR = PROJECT_ROOT / "results" / "single_tests"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="Path to the image to test.")
    ap.add_argument("--crop", action="store_true",
                    help="Treat the image as an already-cropped plate (skip YOLO).")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[X] image not found: {img_path}")
        sys.exit(1)

    import cv2
    from detection.detector import PlateDetector
    from crnn_reader import CRNNReader
    from utils.database import PlateDatabase

    print("Loading models (YOLOv10 + CRNN + DB)...")
    detector = PlateDetector(WEIGHTS)
    reader = CRNNReader(CRNN_WEIGHTS, CHARSET_TXT)
    db = PlateDatabase(DB_PATH)

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"[X] could not read image: {img_path}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f" TESTING: {img_path.name}")
    print("=" * 60)
    print(f"Image size : {image.shape[1]}x{image.shape[0]} px")

    # STAGE 1 — detect (or crop mode)
    if args.crop:
        h, w = image.shape[:2]
        detections = [{"bbox": (0, 0, w, h), "confidence": 1.0, "crop": image.copy()}]
        print("STAGE 1 (YOLO): crop mode — whole image treated as the plate")
    else:
        t = time.perf_counter()
        detections = detector.detect(image)
        print(f"STAGE 1 (YOLO): {len(detections)} plate(s) in "
              f"{(time.perf_counter() - t) * 1000:.1f} ms")
        for d in detections:
            print(f"                box={d['bbox']} conf={d['confidence']:.2f}")

    if not detections:
        print("STAGE 1: no plate detected — nothing to read. DECISION = ENTRY_DENIED")
        db.close()
        return

    actions = []
    for i, det in enumerate(detections):
        t = time.perf_counter()
        text, conf = reader.read(det["crop"])
        text = text or "(unreadable)"
        ms = (time.perf_counter() - t) * 1000
        reg = db.is_registered(text)
        # SRS REC-005 confidence gate
        if conf < 0.70:
            action = "REVIEW_REQUIRED"
        elif reg:
            action = "ENTRY_ALLOWED"
        else:
            action = "ENTRY_DENIED"
        actions.append(action)
        print(f"\nPlate #{i + 1}")
        print(f"  STAGE 2 (CRNN): text = \"{text}\"  conf={conf:.2f}  ({ms:.1f} ms)")
        print(f"  STAGE 3 (DB):   registered? -> {reg}")
        print(f"  STAGE 4 (GATE): {action}"
              + ("  (open gate)" if action == "ENTRY_ALLOWED" else "  (keep closed)"))

    # save annotated image
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    annotated = detector.draw_boxes(image, detections, actions)
    out = OUT_DIR / f"{img_path.stem}_result.jpg"
    cv2.imwrite(str(out), annotated)
    print(f"\nAnnotated image saved -> {out}")
    db.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] failed: {exc}")
        sys.exit(1)
