#!/usr/bin/env python3
"""
scripts/detection/train_number_detector.py
===========================================
Train a 1-class YOLOv10 detector that finds the PLATE NUMBER line
(the second-stage detector that isolates what CRNN must read).

Prerequisite:
    python scripts/detection/prepare_number_dataset.py --src <roboflow export>

Output:
    models/detection/number_best.pt      (does NOT touch best.pt)

Run:
    python scripts/detection/train_number_detector.py
    python scripts/detection/train_number_detector.py --epochs 150 --no-amp
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
DATA_YAML = PROJECT_ROOT / "data" / "number_detect" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
RUN_NAME = "number_detector"
WEIGHTS_OUT = PROJECT_ROOT / "models" / "detection" / "number_best.pt"


def run_training(model_name, batch, args, amp):
    from ultralytics import YOLO
    try:
        import torch
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    model = YOLO(model_name)
    return model.train(
        data=str(DATA_YAML), epochs=args.epochs, imgsz=args.imgsz, batch=batch,
        optimizer="SGD", patience=args.patience, device=args.device, amp=amp,
        workers=(0 if sys.platform == "win32" else args.workers),
        project=str(RUNS_DIR), name=RUN_NAME, exist_ok=True, verbose=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolov10n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.set_defaults(amp=True)
    args = ap.parse_args()

    if not DATA_YAML.exists():
        print(f"[X] {DATA_YAML} not found. Run prepare_number_dataset.py first.")
        sys.exit(1)

    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        print("[X] ultralytics not installed.")
        sys.exit(1)

    WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print(" TRAIN NUMBER DETECTOR (YOLOv10, 1 class = plate_number)")
    print(f"   epochs {args.epochs} | imgsz {args.imgsz} | batch {args.batch} | device {args.device}")
    print("=" * 60)

    try:
        results = run_training(args.model, args.batch, args, amp=args.amp)
    except (RuntimeError, Exception) as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if "cudnn" in msg or "stream_mismatch" in msg:
            print("\n[!] cuDNN crash — retrying with amp=False, batch 8...\n")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            results = run_training(args.model, min(args.batch, 8), args, amp=False)
        elif "out of memory" in msg and args.batch > 8:
            print("\n[!] OOM — retrying at batch 8...\n")
            results = run_training(args.model, 8, args, amp=args.amp)
        else:
            print(f"\n[X] training failed: {exc}")
            raise

    run_dir = Path(getattr(results, "save_dir", RUNS_DIR / RUN_NAME))
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, WEIGHTS_OUT)
        print(f"\n[OK] best weights -> {WEIGHTS_OUT}")

    # quick metrics on the val split
    try:
        from ultralytics import YOLO
        m = YOLO(str(WEIGHTS_OUT)).val(data=str(DATA_YAML), device=args.device, verbose=False)
        print("-" * 60)
        print(f" mAP50     : {m.box.map50:.4f}")
        print(f" mAP50-95  : {m.box.map:.4f}")
        print(f" Precision : {m.box.mp:.4f}")
        print(f" Recall    : {m.box.mr:.4f}")
        print("-" * 60)
    except Exception as exc:
        print(f"[!] could not compute metrics: {exc}")

    print(f"Run folder: {run_dir}")
    print("\nNext: re-crop number lines for CRNN with the new detector.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] training failed: {exc}")
        raise
