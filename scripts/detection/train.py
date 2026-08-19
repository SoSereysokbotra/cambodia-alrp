#!/usr/bin/env python3
"""
train_yolov10_week2.py
======================
STEP 2 of Week 2 — train YOLOv10-nano on the 3,299 Cambodian plates.

Config (per spec):
    model      : yolov10n.pt  (pretrained nano)
    data       : data/annotated/data.yaml
    epochs     : 100
    imgsz      : 640
    batch      : 16  (auto-retries at 8 on CUDA out-of-memory)
    optimizer  : SGD
    patience   : 15  (early stopping)
    device     : 0   (GPU)

On completion:
    * copies best.pt -> models/detection/best.pt
    * prints final mAP50, mAP50-95, Precision, Recall

Run (AFTER prepare_training_week2.py passes):
    python train_yolov10_week2.py
    python train_yolov10_week2.py --epochs 150      # if mAP50 < 0.80
    python train_yolov10_week2.py --batch 8         # low-VRAM GPU
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

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), Path(__file__).resolve().parents[2])
DATA_YAML = PROJECT_ROOT / "data" / "annotated" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
RUN_NAME = "plate_detector_week2"
WEIGHTS_OUT = PROJECT_ROOT / "models" / "detection"


def run_training(model_name: str, batch: int, args, amp: bool) -> object:
    """Kick off one training run. Returns the ultralytics results object."""
    from ultralytics import YOLO
    # cuDNN benchmark autotuning has been linked to CUDNN_STATUS_*_STREAM_MISMATCH
    # crashes on some Windows laptop GPUs — turn it off for stability.
    try:
        import torch
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    model = YOLO(model_name)  # auto-downloads yolov10n.pt if missing
    return model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        optimizer="SGD",
        patience=args.patience,
        device=args.device,
        amp=amp,               # mixed precision — disable if cuDNN errors
        workers=args.workers,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        verbose=True,          # epoch-by-epoch loss + mAP printout
        # ROTATION-INVARIANCE augmentation (MODEL_IMPROVEMENT_PLAN Step 4).
        # Default 0.0 = the original behaviour (upright only). Pass --degrees 180
        # --flipud 0.5 to train a detector that finds plates at ANY angle: the image
        # (and its boxes) are randomly rotated up to +-degrees and flipped vertically,
        # so the model sees plates at every orientation instead of only upright.
        degrees=args.degrees,
        flipud=args.flipud,
    )


def resume_training(args) -> object:
    """Resume the interrupted run from its last checkpoint."""
    from ultralytics import YOLO
    try:
        import torch
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    last = RUNS_DIR / args.name / "weights" / "last.pt"
    if not last.exists():
        raise SystemExit(f"[X] No checkpoint to resume from at {last}")
    print(f"Resuming from checkpoint: {last}")
    model = YOLO(str(last))
    return model.train(resume=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument("--model", default="yolov10n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="0", help="'0' GPU, 'cpu', etc.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Dataloader workers. Try 0 if you hit Windows loader errors.")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="Disable mixed precision (fixes some cuDNN crashes).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the interrupted run from its last checkpoint.")
    parser.add_argument("--degrees", type=float, default=0.0,
                        help="max random rotation in degrees (180 = any angle). "
                             "0 = original upright-only behaviour.")
    parser.add_argument("--flipud", type=float, default=0.0,
                        help="probability of a vertical flip (0.5 helps upside-down).")
    parser.add_argument("--name", default=RUN_NAME, help="run folder name under runs/detect/")
    parser.add_argument("--out", type=Path, default=WEIGHTS_OUT / "best.pt",
                        help="where to copy the best weights. Use a NEW name (e.g. "
                             "models/detection/best_rot.pt) to keep best.pt intact.")
    parser.set_defaults(amp=True)
    args = parser.parse_args()

    if not args.data.exists():
        print(f"[X] {args.data} not found. Run prepare_training_week2.py first.")
        sys.exit(1)

    try:
        from ultralytics import YOLO  # noqa: F401
        import torch
    except ImportError:
        print("[X] ultralytics/torch not installed. Activate venv and: pip install ultralytics")
        sys.exit(1)

    WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" WEEK 2 — STEP 2: TRAIN YOLOv10-nano")
    print(f"   data    : {args.data}")
    print(f"   model   : {args.model}")
    print(f"   epochs  : {args.epochs} | imgsz {args.imgsz} | batch {args.batch}")
    print(f"   opt     : SGD | patience {args.patience} | device {args.device}")
    print("=" * 60)
    print("Expected: mAP50 ~0.82-0.90, ~2-4h on GPU. Let it run.\n")

    # --- train (with resume) ---
    # NOTE on recovery: a GPU crash MID-TRAINING (cuDNN stream-mismatch, OOM)
    # corrupts the CUDA context and does NOT free the card's memory, so retrying in
    # the SAME process always fails with a second "out of memory" (this bit us once).
    # The only reliable recovery is a FRESH process — either --resume from the
    # checkpoint, or rerun with the safer flags below. So on a crash we save nothing
    # to chance: print the exact recovery command and exit, instead of a doomed
    # in-process retry.
    if args.resume:
        results = resume_training(args)
    else:
        try:
            results = run_training(args.model, args.batch, args, amp=args.amp)
        except (RuntimeError, Exception) as exc:  # noqa: BLE001
            msg = str(exc).lower()
            ckpt = RUNS_DIR / args.name / "weights" / "last.pt"
            resumable = ckpt.exists()
            print(f"\n[X] Training GPU-crashed: {exc}")
            print("-" * 60)
            if "cudnn" in msg or "stream_mismatch" in msg:
                print(" Cause: a flaky Windows-laptop-GPU cuDNN error (intermittent,")
                print("        NOT your data). Mixed precision is the usual trigger.")
            elif "out of memory" in msg:
                print(" Cause: the 4 GB GPU ran out of memory.")
            print(" A crashed run can't be retried in THIS process (its GPU memory")
            print(" is not released). Start a FRESH process with one of these:\n")
            if resumable:
                print(f"   # continue from the last checkpoint (keeps progress):")
                print(f"   python {Path(__file__).name} --resume --name {args.name}\n")
            print(f"   # or restart crash-safe (mixed precision off, smaller batch):")
            print(f"   python {Path(__file__).name} --model {args.model} "
                  f"--degrees {args.degrees:g} --flipud {args.flipud:g} "
                  f"--epochs {args.epochs} --name {args.name} --out {args.out} "
                  f"--no-amp --batch 8")
            print("-" * 60)
            sys.exit(1)

    # --- locate + copy best.pt ---
    run_dir = Path(getattr(results, "save_dir", RUNS_DIR / args.name))
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        dest = args.out
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dest)
        print(f"\n[OK] best weights -> {dest}")
    else:
        print(f"\n[!] best.pt not found under {run_dir}. Check the run folder.")
        dest = best

    # --- final metrics (validate best on the val split) ---
    try:
        from ultralytics import YOLO
        best_model = YOLO(str(dest))
        m = best_model.val(data=str(args.data), device=args.device, verbose=False)
        print("\n" + "-" * 60)
        print(" FINAL METRICS (val split)")
        print(f"   mAP50     : {m.box.map50:.4f}")
        print(f"   mAP50-95  : {m.box.map:.4f}")
        print(f"   Precision : {m.box.mp:.4f}")
        print(f"   Recall    : {m.box.mr:.4f}")
        print("-" * 60)
    except Exception as exc:  # noqa: BLE001
        print(f"[!] Could not compute final metrics automatically: {exc}")

    print("\nTraining complete.")
    print(f"Run folder (plots, results.csv): {run_dir}")
    print("\nNext: python evaluate_week2.py")


if __name__ == "__main__":
    main()
