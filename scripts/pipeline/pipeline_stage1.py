#!/usr/bin/env python3
"""
scripts/alpr_pipeline_week3.py
==============================
STEP 2 of Week 3 — the end-to-end ALPR pipeline.

Flow per image:
    load -> DETECT (YOLOv10) -> crop -> [CRNN placeholder] ->
    whitelist lookup -> ENTRY_ALLOWED / ENTRY_DENIED -> log -> annotate -> save

CRNN is NOT built yet, so plate_text is a placeholder ("PLATE_i_DETECTED").
Because a placeholder never matches a registered Khmer plate, every plate is
correctly treated as UNKNOWN and DENIED — the fail-safe default (zero
unauthorised openings). Week 9 slots real CRNN text in between crop and lookup;
nothing else changes.

Run (processes 5 test images by default):
    python scripts/alpr_pipeline_week3.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from detection.detector import PlateDetector   # noqa: E402
from utils.database import PlateDatabase        # noqa: E402

DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "detection" / "best.pt"
DEFAULT_DB = PROJECT_ROOT / "plates.db"
RESULTS_DIR = PROJECT_ROOT / "results" / "week3_pipeline"
CROPS_DIR = RESULTS_DIR / "crops"
ANNOTATED_DIR = RESULTS_DIR / "annotated"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ALPRPipeline:
    """Detect -> lookup -> decide -> log -> visualise."""

    def __init__(self, weights_path=DEFAULT_WEIGHTS, db_path=DEFAULT_DB) -> None:
        self.detector = PlateDetector(weights_path)
        self.db = PlateDatabase(db_path)
        for d in (RESULTS_DIR, CROPS_DIR, ANNOTATED_DIR):
            d.mkdir(parents=True, exist_ok=True)
        # session counters
        self._session = {"images": 0, "plates": 0,
                         "allowed": 0, "denied": 0}

    # ------------------------------------------------------------------ #
    def process_image(self, image_path) -> dict:
        """Run the full pipeline on one image. Never raises."""
        import cv2

        image_path = Path(image_path)
        result = {
            "image_path": str(image_path),
            "plates_detected": 0,
            "detections": [],
            "detection_latency_ms": 0.0,
            "total_latency_ms": 0.0,
            "error": None,
        }
        t_start = time.perf_counter()

        try:
            # STEP 1 — load
            image = cv2.imread(str(image_path))
            if image is None:
                result["error"] = f"could not read image: {image_path}"
                return result

            # STEP 2 — detect
            t_det = time.perf_counter()
            detections = self.detector.detect(image)
            result["detection_latency_ms"] = round(
                (time.perf_counter() - t_det) * 1000.0, 1)
            result["plates_detected"] = len(detections)

            # STEP 3 — per plate: placeholder text -> lookup -> decide -> log
            actions = []
            for i, det in enumerate(detections):
                plate_text = f"PLATE_{i + 1}_DETECTED"   # CRNN placeholder
                conf = det["confidence"]
                is_reg = self.db.is_registered(plate_text)
                action = "ENTRY_ALLOWED" if is_reg else "ENTRY_DENIED"
                actions.append(action)

                # save the crop
                crop_path = CROPS_DIR / f"{image_path.stem}_plate_{i + 1}.jpg"
                try:
                    if det["crop"] is not None and det["crop"].size > 0:
                        cv2.imwrite(str(crop_path), det["crop"])
                except Exception as exc:
                    print(f"  [warn] could not save crop: {exc}")

                # audit log (Week 3 has no CRNN yet -> crnn_confidence=0.0)
                self.db.log_read(
                    detected_plate=plate_text, yolo_confidence=conf,
                    crnn_confidence=0.0, action=action,
                    plate_text=(plate_text if is_reg else None),
                    photo_path=str(crop_path))

                result["detections"].append({
                    "plate_text": plate_text,
                    "confidence": round(conf, 4),
                    "is_registered": is_reg,
                    "action": action,
                    "crop_path": str(crop_path),
                })
                self._session["plates"] += 1
                self._session["allowed" if is_reg else "denied"] += 1

            # STEP 4 — annotate (coloured boxes)
            annotated = self.detector.draw_boxes(image, detections, actions)

            # STEP 5 — save annotated image
            out_path = ANNOTATED_DIR / f"{image_path.stem}_annotated.jpg"
            cv2.imwrite(str(out_path), annotated)

            self._session["images"] += 1

        except Exception as exc:
            result["error"] = str(exc)
            print(f"  [error] {image_path.name}: {exc}")

        result["total_latency_ms"] = round(
            (time.perf_counter() - t_start) * 1000.0, 1)
        return result

    # ------------------------------------------------------------------ #
    def process_folder(self, folder_path, limit: int = 10) -> list[dict]:
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            print(f"[X] folder not found: {folder_path}")
            return []

        images = sorted(p for p in folder_path.iterdir()
                        if p.suffix.lower() in IMG_EXTS)[:limit]
        if not images:
            print(f"[X] no images in {folder_path}")
            return []

        results = []
        total = len(images)
        for idx, img in enumerate(images, 1):
            res = self.process_image(img)
            results.append(res)
            if res["error"]:
                print(f"[{idx}/{total}] {img.name} | ERROR: {res['error']}")
            else:
                n = res["plates_detected"]
                actions = {d["action"] for d in res["detections"]} or {"NO_PLATE"}
                action_str = ", ".join(sorted(actions))
                print(f"[{idx}/{total}] {img.name} | Plates: {n} | "
                      f"Action: {action_str}")
        return results

    # ------------------------------------------------------------------ #
    def get_session_stats(self) -> dict:
        stats = dict(self._session)
        stats["avg_detection_latency_ms"] = round(
            self.detector.get_avg_latency_ms(), 1)
        return stats

    def close(self) -> None:
        self.db.close()


def main() -> None:
    print("=" * 70)
    print(" WEEK 3 — STEP 2: ALPR PIPELINE (5 test images)")
    print("=" * 70)

    if not DEFAULT_WEIGHTS.exists():
        print(f"[X] detector weights missing: {DEFAULT_WEIGHTS}")
        sys.exit(1)
    if not DEFAULT_DB.exists():
        print("[X] plates.db missing. Run scripts/setup_database_week3.py first.")
        sys.exit(1)

    pipe = ALPRPipeline(DEFAULT_WEIGHTS, DEFAULT_DB)
    test_dir = PROJECT_ROOT / "data" / "annotated" / "test" / "images"
    print(f"Source images: {test_dir}\n")

    pipe.process_folder(test_dir, limit=5)

    stats = pipe.get_session_stats()
    print("\n" + "-" * 70)
    print(f" Images processed : {stats['images']}")
    print(f" Plates detected  : {stats['plates']}")
    print(f" Entry allowed    : {stats['allowed']}")
    print(f" Entry denied     : {stats['denied']}")
    print(f" Avg detection    : {stats['avg_detection_latency_ms']} ms/image")
    print("-" * 70)
    print(f"Annotated -> {ANNOTATED_DIR}")
    print(f"Crops     -> {CROPS_DIR}")
    pipe.close()
    print("\nNext: python scripts/demo_week3.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] pipeline failed: {exc}")
        sys.exit(1)
