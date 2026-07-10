#!/usr/bin/env python3
"""
scripts/alpr_pipeline_week5.py
==============================
Week-5 pipeline = Week-3 pipeline + REAL CRNN text reading.

Flow per image:
    load -> DETECT (YOLOv10) -> crop -> READ (CRNN) -> whitelist lookup ->
    ENTRY_ALLOWED / ENTRY_DENIED -> log -> annotate -> save

The only change from Week 3 is STEP 3: the placeholder text is replaced by the
CRNN's real reading. When the read text matches a registered plate, the box
turns GREEN and the action is ENTRY_ALLOWED — for real this time.

Run (processes synthetic test plates so ENTRY_ALLOWED is demonstrable):
    python scripts/alpr_pipeline_week5.py
    python scripts/alpr_pipeline_week5.py --source data/annotated/test/images
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "recognition"))

from detection.detector import PlateDetector          # noqa: E402
from utils.database import PlateDatabase               # noqa: E402
from crnn_reader import CRNNReader                      # noqa: E402

WEIGHTS = PROJECT_ROOT / "models" / "detection" / "best.pt"
CRNN_WEIGHTS = PROJECT_ROOT / "models" / "recognition" / "crnn_best.pth"
CHARSET_TXT = PROJECT_ROOT / "models" / "recognition" / "charset.txt"
DB_PATH = PROJECT_ROOT / "plates.db"
RESULTS_DIR = PROJECT_ROOT / "results" / "week5_pipeline"
CROPS_DIR = RESULTS_DIR / "crops"
ANNOTATED_DIR = RESULTS_DIR / "annotated"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# demo whitelist (number-format) — registered so CRNN reads can match.
DEMO_WHITELIST = [
    ("1AB-2345", "Sokhem Ouch", "Honda Civic"),
    ("2CD-6789", "Bopha Mara", "Toyota Camry"),
    ("3EF-0123", "Chan Rith", "Mitsubishi"),
    ("4GH-4567", "Mey Sophea", "Lexus RX"),
    ("5IJ-8901", "Nary Sophel", "Honda Accord"),
    ("6KL-2345", "Dara Piseth", "Suzuki"),
    ("7MN-6789", "Kosal Mony", "Toyota Vios"),
    ("8OP-0123", "Sreyleak Pov", "Hyundai Tucson"),
]


class ALPRPipeline:
    """Detect -> READ (CRNN) -> lookup -> decide -> log -> visualise."""

    def __init__(self, weights_path=WEIGHTS, db_path=DB_PATH,
                 crnn_weights=CRNN_WEIGHTS, charset_path=CHARSET_TXT,
                 assume_crop: bool = False) -> None:
        """
        assume_crop : if True, the input image is ALREADY a cropped plate
                      (e.g. synthetic data). YOLO is skipped and the whole
                      image is fed to CRNN. Use for synthetic demos; leave
                      False for real gate photos (full YOLO -> CRNN flow).
        """
        self.assume_crop = assume_crop
        self.detector = PlateDetector(weights_path)
        self.reader = CRNNReader(crnn_weights, charset_path)
        self.db = PlateDatabase(db_path)
        self._ensure_whitelist()
        for d in (RESULTS_DIR, CROPS_DIR, ANNOTATED_DIR):
            d.mkdir(parents=True, exist_ok=True)
        self._session = {"images": 0, "plates": 0, "allowed": 0, "denied": 0}

    def _ensure_whitelist(self) -> None:
        """Register the number-format demo plates (idempotent)."""
        for num, owner, veh in DEMO_WHITELIST:
            self.db.add_plate(num, owner, veh)

    # ------------------------------------------------------------------ #
    def process_image(self, image_path) -> dict:
        import cv2
        image_path = Path(image_path)
        result = {
            "image_path": str(image_path),
            "plates_detected": 0,
            "detections": [],
            "detection_latency_ms": 0.0,
            "recognition_latency_ms": 0.0,
            "total_latency_ms": 0.0,
            "error": None,
        }
        t_start = time.perf_counter()
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                result["error"] = f"could not read image: {image_path}"
                return result

            # STEP 2 — detect (or treat the whole image as the plate crop)
            t_det = time.perf_counter()
            if self.assume_crop:
                h, w = image.shape[:2]
                detections = [{"bbox": (0, 0, w, h), "confidence": 1.0,
                               "crop": image.copy()}]
            else:
                detections = self.detector.detect(image)
            result["detection_latency_ms"] = round((time.perf_counter() - t_det) * 1000, 1)
            result["plates_detected"] = len(detections)

            # STEP 3 — read + lookup + decide + log
            actions = []
            rec_ms = 0.0
            for i, det in enumerate(detections):
                t_rec = time.perf_counter()
                plate_text = self.reader.read(det["crop"]) or "(unreadable)"
                rec_ms += (time.perf_counter() - t_rec) * 1000
                conf = det["confidence"]
                is_reg = self.db.is_registered(plate_text)
                action = "ENTRY_ALLOWED" if is_reg else "ENTRY_DENIED"
                actions.append(action)

                crop_path = CROPS_DIR / f"{image_path.stem}_plate_{i + 1}.jpg"
                try:
                    if det["crop"] is not None and det["crop"].size > 0:
                        cv2.imwrite(str(crop_path), det["crop"])
                except Exception as exc:
                    print(f"  [warn] could not save crop: {exc}")

                self.db.log_read(plate_text, conf, is_reg, action,
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

            result["recognition_latency_ms"] = round(rec_ms, 1)

            # STEP 4/5 — annotate + save (labels show the READ text)
            annotated = self._draw(image, detections, actions, result["detections"])
            out_path = ANNOTATED_DIR / f"{image_path.stem}_annotated.jpg"
            cv2.imwrite(str(out_path), annotated)
            self._session["images"] += 1

        except Exception as exc:
            result["error"] = str(exc)
            print(f"  [error] {image_path.name}: {exc}")

        result["total_latency_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
        return result

    def _draw(self, image, detections, actions, det_info):
        """Coloured boxes with the READ plate text as the label."""
        import cv2
        out = image.copy()
        colors = {"ENTRY_ALLOWED": (0, 255, 0), "ENTRY_DENIED": (0, 0, 255)}
        for i, det in enumerate(detections):
            action = actions[i] if i < len(actions) else "ENTRY_DENIED"
            color = colors.get(action, (0, 255, 255))
            x1, y1, x2, y2 = det["bbox"]
            text = det_info[i]["plate_text"]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{text} | {action}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ly = max(y1, th + 6)
            cv2.rectangle(out, (x1, ly - th - 6), (x1 + tw + 4, ly), color, -1)
            cv2.putText(out, label, (x1 + 2, ly - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return out

    # ------------------------------------------------------------------ #
    def process_folder(self, folder_path, limit: int = 10) -> list[dict]:
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            print(f"[X] folder not found: {folder_path}")
            return []
        images = sorted(p for p in folder_path.iterdir()
                        if p.suffix.lower() in IMG_EXTS)[:limit]
        results = []
        for idx, img in enumerate(images, 1):
            res = self.process_image(img)
            results.append(res)
            if res["error"]:
                print(f"[{idx}/{len(images)}] {img.name} | ERROR: {res['error']}")
            else:
                texts = ", ".join(d["plate_text"] for d in res["detections"]) or "-"
                acts = ", ".join(sorted({d["action"] for d in res["detections"]})) or "NO_PLATE"
                print(f"[{idx}/{len(images)}] {img.name} | "
                      f"Text: {texts} | {acts}")
        return results

    def get_session_stats(self) -> dict:
        s = dict(self._session)
        s["avg_detection_latency_ms"] = round(self.detector.get_avg_latency_ms(), 1)
        s["avg_recognition_latency_ms"] = round(self.reader.get_avg_latency_ms(), 1)
        return s

    def close(self) -> None:
        self.db.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path,
                    default=PROJECT_ROOT / "data" / "synthetic" / "test",
                    help="Folder of images to process (default: synthetic test).")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--detect", action="store_true",
                    help="Force full YOLO detection (use for real gate photos). "
                         "Default auto-uses crop mode for synthetic inputs.")
    args = ap.parse_args()

    # Auto: synthetic inputs are already-cropped plates -> skip YOLO.
    assume_crop = (not args.detect) and ("synthetic" in str(args.source).lower())

    print("=" * 70)
    print(" WEEK 5 — ALPR PIPELINE (YOLOv10 + CRNN)")
    print("=" * 70)
    for pth, name in [(WEIGHTS, "detector"), (CRNN_WEIGHTS, "CRNN"), (DB_PATH, "database")]:
        if not pth.exists():
            print(f"[X] missing {name}: {pth}")
            sys.exit(1)

    pipe = ALPRPipeline(assume_crop=assume_crop)
    mode = "crop mode (input is already a plate)" if assume_crop else "full YOLO->CRNN"
    print(f"Source images: {args.source}")
    print(f"Mode         : {mode}\n")
    pipe.process_folder(args.source, limit=args.limit)

    s = pipe.get_session_stats()
    e2e = s["avg_detection_latency_ms"] + s["avg_recognition_latency_ms"]
    print("\n" + "-" * 70)
    print(f" Images processed   : {s['images']}")
    print(f" Plates detected    : {s['plates']}")
    print(f" Entry allowed      : {s['allowed']}")
    print(f" Entry denied       : {s['denied']}")
    print(f" Avg YOLO latency   : {s['avg_detection_latency_ms']} ms")
    print(f" Avg CRNN latency   : {s['avg_recognition_latency_ms']} ms")
    print(f" End-to-end (approx): {e2e:.1f} ms  [budget 300 ms]")
    print("-" * 70)
    print(f"Annotated -> {ANNOTATED_DIR}")
    pipe.close()
    print("\nNext: python scripts/demo_week5.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] pipeline failed: {exc}")
        sys.exit(1)
