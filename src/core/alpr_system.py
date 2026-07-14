"""
src/core/alpr_system.py
=======================
The integrated ALPR system: camera -> YOLOv10 -> CRNN -> DB -> gate -> log.

Everything is wired from configs/system_config.yaml (no hard-coded paths).
Designed to run on a laptop with no camera and no ESP32 (image-folder source
+ mock gate controller).
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

# --- make the sibling src packages importable, regardless of caller --- #
_CORE = Path(__file__).resolve()
PROJECT_ROOT = _CORE.parents[2]
SRC = _CORE.parents[1]
for p in (str(SRC), str(SRC / "recognition")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from detection.detector import PlateDetector        # noqa: E402
from utils.database import PlateDatabase             # noqa: E402
from utils.rtsp_reader import RTSPReader             # noqa: E402
from utils.mqtt_controller import create_gate_controller  # noqa: E402
from crnn_reader import CRNNReader                   # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve(path_str: str) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


class ALPRSystem:
    def __init__(self, config_path: str = "configs/system_config.yaml") -> None:
        self.config = self._load_config(config_path)

        cfg = self.config
        self.assume_crop = bool(cfg.get("assume_crop", False))
        self.open_duration = int(cfg.get("mqtt", {}).get("open_duration_sec", 3))
        # SRS REC-005: reads below this confidence -> REVIEW_REQUIRED (gate stays shut)
        self.crnn_conf_threshold = float(
            cfg.get("gate", {}).get("crnn_confidence_threshold", 0.70))

        # --- models / db / camera / gate --- #
        self.detector = PlateDetector(_resolve(cfg["yolo_weights"]),
                                      conf=cfg.get("gate", {}).get(
                                          "yolo_confidence_threshold", 0.5))
        self.reader = CRNNReader(_resolve(cfg["crnn_weights"]),
                                 _resolve(cfg["charset_path"]))

        # Phase 3 (optional): province classifier -> compose "provinceKhmer number".
        # If the classifier isn't trained yet, fall back to number-only.
        self.province_classifier = None
        self._compose_plate = None
        prov_w = _resolve("models/recognition/province_classifier_best.pth")
        if cfg.get("use_province", True) and prov_w.exists():
            try:
                from province_classifier import ProvinceClassifier
                from province_map import compose_plate
                self.province_classifier = ProvinceClassifier(prov_w)
                self._compose_plate = compose_plate
                print("[ALPRSystem] province classifier ON (province + number)")
            except Exception as exc:
                print(f"[ALPRSystem] province classifier unavailable: {exc}")

        self.database = PlateDatabase(_resolve(cfg["db_path"]))
        self.camera = RTSPReader(self._camera_source(cfg["camera_source"]))
        self.gate = create_gate_controller(cfg)

        # --- session state --- #
        self.session_processed = 0
        self.session_allowed = 0
        self.session_denied = 0
        self.session_review = 0
        self.session_start = time.time()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = _resolve(cfg.get("logging", {}).get("log_dir", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        out_base = _resolve(cfg.get("output", {}).get("output_dir", "outputs"))
        self.session_dir = out_base / f"session_{ts}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.save_annotated = bool(cfg.get("output", {}).get("save_annotated", True))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = _resolve(config_path)
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        try:
            import yaml
            return yaml.safe_load(path.read_text(encoding="utf-8"))
        except ImportError as exc:
            raise ImportError("PyYAML required: pip install pyyaml") from exc

    @staticmethod
    def _camera_source(src):
        """Webcam index stays int; folder/file resolved to project root."""
        s = str(src)
        if s.isdigit():
            return int(s)
        if s.startswith(("rtsp://", "http://", "https://")):
            return s
        return str(_resolve(s))

    # ------------------------------------------------------------------ #
    def process_frame(self, frame) -> dict:
        t0 = time.perf_counter()

        # Stage 1 — detection (or crop mode)
        t1 = time.perf_counter()
        if self.assume_crop:
            h, w = frame.shape[:2]
            detections = [{"bbox": (0, 0, w, h), "confidence": 1.0,
                           "crop": frame.copy()}]
        else:
            detections = self.detector.detect(frame)
        yolo_ms = (time.perf_counter() - t1) * 1000

        plates_result = []
        for det in detections:
            # Stage 2 — CRNN reads the number (returns text + confidence, REC-004)
            t2 = time.perf_counter()
            number, crnn_conf = self.reader.read(det["crop"])
            number = number or "(unreadable)"
            # Phase 3 — classify province and compose "provinceKhmer number"
            prov_id = prov_conf = None
            if self.province_classifier is not None:
                prov_id, prov_conf = self.province_classifier.predict(det["crop"])
                plate_text = self._compose_plate(prov_id, number)
            else:
                plate_text = number
            crnn_ms = (time.perf_counter() - t2) * 1000

            # Stage 3 — DB lookup
            t3 = time.perf_counter()
            is_reg = self.database.is_registered(plate_text)
            db_ms = (time.perf_counter() - t3) * 1000

            # Stage 4 — gate decision (SRS REC-005 confidence gate + SEC-005 fail-safe)
            if crnn_conf < self.crnn_conf_threshold:
                # low confidence -> never open; flag for human review
                action = "REVIEW_REQUIRED"
                self.session_review += 1
            elif is_reg:
                self.gate.open_gate(plate_text, self.open_duration)
                action = "ENTRY_ALLOWED"
                self.session_allowed += 1
            else:
                action = "ENTRY_DENIED"
                self.session_denied += 1

            self.database.log_read(
                detected_plate=plate_text,
                yolo_confidence=det["confidence"],
                crnn_confidence=crnn_conf,
                action=action,
                plate_text=(plate_text if (is_reg and action == "ENTRY_ALLOWED") else None),
            )

            plates_result.append({
                "plate_text": plate_text,
                "number": number,
                "province_id": prov_id,
                "province_confidence": (round(prov_conf, 4) if prov_conf is not None else None),
                "confidence": det["confidence"],
                "crnn_confidence": round(crnn_conf, 4),
                "action": action,
                "yolo_ms": yolo_ms,
                "crnn_ms": crnn_ms,
                "db_ms": db_ms,
                "bbox": det["bbox"],
            })

        actions = [p["action"] for p in plates_result]
        try:
            annotated = self.detector.draw_boxes(frame, detections, actions)
        except Exception:
            annotated = frame

        total_ms = (time.perf_counter() - t0) * 1000
        self.session_processed += 1

        return {
            "frame": annotated,
            "plates": plates_result,
            "total_ms": total_ms,
            "yolo_ms": yolo_ms,
            "plates_count": len(plates_result),
        }

    # ------------------------------------------------------------------ #
    def run_on_images(self, image_folder, limit: int | None = None,
                      save: bool = True) -> list[dict]:
        import cv2
        folder = _resolve(str(image_folder)) if not Path(image_folder).is_absolute() \
            else Path(image_folder)
        if not folder.is_dir():
            print(f"[X] not a folder: {folder}")
            return []
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
        if limit:
            images = images[:limit]

        results = []
        total = len(images)
        for idx, img_path in enumerate(images, 1):
            try:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    print(f"[{idx}/{total}] {img_path.name} | READ ERROR")
                    continue
                res = self.process_frame(frame)
                if save and self.save_annotated:
                    out = self.session_dir / img_path.name
                    cv2.imwrite(str(out), res["frame"])
                results.append(res)
                print(f"[{idx}/{total}] {img_path.name} | "
                      f"Plates:{res['plates_count']} | {res['total_ms']:.0f}ms")
            except Exception as exc:
                print(f"[{idx}/{total}] {img_path.name} | ERROR: {exc}")
        return results

    def run_video(self, show_display: bool = True, save_video: bool = False) -> None:
        import cv2
        self.camera.start()
        frame_times: list[float] = []
        writer = None
        try:
            while True:
                frame = self.camera.get_frame()
                if frame is None:
                    if not self.camera.is_connected():
                        print("[ALPRSystem] source ended / disconnected.")
                        break
                    continue
                res = self.process_frame(frame)
                frame_times.append(res["total_ms"])
                fps = 1000.0 / max(np.mean(frame_times[-30:]), 1e-6)

                out_frame = res["frame"]
                cv2.putText(out_frame, f"FPS: {fps:.1f} | {res['total_ms']:.0f}ms",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(out_frame,
                            f"Allowed:{self.session_allowed} Denied:{self.session_denied}",
                            (out_frame.shape[1] - 320, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if save_video:
                    if writer is None:
                        h, w = out_frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(self.session_dir / "output.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                    writer.write(out_frame)

                if show_display:
                    cv2.imshow("ALPR System", out_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            print("\n[ALPRSystem] interrupted.")
        finally:
            self.camera.stop()
            if writer is not None:
                writer.release()
            if show_display:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    def get_session_stats(self) -> dict:
        return {
            "processed": self.session_processed,
            "allowed": self.session_allowed,
            "denied": self.session_denied,
            "review": self.session_review,
            "uptime_sec": round(time.time() - self.session_start, 1),
            "gate_status": self.gate.get_status(),
            "db_stats": self.database.get_stats(),
        }

    def health_check(self) -> dict:
        health = {}
        # GPU
        try:
            import torch
            health["gpu"] = ("OK" if torch.cuda.is_available() else "CPU only")
        except Exception as exc:
            health["gpu"] = f"ERROR: {exc}"
        # YOLO
        try:
            health["yolo"] = "OK" if self.detector.model is not None else "ERROR"
        except Exception as exc:
            health["yolo"] = f"ERROR: {exc}"
        # CRNN
        try:
            health["crnn"] = "OK" if self.reader.model is not None else "ERROR"
        except Exception as exc:
            health["crnn"] = f"ERROR: {exc}"
        # DB
        try:
            n = self.database.get_stats().get("total_registered", 0)
            health["database"] = f"OK ({n} plates)"
        except Exception as exc:
            health["database"] = f"ERROR: {exc}"
        # gate
        try:
            health["mqtt"] = self.gate.get_status()
        except Exception as exc:
            health["mqtt"] = f"error: {exc}"
        return health

    def close(self) -> None:
        try:
            self.database.close()
        except Exception:
            pass
        try:
            if hasattr(self.gate, "stop"):
                self.gate.stop()
        except Exception:
            pass
