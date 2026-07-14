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

        # Two-detector flow: best.pt (self.detector) finds the Khmer PROVINCE
        # line (fed to the province classifier); number_best.pt finds the NUMBER
        # line (fed to the CRNN). They detect different lines of the same plate,
        # so the CRNN must read the NUMBER crop, not the province crop.
        # If number weights are absent, fall back to reading self.detector's crop.
        self.number_detector = None
        num_w = cfg.get("number_weights")
        if num_w:
            num_path = _resolve(num_w)
            if num_path.exists():
                try:
                    gate_cfg = cfg.get("gate", {})
                    self.number_detector = PlateDetector(
                        num_path,
                        conf=gate_cfg.get("number_confidence_threshold",
                                          gate_cfg.get("yolo_confidence_threshold", 0.4)))
                    print("[ALPRSystem] number detector ON (two-detector flow)")
                except Exception as exc:
                    print(f"[ALPRSystem] number detector unavailable: {exc}")
            else:
                print(f"[ALPRSystem] number weights not found: {num_path}")

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
        self.session_override = 0            # MAN-001 manual opens
        self.estop_active = False            # MAN-002 emergency stop (fail-safe closed)
        self.session_start = time.time()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = _resolve(cfg.get("logging", {}).get("log_dir", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        out_base = _resolve(cfg.get("output", {}).get("output_dir", "outputs"))
        self.session_dir = out_base / f"session_{ts}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.save_annotated = bool(cfg.get("output", {}).get("save_annotated", True))

        # --- health monitoring (SRS Phase 9: HLT-001/002/003, PERF-004) --- #
        health = cfg.get("health", {}) or {}
        self.metrics_every = int(health.get("metrics_every_frames", 100))
        self.latency_alert_ms = float(health.get("latency_alert_ms", 500))   # HLT-003
        self.disconnect_alert_sec = float(health.get("disconnect_alert_sec", 15))
        self.alert_log = self.log_dir / "alerts.log"
        self._latency_hist: list[float] = []      # recent total_ms for FPS/latency
        self._connected_frames = 0                # for uptime_percent (AVAIL-001)
        self._sampled_frames = 0
        self._last_alert: dict[str, float] = {}   # alert-type -> last time (dedupe)

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

    @staticmethod
    def _match_number(prov_bbox, number_dets):
        """Pick the number-line detection that belongs to a province-line box.

        The number line sits directly under the province line and is
        horizontally aligned with it, so we score each candidate by horizontal
        overlap (relative to the number box width), preferring boxes that lie
        below the province box, with detection confidence as a tie-breaker.
        Returns the best number detection dict, or None if there are none.
        """
        if not number_dets:
            return None
        px1, _py1, px2, py2 = prov_bbox
        pcy = (_py1 + py2) / 2.0
        best, best_score = None, -1.0
        for nd in number_dets:
            nx1, ny1, nx2, _ny2 = nd["bbox"]
            overlap = max(0, min(px2, nx2) - max(px1, nx1))
            nw = max(1, nx2 - nx1)
            h_overlap = overlap / nw                     # 0..1 horizontal alignment
            below = 1.0 if ny1 >= pcy else 0.6           # number line is below province
            score = h_overlap * below + float(nd["confidence"]) * 0.05
            if score > best_score:
                best_score, best = score, nd
        return best

    # ------------------------------------------------------------------ #
    def process_frame(self, frame) -> dict:
        t0 = time.perf_counter()

        # Stage 1 — detection (or crop mode)
        t1 = time.perf_counter()
        if self.assume_crop:
            h, w = frame.shape[:2]
            detections = [{"bbox": (0, 0, w, h), "confidence": 1.0,
                           "crop": frame.copy()}]
            number_dets = []
        else:
            # best.pt -> province-line boxes; number_best.pt -> number-line boxes
            detections = self.detector.detect(frame)
            number_dets = (self.number_detector.detect(frame)
                           if self.number_detector is not None else [])
        yolo_ms = (time.perf_counter() - t1) * 1000

        plates_result = []
        for det in detections:
            # Pick the NUMBER crop for the CRNN (two-detector flow).
            if self.assume_crop:
                number_crop = det["crop"]          # whole pre-cropped plate
                number_conf_det = det["confidence"]
            elif self.number_detector is not None:
                num_det = self._match_number(det["bbox"], number_dets)
                number_crop = num_det["crop"] if num_det else None
                number_conf_det = num_det["confidence"] if num_det else 0.0
            else:
                number_crop = det["crop"]          # legacy fallback: province crop
                number_conf_det = det["confidence"]

            # Stage 2 — CRNN reads the number (returns text + confidence, REC-004)
            t2 = time.perf_counter()
            if number_crop is not None and getattr(number_crop, "size", 0):
                number, crnn_conf = self.reader.read(number_crop)
            else:
                number, crnn_conf = "", 0.0        # no number line found
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
            if self.estop_active:
                # MAN-002: emergency stop overrides everything — gate stays closed.
                action = "REVIEW_REQUIRED"
                self.session_review += 1
            elif crnn_conf < self.crnn_conf_threshold:
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
                "number_confidence": round(float(number_conf_det), 4),
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

    # ------------------------------------------------------------------ #
    # Health monitoring & alerts (SRS Phase 9 — HLT-001/002/003, PERF-004)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _gpu_memory_mb() -> float:
        try:
            import torch
            if torch.cuda.is_available():
                return round(torch.cuda.memory_reserved() / (1024 * 1024), 1)
        except Exception:
            pass
        return 0.0

    def _alert(self, kind: str, message: str, severity: str = "WARNING",
               dedupe_sec: float = 30.0) -> None:
        """Raise an alert to console + logs/alerts.log (HLT-003). Repeated alerts
        of the same `kind` are suppressed for `dedupe_sec` to avoid flooding."""
        now = time.time()
        if now - self._last_alert.get(kind, 0.0) < dedupe_sec:
            return
        self._last_alert[kind] = now
        line = (f"{datetime.now().isoformat(timespec='seconds')} "
                f"[{severity}] {kind}: {message}")
        print(f"[ALERT] {line}")
        try:
            with open(self.alert_log, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            print(f"[ALPRSystem] alert-log write failed: {exc}")

    def _uptime_percent(self) -> float:
        """Fraction of sampled frames the camera was connected (AVAIL-001)."""
        if self._sampled_frames == 0:
            return 100.0
        return round(100.0 * self._connected_frames / self._sampled_frames, 2)

    def log_metrics_sample(self) -> dict:
        """Collect one health sample and write it to system_metrics (HLT-002).
        Returns the sample dict. Safe to call without a live camera."""
        recent = self._latency_hist[-30:]
        avg_latency = float(np.mean(recent)) if recent else 0.0
        fps = 1000.0 / max(avg_latency, 1e-6) if recent else 0.0
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
        except Exception:
            cpu = 0.0
        rtsp_ok = False
        try:
            rtsp_ok = bool(self.camera.is_connected())
        except Exception:
            pass
        sample = {
            "fps": round(fps, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "gpu_memory_mb": self._gpu_memory_mb(),
            "cpu_usage_percent": round(cpu, 1),
            "rtsp_connected": rtsp_ok,
            "total_detections_today": self.database.get_stats().get("total_reads", 0),
            "uptime_percent": self._uptime_percent(),
        }
        ok = self.database.log_metrics(**sample)
        if not ok:
            self._alert("db_error", "failed to write system_metrics row", "ERROR")
        # latency SLA breach (HLT-003 / PERF-004: end-to-end must stay < 500 ms)
        if avg_latency > self.latency_alert_ms:
            self._alert("high_latency",
                        f"avg latency {avg_latency:.0f} ms > {self.latency_alert_ms:.0f} ms",
                        "WARNING")
        return sample

    # ------------------------------------------------------------------ #
    # Operator controls (SRS Phase 7 — MAN-001, MAN-002)
    # ------------------------------------------------------------------ #
    def manual_override(self) -> None:
        """MAN-001: operator manually opens the gate; logged as MANUAL_OVERRIDE.
        Ignored while an emergency stop is active (safety takes precedence)."""
        if self.estop_active:
            print("[ALPRSystem] manual override ignored — EMERGENCY STOP active")
            return
        self.gate.open_gate("MANUAL_OVERRIDE", self.open_duration)
        self.session_override += 1
        self.database.log_read(
            detected_plate="MANUAL_OVERRIDE", yolo_confidence=1.0,
            crnn_confidence=1.0, action="MANUAL_OVERRIDE", location="Main Gate",
        )
        print("[ALPRSystem] MANUAL_OVERRIDE — gate opened by operator")

    def emergency_stop(self) -> None:
        """MAN-002: toggle emergency stop. When active the gate is held closed
        (fail-safe) and no read can open it until the operator clears it."""
        self.estop_active = not self.estop_active
        if self.estop_active:
            self.gate.emergency_stop()          # logs [GATE] EMERGENCY_STOP
            print("[ALPRSystem] *** EMERGENCY STOP ACTIVE *** gate locked closed")
        else:
            self.gate.close_gate()
            print("[ALPRSystem] emergency stop cleared — normal operation resumed")

    def run_video(self, show_display: bool = True, save_video: bool = False) -> None:
        import cv2
        self.camera.start()
        frame_times: list[float] = []
        writer = None
        disconnected_since: float | None = None      # start of a stream outage
        try:
            while True:
                frame = self.camera.get_frame()
                if frame is None:
                    if not self.camera.is_connected():
                        # AVAIL-001 / HLT-003: alert on a sustained disconnect,
                        # but only give up once it's clearly ended.
                        if disconnected_since is None:
                            disconnected_since = time.time()
                        outage = time.time() - disconnected_since
                        if outage > self.disconnect_alert_sec:
                            self._alert("stream_disconnect",
                                        f"camera disconnected for {outage:.0f}s",
                                        "CRITICAL")
                        if outage > max(self.disconnect_alert_sec * 3, 30):
                            print("[ALPRSystem] source ended / disconnected.")
                            break
                        continue
                    continue
                disconnected_since = None
                try:
                    res = self.process_frame(frame)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        self._alert("gpu_oom", "CUDA out of memory during inference",
                                    "CRITICAL")
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        continue
                    raise
                frame_times.append(res["total_ms"])
                # feed the health monitor + sample metrics every N frames (HLT-002)
                self._latency_hist.append(res["total_ms"])
                self._sampled_frames += 1
                if self.camera.is_connected():
                    self._connected_frames += 1
                if self.session_processed % self.metrics_every == 0:
                    self.log_metrics_sample()
                fps = 1000.0 / max(np.mean(frame_times[-30:]), 1e-6)

                out_frame = res["frame"]
                cv2.putText(out_frame, f"FPS: {fps:.1f} | {res['total_ms']:.0f}ms",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(out_frame,
                            f"Allowed:{self.session_allowed} Denied:{self.session_denied}",
                            (out_frame.shape[1] - 320, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                # operator controls hint + emergency-stop banner (MAN-001/002)
                cv2.putText(out_frame, "[o] open  [e] e-stop  [q] quit",
                            (10, out_frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if self.estop_active:
                    cv2.putText(out_frame, "*** EMERGENCY STOP ***",
                                (out_frame.shape[1] // 2 - 180, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

                if save_video:
                    if writer is None:
                        h, w = out_frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(self.session_dir / "output.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                    writer.write(out_frame)

                if show_display:
                    cv2.imshow("ALPR System", out_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("o"):          # MAN-001 manual override open
                        self.manual_override()
                    elif key == ord("e"):          # MAN-002 emergency stop toggle
                        self.emergency_stop()
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
            "manual_override": self.session_override,
            "estop_active": self.estop_active,
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
