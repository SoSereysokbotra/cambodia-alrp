"""
src/core/alpr_system.py
=======================
The integrated ALPR system: camera -> YOLOv10 -> CRNN -> DB -> gate -> log.

Everything is wired from configs/system_config.yaml (no hard-coded paths).
Designed to run on a laptop with no camera and no ESP32 (image-folder source
+ mock gate controller).
"""

from __future__ import annotations

import re
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
from utils.logger import get_logger                  # noqa: E402
from crnn_reader import CRNNReader                   # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_OTHER_CLASS = 25   # province_map.OTHER_CLASS: 'other' (no Khmer prefix composed)


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
        # ROADMAP 1.2: whitelist-constrained matching. A confident read that is
        # NOT an exact whitelist hit but is within `match_max_distance` edits of a
        # registered plate is routed to REVIEW_REQUIRED (a candidate correction) —
        # it NEVER auto-opens the gate (that stays exact-match only, fail-safe).
        self.constrained_matching = bool(
            cfg.get("gate", {}).get("constrained_matching", True))
        self.match_max_distance = int(
            cfg.get("gate", {}).get("match_max_distance", 1))
        # ROADMAP 2.2: province<->number consistency check. Flags a confident,
        # non-whitelisted read to REVIEW when the two branches disagree.
        self.consistency_check = bool(
            cfg.get("gate", {}).get("consistency_check", True))
        self.province_confidence_min = float(
            cfg.get("gate", {}).get("province_confidence_min", 0.55))
        self.number_alignment_min = float(
            cfg.get("gate", {}).get("number_alignment_min", 0.20))
        # Parking mode (open parking): entry records a session, exit clears it.
        gate_cfg = cfg.get("gate", {})
        self.parking_mode = bool(gate_cfg.get("parking_mode", False))
        self.parking_camera_role = str(gate_cfg.get("parking_camera_role", "auto")).lower()
        self.parking_exit_fuzzy = bool(gate_cfg.get("parking_exit_fuzzy", True))
        self.parking_stale_hours = float(gate_cfg.get("parking_stale_hours", 12))
        # Permit parking (hybrid): entry requires a whitelist "permit" but the
        # inside-session is still cleared on exit (permit kept).
        self.parking_require_permit = bool(gate_cfg.get("parking_require_permit", False))

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
        if self.parking_mode:
            n_stale = self.database.expire_stale_sessions(self.parking_stale_hours)
            inside = self.database.count_inside()
            print(f"[ALPRSystem] PARKING MODE on (role={self.parking_camera_role}) "
                  f"| {inside} car(s) inside"
                  + (f", cleared {n_stale} stale" if n_stale else ""))
        cam_cfg = cfg.get("camera", {}) or {}       # SRS Phase 5 (VID-001/004)
        self.frame_timeout = float(cam_cfg.get("frame_timeout_sec", 2.0))
        self.camera = RTSPReader(
            self._camera_source(cfg["camera_source"]),
            queue_size=int(cam_cfg.get("queue_size", 5)),
            reconnect_interval=float(cam_cfg.get("reconnect_interval_sec", 5.0)),
            max_reconnect_attempts=int(cam_cfg.get("max_reconnect_attempts", 10)),
        )
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

        # --- Phase 6: photo capture, structured logging, crop padding --- #
        self._photo_rel = cfg.get("output", {}).get("photo_dir", "photos")
        self.photo_dir = _resolve(self._photo_rel)
        self.photo_dir.mkdir(parents=True, exist_ok=True)
        self.crop_pad = float(cfg.get("detection", {}).get("crop_padding", 0.10))  # DET-005
        self.location = cfg.get("gate", {}).get("location", "Main Gate")
        self.logger = get_logger(self.log_dir)                                     # LOG-003

        # --- live read de-duplication (one row per car, kept at the best read) --- #
        log_cfg = cfg.get("logging", {}) or {}
        self.dedup_enabled = bool(log_cfg.get("dedup_enabled", True))
        self.dedup_gap_sec = float(log_cfg.get("dedup_gap_sec", 3.0))
        self.dedup_merge_edits = int(log_cfg.get("dedup_merge_edits", 2))
        self._live_dedup = False        # turned on by run_video (not offline/demo)
        self._visit = None              # current car's aggregated read {id,best_conf,...}
        self.logger.info("ALPRSystem initialised (crop_pad=%.2f, photos=%s)",
                         self.crop_pad, self._photo_rel)

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
        Returns (best_number_detection, quality) — quality is a dict with the
        winning pairing's `h_overlap` (0..1) and `below` (1.0/0.6), used by the
        ROADMAP 2.2 consistency check. Returns (None, None) if there are none.
        """
        if not number_dets:
            return None, None
        px1, _py1, px2, py2 = prov_bbox
        pcy = (_py1 + py2) / 2.0
        pw = max(1, px2 - px1)
        best, best_score, best_q = None, -1.0, None
        for nd in number_dets:
            nx1, ny1, nx2, _ny2 = nd["bbox"]
            overlap = max(0, min(px2, nx2) - max(px1, nx1))
            nw = max(1, nx2 - nx1)
            h_overlap = overlap / nw                     # 0..1 (used for ranking)
            below = 1.0 if ny1 >= pcy else 0.6           # number line is below province
            score = h_overlap * below + float(nd["confidence"]) * 0.05
            if score > best_score:
                best_score, best = score, nd
                # `align` is scale-invariant (overlap vs the SMALLER box), so a
                # normally-wider number line isn't wrongly penalised — used by the
                # ROADMAP 2.2 consistency check to catch true mis-pairings only.
                best_q = {"h_overlap": h_overlap, "below": below,
                          "align": overlap / min(pw, nw)}
        return best, best_q

    # ------------------------------------------------------------------ #
    def _parking_gate(self, plate_text: str) -> tuple[str, str | None]:
        """Parking decision for one confident read. Returns (action, event).

        Single camera ("auto"): a plate already INSIDE is an EXIT (open + delete
        the session -> plate cleared); otherwise it's an ENTRY (open + record).
        Two cameras: force role="entry" or "exit". Exit is fuzzy-matched (a plate
        the CRNN misread by one char still clears the right car). The gate always
        opens on exit — a parking gate never traps a car.

        Permit mode (`parking_require_permit`): ENTRY is allowed only for a plate
        in the whitelist; an un-permitted car is DENIED and no session is created.
        Exit still just clears the inside-session (the permit is untouched).
        """
        role = self.parking_camera_role
        match = plate_text if self.database.is_inside(plate_text) else None
        if match is None and self.parking_exit_fuzzy:
            near = self.database.nearest_inside(plate_text, 1)
            if near is not None and near[1] > 0:
                match = near[0]

        treat_as_exit = (match is not None) if role == "auto" else (role == "exit")
        if treat_as_exit:
            target = match or plate_text
            self.gate.open_gate(target, self.open_duration)
            cleared = self.database.close_parking_session(target)
            self.session_allowed += 1
            return "ENTRY_ALLOWED", ("EXIT" if cleared else "EXIT_UNKNOWN")

        # ENTRY. In permit mode, only a whitelisted plate may enter.
        if self.parking_require_permit and not self.database.is_registered(plate_text):
            self.session_denied += 1
            return "ENTRY_DENIED", None
        self.gate.open_gate(plate_text, self.open_duration)
        self.database.open_parking_session(plate_text)
        self.session_allowed += 1
        return "ENTRY_ALLOWED", "ENTRY"

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
            # DET-005: 10% crop padding for a margin of context.
            detections = self.detector.detect(frame, pad=self.crop_pad)
            number_dets = (self.number_detector.detect(frame, pad=self.crop_pad)
                           if self.number_detector is not None else [])
        yolo_ms = (time.perf_counter() - t1) * 1000

        plates_result = []
        for det in detections:
            # Pick the NUMBER crop for the CRNN (two-detector flow).
            match_q = None                          # ROADMAP 2.2 pairing quality
            if self.assume_crop:
                number_crop = det["crop"]          # whole pre-cropped plate
                number_conf_det = det["confidence"]
            elif self.number_detector is not None:
                num_det, match_q = self._match_number(det["bbox"], number_dets)
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

            # ROADMAP 2.2 — province <-> number cross-validation. The two branches
            # run independently; flag internal inconsistencies so a confident-but-
            # -unreliable composed read is sent to a human instead of silently
            # (mis)acted on. Signals: (a) the number box is poorly aligned under
            # the province box (likely mis-paired in a multi-plate frame), or
            # (b) a province PREFIX is being composed from an uncertain classifier.
            consistency_reasons = []
            if self.consistency_check:
                if (match_q is not None
                        and match_q.get("align", 1.0) < self.number_alignment_min):
                    consistency_reasons.append("weak-number-alignment")
                if (prov_id is not None and prov_id != _OTHER_CLASS
                        and prov_conf is not None
                        and prov_conf < self.province_confidence_min):
                    consistency_reasons.append("uncertain-province")

            # Stage 3 — DB lookup (whitelist mode only; parking mode uses sessions)
            t3 = time.perf_counter()
            is_reg = False
            suggested = None
            if not self.parking_mode:
                is_reg = self.database.is_registered(plate_text)
                # ROADMAP 1.2: only look for a near match when it isn't an exact
                # hit and the read is confident — used ONLY to flag a review candidate.
                if (self.constrained_matching and not is_reg
                        and crnn_conf >= self.crnn_conf_threshold):
                    near = self.database.nearest_registered(plate_text, self.match_max_distance)
                    if near is not None and near[1] > 0:
                        suggested = near[0]          # (matched_plate, distance>0)
            db_ms = (time.perf_counter() - t3) * 1000

            # Stage 4 — gate decision (SRS REC-005 confidence gate + SEC-005 fail-safe)
            parking_event = None
            if self.estop_active:
                # MAN-002: emergency stop overrides everything — gate stays closed.
                action = "REVIEW_REQUIRED"
                self.session_review += 1
            elif crnn_conf < self.crnn_conf_threshold:
                # low confidence -> never open; flag for human review
                action = "REVIEW_REQUIRED"
                self.session_review += 1
            elif self.parking_mode:
                # OPEN PARKING: entry records a session, exit clears it. No whitelist.
                action, parking_event = self._parking_gate(plate_text)
            elif is_reg:
                self.gate.open_gate(plate_text, self.open_duration)
                action = "ENTRY_ALLOWED"
                self.session_allowed += 1
            elif suggested is not None:
                # ROADMAP 1.2: confident read one edit from a registered plate —
                # likely a legit plate misread by a character. Gate STAYS CLOSED;
                # surface it for a human instead of a silent ENTRY_DENIED.
                action = "REVIEW_REQUIRED"
                self.session_review += 1
            elif consistency_reasons:
                # ROADMAP 2.2: confident, not a whitelist match, but the province
                # and number branches are internally inconsistent -> the composed
                # read is unreliable. Flag for review rather than a silent DENY.
                # (Never downgrades a confirmed ENTRY_ALLOWED above.)
                action = "REVIEW_REQUIRED"
                self.session_review += 1
            else:
                action = "ENTRY_DENIED"
                self.session_denied += 1

            plates_result.append({
                "plate_text": plate_text,
                "number": number,
                "is_registered": is_reg,
                "consistency_reasons": consistency_reasons,   # ROADMAP 2.2
                "parking_event": parking_event,               # ENTRY / EXIT (parking mode)
                "province_id": prov_id,
                "province_confidence": (round(prov_conf, 4) if prov_conf is not None else None),
                "confidence": det["confidence"],
                "number_confidence": round(float(number_conf_det), 4),
                "crnn_confidence": round(crnn_conf, 4),
                "action": action,
                "suggested_plate": suggested,     # ROADMAP 1.2: review candidate
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

        # Phase 6.1 (LOG-002/SEC-003): persist an evidence photo + audit row.
        # LIVE (run_video): de-duplicate — one row per car, upgraded to its best
        #   read, so a plate held in view for seconds is a SINGLE clean entry.
        # OFFLINE (demo/benchmark): every frame is a distinct sample -> log each.
        if self._live_dedup and self.dedup_enabled:
            if plates_result:
                self._dedup_persist(plates_result[0], annotated, time.time())
        else:
            photo_path = self._save_photo(annotated, plates_result[0]) if plates_result else None
            for p in plates_result:
                self._log_one(p, photo_path)

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
    # Evidence photo (SRS Phase 6 — LOG-002, SEC-003)
    # ------------------------------------------------------------------ #
    def _save_photo(self, annotated, plate: dict) -> str | None:
        """Save the annotated full frame as photos/plate_{ts}_{PLATE}.jpg and
        return its project-relative path for the DB (SRS LOG-002 naming)."""
        try:
            import cv2
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]     # ms precision
            tag = re.sub(r"[^0-9A-Za-z-]", "", plate.get("number") or "") or "UNREAD"
            path = self.photo_dir / f"plate_{ts}_{tag}.jpg"
            cv2.imwrite(str(path), annotated)
            try:
                return path.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                return str(path)
        except Exception as exc:
            self.logger.error("photo save failed: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Audit persistence + live de-duplication
    # ------------------------------------------------------------------ #
    def _audit_fields(self, p: dict) -> tuple[str, str | None]:
        """(location tag, plate_text-to-store) shared by log + update."""
        loc = (f"{self.location} | {p['parking_event']}"
               if p.get("parking_event") else self.location)
        matched = ((p["is_registered"] or p.get("parking_event"))
                   and p["action"] == "ENTRY_ALLOWED")
        return loc, (p["plate_text"] if matched else None)

    def _log_one(self, p: dict, photo_path: str | None) -> int | None:
        """Insert one audit row for a read; returns its id."""
        loc, matched_text = self._audit_fields(p)
        rid = self.database.log_read(
            detected_plate=p["plate_text"], yolo_confidence=p["confidence"],
            crnn_confidence=p["crnn_confidence"], action=p["action"],
            plate_text=matched_text, location=loc, photo_path=photo_path)
        _sugg = f" | did-you-mean={p['suggested_plate']}" if p.get("suggested_plate") else ""
        _evt = f" | {p['parking_event']}" if p.get("parking_event") else ""
        self.logger.info("%s%s | %s | crnn=%.2f | %s%s", p["action"], _evt,
                         p["plate_text"], p["crnn_confidence"], photo_path or "-", _sugg)
        return rid

    def _dedup_persist(self, p: dict, annotated, now: float) -> None:
        """Live: keep ONE audit row per car (per visit), upgraded to the best read.

        A read whose text is within `dedup_merge_edits` of the current visit and
        seen within `dedup_gap_sec` is the SAME car: it does not add a new row —
        it only replaces the existing row if it's more confident. A different plate
        (or a gap in time) starts a new visit / row. This turns the per-frame flood
        into a single, correct entry that staff can act on.
        """
        plate, conf = p["plate_text"], p["crnn_confidence"]
        v = self._visit
        same = (v is not None and (now - v["last_time"]) <= self.dedup_gap_sec
                and self.database._edit_distance(plate, v["best_plate"])
                <= self.dedup_merge_edits)
        if not same:
            photo = self._save_photo(annotated, p)
            rid = self._log_one(p, photo)
            self._visit = {"id": rid, "best_plate": plate, "best_conf": conf,
                           "last_time": now, "photo": photo}
        else:
            v["last_time"] = now
            if conf > v["best_conf"] and v["id"] is not None:
                # better read of the SAME car -> upgrade its single row + photo
                photo = self._save_photo(annotated, p)
                loc, matched_text = self._audit_fields(p)
                self.database.update_read(v["id"], detected_plate=plate,
                                          crnn_confidence=conf, action=p["action"],
                                          plate_text=matched_text, photo_path=photo)
                if v.get("photo"):
                    try:
                        (PROJECT_ROOT / v["photo"]).unlink(missing_ok=True)
                    except Exception:
                        pass
                v.update(best_plate=plate, best_conf=conf, photo=photo)

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
        self._live_dedup = True          # collapse repeated reads of one car into one row
        self.camera.start()
        frame_times: list[float] = []
        writer = None
        disconnected_since: float | None = None      # start of a stream outage
        try:
            while True:
                frame = self.camera.get_frame(self.frame_timeout)
                capture_ms = self.camera.frame_timestamp_ms     # VID-002
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
                # VID-002: frame age = now - acquisition timestamp (proves per-frame ts)
                age_ms = (time.time() * 1000.0 - capture_ms) if capture_ms else 0.0
                cv2.putText(out_frame,
                            f"FPS: {fps:.1f} | {res['total_ms']:.0f}ms | age {age_ms:.0f}ms",
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
