# SRS Alignment — Implementation Plan
## Cambodian ALPR System — revising the project to conform to `srs.md` and `database.md`

**Purpose:** This document is the master plan to bring the current proof-of-concept
into conformance with the Software Requirements Specification (`docs/srs.md`) and the
database schema (`docs/database.md`). Every task references the SRS requirement ID it
satisfies, the files it touches, and its acceptance criteria.

**How to read this:** Work top to bottom. Phases are ordered by priority
(CRITICAL safety/scope first). Do not start a later phase until the phase it depends
on is done and verified.

---

## 0. Current State vs. SRS (baseline)

| Area | SRS target | Current | Verdict |
|------|-----------|---------|---------|
| YOLOv10 mAP | ≥ 0.80 | 0.9664 | ✅ exceeds |
| YOLO latency | < 50 ms | 8 ms | ✅ |
| End-to-end latency | < 500 ms | ~32 ms | ✅ exceeds |
| FPS | ≥ 15 | ~30 | ✅ |
| CRNN CER | ≤ 10% (real) | 0% (synthetic only) | ⚠️ synthetic only |
| Recognition scope | Khmer + Latin (50+ chars) | Latin+digits (38), no Khmer | ❌ |
| Confidence gate (REC-005) | REVIEW_REQUIRED < 0.70 | none | ❌ CRITICAL |
| Live input | smartphone RTSP | image folder | ⚠️ code only |
| DB schema | `database.md` | partial | ❌ |
| Manual override / E-stop | required | none | ❌ |
| Health metrics / alerts | required | none | ❌ |
| Operating system | Ubuntu (PORT-002) | Windows 11 | ⚙️ kept on Windows (approved deviation) |

**Guiding principle:** preserve the parts that already exceed spec (detection,
latency). Do **not** retrain YOLOv10. Focus effort on recognition scope, the safety
confidence-gate, schema, live input, and operator/monitoring features.

---

## Phase 1 — Recognition Confidence + Safety Gate  `[CRITICAL]`

Satisfies: **REC-004, REC-005, SEC-001, SEC-005, GTC-003/004**.
This is first because REC-005 is a CRITICAL safety requirement and everything in the
decision path depends on having a confidence value.

### Task 1.1 — CRNN emits a confidence score  (REC-004)
- **Files:** `src/recognition/crnn_reader.py`, `src/recognition/crnn_model.py`
- **Change:** `CRNNReader.read()` returns `(text, confidence)` instead of just `text`.
  Compute confidence as the **mean of per-timestep max softmax probabilities** over the
  non-blank decoded characters (per SRS REC-004: "average of per-character confidences").
- **Acceptance:** `read(crop)` returns a float in `[0,1]`; a clear plate scores high
  (> 0.9), a blank/garbage crop scores low (< 0.5).

### Task 1.2 — REVIEW_REQUIRED decision logic  (REC-005)
- **Files:** `src/core/alpr_system.py`, `scripts/alpr_pipeline_week5.py`
- **Change:** after reading, apply the config threshold `gate.crnn_confidence_threshold`
  (0.70 per SRS, currently 0.60 in config — **update config to 0.70**):
  - `confidence < 0.70` → action = **`REVIEW_REQUIRED`**, gate **stays closed**, logged.
  - `confidence ≥ 0.70` AND registered → `ENTRY_ALLOWED`.
  - `confidence ≥ 0.70` AND not registered → `ENTRY_DENIED`.
- **Acceptance:** a low-confidence read never opens the gate; the audit log shows
  `REVIEW_REQUIRED`; unit test proves all three branches.

### Task 1.3 — Update config threshold
- **Files:** `configs/system_config.yaml`
- **Change:** `gate.crnn_confidence_threshold: 0.70` (was 0.60), matching REC-005.

---

## Phase 2 — Database Schema Alignment  `[CRITICAL]`

Satisfies: **DB-001, DB-002, DB-003, DB-004, DB-005, HLT-002, LOG-001**.
Bring `plates.db` and `PlateDatabase` into exact conformance with `database.md`.

### Task 2.1 — Rewrite table definitions to match `database.md`
- **Files:** `src/utils/database.py`
- **`registered_plates`:** add `notes TEXT`; make `owner_name TEXT NOT NULL`;
  add `status TEXT DEFAULT 'active' CHECK (status IN ('active','suspended','expired'))`.
- **`plate_reads`:** add `plate_text` (the matched whitelist plate), `crnn_confidence`,
  `location TEXT DEFAULT 'Main Gate'`; add
  `action CHECK (action IN ('ENTRY_ALLOWED','ENTRY_DENIED','REVIEW_REQUIRED','MANUAL_OVERRIDE','ERROR'))`;
  add CHECK constraints on `yolo_confidence`/`crnn_confidence` in `[0,1]`.
- **`system_metrics`:** create the table exactly as in `database.md`.

### Task 2.2 — Safe migration of existing `plates.db`
- **Files:** `scripts/migrate_db_week13.py` (new)
- **Change:** back up `plates.db` → create new-schema tables → copy the 16 registered
  plates across → swap in. Never lose existing data.
- **Acceptance:** after migration, `view_database` shows all 16 plates; new columns exist.

### Task 2.3 — Update `log_read()` + method signatures
- **Files:** `src/utils/database.py`, all callers (`alpr_system.py`, pipelines)
- **Change:** `log_read()` accepts `plate_text`, `detected_plate`, `yolo_confidence`,
  `crnn_confidence`, `action`, `location`, `photo_path`. Add
  `suspend_plate(plate_text)` and `log_metrics(...)` methods.
- **Acceptance:** every logged row now records both confidences and the resolved action.

---

## Phase 3 — Khmer Recognition Scope  `[CRITICAL, hardest]`

Satisfies: **REC-002, REC-006**. This is the biggest scope gap (currently number-only).

> **Design decision required (pick one before starting):**
> **Option A (recommended): Two-field plate.** YOLOv10's original 29 **province classes**
> supply the Khmer province; CRNN reads the **Latin/digit number**. The full plate text
> = `provinceKhmer + " " + number`. Avoids rendering/segmenting Khmer in CRNN.
> **Option B: Full Khmer CRNN.** Extend the charset to 50+ (Khmer consonants + vowels +
> Latin + digits) and train CRNN to read the whole plate. Needs correctly-shaped Khmer
> rendering (Pillow + libraqm) and much more data. Higher risk.

### Task 3.1 — (Option A) Province classifier from detection
- **Files:** `scripts/train_province_head.py` (new) or reuse Plate_v4 29-class labels
- **Change:** train/keep a 29-class detector (the original Plate_v4 labels, before the
  1-class collapse) so each plate also yields a province name in Khmer.
- **Acceptance:** given a plate crop, system outputs a province label (Khmer) with prob.

### Task 3.2 — Compose full plate text  (REC-006)
- **Files:** `src/core/alpr_system.py`
- **Change:** `plate_text = normalize(province_khmer + " " + number)`. Implement
  `normalize()` (collapse spaces, standardize separators) per REC-006.
- **Acceptance:** output matches the whitelist format `"ProvinceName NNNXXX"`.

### Task 3.3 — Re-register whitelist in full format
- **Files:** `scripts/setup_database_week3.py`
- **Change:** register the 8 demo plates in `provinceKhmer + number` form (already the
  case); ensure the composed output can match them exactly.

> If **Option B** is chosen instead: extend `CHARSET` in `crnn_model.py`, install
> `libraqm` for Pillow, re-render synthetic plates with full Khmer, retrain CRNN.

---

## Phase 4 — Real-World Recognition (close the domain gap)

Satisfies: **REC-001 (CER ≤ 10% on real), F3, Acceptance Phase 2**.
The current CRNN reads synthetic perfectly but fails on real photos.

### Task 4.1 — Label a real evaluation + fine-tune set
- **Files:** `scripts/crop_plates_for_crnn.py` (exists), `scripts/label_real_crops.py` (new)
- **Change:** extract real crops (done), build a small manual-label tool; label
  **300–500** real plate numbers.
- **Acceptance:** `data/crnn_crops/real_labels.csv` with ≥ 300 verified rows.

### Task 4.2 — Fine-tune CRNN on real crops
- **Files:** `scripts/finetune_crnn_week13.py` (new)
- **Change:** start from `crnn_best.pth`, fine-tune on real labels (+ synthetic mix).
- **Acceptance:** **CER ≤ 10% on a held-out REAL test split** (the actual SRS target).

---

## Phase 5 — Live Smartphone RTSP Input

Satisfies: **VID-001, VID-002, VID-003, VID-004, F1, PERF-002**.

### Task 5.1 — Validate RTSP with a real phone
- **Files:** `src/utils/rtsp_reader.py` (exists), `configs/system_config.yaml`
- **Change:** install IP Webcam (Android) / Iriun; set `camera_source` to the RTSP URL;
  verify auto-reconnect (retry 5 s, max 10 — currently 3, **raise to 10 per VID-001**),
  queue size < 5 (done), ≥ 15 FPS decode.
- **Acceptance:** phone streams to edge PC ≥ 20 FPS for 2 h without disconnect (F1).

### Task 5.2 — Per-frame timestamp (VID-002)
- **Files:** `src/utils/rtsp_reader.py`
- **Change:** attach millisecond acquisition timestamp to each frame.
- **Acceptance:** every processed frame carries its capture time in logs.

---

## Phase 6 — Photo Capture & Audit Completeness

Satisfies: **LOG-001, LOG-002, LOG-003, SEC-003, DB-002**.

### Task 6.1 — Full-frame photo saving
- **Files:** `src/core/alpr_system.py`
- **Change:** on a successful read, save the **full annotated frame** as
  `photos/plate_{YYYYMMDD_HHMMSS}_{PLATE}.jpg` (SRS naming), store `photo_path` in
  `plate_reads`.
- **Acceptance:** every ALLOWED/REVIEW event has a saved photo and a DB `photo_path`.

### Task 6.2 — Structured error log
- **Files:** `src/core/alpr_system.py`, `src/utils/logger.py` (new)
- **Change:** Python `logging` to `logs/alpr.log` with levels (LOG-003).
- **Acceptance:** errors captured with timestamp + severity.

### Task 6.3 — 10% crop padding (DET-005)
- **Files:** `src/detection/detector.py`
- **Change:** expand each bbox by 10% (clamped) before cropping.
- **Acceptance:** crops include a small margin of context.

---

## Phase 7 — Manual Override & Emergency Stop  `[CRITICAL controls]`

Satisfies: **MAN-001, MAN-002**.

### Task 7.1 — Keyboard controls in the live loop
- **Files:** `src/core/alpr_system.py` (`run_video`)
- **Change:** key `o` → manual open (log `MANUAL_OVERRIDE`); key `e` → E-stop
  (`gate.emergency_stop()`, log emergency); `q` → quit (exists).
- **Acceptance:** both actions work within 1 s and are logged.

---

## Phase 8 — Administrative Functions

Satisfies: **ADM-001, ADM-002, ADM-003**.

### Task 8.1 — Admin CLI
- **Files:** `scripts/admin_week13.py` (new)
- **Change:** subcommands — `add`, `suspend`, `list`, `search` (by date/plate/action).
- **Acceptance:** can register (<2 s), suspend, and search the audit log.

---

## Phase 9 — Health Monitoring & Alerts

Satisfies: **HLT-001, HLT-002, HLT-003, PERF-004, AVAIL-001**.

### Task 9.1 — Metrics collection into `system_metrics`
- **Files:** `src/core/alpr_system.py`
- **Change:** periodically (hourly, or every N frames) log FPS, avg latency, GPU MB,
  CPU %, rtsp_connected, uptime → `system_metrics` table (created in Phase 2).
- **Acceptance:** rows appear in `system_metrics`; uptime tracked.

### Task 9.2 — Alert generation
- **Files:** `src/core/alpr_system.py`
- **Change:** raise alerts (console + log) for stream disconnect > 15 s, GPU OOM, DB
  error, latency > 500 ms (HLT-003).
- **Acceptance:** disconnecting the phone triggers an alert within 30 s.

---

## Phase 10 — Control Dashboard

Satisfies: **UI-001, USAB-001, USAB-002**.

### Task 10.1 — Live status dashboard
- **Files:** `scripts/dashboard_week13.py` (new)
- **Change:** show live frame + boxes, gate status, last 20 events, FPS/latency/GPU,
  and Manual-Open / E-stop buttons (PySimpleGUI or an OpenCV overlay).
- **Acceptance:** operator can monitor and control from one screen (USAB-002).

---

## Phase 11 — MQTT Protocol Alignment  `[decision]`

Satisfies: **GTC-002, GTC-003, GTC-004**.

> **Conflict to resolve:** SRS specifies topic `gate/control`, payload plain
> `"GATE_OPEN"`/`"GATE_CLOSE"`. My implementation uses `alpr/{gate_id}/control` with a
> **richer JSON** payload (plate, duration, timestamp) that the ESP32 sketch parses.
> **Recommendation:** keep JSON (more capable, carries the plate for logging) but add a
> config flag `mqtt.srs_compat: true` that publishes the SRS plain-text form on
> `gate/control` as well, so both the SRS spec and the ESP32 are satisfied.

### Task 11.1 — Dual-format publish
- **Files:** `src/utils/mqtt_controller.py`, `configs/system_config.yaml`,
  `hardware/esp32_gate_controller/esp32_gate_controller.ino`
- **Acceptance:** with `srs_compat` on, `gate/control` receives `GATE_OPEN`/`GATE_CLOSE`.

---

## Phase 12 — Config, Portability, Final Acceptance

Satisfies: **MAINT-002, PORT-001/002/003, Section 10 Acceptance Criteria**.

### Task 12.1 — Config file conformance
- **Files:** `configs/system_config.yaml`
- **Change:** align keys/sections with the SRS Appendix C example (smartphone,
  reconnect_interval, frame_timeout, photo_directory, etc.).

### Task 12.2 — OS: keep Windows (accepted SRS deviation)
- **Decision:** the project **stays on Windows** for both development and deployment.
  The SRS names Ubuntu 20.04/22.04 (PORT-002), but Windows is retained by choice — the
  full stack (Python venv, PyTorch+CUDA, YOLOv10, CRNN, OpenCV, SQLite, MQTT) already
  runs and is verified on Windows 11.
- **Files:** `docs/DEPLOYMENT.md` (new)
- **Change:** document the **Windows** setup (venv activation `.\.venv\Scripts\activate`,
  CUDA build, paths). Record this as a **formal, approved deviation** from PORT-002 with
  the rationale (existing working environment, no functional difference). Do **not** port
  to Ubuntu/Jetson.
- **Acceptance:** `docs/DEPLOYMENT.md` describes a reproducible Windows setup; the SRS
  deviation is logged so it is intentional, not an omission.

### Task 12.3 — SRS acceptance test suite
- **Files:** `scripts/srs_acceptance_test.py` (new)
- **Change:** automated checks mapping to SRS §10 (mAP, CER on real, latency p95, FPS,
  zero false positives, fail-safe closed, audit completeness).
- **Acceptance:** prints a PASS/FAIL per SRS requirement ID.

---

## Sequencing & Milestones

| Milestone | Phases | Outcome |
|-----------|--------|---------|
| **M1 — Safety & Data core** | 1, 2 | Confidence gate + REVIEW_REQUIRED; DB matches `database.md` |
| **M2 — Recognition scope** | 3, 4 | Khmer province + number; CER ≤ 10% on **real** plates |
| **M3 — Live operation** | 5, 6, 7 | Real phone stream; photos; manual/E-stop |
| **M4 — Ops & monitoring** | 8, 9, 10 | Admin, metrics, alerts, dashboard |
| **M5 — Conformance** | 11, 12 | MQTT/config aligned; SRS acceptance suite green |

**Recommended order to start:** **Phase 1 → Phase 2** (both CRITICAL, low risk, high
value), then decide the Phase 3 option (province-classifier vs. full-Khmer CRNN).

---

## Risks & Notes

- **Phase 3/4 are the hard ones.** Real-world Khmer recognition needs real labeled data;
  budget the most time here. Everything else is straightforward engineering.
- **Do not retrain YOLOv10** — it already exceeds spec. Reuse the original 29-class
  labels for the province head instead.
- **Schema migration (Phase 2)** touches `plates.db` — always back up first.
- **`models/detection/best.pt` and `models/recognition/crnn_best.pth` stay read-only**
  except where a phase explicitly fine-tunes (Phase 4 writes a *new* checkpoint).
- **Keep the passing parts green:** re-run `scripts/system_test_week12.py` after each
  phase to catch regressions.
- **OS stays Windows** (not Ubuntu). This is a deliberate, documented deviation from
  PORT-002 — all commands, paths, and setup in this plan assume Windows 11 + the project
  `.venv`. No Ubuntu/Jetson porting is in scope.

---

*Plan version 1.0 — aligns implementation to `docs/srs.md` v2.0 and `docs/database.md`.*
