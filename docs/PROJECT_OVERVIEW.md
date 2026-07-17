# Cambodian ALPR — Project Overview (AI-readable context)

> **Purpose of this document.** This is a self-contained description of the
> Cambodian Automatic License Plate Recognition (ALPR) project, written so that
> **another AI (or a new engineer) can read it once and fully understand the
> system** — its goal, architecture, tech stack, data, models, current status,
> and known limitations — and then propose concrete improvements. A dedicated
> section at the end frames where suggestions are most welcome.
>
> Last updated: 2026-07-17.

---

## 1. What the project is

A **gate-automation ALPR system** for Cambodian vehicle license plates. It runs
at a parking/entrance gate: a camera sees a vehicle, the system reads the plate,
checks it against an authorized whitelist, and opens the gate only for a
confident, registered match — otherwise the gate stays closed.

**End-to-end pipeline:**

```
Camera (phone/RTSP/webcam/folder)
  → YOLOv10 plate detection
  → CRNN+CTC number reading  +  province classification
  → compose full plate text  →  SQLite whitelist lookup
  → gate decision (fail-safe: default DENY)
  → MQTT → ESP32 gate hardware
  → audit log + evidence photo
```

**Context:** a university Year-2 Deep Learning project. Driven by a formal SRS
(`docs/srs.md`) and DB schema (`docs/database.md`); brought into conformance via
`docs/SRS_ALIGNMENT_PLAN.md`. The full SRS acceptance suite currently reports
**16/16 requirements PASS**.

### Domain fact that shapes the whole design
Cambodian plates have **three lines**:
1. **Khmer province name** (top) — e.g. ភ្នំពេញ (Phnom Penh)
2. **The number** (middle) — e.g. `2A-0243` (digits + Latin letters + separators)
3. **English province name** (bottom)

The number is the identifying part and renders reliably; Khmer script needs
complex shaping. This is why recognition is **split into two sub-problems**:
read the number with a CRNN, and get the province from a classifier — then
compose them.

---

## 2. Architecture — two detectors + two readers

The most important thing to understand. There are **four trained models**, in two
parallel branches that are fused per plate:

| # | Model | File | Type | Job | Metric |
|---|-------|------|------|-----|--------|
| 1 | Plate/province detector | `models/detection/best.pt` | YOLOv10-n, 1 class | Find the plate region (boxes the **Khmer province line**) | mAP50 **0.9664** |
| 2 | Number detector | `models/detection/number_best.pt` | YOLOv10-n, 1 class | Find the **number line** specifically | mAP50 **0.943** |
| 3 | Province classifier | `models/recognition/province_classifier_best.pth` | ResNet18, 26 classes (25 provinces + "other") | Classify which province from the crop | test acc **97.18%** |
| 4 | CRNN number reader | `models/recognition/crnn_finetuned.pth` | CRNN (CNN+BiLSTM+CTC), 38-char set (digits+Latin+`- `) | Read the number string | real-test CER **10.21%**, word-acc **72.48%** |

**Per-plate flow (in `src/core/alpr_system.py` → `process_frame`):**

```
best.pt        → province crop  → province classifier → provinceKhmer
number_best.pt → number crop    → fine-tuned CRNN     → number string
                    │
                    └─ _match_number() pairs each number box with the province
                       box above it (horizontal overlap + "is-below" geometry)

compose_plate(province_id, number) → e.g. "ភ្នំពេញ 3E-6694"
   → database.is_registered(text)  → gate decision
```

Why two detectors: `best.pt` only reliably boxes the province line, **not** the
number, so a dedicated `number_best.pt` was trained to crop the number for the CRNN.

---

## 3. The 4-stage runtime pipeline (`process_frame`)

`src/core/alpr_system.py`, `process_frame()` — the heart of the system. Each stage
is individually timed (`yolo_ms`, `crnn_ms`, `db_ms`, `total_ms`).

1. **Detect** — run `best.pt` (+ `number_best.pt`); returns boxes + crops.
2. **Read** — CRNN reads the number crop → `(text, confidence)`; province
   classifier labels the province crop; compose full plate text.
3. **DB lookup** — `is_registered(plate_text)` against the SQLite whitelist (exact match).
4. **Decide** (fail-safe, priority order):
   - E-stop active → `REVIEW_REQUIRED` (closed)
   - `crnn_conf < 0.70` → `REVIEW_REQUIRED` (closed)
   - registered & confident → `ENTRY_ALLOWED` (open gate)
   - confident but not registered → `ENTRY_DENIED` (closed)

Every read is logged to SQLite **and** saved as an annotated evidence photo
(`photos/plate_{timestamp}_{PLATE}.jpg`).

**Confidence design (important):** the CRNN confidence is the mean per-character
max-softmax probability, then multiplied by a **length-plausibility factor**
(`min(1, n/5)`) so implausibly short reads (spurious 1–2 char misfires) are
suppressed below the 0.70 gate. See `src/recognition/crnn_reader.py`.

---

### Two gate modes (config-selectable)
The gate decision runs in one of two modes, set by `gate.parking_mode`:

- **Whitelist mode (default, `parking_mode: false`)** — secure access control. The
  gate opens only for a plate that exactly matches the `registered_plates`
  whitelist *and* reads confidently (fail-safe default-deny). Plates are enrolled
  manually (admin panel / enroll / read). Includes the constrained-matching (1.2)
  and province↔number consistency (2.2) review paths.
- **Parking mode (`parking_mode: true`)** — session tracking with clear-on-exit.
  A confident read that is **not** currently inside is an **ENTRY** (gate opens, a
  `parking_sessions` row is created); a read that **is** inside is an **EXIT** (gate
  opens, the row is **deleted** — the plate is cleared from the system). Only cars
  parked *right now* are stored, so storage stays flat. A single camera infers
  entry/exit from session presence (`parking_camera_role: auto`), or two cameras can
  force `entry`/`exit`. Exit matching is fuzzy (tolerates a 1-char misread); stale
  sessions (missed exits) auto-expire after `parking_stale_hours`.
  `python main.py inside` lists cars currently inside. Two entry policies:
  - `parking_require_permit: false` (**open parking**) — the gate opens for *every*
    car on entry. Not access control.
  - `parking_require_permit: true` (**permit parking, hybrid**) — ENTRY is allowed
    only for a whitelisted plate ("permit"), but the inside-session is still deleted
    on exit (the permit is kept). Grant permits with the admin panel's **[Authorize]**
    button; a car is approved once and never re-typed. Combines access control with
    flat storage.

## 4. Technology stack

**Language / DL:** Python 3.10, PyTorch 2.5.1 + CUDA 12.1, Ultralytics (YOLOv10),
torchvision (ResNet18).
**Computer vision:** OpenCV (`opencv-python`) — capture, crop, draw, video I/O.
**Data / storage:** SQLite (`plates.db`) — whitelist + audit log + system metrics.
NumPy, pandas for data handling.
**IoT / hardware:** MQTT (paho-mqtt / Mosquitto) → **ESP32** gate controller
firmware (`hardware/esp32_gate_controller/*.ino`). A **mock gate controller**
lets the whole system run with no hardware.
**Web / UI:** stdlib `http.server` + Jinja2 web admin panel (no heavy web
framework); OpenCV-native live control dashboard.
**Config:** YAML (`configs/system_config.yaml`) — every path/threshold is
external, nothing hard-coded.
**Camera input:** threaded RTSP/HTTP reader (`src/utils/rtsp_reader.py`) with a
bounded latest-frame queue, per-frame timestamps, and auto-reconnect.

**Environment:** Windows 11, NVIDIA RTX 3050 Laptop (4 GB VRAM). The SRS names
Ubuntu; staying on Windows is a **documented, approved deviation** (DEV-001).

---

## 5. Repository layout

```
main.py                         single entry point (interactive menu + subcommands)
configs/system_config.yaml      all runtime settings

src/
  core/alpr_system.py           the pipeline, gate logic, health, evidence photos
  detection/detector.py         YOLOv10 wrapper (detect, crop, draw, warm-up, latency)
  recognition/
    crnn_model.py               CRNN network (CNN+BiLSTM+CTC) + greedy CTC decoder + charset
    crnn_reader.py              inference wrapper: preprocess → infer → confidence+length penalty
    crnn_dataset.py             CRNN training dataset
    province_classifier.py      ResNet18 province classifier (uses saved idx_to_class)
    province_map.py             province id → Khmer name, compose_plate()
  utils/
    database.py                 SQLite: whitelist, audit log, system_metrics
    rtsp_reader.py              threaded camera reader, queue, reconnect, timestamps
    mqtt_controller.py          real MQTT gate + mock gate (same interface)
    logger.py                   structured logging to logs/alpr.log

scripts/
  detection/     train.py, evaluate.py, prepare_dataset.py, train_number_detector.py, ...
  recognition/   crop_numbers.py, finetune_crnn.py, evaluate_crnn_on_real.py, ...
  database/      setup.py, migrate.py, view.py
  system/        dashboard.py, run_demo.py, admin_web.py, srs_acceptance_test.py, system_test.py
  tools/         make_montage.py, import_label_csv.py, test_one_image.py

hardware/esp32_gate_controller/ ESP32 firmware (Arduino .ino)
models/          detection/*.pt, recognition/*.pth + charset.txt + province config json
data/            datasets, number annotations, CRNN crops + real_labels.csv
docs/            srs.md, database.md, SRS_ALIGNMENT_PLAN.md, SRS_DEVIATION_LOG.md,
                 DEPLOYMENT.md, HANDOFF.md, PROJECT_STRUCTURE.md, CONCEPTS_STUDY_GUIDE.md
photos/          per-read evidence images     outputs/  annotated sessions     logs/  runtime logs
```

---

## 6. Data & training approach

- **Detection dataset:** *Plate_v4* (`taki-dk0de`, Roboflow Universe, CC BY 4.0),
  ~3,299 Cambodian plates. Number-line boxes were separately user-annotated
  (`cambodian-plate-number`, Roboflow).
- **CRNN synthetic pre-training:** synthetically rendered plate numbers →
  `crnn_best.pth` (CER 0% on synthetic, but 94.89% CER on *real* — a large domain gap).
- **CRNN real fine-tuning (Phase 4):** progressively labeled real number crops and
  fine-tuned. Test set (149 crops) is **human-labeled**; train labels were
  AI-assisted via montage transcription. Never train on the test split.

| CRNN checkpoint | Real-test CER | Word-acc |
|-----------------|---------------|----------|
| synthetic only (baseline) | 94.89% | 0% |
| fine-tuned on 143 train | 25.93% | 47.65% |
| fine-tuned on 324 train | 20.32% | 51.68% |
| **fine-tuned on 473 train** | **10.21%** | **72.48%** ✅ |

- **Full-pipeline (two-detector) end-to-end on real test images:**
  **70.6% exactly-correct numbers** (101/143). Pipeline latency **~51 ms ≈ 19.6 FPS**.

---

## 7. Current status

**Done and verified:** detection (both detectors), province classifier, CRNN real
fine-tuning (SRS CER/word-acc targets met), full two-detector integration with a
real `ENTRY_ALLOWED` demonstrated, confidence gate + `REVIEW_REQUIRED`, DB schema
aligned to `database.md`, evidence photos + structured logging, manual override +
emergency stop, health metrics + alerts, web admin panel, OpenCV control
dashboard, SRS acceptance suite (**16/16 PASS**).

**Pending / partial:**
- **Live smartphone RTSP** — code complete; needs a real 2-hour phone-stream
  validation run (F1 acceptance) on the user's device.
- **Khmer character recognition in the CRNN itself** — deliberately out of scope;
  Khmer comes from the classifier (DEV-002). Full-Khmer CRNN was not pursued.
- Minor config-key renaming to SRS Appendix C names.

**Known documented deviations:** Windows instead of Ubuntu (DEV-001);
number-only CRNN + province classifier instead of a 50+ char Khmer CRNN (DEV-002);
crop padding kept at 0.0 because padding regressed recognition 70.6%→46.2%
(DEV-004; the "margin of context" intent is met by full-frame evidence photos).

---

## 8. Known limitations / weak spots (candidates for improvement)

An honest list of where the system is thin — useful for anyone proposing upgrades:

1. **Real-world number accuracy is ~70.6% end-to-end.** Good for a student
   project, not yet production-grade. The CRNN CER (10.21%) is at the edge of the
   target; harder real conditions (night, motion blur, angles, dirty plates) are
   under-represented in the ~473 labeled real crops.
2. **Small real-labeled dataset.** Only ~473 real number labels (324 AI-labeled).
   More real, diverse, human-verified data would likely be the single biggest win.
3. **Four separate models in sequence** = more moving parts, more latency, more
   failure modes than a unified approach. No model is exported/optimized (raw
   PyTorch + Ultralytics; no ONNX/TensorRT).
4. **Exact-string whitelist matching** is brittle: one misread character → denied.
   No fuzzy/edit-distance fallback or confirmation flow.
5. **Province branch and number branch can disagree** and there's no
   cross-validation between them (e.g. a plate whose Khmer province is unreadable).
6. **Windows-only, single-machine.** No containerization, no reproducible deploy,
   no CI. Model files are versioned ad-hoc (`.bak.pth` names).
7. **No experiment tracking.** Metrics live in scattered `results/`, `metrics/`,
   `runs/` folders and prose in `HANDOFF.md` rather than a tracker.
8. **Security surface** of the web admin panel (stdlib http.server, no auth) is
   minimal — fine for a LAN demo, not for real deployment.
9. **Evaluation is number-centric.** There's no end-to-end *composed-plate*
   (province+number) accuracy metric against a labeled real set.

---

## 9. Where improvement ideas are most welcome

If you are an AI or engineer reading this to suggest improvements, the highest-value
directions (roughly ordered) are:

1. **Close the real-world accuracy gap** — data strategy (more/diverse real labels,
   targeted hard cases), stronger augmentation, or a better recognizer
   (e.g. attention/transformer OCR such as TrOCR or PaddleOCR) vs. the current CRNN.
2. **Model optimization & deployment** — ONNX/TensorRT export, quantization, and a
   reproducible container so the pipeline runs on cheap edge hardware within budget.
3. **Robustness of the decision layer** — smarter matching than exact string
   (confidence-weighted / edit-distance with a human-confirm path), and
   cross-checking the province and number branches.
4. **Consolidation** — whether a single end-to-end model (or a vision-language
   model) could replace the 4-model chain, and the accuracy/latency trade-off.
5. **MLOps** — experiment tracking (W&B/MLflow), dataset/model versioning (DVC),
   basic CI, and a real end-to-end composed-plate benchmark.
6. **Operational hardening** — auth on the admin panel, alerting/observability, the
   pending live-RTSP stability validation.

**Constraints to respect when proposing changes:**
- Runs on a **4 GB VRAM laptop GPU**; keep it edge-deployable.
- **Fail-safe default-deny** gate behavior is a hard safety requirement — never
  weaken it.
- `models/detection/best.pt` and `models/recognition/crnn_best.pth` are treated as
  **read-only** baselines; new training writes new checkpoints.
- Stays on **Windows** by design (documented deviation).
- Test labels are **human-verified**; never train on the test split.

---

## 10. Quick pointers for a reader

- **Read first:** this file, then `src/core/alpr_system.py` (`process_frame`), then
  `src/recognition/crnn_model.py` (the CTC recognizer).
- **Concepts explained for beginners:** `docs/CONCEPTS_STUDY_GUIDE.md`.
- **Requirements & schema:** `docs/srs.md`, `docs/database.md`.
- **Conformance plan & status:** `docs/SRS_ALIGNMENT_PLAN.md`, `docs/HANDOFF.md`.
- **Approved deviations:** `docs/SRS_DEVIATION_LOG.md`.
- **Run it:** `python main.py` (menu) — dashboard / demo / enroll / admin / accept.
```
