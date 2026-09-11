# Cambodian ALPR — Automatic License Plate Recognition for Gate Control

A real-time system that reads Cambodian vehicle license plates at a parking or
entrance gate, checks them against a whitelist, and opens the gate only for a
confident, registered match. Anything else — an unknown plate, a low-confidence
read, a hardware fault — leaves the gate **closed**.

Built as a Year-2 Deep Learning project. Driven by a formal SRS
([docs/srs.md](docs/srs.md)); the acceptance suite currently passes **16 / 16**
requirements.

```
Camera (phone / RTSP / webcam / folder)
   │
   ├─ YOLOv10  plate detector     → province crop → ResNet18 classifier → "ភ្នំពេញ"
   └─ YOLOv10  number detector    → number crop   → CRNN + CTC reader   → "2A-0243"
                                                            │
                                            compose → "ភ្នំពេញ 2A-0243"
                                                            │
                                     SQLite whitelist lookup → gate decision
                                                            │
                                     MQTT → ESP32 gate  +  audit log  +  evidence photo
```

---

## Why the pipeline looks like this

A Cambodian plate has **three lines**:

| line | content | example |
|---|---|---|
| top | province name in **Khmer script** | ភ្នំពេញ |
| middle | the identifying **number** — digits, Latin letters, separators | `2A-0243` |
| bottom | province name in English | PHNOM PENH |

The number is what identifies the vehicle and renders reliably in any font.
Khmer script needs complex glyph shaping and is a poor fit for a sequence reader.
So recognition is split into two sub-problems and fused per plate:

- **read the number** with a CRNN (CNN → BiLSTM → CTC, 38-character set)
- **classify the province** with a ResNet18 over 26 classes (25 provinces + *other*)

A single detector reliably boxes the province line but not the number line, which
is why there are **two** YOLOv10 detectors rather than one.

---

## Models and measured results

Four trained models. Every number below is from
[metrics/experiment_log.csv](metrics/experiment_log.csv), the project's
append-only measurement log (each row carries the git commit it was measured at).

| model | file | architecture | job | result |
|---|---|---|---|---|
| Plate detector | `models/detection/best.pt` | YOLOv10-n, 1 class | box the plate / province line | mAP50 **0.966** |
| Number detector | `models/detection/number_best.pt` | YOLOv10-n, 1 class | box the number line | mAP50 **0.943** |
| Province classifier | `models/recognition/province_classifier_best.pth` | ResNet18, 26 classes | read the Khmer province line | test acc **97.2 %** |
| Number reader | `models/recognition/crnn_stn5.pth` (deployed) | CRNN + CTC, 38 chars | read the number string | word-acc **84.5 %** upright, 48.5 % upside-down (de-leaked real test, n = 97) |

**End-to-end on real gate photos** (n = 149 plates, measured 2026-08-08):

| metric | value |
|---|---|
| plate detected | 96.0 % |
| number read exactly right | 76.5 % |
| province read right | 95.2 % (n = 147) |
| **false accepts** (wrong plate let through) | **0.0 %** |
| latency per frame, RTX 3050 laptop | ~50–100 ms (10–20 FPS) |

The false-accept figure is the one that matters for a gate: the system is tuned
to refuse rather than guess.

---

## Gate decision — fail-safe by design

`src/core/alpr_system.py → process_frame()` runs four timed stages: detect → read
→ database lookup → decide. The decision is evaluated in priority order and
defaults to closed:

| condition | outcome | gate |
|---|---|---|
| emergency-stop active | `REVIEW_REQUIRED` | closed |
| CRNN confidence < 0.70 | `REVIEW_REQUIRED` | closed |
| registered **and** confident | `ENTRY_ALLOWED` | **open** |
| confident but not registered | `ENTRY_DENIED` | closed |

Confidence is the mean per-character softmax probability multiplied by a
length-plausibility factor, so a spurious one- or two-character misfire cannot
reach the 0.70 threshold.

Two operating modes, selected in `configs/system_config.yaml` under `gate:`:

- **Whitelist mode** (default) — access control. Plates are enrolled by an
  operator; only an exact match opens the gate.
- **Parking mode** (`parking_mode: true`) — session tracking. A confident read of
  a car not currently inside is an *entry*; the same plate seen again is an
  *exit* and the session is cleared. Optionally require a permit
  (`parking_require_permit: true`) so entry still needs a whitelisted plate.

Every read — allowed or not — is written to SQLite and saved as an annotated
evidence photo in `photos/`.

---

## Quick start (Windows, NVIDIA GPU)

```powershell
# 1. Environment — creates .venv, installs PyTorch for your CUDA, then requirements.txt
python scripts/setup/environment.py
.\.venv\Scripts\activate

# 2. Database — creates plates.db and registers a few demo plates
python scripts/database/setup.py

# 3. Readiness test — GPU, all four models, DB, camera source, MQTT, 10-frame pipeline
python scripts/system/system_test.py          # expect: SYSTEM TEST: PASS

# 4. Run it
python main.py                                # interactive menu
```

No camera needed: the default `camera_source` is a folder of test images. To use
a phone as an IP camera, `python main.py camera http://<phone-ip>:8080/video`.

Tested with Python 3.10, PyTorch 2.5.1 + CUDA 12.1. CPU-only works but is slow.

### `main.py` commands

| command | what it does |
|---|---|
| `python main.py dashboard` | live dashboard on the configured camera source |
| `python main.py demo` | run the pipeline over the real test images |
| `python main.py read <image>` | read one image → plate text → offer to enroll |
| `python main.py enroll` | whitelist a plate by typing it |
| `python main.py admin` | browser panel to manage the whitelist |
| `python main.py db` | registered plates + recent reads |
| `python main.py inside` | parking mode: cars currently inside |
| `python main.py stream` | test the phone / IP-camera connection |
| `python main.py accept` | run the 16-check SRS acceptance suite |
| `python main.py camera <url>` | set `camera_source` (URL, `0` for webcam, or a folder) |
| `python main.py gdrive <cmd>` | optional Google Drive backup of models and data |

---

## Repository layout

```
├── main.py                  single entry point (see table above)
├── configs/
│   └── system_config.yaml   every runtime setting — no paths are hard-coded
├── src/                     importable library
│   ├── core/alpr_system.py  ALPRSystem — the integrated pipeline
│   ├── detection/           YOLOv10 wrappers
│   ├── recognition/         CRNN reader, province classifier, plate composition
│   └── utils/               SQLite, camera reader, MQTT, Google Drive
├── scripts/                 runnable entry points, grouped by stage
│   ├── setup/  detection/  recognition/  database/  pipeline/  system/  tools/
├── notebooks/               Colab training notebooks (CRNN + STN, province classifier)
├── models/                  trained weights (git-ignored; see Training)
├── data/                    datasets (git-ignored; see Dataset)
├── metrics/                 experiment_log.csv + benchmark outputs (tracked)
├── docs/                    SRS, schema, deployment, plans, study guide
└── hardware/                ESP32 gate-controller firmware
```

Full annotated tree: [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

---

## Configuration

Everything lives in `configs/system_config.yaml`. The settings you are most likely
to touch:

| key | purpose |
|---|---|
| `camera_source` | URL, `0` (webcam), video file, or an image folder |
| `crnn_weights` | which CRNN checkpoint is deployed |
| `gate.crnn_confidence_threshold` | the 0.70 gate; raise for stricter, lower for more reads |
| `gate.province_crop_trim` | fraction trimmed off each side of the province crop before classifying (tuned end-to-end: 0.125) |
| `gate.parking_mode` / `parking_require_permit` | whitelist vs. parking behaviour |
| `mqtt.enabled` | `false` runs a mock gate; `true` talks to the ESP32 |

---

## Training and reproducing

Heavy training runs on **Google Colab**; the local machine does inference,
evaluation and fine-tuning.

1. `python scripts/tools/make_colab_bundle.py` — packs `src/`, the training
   scripts, the crops and the base weights into `alpr_colab_bundle.zip`.
2. Upload the zip to Drive, open `notebooks/colab_train.ipynb`, run top to bottom.
   The notebook verifies the bundle is current before it trains.
3. Trained weights are copied back to Drive; evaluate locally with the tools in
   `scripts/tools/` (`check_stn.py`, `check_leakage.py`, `test_rotation_reading.py`,
   `score_pipeline.py`).

Detector training: `scripts/detection/train.py`. Province classifier:
`scripts/recognition/train_province_classifier.py`. Synthetic plates for CRNN
pre-training: `scripts/recognition/generate_synthetic.py`.

### Experiment tracking

- **`metrics/experiment_log.csv`** is the system of record. Evaluation scripts
  append to it with `--log`; `finetune_crnn.py` logs its best validation CER
  automatically. Each row is tagged with the commit it was measured at.
- **Weights & Biases** is optional, for live curves during a Colab run:
  `finetune_crnn.py --wandb`. It never replaces the CSV and degrades to a
  warning if the package or API key is absent.

---

## Hardware

The gate is an **ESP32** listening on MQTT
([hardware/esp32_gate_controller/](hardware/esp32_gate_controller/)). With
`mqtt.enabled: false` (the default) the software runs a mock gate, so the whole
pipeline can be developed and demonstrated with no hardware attached. Live-camera
and ESP32 wiring: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## Known limitations (measured, not guessed)

- **Upside-down plates.** Upright reading is 84.5 %; rotated 180° it drops to
  48.5 %. A learnable straightening layer (STN) was tried across three
  checkpoints and measurably did nothing — `check_stn.py` reports FAIL on all of
  them. Fixing this through training, not inference-time tricks, is the current
  open problem.
- **Test-set contamination.** 34.9 % of real test crops share a physical plate
  with a training crop (same car photographed twice). `check_leakage.py` found
  it; a de-leaked split (`real_labels_deleaked.csv`) is used for all numbers
  reported above. Earlier headline figures were inflated by this.
- **Dataset size.** Fine-tuning data is a few hundred real plates. Accuracy on
  unusual fonts, damaged plates, or provinces with few examples is lower.
- **Single-camera parking mode** infers entry vs. exit from session state; two
  vehicles with the same plate misread cannot be disambiguated.

---

## Documentation

| document | contents |
|---|---|
| [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) | the whole system in one read — start here after this README |
| [docs/srs.md](docs/srs.md) | Software Requirements Specification |
| [docs/database.md](docs/database.md) | SQLite schema |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | live camera, phone camera, ESP32 setup |
| [docs/DATA_COLLECTION_GUIDE.md](docs/DATA_COLLECTION_GUIDE.md) | how plates were photographed and labelled |
| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | annotated folder tree |
| [docs/SRS_ALIGNMENT_PLAN.md](docs/SRS_ALIGNMENT_PLAN.md) · [SRS_DEVIATION_LOG.md](docs/SRS_DEVIATION_LOG.md) | how the build was brought into conformance, and approved deviations |
| [docs/MLOPS_IMPLEMENTATION_PLAN.md](docs/MLOPS_IMPLEMENTATION_PLAN.md) | experiment logging and versioning decisions |
| [docs/CONCEPTS_STUDY_GUIDE.md](docs/CONCEPTS_STUDY_GUIDE.md) | the deep-learning concepts used, explained |

---

## Dataset and credits

Datasets are not stored in this repository (`data/` is git-ignored, ~410 MB).

- **Plate detection:** *Plate_v4* by `taki-dk0de` on Roboflow Universe
  (CC BY 4.0) — 3,299 Cambodian plates.
- **Number and province recognition:** real plate crops photographed and labelled
  for this project, plus synthetic plates from `generate_synthetic.py`.
- **Full dataset archive:** _add your Google Drive / Roboflow link here_

Base weights: YOLOv10-n (Ultralytics), ResNet18 (torchvision).
