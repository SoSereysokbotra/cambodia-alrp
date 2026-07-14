# 🗂️ Project Structure

Professional layout for the Cambodian ALPR system. `src/` is an importable
package (the library); `scripts/` holds runnable entry points grouped by purpose.

```
Cambodian ALPR Project/
│
├── README.md                     # project overview + quick start
├── .gitignore
├── plates.db                     # SQLite runtime DB (whitelist + audit log)
│
├── configs/
│   └── system_config.yaml        # all runtime settings (no hardcoded paths)
│
├── src/                          # ── LIBRARY (importable package) ──
│   ├── detection/                #   YOLOv10 plate detector
│   │   └── detector.py           #     PlateDetector
│   ├── recognition/              #   CRNN text reader (Stage 2)
│   │   ├── crnn_model.py         #     CRNN + CTCDecoder + charset
│   │   ├── crnn_dataset.py       #     PlateDataset + collate_fn
│   │   ├── crnn_reader.py        #     CRNNReader (crop -> text, confidence)
│   │   └── province_map.py       #     29 Khmer provinces + compose_plate
│   ├── utils/                    #   infrastructure
│   │   ├── database.py           #     PlateDatabase (schema per docs/database.md)
│   │   ├── rtsp_reader.py        #     RTSPReader (webcam/RTSP/video/folder)
│   │   └── mqtt_controller.py    #     MQTT + mock gate control
│   └── core/
│       └── alpr_system.py        #     ALPRSystem — the integrated orchestrator
│
├── scripts/                      # ── RUNNABLE ENTRY POINTS (grouped) ──
│   ├── setup/
│   │   ├── environment.py        #   create venv, install deps, GPU check
│   │   └── roboflow.py           #   Roboflow dataset download
│   ├── detection/
│   │   ├── prepare_dataset.py    #   build 1-class detection dataset
│   │   ├── validate_dataset.py   #   verify data before training
│   │   ├── train.py              #   train YOLOv10
│   │   ├── train_quickstart.py   #   minimal training helper
│   │   ├── evaluate.py           #   test-set metrics
│   │   ├── test_inference.py     #   visual detection check
│   │   ├── summary.py            #   detection results summary
│   │   └── auto_label.py         #   auto-label your own photos
│   ├── recognition/
│   │   ├── generate_synthetic.py #   synthetic plate generator (labelled)
│   │   ├── crop_plates.py        #   extract real crops (for fine-tuning)
│   │   ├── train.py              #   train CRNN (CTC)
│   │   ├── evaluate.py           #   CER + word accuracy
│   │   └── test_inference.py     #   CRNN latency check
│   ├── database/
│   │   ├── setup.py              #   create DB + register demo plates
│   │   ├── migrate.py            #   migrate DB to docs/database.md schema
│   │   └── view.py               #   inspect DB contents
│   ├── pipeline/
│   │   ├── pipeline_stage1.py    #   detect -> DB -> gate (placeholder text)
│   │   ├── pipeline_full.py      #   detect -> CRNN -> DB -> gate
│   │   ├── demo_stage1.py        #   Stage-1 demo
│   │   └── demo_full.py          #   Stage-1 + Stage-2 demo
│   ├── system/
│   │   ├── run_demo.py           #   MAIN integrated demo (any source)
│   │   ├── system_test.py        #   6-component readiness test
│   │   ├── latency_profiler.py   #   per-stage latency profile
│   │   └── srs_acceptance_test.py#   SRS §10 acceptance checks
│   └── tools/
│       ├── test_one_image.py     #   full trace on a single image
│       └── test_confidence_gate.py# REC-005 confidence-gate unit test
│
├── models/
│   ├── detection/best.pt         # trained YOLOv10 (READ ONLY)
│   ├── recognition/              # crnn_best.pth + charset.txt (READ ONLY)
│   └── pretrained/               # base weights (yolov10n.pt, etc.)
│
├── data/                         # (git-ignored) datasets
│   ├── annotated/                #   YOLO detection data (Plate_v4)
│   ├── synthetic/                #   generated CRNN training data
│   └── crnn_crops/               #   real crops for fine-tuning
│
├── docs/                         # documentation
│   ├── srs.md                    #   Software Requirements Specification
│   ├── database.md               #   DB schema
│   ├── PROJECT_STRUCTURE.md      #   (this file)
│   ├── DATA_COLLECTION_GUIDE.md
│   ├── DEPLOYMENT.md             #   Windows setup + run guide
│   ├── SRS_ALIGNMENT_PLAN.md     #   plan to conform to the SRS
│   └── SRS_DEVIATION_LOG.md      #   approved deviations
│
├── hardware/
│   └── esp32_gate_controller/    #   ESP32 firmware (.ino)
│
├── metrics/                      # saved metrics + summaries (JSON/TXT)
├── results/                      # annotated outputs, crops
├── outputs/                      # per-session run outputs (git-ignored)
├── runs/                         # training runs (git-ignored)
├── logs/                         # logs (git-ignored)
├── backups/                      # DB backups (git-ignored)
├── third_party/                  # cloned reference repos (git-ignored)
└── .venv/                        # virtual environment (git-ignored)
```

## Conventions

- **`src/` = library, `scripts/` = entry points.** Reusable code lives in `src/`
  and is imported; `scripts/` files are thin runnables grouped by pipeline stage.
- **Scripts are location-independent.** Each computes the project root by walking
  up to the folder containing `src/`, so they can be moved without breaking paths.
- **Config over hardcoding.** Paths/thresholds come from
  `configs/system_config.yaml`.
- **Read-only models.** `models/detection/best.pt` and
  `models/recognition/crnn_best.pth` are never overwritten by scripts.
- **Git-ignored:** `.venv/`, `data/`, `models/`, `runs/`, `outputs/`, `logs/`,
  `backups/`, `third_party/` — large or regenerable.

## How to run (from the project root, venv active)

```powershell
python scripts/system/system_test.py           # verify all components
python scripts/system/run_demo.py --limit 20   # main demo
python scripts/tools/test_one_image.py <img> --crop
```
