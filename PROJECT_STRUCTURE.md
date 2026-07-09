# 🗂️ Project Structure Guide — Cambodian ALPR

This document defines the **folder hierarchy** for the whole project and explains
**what goes where**. `setup_week1.py` creates all of these folders automatically.
Keep this layout stable — the two-stage pipeline (YOLOv10 detection + CRNN reading),
scripts, and future web/API modules all assume it.

---

## Full Hierarchy

```
Cambodian ALPR Project/
│
├── setup_week1.py              # Week 1 environment bootstrapper
├── roboflow_setup.py           # Roboflow guide + dataset downloader
├── DATA_COLLECTION_GUIDE.md    # How to collect the 100 photos
├── PROJECT_STRUCTURE.md        # (this file)
├── requirements.txt            # (generated later: pip freeze of the venv)
├── README.md                   # (project entry point, add over time)
│
├── .venv/                      # Virtual environment (NOT committed to git)
│
├── data/                       # ── ALL DATA lives here ──
│   ├── raw/                    # Original renamed phone photos (source of truth)
│   │   ├── by_angle/           #   optional review copies, sorted by angle
│   │   │   ├── front/
│   │   │   ├── angled_left/
│   │   │   ├── angled_right/
│   │   │   └── rear/
│   │   └── by_lighting/        #   optional review copies, sorted by lighting
│   │       ├── daylight/
│   │       ├── low_light/
│   │       └── backlit/
│   ├── interim/                # Cleaned / renamed / deduped images
│   ├── annotated/              # Roboflow YOLO export (train/valid/test + data.yaml)
│   └── metadata/               # metadata_log.csv and other logs
│
├── models/                     # ── TRAINED WEIGHTS ──
│   ├── detection/              # YOLOv10 plate-detection weights (best.pt)
│   ├── recognition/            # CRNN text-recognition weights
│   └── pretrained/             # Downloaded base checkpoints (yolov10n.pt, etc.)
│
├── src/                        # ── SOURCE CODE (importable package) ──
│   ├── detection/              # YOLOv10 wrappers: train / infer plate boxes
│   ├── recognition/            # CRNN wrappers: read text from plate crops
│   ├── data/                   # Dataset prep, augmentation, conversion utils
│   └── utils/                  # Config, logging, image helpers, metrics
│
├── configs/                    # YAML/JSON configs (model, training, paths)
├── notebooks/                  # Jupyter exploration & visual sanity checks
├── scripts/                    # One-off runnable scripts (train.py, eval.py, ...)
├── third_party/                # Cloned reference repos (read-only)
│   ├── yolov10/                #   github.com/THU-MIG/yolov10
│   └── crnn.pytorch/           #   github.com/meijieru/crnn.pytorch
│
├── outputs/                    # Inference results, predictions, exported crops
├── logs/                       # Training logs, run logs
└── docs/                       # Extra documentation, diagrams, notes
```

---

## Where Things Go — Quick Reference

| You have... | Put it in... |
|-------------|--------------|
| Original photos off the phone (renamed) | `data/raw/` |
| Your CSV metadata log | `data/metadata/metadata_log.csv` |
| Roboflow YOLO export (train/valid/test) | `data/annotated/` |
| Downloaded YOLOv10 base weights | `models/pretrained/` |
| Your trained plate detector | `models/detection/` |
| Your trained CRNN reader | `models/recognition/` |
| Reusable Python code | `src/` (as an importable package) |
| A runnable training/eval script | `scripts/` |
| Experiment notebooks | `notebooks/` |
| Config files (hyperparameters, paths) | `configs/` |
| Reference repos you cloned | `third_party/` |
| Prediction images / cropped plates | `outputs/` |

---

## Key Conventions

**1. `data/raw/` is sacred.**
It is the untouched source of truth. Never annotate, rename in place, or delete from
it after collection. Do all processing on copies (`data/interim/`, `data/annotated/`).

**2. `data/`, `models/`, `.venv/` are NOT committed to git.**
They are large / regenerable. Add a `.gitignore`:
```gitignore
.venv/
data/raw/
data/interim/
data/annotated/
models/
outputs/
logs/
__pycache__/
*.pt
*.pth
```
> Keep `data/metadata/metadata_log.csv` **in** git — it's small and valuable history.
> `.gitkeep` files preserve empty folder structure if you do commit.

**3. `src/` = library, `scripts/` = entry points.**
`src/` holds reusable, importable modules (no side effects on import). `scripts/`
holds thin runnable files that import from `src/`. This keeps the codebase modular —
matching the project's architecture goals (AI, data, hardware, config as separate
modules).

**4. `third_party/` is read-only reference.**
Cloned repos (YOLOv10, CRNN) are for reading and, when needed, training with their
tooling. Don't edit them in place — wrap them from `src/` instead.

**5. Weights are versioned by stage.**
`models/detection/` (YOLOv10) and `models/recognition/` (CRNN) stay separate so the
two-stage pipeline can load each independently.

---

## How This Maps to the Pipeline

```
  Camera frame
       │
       ▼
  [ src/detection ]  ── loads ──►  models/detection/best.pt   (YOLOv10)
       │  crops plate region
       ▼
  [ src/recognition ] ─ loads ──►  models/recognition/*.pth   (CRNN)
       │  reads Khmer + Latin text
       ▼
  business logic / DB / MQTT gate control   (later phases)
```

Week 1 fills **`data/raw/`** and **`data/annotated/`**.
Week 2 produces **`models/detection/`**. CRNN and the rest follow.
