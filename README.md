# Cambodian ALPR System

Automatic License Plate Recognition for parking-gate automation in Cambodia.
A two-stage deep-learning pipeline detects a license plate, reads its number,
checks it against a whitelist, and controls a gate — with a fail-safe default of
**deny**.

```
Camera → YOLOv10 (detect) → CRNN (read) → SQLite (whitelist) → Gate decision → MQTT/ESP32
```

## Highlights

| Stage | Model | Result |
|-------|-------|--------|
| Detection | YOLOv10-nano | **mAP50 0.9664**, 8 ms |
| Recognition | CRNN + CTC | **CER 0.00%** (synthetic), ~29 ms |
| End-to-end | full pipeline | **~32 ms → ~30 FPS** |

Safety: exact whitelist match only; low-confidence reads (< 0.70) →
`REVIEW_REQUIRED` (gate stays shut); errors default to closed.

## Quick start (Windows, GPU)

```powershell
# 1. environment (creates .venv, installs PyTorch+CUDA, YOLOv10, deps)
python scripts/setup/environment.py
.\.venv\Scripts\activate

# 2. database (whitelist + audit log)
python scripts/database/setup.py

# 3. verify everything
python scripts/system/system_test.py        # -> SYSTEM TEST: PASS

# 4. run the demo (image folder — no camera needed)
python scripts/system/run_demo.py --limit 20

# 5. test one image with a full stage-by-stage trace
python scripts/tools/test_one_image.py <path-to-image> --crop
```

See **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** for live-camera and ESP32 setup.

## Project layout

- `src/` — the importable library (detection, recognition, utils, core)
- `scripts/` — runnable entry points grouped by stage (setup, detection,
  recognition, database, pipeline, system, tools)
- `configs/system_config.yaml` — all runtime settings
- `docs/` — SRS, schema, structure, deployment, and the SRS-alignment plan
- `hardware/` — ESP32 gate-controller firmware

Full tree: **[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)**

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/srs.md](docs/srs.md) | Software Requirements Specification |
| [docs/database.md](docs/database.md) | Database schema |
| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | Full folder layout |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Windows setup + run guide |
| [docs/SRS_ALIGNMENT_PLAN.md](docs/SRS_ALIGNMENT_PLAN.md) | Plan to conform to the SRS |
| [docs/SRS_DEVIATION_LOG.md](docs/SRS_DEVIATION_LOG.md) | Approved deviations |

## Dataset & credits

- Detection: **Plate_v4** (`taki-dk0de`, Roboflow Universe, CC BY 4.0) — 3,299
  Cambodian plates.
- Recognition: synthetically generated plates (this repo).

## Status

Detection + recognition + full integration complete and verified (~30 FPS).
Ongoing SRS-alignment work (Khmer province recognition, real-photo fine-tuning,
live camera, operator controls) is tracked in
[docs/SRS_ALIGNMENT_PLAN.md](docs/SRS_ALIGNMENT_PLAN.md).
