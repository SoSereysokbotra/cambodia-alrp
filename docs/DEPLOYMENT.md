# Deployment Guide (Windows) — Cambodian ALPR System

Target OS: **Windows 11** (approved deviation from SRS PORT-002 — see
`docs/SRS_DEVIATION_LOG.md`). All commands below are PowerShell.

---

## 1. Prerequisites

- Windows 11 with an **NVIDIA GPU** (RTX 3050 or better) + recent driver.
- **Python 3.9+** (project verified on 3.10).
- **Git** (for cloning the reference repos).
- Optional hardware: an **ESP32** + relay, and an Android/iOS phone for the camera.

---

## 2. Environment setup

```powershell
# from the project root
python setup_week1.py            # creates .venv, installs PyTorch+CUDA, YOLOv10, deps

# activate the environment (do this EVERY new terminal)
.\.venv\Scripts\activate         # prompt shows (.venv)

# verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # -> CUDA: True
```

If `CUDA: False`, reinstall the correct build: `python setup_week1.py --cuda 121`
(or `--cuda 118`).

---

## 3. Models & database

The trained models must be present (do not overwrite — READ ONLY):
- `models/detection/best.pt`         (YOLOv10 detector)
- `models/recognition/crnn_best.pth` + `models/recognition/charset.txt`

Prepare the database (schema per `docs/database.md`):

```powershell
python scripts/setup_database_week3.py   # creates plates.db + demo whitelist
python scripts/migrate_db_week13.py      # if upgrading an older plates.db
python scripts/view_database_week3.py    # inspect
```

---

## 4. Configuration

All runtime settings live in **`configs/system_config.yaml`** (no hardcoded
paths). Key fields:

```yaml
camera_source: "data/annotated/test/images/"   # 0=webcam, rtsp url, video, or folder
gate:
  crnn_confidence_threshold: 0.70   # REC-005: below -> REVIEW_REQUIRED
mqtt:
  enabled: false                    # true to use a real broker/ESP32
  broker_host: "localhost"
```

---

## 5. Run the system

```powershell
# component + pipeline self-test
python scripts/system_test_week12.py

# latency profile
python scripts/latency_profiler_week11.py

# main demo — image folder (no camera needed)
python scripts/run_demo_week12.py --limit 20

# test a single image with a full stage-by-stage trace
python scripts/test_one_image.py <path-to-image> --crop
```

---

## 6. (Optional) Live smartphone camera

1. Install **IP Webcam** (Android) or **Iriun** (iOS/Android); start the server.
2. Note the stream URL, e.g. `rtsp://192.168.1.5:8554/h264_pcm.sdp`.
3. Set `camera_source` in the config to that URL (or pass `--source`), then:
   ```powershell
   python scripts/run_demo_week12.py --source "rtsp://192.168.1.5:8554/h264_pcm.sdp"
   ```
Phone and PC must be on the **same Wi-Fi**.

---

## 7. (Optional) Real gate via MQTT + ESP32

1. Install **Mosquitto** for Windows and start the broker (`localhost:1883`).
2. Flash `hardware/esp32_gate_controller/esp32_gate_controller.ino` (fill in WiFi
   SSID/password and the **PC's IP** as `MQTT_BROKER`). Requires Arduino libraries:
   `WiFi`, `PubSubClient`, `ArduinoJson`.
3. In `configs/system_config.yaml` set `mqtt.enabled: true`.
4. Run the system; if the broker is unreachable it auto-falls back to the mock
   gate (commands still logged to `logs/mock_gate_log.txt`).

---

## 8. Notes

- **Always activate the venv** before running anything (the #1 cause of a
  CPU-only PyTorch crash).
- Model weights and the dataset are git-ignored; keep local backups.
- Windows-vs-Ubuntu is the only OS difference; no code changes are needed.
