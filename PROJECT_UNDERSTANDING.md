# PROJECT UNDERSTANDING — Cambodian ALPR

> Written 2026-09-11 from the actual code, configs, data folders, metrics and git
> history — not from the README alone. Every number is tagged with where it came
> from. Numbers marked **[measured today]** were produced by running the project's
> own scripts during this analysis. Numbers marked **[log]** come from
> `metrics/experiment_log.csv`. Numbers marked **[doc]** come from a document in
> `docs/` and were *not* re-measured. Anything unclear, stale, or inconsistent is
> flagged in §12 — read that section before presenting.

---

## 1. What the project is, in one paragraph

A real-time **Automatic License Plate Recognition (ALPR) system for a gate**. A
camera watches the gate; the software finds the plate, reads it, looks it up in
a whitelist stored in SQLite, and only if the read is *confident* **and** the
plate is *registered* does it send an OPEN command (over MQTT) to an ESP32 that
drives the gate relay. Every read — allowed or not — is written to an audit
table and saved as an annotated evidence photo. If anything is doubtful the
gate stays closed. It is a Year-2 Deep Learning course project (Kirirom
Institute of Technology) built against a formal SRS (`docs/srs.md`).

The deep-learning content is **four trained neural networks** (two YOLOv10
detectors, a CRNN sequence reader, a ResNet18 classifier) plus a disciplined
measurement story (test-set leakage audit, negative results reported honestly).

---

## 2. The domain fact that shapes the entire design

A Cambodian plate has **three lines**:

| line | content | example |
|---|---|---|
| top | province name in **Khmer script** | ភ្នំពេញ |
| middle | the **number** — digit, Latin letter(s), dash, four digits | `2A-0243` |
| bottom | province name in English | PHNOM PENH |

Two consequences:

1. **The number is read by a sequence model (CRNN); the province is *classified*, not read.**
   Khmer script needs complex glyph shaping that simple image libraries render
   wrongly, so synthetic Khmer training data was impractical. The number, by
   contrast, uses only digits + Latin + `-`, which renders reliably from any
   font. So the CRNN's alphabet is only 38 characters (`0-9`, `A-Z`, `-`, space)
   and the province is one of 26 classes (25 provinces + "other").
   The final plate string is *composed*: `"ភ្នំពេញ 2A-0243"`.

2. **There are two detectors, not one.** The public *Plate_v4* dataset's boxes
   sit on the **Khmer province line**, so the detector trained on it
   (`best.pt`) boxes the province line, not the number. A second detector
   (`number_best.pt`) was trained on ~280 hand-annotated images to box the
   **number line** specifically.

The plate-number *grammar* is also encoded explicitly in
`src/recognition/plate_format.py`: measured on 622 human labels, 98.6% of
plates match one of three shapes — `D L - D D D D` (55.5%), `D - D D D D`
(39.9%), `D L L - D D D D` (3.2%) — plus rare vanity plates (`HENGHENG`,
`COVI19`). A read that matches no legal shape has its confidence forced to 0.

---

## 3. Architecture

```
                      camera frame (phone / RTSP / webcam / image folder)
                                         │
              ┌──────────────────────────┴──────────────────────────┐
              ▼                                                     ▼
   YOLOv10-n  best.pt                                    YOLOv10-n  number_best.pt
   (boxes the PROVINCE line)                             (boxes the NUMBER line)
              │                                                     │
              │  _match_number(): pair each number box with the      │
              │  province box above it (horizontal overlap + below)  │
              ▼                                                     ▼
   _tighten_crop(trim=0.125)                             grayscale, resize 320×64
              ▼                                                     ▼
   ResNet18 province classifier                          CRNN (CNN→BiLSTM→CTC)
   26 classes → id + softmax conf                        → text + per-char confs
              │                                                     │
              └──────────────► compose_plate(id, number) ◄──────────┘
                               "ភ្នំពេញ 2A-0243"
                                         │
                       SQLite  registered_plates  (exact match, status='active')
                                         │
                       gate decision (fail-safe, priority ordered — §5)
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
     MQTT → ESP32 relay        plate_reads audit row       photos/plate_<ts>_<NUM>.jpg
     (or mock gate)            (+ system_metrics)          (annotated full frame)
```

### The four models

| # | Model | File | Architecture | Input | Output |
|---|---|---|---|---|---|
| 1 | Plate/province detector | `models/detection/best.pt` | YOLOv10-nano, 1 class `license_plate`, 2.27 M params | full frame, 640 px | boxes on the Khmer province line |
| 2 | Number detector | `models/detection/number_best.pt` | YOLOv10-nano, 1 class `plate_number` | full frame, 640 px | boxes on the number line |
| 3 | Province classifier | `models/recognition/province_classifier_best.pth` (+ `_config.json`) | ResNet18, 26-way head, 11.2 M params | RGB 128×128, ImageNet mean/std | class id + softmax confidence |
| 4 | Number reader (deployed) | `models/recognition/crnn_stn5.pth` | CRNN: 7 conv layers → 2-layer BiLSTM(256) → linear(39) → log-softmax, + an (inert) STN, 9.0 M params | grayscale 320×64, scaled to [-1, 1] | 81 timesteps × 39 classes, greedy-CTC-decoded to a string |

All four are pure PyTorch/Ultralytics at runtime. ONNX exports exist in
`models/onnx/` (parity-verified) but **the runtime does not use them**.

### Source layout (what actually matters)

| path | role |
|---|---|
| `main.py` | single entry point: interactive menu + subcommands (`dashboard`, `demo`, `read`, `enroll`, `admin`, `db`, `inside`, `accept`, `camera`, `gdrive`) |
| `configs/system_config.yaml` | every path, threshold and mode; nothing is hard-coded |
| `src/core/alpr_system.py` | `ALPRSystem` — loads everything, `process_frame()` is the pipeline, `run_video()` is the live loop |
| `src/detection/detector.py` | `PlateDetector` wraps Ultralytics YOLO: load once, warm-up, `detect()` → `[{bbox, confidence, crop}]`, `draw_boxes()` |
| `src/recognition/crnn_model.py` | `CRNN`, `STN`, `CTCDecoder`, `load_crnn()` (auto-detects STN weights), the 38-char `CHARSET` |
| `src/recognition/crnn_reader.py` | `CRNNReader`: preprocess → infer → confidence formula → format validation |
| `src/recognition/plate_format.py` | the plate-number grammar (`is_valid`, `signature`, `normalise`) |
| `src/recognition/province_classifier.py` | `ProvinceClassifier` (ResNet18, `idx_to_class` remap) |
| `src/recognition/province_map.py` | class id ↔ Khmer/Latin names, `compose_plate()`, `normalize_plate()` |
| `src/utils/database.py` | `PlateDatabase`: whitelist, audit log, parking sessions, metrics, Levenshtein near-match |
| `src/utils/rtsp_reader.py` | threaded latest-frame camera reader for webcam / RTSP / HTTP / video / image folder; reconnect; per-frame timestamp |
| `src/utils/mqtt_controller.py` | `MQTTGateController` (paho) and `MockMQTTGateController`; factory falls back to mock |
| `src/utils/logger.py` | rotating UTF-8 log `logs/alpr.log` |
| `src/utils/gdrive_storage.py` | optional Google Drive backup/sync (OAuth, background uploader) |
| `hardware/esp32_gate_controller/*.ino` | ESP32 firmware: Wi-Fi + MQTT subscribe → relay + LEDs, publishes status back |
| `scripts/` | training, evaluation, tooling — grouped by stage (§7) |
| `notebooks/` | three Colab notebooks (CRNN+STN training, province rotation training, transfer-learning study) |
| `metrics/experiment_log.csv` | append-only measurement log, each row tagged with the git commit |

---

## 4. How one frame is processed (`ALPRSystem.process_frame`)

Read this with `src/core/alpr_system.py` lines 392–569 open.

**Stage 1 — Detect** (timed as `yolo_ms`)
- `best.pt` runs on the frame → province-line boxes. `number_best.pt` runs on the
  same frame → number-line boxes. Both use `pad=self.crop_pad`, which the config
  sets to **0.0** (see §12, item 4).
- Thresholds: `yolo_confidence_threshold: 0.5`, `number_confidence_threshold: 0.4`.

**For each province box:**
- `_match_number(prov_bbox, number_dets)` picks the number box that belongs to
  it: score = horizontal-overlap-fraction × (1.0 if the number box is below the
  province box's centre, else 0.6) + 0.05 × detector confidence. It also returns
  an `align` quality (overlap ÷ smaller width) used by the consistency check.

**Stage 2 — Read** (timed as `crnn_ms`)
- The number crop → `CRNNReader.read_detailed()` → `(number, crnn_conf, char_confs)`.
- The province crop is first **trimmed by 12.5 % on each side**
  (`_tighten_crop`) because the classifier was trained on tight crops of the
  Khmer line and is confidently wrong on loose ones (swept end-to-end on 143
  scenes: trim 0.0 → 90.2 %, 0.125 → 98.6 % province accuracy — comment in
  `alpr_system.py` lines 72–86). Then → `ProvinceClassifier.predict()` →
  `(prov_id, prov_conf)`.
- `plate_text = compose_plate(prov_id, number)` → `"ភ្នំពេញ 2A-0243"`; for
  class 25 ("other") there is no prefix.

**Stage 3 — Database** (timed as `db_ms`)
- `is_registered(plate_text)`: exact string match against `registered_plates`
  where `status = 'active'`. **Only this can open the gate.**
- If not registered *and* confident: `nearest_registered()` looks for a
  whitelisted plate within 1 Levenshtein edit. A hit becomes a *suggestion* for
  human review; it never opens the gate.

**Stage 4 — Decide** — see §5.

**Persist**
- Offline (`demo`, benchmarks): every plate gets its own `plate_reads` row and
  photo.
- Live (`run_video`): **de-duplication** — consecutive reads within 3 s whose
  *number* is within 2 edits are treated as the same car; one row is kept and
  upgraded to the most confident read. Keyed on the number, not the composed
  text, because the province classifier flickers frame-to-frame on video.

The result dict carries every intermediate: `plate_text`, `number`,
`province_id`, `province_confidence`, `crnn_confidence`, `char_confidences`,
`weakest_char`, `action`, `suggested_plate`, `match_align`, per-stage timings.

---

## 5. The gate decision — fail-safe by construction

Evaluated in this order in `process_frame` (lines 481–513); the first matching
branch wins, and the default is closed:

| priority | condition | action | gate |
|---|---|---|---|
| 1 | emergency stop active (`e` key) | `REVIEW_REQUIRED` | closed |
| 2 | `crnn_conf < 0.70` (config `gate.crnn_confidence_threshold`) | `REVIEW_REQUIRED` | closed |
| 3 | parking mode on | `_parking_gate()` (see below) | — |
| 4 | exact whitelist match, status active | `ENTRY_ALLOWED` | **open** |
| 5 | confident, within 1 edit of a registered plate | `REVIEW_REQUIRED` (with `suggested_plate`) | closed |
| 6 | confident, consistency check flagged (currently **disabled** in config) | `REVIEW_REQUIRED` | closed |
| 7 | otherwise | `ENTRY_DENIED` | closed |

**The confidence number** (`crnn_reader.py::_infer`):

```
raw_conf   = mean of max-softmax over non-blank timesteps        (confidence_mode: mean)
length_f   = min(1, n_chars / 5)                                  (short misfires are penalised)
conf       = raw_conf × length_f
if format_validation and not is_valid(text):  conf = 0.0          (impossible shape → review)
```

Modes `min` and `geometric` exist; both were measured in July (AUC 0.847 /
0.855 vs 0.846 for mean — not significant) and **mean stays the default** [log].

**Two operating modes** (`gate.parking_mode`):

- **Whitelist mode** (default, `parking_mode: false`) — access control as above.
- **Parking mode** (`parking_mode: true`) — no whitelist needed. A confident read
  of a plate not currently in `parking_sessions` is an **ENTRY** (gate opens,
  row inserted); the same plate seen again is an **EXIT** (gate opens, row
  deleted — fuzzy 1-edit match allowed on exit). Sessions older than
  `parking_stale_hours` (12) are purged at startup. With
  `parking_require_permit: true`, entry additionally requires a whitelisted
  plate. `python main.py inside` lists cars inside.

**Operator controls** (live dashboard / `run_video`): `o` = manual override
(opens gate, logs `MANUAL_OVERRIDE`, ignored during e-stop), `e` = toggle
emergency stop (gate held closed until cleared), `q` = quit.

---

## 6. The models in depth

### 6.1 YOLOv10-nano detectors

- **Training** (`scripts/detection/train.py`, `train_number_detector.py`):
  Ultralytics `YOLO("yolov10n.pt").train(...)`, 100 epochs, imgsz 640, batch 16,
  **SGD**, early-stop patience 15/20, mixed precision on, `cudnn.benchmark=False`
  (a Windows-laptop cuDNN crash workaround). Best weights copied to `models/detection/`.
- **Plate detector data**: *Plate_v4* (Roboflow Universe, `taki-dk0de`, CC BY 4.0),
  re-exported as 1 class. Splits on disk: **2 054 train / 741 valid / 504 test**
  images. Boxes are on the Khmer province line.
- **Number detector data**: `data/number_detect/` — **196 train / 56 valid / 28 test**
  images with number-line boxes annotated by the project author on Roboflow
  (`cambodian-plate-number`). The test split is tiny (28 images, 29 boxes), so
  its mAP has wide error bars.
- **Rotation variant tried and rolled back**: `best_rot.pt` (trained with
  `--degrees 180 --flipud 0.5`) found rotated plates but dropped upright
  detection 96 % → 86 % end-to-end, and the readers couldn't read the rotated
  crops anyway. Deployed `best.pt` is byte-identical to `best_before_rot.pt`
  (md5 verified today).

### 6.2 CRNN number reader (`crnn_model.py`)

- **Shape walk** (verified by running the model): input `(1, 1, 64, 320)` →
  CNN with asymmetric pooling (height shrinks 64→3, width stays 320→81) →
  `(1, 512, 3, 81)` → residual height averaged to 1 → `(81, 1, 512)` as a
  sequence → 2-layer bidirectional LSTM (hidden 256) → linear to 39 classes
  (38 chars + CTC blank) → `log_softmax`. So the reader emits **81 timesteps**
  for a ~7-character plate; CTC handles the alignment.
- **Decoding**: greedy — argmax per timestep, collapse repeats, drop blanks
  (`CTCDecoder._collapse`). No beam search, no language model.
- **Per-character confidence**: the softmax max at the *first* timestep of each
  emitted run (`_char_confidences`) — aligned 1:1 with the decoded text, so
  `weakest_char` can point a reviewer at the doubtful character.
- **Training recipe** (two stages):
  1. `scripts/recognition/train.py` — from scratch on **synthetic** plates
     (`generate_synthetic.py`: 6 000 / 1 000 / 800 images on disk; the Colab
     run used 20 000). CTC loss, Adam 1e-3, StepLR. Result `crnn_best.pth`,
     CER 0 % on synthetic but **94.9 % CER on real crops** [log] — a pure
     domain gap.
  2. `scripts/recognition/finetune_crnn.py` — full fine-tune from `crnn_best.pth`
     at lr 1e-4 (cosine schedule) on real crops (oversampled ×6) mixed with
     ~3 000–16 000 synthetic, with brightness/blur/noise/±4° augmentation;
     optional `--rotate180 P` flips crops 180° with probability P; optional
     `--stn` adds the spatial transformer; early-stop on a held-out real
     validation slice. It **raises an exception if any `/test/` row reaches
     training** — leakage guard by construction. Best validation CER is
     auto-logged to `experiment_log.csv`.
- **The STN (Spatial Transformer)**: a small localisation CNN predicting a 2×3
  affine matrix, initialised to identity, applied with `affine_grid` /
  `grid_sample` before the CNN. The intent was for it to learn to rotate
  upside-down crops upright from the CTC loss alone. **It does not.** On
  `crnn_stn2/4/5` it predicts a near-identity matrix (`a00 ≈ +1.06…+1.09`) for
  upright *and* flipped input, flip fraction 0.000 — `check_stn.py` reports
  FAIL on all three [log, 2026-08-21]. The deployed `crnn_stn5.pth` still
  *contains* 25 STN tensors (auto-detected by `load_crnn`), but its upside-down
  ability comes from the `--rotate180 0.5` augmentation, not the STN. A
  supervised attempt (`--stn-supervise`, `crnn_stn6`) also failed: the
  diagonal has to pass through 0 (a singular matrix) to reach −1 and gradient
  descent stalls at ≈ +0.55 [doc: PRESENTATION_GUIDE §3 Finding 2; `crnn_stn6`
  is not on disk].

### 6.3 ResNet18 province classifier

- **Data**: `data/province_crops/{train,val,test}/<class_id>/` — **2 515 / 529 / 567**
  128 px crops, 26 folders, built by `build_province_dataset.py` from Plate_v4's
  original 29 labels (`plate_v4_to_class()` folds Cambodia/Police/RCAF/State
  into class 25 "other"). Class imbalance is real: Phnom Penh has 254 training
  crops, Mondul Kiri 18.
- **Training** (`train_province_classifier.py`): cross-entropy, Adam 1e-3,
  StepLR, 30 epochs. The deployed model was trained **ImageNet-pretrained
  (full fine-tune) with framing augmentation** (`RandomResizedCrop` scale
  0.6–1.0 + `RandomAffine` ±6°, translate 12 %, black fill + `ColorJitter`).
  That augmentation was added on 2026-08-07 specifically to cure live
  "province flicker": under simulated detector-box jitter the old model changed
  its answer on 70 % of plates; the framing-augmented one on 6 %, with clean
  accuracy unchanged [doc: MODEL_IMPROVEMENT_PLAN Step 3]. It was promoted to
  `province_classifier_best.pth` (file sizes and dates match `_framing.pth`).
- **The `idx_to_class` gotcha**: `torchvision.datasets.ImageFolder` sorts class
  folders lexicographically (`'0','1','10','11',…,'2',…`), so model output
  index ≠ province id. The training script saves the true mapping in
  `province_classifier_config.json` and the classifier remaps at predict time.
  Never bypass it.
- **Transfer-learning study** (`notebooks/colab_transfer_learning.ipynb`): same
  data/epochs, three runs — from scratch 94.2 %, ImageNet **feature extraction
  (frozen backbone) 51.3 %**, ImageNet fine-tune 94.5 % [doc: PRESENTATION_GUIDE
  Finding 4]. Conclusion: ImageNet features alone don't separate Khmer glyphs;
  domain training is what matters. Those three checkpoints are not on disk.
- **Rotation-trained variant** `province_classifier_rot2.pth` (trained with
  `--rotate180`) exists on disk: 92.4 % upright / 93.3 % upside-down [doc]. It is
  **not deployed**; the deployed model reads upside-down provinces at 19.4 %
  [measured today].

---

## 7. Data — sources, splits, and labelling provenance

| dataset | where | source | train / val / test |
|---|---|---|---|
| Plate detection scenes | `data/annotated/` | Plate_v4, Roboflow (CC BY 4.0) | 2 054 / 741 / 504 |
| Number-line detection | `data/number_detect/` | author-annotated subset on Roboflow | 196 / 56 / 28 |
| Real number crops for CRNN | `data/crnn_crops/` + `real_labels.csv` | **cut from the Plate_v4 scenes above** by `crop_numbers.py` with `number_best.pt` | 792 / 167 / **149** labelled rows (1 803 / – / 436 crop files) |
| De-leaked CRNN labels | `real_labels_deleaked.csv` | `check_leakage.py --write-clean` | 716 / 157 / 149 |
| Hard-condition augmented copies | `data/crnn_crops_augmented/` (2 716 rows) | `augment_real_crops.py`, train-only | – |
| Synthetic plates | `data/synthetic/` | `generate_synthetic.py` (PIL render + perspective/lighting/blur/noise/JPEG) | 6 000 / 1 000 / 800 on disk; 20 000 generated on Colab |
| Province crops | `data/province_crops/` | Plate_v4 boxes, 29 → 26 classes | 2 515 / 529 / 567 |
| Province ground truth for end-to-end scoring | `data/crnn_crops/province_test_labels.csv` | hand-labelled, 149 scenes | test only |

Important provenance facts:

- **The 149 CRNN test labels are human-labelled; most train labels were
  AI-transcribed** from montage sheets (`make_montage.py` → transcribe →
  `import_label_csv.py`) and later harvested via active learning
  (`harvest_active.py`). This is stated in `docs/HANDOFF.md` §7 and
  `PROJECT_OVERVIEW.md` §6.
- **Test-set leakage was found and corrected.** Because Plate_v4 contains the
  same physical car photographed several times, 52 of the 149 test crops
  (34.9 %) share a plate *number* with a train/valid crop under a different
  filename, plus 4 byte-identical duplicates. A path-based guard cannot catch
  this. `check_leakage.py` measures it and writes `real_labels_deleaked.csv`
  (contaminated train/valid rows dropped, test rows kept). `crnn_stn5` was
  trained on the de-leaked file. On the *clean* 97 test crops the
  upside-down score of `crnn_stn4` fell from a headline 69.1 % to 56.7 % —
  the signature of memorisation [log].
- **`data/raw/` is essentially empty** (8 files). The "collect 100 plates in
  Phnom Penh" plan in `DATA_COLLECTION_GUIDE.md` was not the source of the
  training data; the public Roboflow dataset was. (See §12, item 3.)

---

## 8. Measured results

### 8.1 Verified today (2026-09-11) with the deployed configuration

Deployed config: `crnn_weights: crnn_stn5.pth`, `province_classifier_best.pth`,
`province_crop_trim: 0.125`, `format_validation: true`, `consistency_check: false`.

| what | script | result |
|---|---|---|
| Plate detector, 504-image test split | `YOLO(best.pt).val(split="test")` | **mAP50 0.9664**, mAP50-95 0.7259, P 0.954, R 0.944 |
| Number detector, 28-image test split | `YOLO(number_best.pt).val(split="test")` | **mAP50 0.9567**, mAP50-95 0.693, P 0.796, R 0.939 |
| **End-to-end on 149 labelled full scenes** | `scripts/tools/score_pipeline.py` | detector found **143/149**; **number correct 83.9 % (125/149)**; **province correct 94.0 % (140/149)**; **both correct 82.6 % (123/149)** — a detector miss counts as a failure |
| CRNN `crnn_stn5.pth`, 149 test crops | `test_rotation_reading.py` | **upright 87.2 % (130/149)**, **upside-down 51.7 % (77/149)** |
| CRNN `crnn_finetuned.pth` (previous deployed) | same | upright 80.5 %, upside-down 0.0 % |
| Province classifier, 567 test crops, trim 0.125 | `test_province_rotation.py` | **upright 96.1 % (545/567)**, upside-down 19.4 % |

Of the 24 number errors in the end-to-end run: 6 are detector misses; of the
18 misreads, 5 are vanity plates whose ground truth is not a legal plate shape
(`HENGHENG`, `COVI19`, `SELAGTR`, `HYWAZA9`, `ELDC865`), 2 are label artefacts
where the plate was partly out of frame (`6667`, `-7495`), and the remaining 11
are single-character confusions on standard plates (`3A→3E`, `2803→2603`,
`1Q→1O`). Two further scenes had the number right but the province wrong.
`results/pipeline_mistakes.csv` now holds this run's mistake list (it
previously held the upside-down run's list from 2026-08-24).

### 8.2 From `metrics/experiment_log.csv` (not re-run today)

| what | value | date / note |
|---|---|---|
| Test-set contamination | 34.9 % of test crops (52/149) | 2026-08-21 |
| `crnn_stn5` on the **clean** 97 crops | upright **84.5 %**, upside-down **48.5 %** | 2026-08-21 — the README's headline |
| `crnn_stn5` on the 52 leaked crops | upright 92.3 %, upside-down 57.7 % | shows the memorisation gap |
| STN verdict, `crnn_stn2/4/5` | flip fraction 0.000, FAIL | 2026-08-21 |
| CRNN CER history on real test | 94.9 % (synthetic only) → 25.9 % (143 labels) → 20.3 % (324) → 10.2 % (473) | Jul 2026 |
| Province classifier test accuracy (training-time eval, no trim) | 97.18 % | Jul 2026 |
| `benchmark_composed.py` with the *old* CRNN (`crnn_finetuned.pth`), n = 149 | number 76.5 %, CER 10.1 %, province 95.2 % (n = 147), composed 77.6 % | 2026-08-08 — **this is the README's end-to-end table** |
| False-accept harness (whitelist = 137 fabricated 1-edit neighbours of the true plates) | **0 opens, FAR 0.0 %** | 2026-08-08 |
| Gate simulation, `intruder_false_open` | **1** of 143 detected | 2026-08-08 — see §12 item 5 |
| Consistency check (2.2) precision on confident reads | 14.3 % vs 14.5 % base rate → no predictive value → **disabled** | 2026-07-23 |
| Confidence mode min / geometric vs mean (AUC) | 0.847 / 0.855 vs 0.846 → keep mean | 2026-07-23 |
| Visit-level fusion (char vote) vs best-confidence frame | 73.6 % vs 75.7 % → fusion worse; oracle ceiling 77.9 % | 2026-07-23 |
| ONNX parity, all four models | max abs diff ≤ 1.1e-5 | 2026-07-17 |

### 8.3 From docs only (no log row, not re-measured)

| what | value | source |
|---|---|---|
| Latency, full two-detector pipeline, RTX 3050 | avg 47–51 ms, p95 69 ms, ~20 FPS | `srs_acceptance.json` (2026-07-15), HANDOFF |
| SRS acceptance suite | 16 / 16 pass (MAN-002 marked "skipped" but counted pass) | `srs_acceptance.json`, 2026-07-15 |
| Province flicker under box jitter, before → after framing aug | 70 % → 6 % of plates | MODEL_IMPROVEMENT_PLAN |
| Province classifier under degradation (blur, 6× downscale, JPEG q20, combined) | 97.2 % → 91.7 % worst case | MODEL_IMPROVEMENT_PLAN |
| Upside-down **end-to-end** (whole scenes rotated) | detector finds 44/149 (30 %); number 16.8 %, province 3.4 % | PRESENTATION_GUIDE Finding 3 — the detector, not the reader, is the bottleneck |
| Transfer-learning study | scratch 94.2 %, feature-extraction 51.3 %, fine-tune 94.5 % | PRESENTATION_GUIDE Finding 4 |
| McNemar's test, stn2→stn4 upside-down (clean) | p = 0.012 significant; stn4→stn5 p = 0.13 not significant | PRESENTATION_GUIDE §4 |
| `orientation_search` inference wrapper (rejected) | ~76 % at any angle | HANDOFF_ROTATION |

---

## 9. What was tried and rejected (negative results — present these)

| idea | what happened | evidence |
|---|---|---|
| Naive 180° flip augmentation on the CRNN, no extra data | upright fell 80.5 % → 62 %, upside-down only 12 % | HANDOFF_ROTATION |
| Rotation-training the plate detector (`best_rot.pt`) | detected rotated plates but upright detection fell 96 % → 86 %; readers still failed on the crops; rolled back | experiment_log 2026-08-07/08, MODEL_IMPROVEMENT_PLAN Step 4 |
| Spatial Transformer learning rotation from CTC loss | predicts identity for both orientations; contributed nothing | `check_stn.py`, log 2026-08-21 |
| Supervising the STN transform directly (`crnn_stn6`) | diagonal stalled at +0.55; accuracy fell to 77.9 % / 37.6 % | PRESENTATION_GUIDE Finding 2 |
| Inference-time orientation search (try 0/90/180/270, keep best) | works (~76 %) but **rejected by the author** — the goal is for the *model* to learn, not for wrapper logic to hide the problem; kept off (`orientation_search: false`) | HANDOFF_ROTATION |
| `min` / `geometric` confidence instead of `mean` | no significant AUC change | log 2026-07-23 |
| Province↔number consistency flag (2.2) | precision equal to base rate; turned off | log 2026-07-23 |
| Multi-frame character voting per visit | worse than taking the most confident frame | log 2026-07-23 |
| 10 % crop padding (SRS DET-005) | number accuracy 70.6 % → 46.2 %; padding kept at 0 (DEV-004) | SRS_DEVIATION_LOG |
| Fine-tune on 521 labels (Phase 5 candidate) | word-acc 70.5 % vs deployed 72.5 % → not shipped | log 2026-07-23 |
| ImageNet feature extraction (frozen backbone) | 51.3 % — underfits Khmer glyphs | Finding 4 |
| "6/9 ambiguity explains the upside-down ceiling" | plates *with* 6 or 9 score slightly better (53.9 % vs 48.3 %); real failure is dropped/substituted characters | Finding 5 |

---

## 10. Engineering around the models

- **Camera input** (`rtsp_reader.py`): a daemon thread reads continuously and
  keeps only the *latest* frame, so inference never falls behind the stream.
  Modes: webcam index, `rtsp://` / `http://` stream, video file, or an image
  folder (loops, throttled to ~30 FPS). Reconnects 10× every 5 s (SRS VID-001);
  each frame carries its acquisition timestamp so the dashboard can show frame
  age (VID-002). Default source is `data/annotated/test/images/`, so **no camera
  is needed to demo**.
- **Gate output** (`mqtt_controller.py`): with `mqtt.enabled: false` (default) a
  mock controller prints and appends to `logs/mock_gate_log.txt`. With
  `enabled: true` it publishes JSON `{"command":"OPEN","plate":…,"duration":3,
  "timestamp":…}` on `alpr/main_gate/control` via paho-mqtt and falls back to
  the mock if the broker is unreachable within 2 s.
- **ESP32 firmware**: subscribes to that topic, parses JSON with ArduinoJson,
  drives `RELAY_PIN 5` + green/red LEDs, holds the relay for `duration`
  seconds (blocking `delay`), publishes `{"status":"opened"/"closed"/
  "emergency_stop"}` on `alpr/main_gate/status`. Relay is LOW on boot
  (fail-safe closed). Wi-Fi/broker credentials are placeholders.
- **Database** (`plates.db`, schema in `docs/database.md`): `registered_plates`
  (whitelist with `status` active/suspended/expired), `plate_reads` (audit:
  detected text, matched text, both confidences, action, photo path),
  `parking_sessions` (cars currently inside — added beyond the doc'd schema),
  `system_metrics` (fps, latency, GPU MB, CPU %, stream connected, uptime).
- **Evidence & logs**: every read saves the annotated full frame to
  `photos/plate_<timestamp>_<NUMBER>.jpg`; structured log in `logs/alpr.log`;
  alerts (latency > 500 ms, stream down > 15 s, GPU OOM, DB error) in
  `logs/alerts.log`; health sample every 100 frames.
- **Admin panel** (`scripts/system/admin_web.py`): stdlib `http.server` +
  Jinja2, port 5000, password-protected (PBKDF2-SHA256, 200 k iterations,
  server-side session cookie; refuses to start without a password). Add /
  suspend / reactivate / delete plates, search the audit log. Khmer renders in
  the browser.
- **Dashboard** (`scripts/system/dashboard.py`): OpenCV window — video with
  boxes, gate banner, FPS/latency/GPU, counters, recent events, clickable
  MANUAL-OPEN / E-STOP.
- **Google Drive** (`gdrive_storage.py`, `gdrive.enabled: true`): background
  uploader for photos/outputs/logs and periodic DB backup; requires OAuth
  credential files that are git-ignored.
- **Colab workflow**: `make_colab_bundle.py` zips `src/`, scripts, crops and
  base weights (`alpr_colab_bundle.zip`, 151 MB in the root); the notebooks
  verify the bundle contains the expected code before training; weights are
  copied back to Drive and evaluated locally.
- **Experiment tracking**: `experiment_log.py` — one CSV, columns
  `timestamp, git_commit, component, metric, value, split, notes`. Evaluation
  scripts append with `--log`; `finetune_crnn.py` logs automatically. W&B is an
  optional `--wandb` flag for live curves only.

---

## 11. Technologies (as actually used)

| layer | technology |
|---|---|
| Language / runtime | Python 3.10, Windows 11, NVIDIA RTX 3050 Laptop 4 GB |
| Deep learning | PyTorch 2.5.1 + CUDA 12.1, torchvision (ResNet18), Ultralytics (YOLOv10-n) |
| Vision / data | OpenCV, NumPy, Pillow (synthetic rendering), pandas |
| Training compute | Google Colab T4 for heavy runs; local GPU for evaluation and short fine-tunes |
| Storage | SQLite (`plates.db`), YAML config, CSV experiment log, optional Google Drive |
| IoT | MQTT (paho-mqtt; Mosquitto broker), ESP32 (Arduino: WiFi, PubSubClient, ArduinoJson) |
| UI | OpenCV dashboard; stdlib HTTP server + Jinja2 admin panel |
| Export / MLOps | ONNX export + onnxruntime parity check (not used at runtime); optional Weights & Biases; DVC *planned, not implemented* |

---

## 12. Things that are stale, inconsistent, or unimplemented — know these before you present

1. **The README's end-to-end table (76.5 % number, 2026-08-08) is stale.** It was
   measured by `benchmark_composed.py` when `crnn_finetuned.pth` was deployed.
   The config switched to `crnn_stn5.pth` on 2026-09-11 (commit `f0d0860`) and no
   `benchmark_composed.py` row was logged after that. The current deployed
   pipeline, scored today with `score_pipeline.py`, reads **83.9 %** of numbers
   over 149 scenes (detector misses counted as failures). The two scripts
   differ slightly (benchmark_composed also runs the false-accept harness and
   composed-plate metric), so quote them as what they are.
2. **`metrics/week2_metrics.json` says mAP50 0.9016**, written on 2026-08-19 —
   most likely while the rotation detector was being evaluated. The deployed
   `best.pt` measures **0.9664** today. `srs_acceptance_test.py` reads DET-001
   from that JSON, so re-running the acceptance suite would print 0.9016 (still
   a pass). Re-run `scripts/detection/evaluate.py` to refresh the file.
3. **README says the recognition data was "photographed and labelled for this
   project".** The crops in `data/crnn_crops/` are cut from the Plate_v4
   Roboflow scenes (20/20 sampled test crops have the same-named scene in
   `data/annotated/test/images`; `data/raw/` holds 8 files). `PRESENTATION_GUIDE.md`
   §5 states this correctly: "a public Roboflow dataset, not purpose-collected."
   What *was* produced in-house: the number-line boxes (~280 images), the
   149 human test labels, the province ground truth for 149 scenes, and the
   AI-assisted train labels.
4. **Comments in `alpr_system.py` say "the detector pads every box by 10 % per
   side"** (lines 73–75, 408) but `configs/system_config.yaml` sets
   `detection.crop_padding: 0.0` (DEV-004), so `crop_pad` is 0. The 12.5 %
   trim value was swept *end-to-end through the real pipeline*, so the number is
   still valid — but the stated reason is inconsistent with the config.
5. **"0 false accepts" needs a precise sentence.** It is true on the harness
   where the whitelist holds 137 *fabricated* 1-edit neighbours of the true
   plates. The same benchmark's gate simulation (whitelist = all true test
   numbers) recorded **`intruder_false_open: 1`** — one confident read of plate
   X exactly matched a *different real* plate Y in the set. The script itself
   labels this a "pre-existing exact-match risk". If two such cars are both
   registered, a misread can open the gate. Say: "zero false accepts against
   near-miss plates; one cross-plate misread out of 143 in simulation."
6. **SRS acceptance "16/16" is from 2026-07-15**, before the current models.
   MAN-002 (e-stop) was *skipped* (needs an enrolled plate) and counted as pass.
   Re-run `python main.py accept` before claiming it.
7. **Stale docs**: `PROJECT_OVERVIEW.md` (2026-07-17) says there is no ONNX
   export, no experiment tracking, no admin auth, and names `crnn_finetuned.pth`
   as deployed — all since changed. `HANDOFF_ROTATION_UPSIDEDOWN.md` (2026-08-19)
   says the STN "worked"; the 2026-08-21 measurement showed it does nothing.
   `HANDOFF.md` predates the two-detector integration. Treat
   `PRESENTATION_GUIDE.md` and `experiment_log.csv` as the current truth.
8. **Number-detector mAP50 is quoted three ways**: 0.943 (README/log history),
   0.9567 (test split, today), 0.9592 (best val epoch). The test split is 28
   images, so all of these are noisy.
9. **Province accuracy is quoted two ways**: 97.18 % (training-time test eval on
   untrimmed crops, Jul) vs 96.1 % (today, `test_province_rotation.py`, trim
   0.125). Different preprocessing, both real.
10. **The 2-hour live phone-stream stability run (SRS F1 / VID-001) has no
    recorded evidence.** `results/stream_snapshot.jpg` (2026-07-24) and a
    commented-out phone URL in the config show the stream was connected at
    least once.
11. **Not implemented**: DVC model versioning (MLOps Phase B); the
    `mqtt.srs_compat` plain-text MQTT bridge promised in DEV-003; a
    `gate.format_repair` path (`plate_format.normalise` exists but is
    deliberately unwired); Khmer character recognition inside the CRNN (SRS
    REC-002, resolved by DEV-002 via the classifier); TensorRT / quantisation;
    the ESP32 has never been wired to a real gate in the repo's evidence
    (mock gate is the default).
12. **Unused / leftover files**: `yolo26n.pt`, `yolov10n.pt` (base weights in
    root), `crnn_ft_phase5*.pth`, `crnn_augmented_candidate.pth`,
    `crnn_finetuned_324.bak.pth`, `province_classifier_rot*.pth`,
    `models/onnx/*`, `make_label_sheet.py.bak`, several `runs/detect/val*`
    folders. None affect runtime.
13. **`crnn_stn5.pth` ships with inert STN weights.** If asked "is the STN
    used?", the honest answer is: the layer runs (it is in the forward pass)
    but predicts ≈ identity, so it is a no-op.
14. The SRS's stated `REC-001` target is CER ≤ 10 %; the acceptance test checks
    ≤ 15 %. The July measurement (10.21 %) passes the test's threshold but is
    just above the SRS wording. Today's deployed CRNN is better (87.2 % word
    accuracy; CER not re-measured today).

---

## 13. Concepts you should be able to explain (with where they live)

| concept | one-line explanation | where |
|---|---|---|
| **Object detection / YOLO** | one CNN pass predicts boxes + class + confidence for the whole image; YOLOv10 is NMS-free | `detector.py` |
| **mAP@50** | mean average precision at IoU ≥ 0.5: area under the precision-recall curve when a box counts as correct if it overlaps ground truth by ≥ 50 % | `evaluate.py`, Ultralytics `val()` |
| **CRNN** | CNN extracts a column-wise feature sequence, BiLSTM reads it left→right and right→left, a linear layer scores every character at every column | `crnn_model.py` |
| **CTC (Connectionist Temporal Classification)** | lets a variable-length text be trained against a fixed 81-step output without character-position labels, via a "blank" token and collapse-repeats decoding | `nn.CTCLoss`, `CTCDecoder` |
| **Greedy decoding** | argmax per timestep → merge repeats → drop blanks (no beam search here) | `CTCDecoder._collapse` |
| **CER / word accuracy** | CER = edit distance ÷ target length summed over the set; word accuracy = fraction of exact-match strings | `finetune_crnn.py::cer`, `evaluate_crnn_on_real.py` |
| **Domain gap** | synthetic-trained CRNN got 0 % CER on synthetic but 95 % CER on real crops until fine-tuned | log history |
| **Fine-tuning vs feature extraction** | fine-tune = all weights train from a pretrained start; feature extraction = backbone frozen, only the head trains (51 % here) | `train_province_classifier.py --pretrained [--freeze]` |
| **Softmax confidence & calibration** | max softmax ≈ confidence; the 0.70 gate uses mean over characters × length factor; AUC used to compare modes | `crnn_reader.py` |
| **Spatial Transformer Network** | a sub-network predicts an affine warp applied to the input, trained end-to-end; here it converged to identity | `crnn_model.py::STN`, `check_stn.py` |
| **Data leakage / near-duplicates** | same object in train and test inflates scores; audited by plate number, not filename | `check_leakage.py` |
| **Augmentation** | random brightness/blur/noise/rotation/flip/crop at train time to widen the data distribution | `finetune_crnn.py::augment`, `make_loaders()` |
| **Fail-safe design** | default action is closed; every uncertain path routes to `REVIEW_REQUIRED`; only exact match + confidence opens | `process_frame` stage 4 |
| **Levenshtein distance** | minimum single-character edits between strings; used for review suggestions and live de-dup, never to open the gate | `database.py::_edit_distance` |
| **McNemar's test** | paired significance test on the same items scored by two models | PRESENTATION_GUIDE §4 |

---

## 14. Suggested 60-second version for the presentation

> Cambodian plates carry a Khmer province line and a Latin/digit number line.
> I read the number with a CRNN trained with CTC loss and classify the province
> with a ResNet18, each fed by its own YOLOv10 detector, then compose the plate
> string and check it against a SQLite whitelist. The gate opens only on an exact
> match with reader confidence ≥ 0.70; everything else stays closed and goes to
> review, with a photo and an audit row. On 149 held-out real scenes the full
> pipeline gets the number right 83.9 % of the time and the province 94.0 %.
> The part I'm proudest of is the measurement: I found that a third of my test
> set shared plates with training data, removed the contamination, and reported
> the lower number; and I built a tool that proved my own Spatial Transformer
> layer does nothing, so the upside-down reading ability — 0 % → 51.7 % — is
> attributed to augmentation, not to that layer.

---

## 15. Commands that reproduce the numbers in this document

```powershell
.\.venv\Scripts\activate
python scripts\tools\score_pipeline.py                                           # 83.9 / 94.0 / 82.6
python scripts\tools\test_rotation_reading.py --weights models\recognition\crnn_stn5.pth   # 87.2 / 51.7
python scripts\tools\test_province_rotation.py --weights models\recognition\province_classifier_best.pth  # 96.1 / 19.4
python scripts\tools\check_leakage.py --audit                                    # 34.9 %
python scripts\tools\check_stn.py --weights models\recognition\crnn_stn5.pth     # FAIL
python scripts\detection\evaluate.py                                             # refreshes week2_metrics.json → 0.9664
python scripts\tools\experiment_log.py --show                                    # the full measurement history
python main.py demo --limit 20                                                   # visual demo on real images
```
