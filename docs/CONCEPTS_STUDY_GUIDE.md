# Cambodian ALPR — Concepts Study Guide

A learn-your-own-project guide. Every concept below is something that already
exists in **your** code. Each section says **what it is**, **where it lives in
your repo**, and — where useful — a **simple worked calculation** you can do by
hand so the idea becomes concrete.

> How to use this: read a section, open the file it points to, then re-read the
> code with the concept in mind. Do the little calculations with pen and paper.

---

## Table of contents

1. [The big picture — the pipeline](#1-the-big-picture--the-pipeline)
2. [Images as numbers (NumPy + OpenCV)](#2-images-as-numbers-numpy--opencv)
3. [PyTorch fundamentals](#3-pytorch-fundamentals)
4. [Stage 1 — YOLO object detection](#4-stage-1--yolo-object-detection)
5. [Detection metrics: IoU, precision/recall, mAP](#5-detection-metrics-iou-precisionrecall-map)
6. [Stage 2 — CRNN: CNN + BiLSTM + CTC](#6-stage-2--crnn-cnn--bilstm--ctc)
7. [Softmax, log-softmax, and confidence](#7-softmax-log-softmax-and-confidence)
8. [CTC decoding — the core idea](#8-ctc-decoding--the-core-idea)
9. [Your CRNN confidence formula (with length penalty)](#9-your-crnn-confidence-formula-with-length-penalty)
10. [Stage 3 — the database & whitelist](#10-stage-3--the-database--whitelist)
11. [Stage 4 — the gate decision & fail-safe logic](#11-stage-4--the-gate-decision--fail-safe-logic)
12. [The engineering glue (camera, MQTT, config, logging)](#12-the-engineering-glue)
13. [Performance: latency & FPS](#13-performance-latency--fps)
14. [A suggested study order + checklist](#14-a-suggested-study-order--checklist)

---

## 1. The big picture — the pipeline

Your whole system is a **4-stage pipeline**. One frame goes in, a gate decision
comes out.

```
Camera → [1] YOLO detect → [2] CRNN read → [3] DB lookup → [4] gate decision → log
```

**Where:** `src/core/alpr_system.py`, method `process_frame()` (around line 198).
This method *is* the project. Everything else feeds it or is called by it.

The code even times each stage separately:

```python
yolo_ms  = ...   # how long detection took
crnn_ms  = ...   # how long reading took
db_ms    = ...   # how long the DB lookup took
total_ms = ...   # the whole frame
```

**Concept — pipeline / stages:** a complex task is split into independent steps,
each with one job. The output of step *N* is the input of step *N+1*.

---

## 2. Images as numbers (NumPy + OpenCV)

Before any AI, you must understand: **an image is just a grid of numbers.**

- A grayscale image of height `H` and width `W` is an `H × W` array. Each cell is
  a pixel brightness from `0` (black) to `255` (white).
- A color image is `H × W × 3`. The 3 channels in OpenCV are **B, G, R** (blue,
  green, red) — note: **not** RGB. This is why your code sometimes converts:
  `cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)` in `crnn_reader.py:61`.

**Where in your code:**
- `cv2.imread` / `cv2.imwrite` — load/save images (`alpr_system.py`).
- `image[cy1:cy2, cx1:cx2]` — *cropping* is just array slicing (`detector.py:138`).
- `cv2.resize`, `cv2.putText`, `cv2.rectangle` — resize and draw.

**Simple calculation — how big is one frame?**
A 640×640 color frame holds:

```
640 × 640 × 3 = 1,228,800 numbers (bytes) ≈ 1.2 MB uncompressed
```

**Normalization** (turning pixels 0–255 into a small range the network likes).
Your CRNN does this in `crnn_reader.py:63-64`:

```python
arr = resized.astype("float32") / 255.0   # now in [0, 1]
arr = (arr - 0.5) / 0.5                    # now in [-1, 1]
```

Worked example for one pixel of value `200`:

```
200 / 255       = 0.784
(0.784 - 0.5)/0.5 = 0.568   → the value the network actually sees
```

A pixel of `0` → `-1.0`; a pixel of `255` → `+1.0`. So the formula maps the
`[0,255]` range onto `[-1,+1]`, centered at 0. Networks train better on centered,
small-range inputs.

---

## 3. PyTorch fundamentals

**PyTorch** is the deep-learning framework both your models are built in.
The must-know pieces, all visible in `src/recognition/crnn_model.py`:

| Concept | What it means | Where |
|--------|----------------|-------|
| **Tensor** | An n-dimensional array (like NumPy) that can live on the GPU | everywhere |
| `nn.Module` | Base class for a model; you subclass it | `class CRNN(nn.Module)` line 46 |
| `__init__` | Where you **define the layers** | lines 49–75 |
| `forward()` | Where you **use the layers** on input `x` | lines 77–93 |
| `state_dict` | The learned weights, saved to a `.pth` file | `load_crnn`, line 148 |
| `.eval()` | Switch to inference mode (no training behavior) | line 149 |
| `torch.no_grad()` | "Don't track gradients" — faster inference | `crnn_reader.py:73` |
| `.to(device)` | Move tensor/model to `"cuda"` (GPU) or `"cpu"` | `crnn_reader.py:66` |

**Tensor shapes** are the thing students trip on. Read the shape comments in
`crnn_model.py` `forward()` — every line notes the shape, e.g. `(b, c, h, w)`.
That `(batch, channels, height, width)` order is the PyTorch image convention.

`torch.cuda.is_available()` (`crnn_reader.py:30`) is how your code auto-picks GPU
vs CPU — the GPU is why detection is ~8 ms instead of hundreds.

---

## 4. Stage 1 — YOLO object detection

**Where:** `src/detection/detector.py`, class `PlateDetector`. It wraps
**Ultralytics YOLOv10** (`from ultralytics import YOLO`, line 64).

**Classification vs detection** — the key distinction:
- *Classification* answers "what is in this image?" → one label.
- *Detection* answers "what is in this image **and where**?" → a list of
  **bounding boxes**, each with a label and a confidence.

Your `detect()` returns exactly that (line 99-103):

```python
[{"bbox": (x1, y1, x2, y2),   # box corners in pixels
  "confidence": float,         # 0..1, how sure YOLO is
  "crop": <the pixels inside>}]
```

**Concepts to learn here:**
- **Bounding box** `(x1,y1,x2,y2)` = top-left and bottom-right corners.
- **Confidence threshold** — `conf=0.5` (line 42). Boxes below this are thrown
  away. Your config sets `yolo_confidence_threshold: 0.50`.
- **Clamping** (lines 126-127): making sure a box never goes outside the image.
- **Warm-up** (lines 75-85): the first GPU call is always slow, so the code runs
  one throwaway inference and doesn't count it. Good real-world detail.
- **Transfer learning**: you didn't train YOLO from zero — you started from
  `yolov10n.pt` (a model pretrained on millions of images) and fine-tuned it on
  Cambodian plates. The `n` means "nano", the smallest/fastest size.

**Your clever trick — two detectors** (`alpr_system.py:63-83`): `best.pt` finds
the **Khmer province line**, `number_best.pt` finds the **number line**. They
detect *different rows of the same plate*. `_match_number()` (line 171) then pairs
each number box with the province box **above** it using horizontal overlap. Read
that scoring function — it's pure geometry, no AI.

---

## 5. Detection metrics: IoU, precision/recall, mAP

Your README reports **mAP50 = 0.9664**. Here's what that number means, built up
from scratch with calculations.

### IoU (Intersection over Union)
How much do two boxes overlap? `IoU = area_of_overlap / area_of_union`.

**Worked example.** Predicted box `A = (0,0,10,10)`, ground-truth box
`B = (5,0,15,10)` (both 10×10, shifted right by 5):

```
Overlap  = width 5 × height 10           = 50
Area A   = 10 × 10 = 100
Area B   = 10 × 10 = 100
Union    = 100 + 100 − 50                = 150
IoU      = 50 / 150                       = 0.333
```

A detection usually "counts as correct" if `IoU ≥ 0.5`. That `0.5` is the **"50"**
in **mAP50**.

### Precision & Recall
Given all your predictions vs the true plates:

```
Precision = TP / (TP + FP)   "of the boxes I predicted, how many were right?"
Recall    = TP / (TP + FN)   "of the real plates, how many did I find?"
```

(TP = true positive, FP = false positive, FN = missed.)

**Worked example.** 100 real plates. Model outputs 90 boxes: 85 correct (TP),
5 wrong (FP), and it missed 15 (FN):

```
Precision = 85 / (85 + 5)  = 85/90  = 0.944
Recall    = 85 / (85 + 15) = 85/100 = 0.850
```

### AP and mAP
- **AP (Average Precision)** = the area under the precision–recall curve for one
  class (a number between 0 and 1).
- **mAP** = **mean** AP averaged over all classes. With one class (plate), mAP = AP.
- **mAP50** = mAP computed using the `IoU ≥ 0.5` rule.

So your **0.9664** means: across the test set, at the 0.5-overlap bar, your
detector's precision-recall area is 96.6%. Very strong.

**Where:** `scripts/detection/evaluate.py` and `train.py` compute these via
Ultralytics.

---

## 6. Stage 2 — CRNN: CNN + BiLSTM + CTC

**Where:** `src/recognition/crnn_model.py` (the network) and
`src/recognition/crnn_reader.py` (the inference wrapper).

CRNN = **C**onvolutional **R**ecurrent **N**eural **N**etwork. It reads a whole
line of text (e.g. `1AB-2345`) from a crop, in one shot. Three parts:

### (a) CNN — sees visual features
`crnn_model.py:58-70`. A stack of `Conv2d → BatchNorm → ReLU → MaxPool`.

- **Convolution (`Conv2d`)**: slides a small filter over the image to detect
  patterns (edges, curves, then strokes of characters).
- **ReLU**: the activation `f(x) = max(0, x)` — keeps positives, zeros negatives.
  This is what makes the network *non-linear* (able to learn complex shapes).
- **BatchNorm**: rescales activations to keep training stable.
- **MaxPool**: downsamples by keeping the max in each little window.

**The clever part — asymmetric pooling** (lines 65, 68):
`nn.MaxPool2d((2,2),(2,1),(0,1))` shrinks **height** faster than **width**. Why?
Because **width is the reading direction** (the "time" axis). You want to squeeze
height down to 1 row, but keep many columns — one column ≈ one character slot.

**Simple calculation — convolution output size.** For a conv/pool layer:

```
out = floor( (in + 2·padding − kernel) / stride ) + 1
```

Example, a `MaxPool2d(2,2)` (kernel 2, stride 2, pad 0) on height 64:

```
out = floor((64 + 0 − 2) / 2) + 1 = floor(62/2)+1 = 31 + 1 = 32
```

So 64 → 32 → 16 → 8 … the height keeps halving until it reaches 1, exactly as the
shape comments in the code say (`H/2 W/2`, `H/4 W/4`, …).

### (b) BiLSTM — reads the sequence
`crnn_model.py:73`. After the CNN, the image is a **sequence of feature columns**
(shape `(width, batch, features)`, line 89). Two stacked **bidirectional LSTMs**
read this sequence left→right *and* right→left, so each position "knows" its
neighbors on both sides (context helps: `0` vs `O`, `1` vs `I`).

- **RNN** = a network with memory that processes a sequence step by step.
- **LSTM** = a better RNN that remembers long-range info without forgetting.
- **Bidirectional** = two LSTMs, one each direction, outputs concatenated →
  that's why the final layer is `n_hidden * 2` (line 75).

### (c) CTC head — turns the sequence into text
`self.fc` (line 75) maps each of the sequence positions to a score over your
**character set** + a special **blank**. Decoding explained in §8.

**Character set** (`crnn_model.py:32-37`):

```python
DIGITS = "0123456789"                  # 10
LATIN  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 26
SEP    = "- "                          # 2 (dash, space)
CHARSET = DIGITS + LATIN + SEP         # 38 characters
BLANK   = 38                           # the blank is index 38 (the 39th class)
N_CLASSES = 39
```

So the network's final layer outputs **39 numbers per column** — a score for each
possible character plus "blank".

---

## 7. Softmax, log-softmax, and confidence

The network's raw outputs (called **logits**) are arbitrary numbers. **Softmax**
turns a row of them into probabilities that sum to 1.

```
softmax(z_i) = e^(z_i) / Σ_j e^(z_j)
```

**Worked example.** Suppose for one column the logits over just 3 classes are
`[2.0, 1.0, 0.1]`:

```
e^2.0 = 7.389
e^1.0 = 2.718
e^0.1 = 1.105
sum   = 11.212

softmax = [7.389/11.212, 2.718/11.212, 1.105/11.212]
        = [0.659, 0.242, 0.099]        (sums to 1.0 ✓)
```

So the model is **65.9%** confident of the first class for that column.

**log-softmax** is just `log(softmax)`; your `forward()` returns it
(`crnn_model.py:93`) because CTC math is done in log-space (more numerically
stable). In `crnn_reader.py:76` the code undoes it with `.exp()` to get
probabilities back when it needs a human-readable confidence.

---

## 8. CTC decoding — the core idea

**This is the single most important concept in your recognition stage.** Learn
this one well.

**The problem CTC solves:** the CNN produces, say, 40 columns, but the plate has
only 7 characters. You never labeled *which* column is which character. How do you
line them up? CTC's answer: don't. Let the network emit a character or a **blank**
(`-`) at every column, then **collapse** the result with two rules:

1. **Merge repeated characters** that are next to each other.
2. **Remove all blanks.**

The blank's real job is to *separate* two genuinely-repeated characters (like the
`LL` in "HELLO") so they don't get merged.

**Where in your code:** `CTCDecoder._collapse()` in `crnn_model.py:118-124`:

```python
out, prev = [], -1
for idx in seq:
    if idx != blank and idx != prev:   # not blank, not a repeat
        out.append(charset[idx])
    prev = idx
```

**Worked example.** Let `-` be blank. Suppose the per-column argmax gives:

```
raw sequence:  3 3 - A A - - 1 1 1
```

Apply the rules step by step:

```
3 3   → merge repeats            → 3
-     → drop blank               → (nothing)
A A   → merge repeats            → A
- -   → drop blanks              → (nothing)
1 1 1 → merge repeats            → 1
```

Result: **`3A1`**.

Now see why blank matters. To output `"33"` (two real 3s), the network must emit a
blank *between* them: `3 - 3` collapses to `3 3` = `33`. Without the blank,
`3 3` would merge to a single `3`. That is the whole trick.

Your decoder is **greedy** (`decode` = argmax then collapse, lines 106-116): it
just takes the most likely character in each column. (A fancier "beam search"
decoder exists but you don't need it — greedy is fast and works well here.)

---

## 9. Your CRNN confidence formula (with length penalty)

**Where:** `crnn_reader.py:68-86`. After decoding, the code produces a single
confidence number used later by the gate. Two steps:

**Step 1 — average per-character confidence.** For every column that emitted a
real character (not blank), take that column's max softmax probability, then
average them:

```python
maxp, argmax = probs.max(dim=1)     # best prob per column
mask = argmax != blank              # keep only character columns
raw_conf = maxp[mask].mean()        # average them
```

**Worked example.** Say 4 character columns had max-probs `[0.99, 0.95, 0.80, 0.90]`:

```
raw_conf = (0.99 + 0.95 + 0.80 + 0.90) / 4 = 3.64 / 4 = 0.91
```

**Step 2 — length plausibility penalty** (lines 84-86). A real Cambodian plate
number is ~5+ characters. A crop that reads as 1–2 characters is probably junk, so
its confidence is scaled down:

```python
n = len(text without spaces/dashes)
length_factor = min(1.0, n / 5)      # min_plausible_len = 5
final_conf = raw_conf * length_factor
```

**Worked examples:**

```
Read "1AB-2345" → n = 7 → length_factor = min(1, 7/5) = 1.0
   final = 0.91 × 1.0 = 0.91   → above 0.70 gate → allowed if registered

Read "7"        → n = 1 → length_factor = min(1, 1/5) = 0.20
   final = 0.91 × 0.20 = 0.182 → below 0.70 → REVIEW_REQUIRED (gate stays shut)
```

That is exactly how a spurious 1-character misread on a blank wall gets
suppressed. This is a great, concrete example of **engineering around model
uncertainty** — worth understanding fully.

---

## 10. Stage 3 — the database & whitelist

**Where:** `src/utils/database.py` (class `PlateDatabase`), backed by **SQLite**
(a file-based SQL database — your `plates.db`).

Two jobs:
- **Whitelist lookup** — `is_registered(plate_text)` (`alpr_system.py:248`): is
  this plate authorized? An **exact string match**.
- **Audit log** — `log_read(...)` (`alpr_system.py:293`): every decision is
  recorded (plate, confidence, action, timestamp, evidence photo path).

**Concepts:** relational tables, rows/columns, SQL `SELECT`/`INSERT`, and why an
**audit trail** matters for a security system (you can prove what happened).

Schema is documented in `docs/database.md`.

---

## 11. Stage 4 — the gate decision & fail-safe logic

**Where:** `alpr_system.py:252-266`. This is small but is the **safety core** of
the project. In priority order:

```python
if self.estop_active:                       # emergency stop pressed
    action = "REVIEW_REQUIRED"              # → gate stays CLOSED
elif crnn_conf < self.crnn_conf_threshold:  # not confident enough (< 0.70)
    action = "REVIEW_REQUIRED"              # → gate stays CLOSED
elif is_reg:                                # confident AND on whitelist
    self.gate.open_gate(...)
    action = "ENTRY_ALLOWED"                # → gate OPENS
else:                                       # confident but not whitelisted
    action = "ENTRY_DENIED"                 # → gate stays CLOSED
```

**Concept — fail-safe / default-deny:** the gate opens **only** when *everything*
is right (confident read **and** exact whitelist match **and** no e-stop). Every
other path — low confidence, unknown plate, error, emergency — keeps the gate
**closed**. In security, the safe default is "no".

Notice the **three-way outcome**, not two: `ALLOWED`, `DENIED`, and
`REVIEW_REQUIRED`. The third is "I'm not sure — a human should look." This is a
key real-world ML pattern: **let the model abstain when unsure** rather than
forcing a risky yes/no.

---

## 12. The engineering glue

These aren't AI, but they make it a *system*. Each is worth a small study.

| Concept | Where | What to learn |
|--------|-------|----------------|
| **YAML config** | `configs/system_config.yaml` + `_load_config` | Externalizing settings so nothing is hard-coded |
| **Threaded camera reader** | `src/utils/rtsp_reader.py` | Threads, a bounded queue keeping only the latest frame, auto-reconnect |
| **MQTT / IoT gate** | `src/utils/mqtt_controller.py` | Publish/subscribe messaging; the **mock vs real** controller pattern (runs with no hardware) |
| **Structured logging** | `src/utils/logger.py` | Recording events to files at levels (INFO/WARNING/ERROR) |
| **Health monitoring** | `alpr_system.py:407-440` | FPS, latency, GPU memory, uptime %; alerting on SLA breaches |
| **Flask web admin** | `scripts/system/admin_web.py` | A tiny web server to manage the whitelist in a browser |
| **Evidence photos** | `alpr_system.py:354-369` | Saving a timestamped annotated image per read |

**Mock vs real pattern** (`create_gate_controller`, `alpr_system.py:109`): the
same interface (`open_gate`, `close_gate`, …) has a real MQTT implementation *and*
a fake one that just prints. That's why the whole system runs on your laptop with
no ESP32. Learn this pattern — it's everywhere in good software.

---

## 13. Performance: latency & FPS

Your README claims **~32 ms → ~30 FPS**. Here's the relationship, with a
calculation.

**Latency** = time to process one frame (milliseconds).
**FPS** (frames per second) = how many frames you handle per second.

```
FPS = 1000 / latency_ms
```

**Worked example** using your numbers:

```
YOLO detect : ~8 ms
CRNN read   : ~29 ms   (they overlap/pipeline; end-to-end ≈ 32 ms)
DB lookup   : <1 ms

FPS = 1000 / 32 ≈ 31 frames per second
```

**Where in code:** `alpr_system.py:518` literally computes
`fps = 1000.0 / mean(recent_latencies)`, and the health monitor alerts if average
latency exceeds `latency_alert_ms` (500 ms) — that's your **SLA** (service-level
agreement / performance budget).

**Why a moving average?** The code averages the **last 30 frames**
(`frame_times[-30:]`) so a single slow frame doesn't make the FPS number jump
around. Smoothing a noisy measurement is a common, useful trick.

---

## 14. A suggested study order + checklist

Work top to bottom. Tick a box when you can **explain it to a friend** *and*
**point to it in your code**.

**Foundations**
- [ ] An image is an `H×W×3` array of 0–255 numbers; OpenCV uses BGR
- [ ] Normalization: map `[0,255] → [-1,1]` (do the pixel=200 calc by hand)
- [ ] PyTorch: `nn.Module`, `__init__` vs `forward`, tensor shapes, `.eval()`, `no_grad`

**Detection**
- [ ] Detection = boxes + confidence (vs classification = one label)
- [ ] Confidence threshold and clamping
- [ ] IoU by hand (do the 0.333 example)
- [ ] Precision, recall, and what mAP50 = 0.9664 means

**Recognition (the heart)**
- [ ] CNN: Conv → ReLU → BatchNorm → MaxPool; conv output-size formula
- [ ] Why asymmetric pooling keeps width (the time axis)
- [ ] RNN → LSTM → BiLSTM and why bidirectional helps
- [ ] Softmax by hand (do the `[2.0,1.0,0.1]` example)
- [ ] **CTC collapse rules** (do the `3 3 - A A - - 1 1 1 → 3A1` example)
- [ ] Why the blank token lets you spell "33"
- [ ] Your confidence formula + length penalty (do both worked examples)

**System**
- [ ] The 4-stage `process_frame` pipeline end to end
- [ ] Fail-safe / default-deny gate logic and the 3rd outcome (REVIEW_REQUIRED)
- [ ] SQLite whitelist + audit log
- [ ] YAML config, threaded camera, MQTT mock-vs-real, logging, health
- [ ] Latency ↔ FPS (`FPS = 1000/ms`); moving average

---

### Where each concept lives (quick map)

```
src/core/alpr_system.py        pipeline, gate logic, health, evidence photos
src/detection/detector.py      YOLO wrapper, bbox, crop, draw, warm-up
src/recognition/crnn_model.py  CNN+BiLSTM+CTC network, CTC decoder, charset
src/recognition/crnn_reader.py preprocessing, inference, confidence + length penalty
src/recognition/province_classifier.py  plain image classification (start here — simplest)
src/utils/database.py          SQLite whitelist + audit log
src/utils/rtsp_reader.py       threaded camera, queue, reconnect
src/utils/mqtt_controller.py   IoT gate, mock-vs-real pattern
configs/system_config.yaml     all settings, no hard-coding
```

> Tip: the **easiest** deep-learning file to start reading is
> `province_classifier.py` (plain classification). The **most valuable** is
> `crnn_model.py` (CTC). Save it for when you're comfortable with the rest.
