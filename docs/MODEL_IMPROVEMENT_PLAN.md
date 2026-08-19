# Deep Learning Model Improvement Plan

> **Scope: the models only.** This project is a Deep Learning course project — the
> deliverable is model accuracy, not the gate/database/admin business logic. That
> wrapper is explicitly OUT of scope here. Every step below changes or measures a
> neural network, nothing else.
>
> Started: 2026-08-07. Companion to `docs/IMPROVEMENT_PLAN_V2.md` (which covered the
> pipeline/logic work). This file tracks the pure model work so the next step is
> always recoverable.

---

## The 4 models and their real state

| Model | Reported | Real state | Work needed |
|-------|----------|-----------|-------------|
| Plate detector (YOLOv10) | mAP50 **0.966** | genuinely strong | none |
| Number-line detector (YOLOv10) | mAP50 **0.943** | genuinely strong | none |
| **Province classifier** (ResNet18, 26 classes) | **97.18%** | ⚠️ inflated — flickers on live video; measured on clean crops only | **fix domain gap** |
| **CRNN number reader** (CNN+BiLSTM+CTC) | word-acc **72.48%**, CER 10.21% | the main accuracy bottleneck | **more data + retrain** |

**The core problem in one sentence:** both weak models were trained *and* tested on
clean images, but deployed on messy live frames (blur, angle, low-res). They memorised
clean data. This is **domain shift** — the real subject of this project.

---

## Steps

### Step 1 — Verify the CRNN fine-tune from the +206-label campaign  ⬅ NEXT
- **Type:** measurement only. No training, no file changes.
- **What:** evaluate `models/recognition/crnn_ft_phase5b.pth` (trained on 679 real
  labels, up from 473) against the deployed `crnn_finetuned.pth` on the **unchanged
  149-frame test set**.
- **Question it answers:** did +206 real labels make the number model better?
- **Deliverable:** word-accuracy + CER, candidate vs deployed, in the results table below.
- [ ] done

### Step 2 — Measure the province classifier's TRUE accuracy under live conditions
- **Type:** measurement only. No training, no file changes.
- **What:** take the province classifier's clean test crops, measure clean accuracy
  (expect ≈97%), then apply deployment-like degradation (motion blur, downscale,
  JPEG, slight rotation) and re-measure. The gap = the domain shift, as a number.
- **Question it answers:** how bad is the flicker really? What's the real live accuracy?
- **Deliverable:** clean-vs-degraded accuracy table below.
- [ ] done

### Step 3 — Retrain the weaker model with deployment-matched augmentation
- **Type:** training (~1 h on the RTX 3050). Writes a NEW candidate weights file.
- **What:** retrain whichever model Steps 1–2 show is weakest, adding augmentation that
  mirrors live conditions (blur, downscale, perspective, low light) so it stops
  overfitting to clean crops.
- **Rule:** the deployed model is only replaced if the new one is **better on the
  held-out test set**. Candidate saved under a new filename until then.
- [ ] done

---

## Ground rules (so this doesn't drift)

- **Models only.** No gate, database, dedup, admin, or config-logic work in this plan.
- **Measure before changing.** Steps 1–2 are pure measurement and must run before any
  retrain.
- **Never train on the 149-frame human-verified test split.**
- **A model is promoted only on a test-set improvement**, never on training/val loss.
- Read-only baselines stay untouched: `models/detection/best.pt`,
  `models/recognition/crnn_best.pth`.

---

## Results (filled in as steps complete)

### Step 1 — CRNN candidate vs deployed (n=149 test) ✅ 2026-08-07
| Model | word accuracy | CER |
|-------|--------------|-----|
| deployed `crnn_finetuned.pth` | **80.54%** | 5.11% |
| candidate `crnn_ft_phase5b.pth` (679 labels) | **81.88%** | 4.70% |

**Findings:**
- Both models are **far better than the 72.48% / 10.21% in the old docs** — the number
  reader is in good shape on clean crops (~81% word-acc, ~5% CER).
- The 679-label candidate is marginally better (+1.3 pp word-acc), but **within noise on
  n=149** — consistent with earlier findings that this test set is too small to resolve
  a gain this size.
- **Provenance caveat:** the deployed weights are dated **2026-08-05**, *newer* than both
  candidates (Jul 23/24). So "deployed" was retrained since my campaign — this is not a
  clean +206-label A/B, and the old 72.48% baseline is stale.

### Step 2 — Province classifier, clean vs degraded ✅ 2026-08-07  (567 test crops)
| Condition | accuracy | drop |
|-----------|----------|------|
| clean | **97.2%** | — |
| motion blur | 92.8% | −4.4 |
| downscale 6× | 94.5% | −2.6 |
| gaussian blur | 97.0% | −0.2 |
| JPEG q20 | 97.4% | 0 |
| rotate 8° | 96.3% | −0.9 |
| **LIVE combo** (downscale + blur + JPEG) | **91.7%** | **−5.5** |

**The important finding — the province classifier is NOT the flicker's cause.** Even
under heavy combined degradation it holds **91.7%**. Image quality (blur/noise/low-res)
does *not* break it, so it cannot explain the wild frame-to-frame flicker between 4
different provinces seen on live video.

**So what does cause the flicker?** Not classifier fragility. The remaining suspects,
which this synthetic test could NOT capture (it degraded *clean* crops, holding the crop
framing fixed):
1. **Inconsistent detector crops** — on live video the plate detector's province-line
   box wobbles frame-to-frame, feeding the classifier different content each frame.
2. **Live conditions not simulated** — extreme angle, glare, partial occlusion, the box
   catching the number line or background.

Both require **real live province crops** to confirm — they cannot be reproduced from
the clean test set.

**Consequence for Step 3:** do NOT retrain the province classifier for robustness it
already has. Measuring first prevented fixing the wrong model.

### Step 2b — Detector-crop-consistency test ✅ 2026-08-07  (142 plates, ~45 jitters each)
Tested the theory directly: keep the plate identical, **jitter the detector's box**
(shift ±10%, scale ±8% — the amount a detector wobbles frame-to-frame on live video)
and see if the province prediction changes.

| Metric | Value |
|--------|-------|
| province accuracy on the **base** (well-framed) box | **98.6%** |
| province FLIP rate when only the BOX moved | **11.5%** |
| accuracy on jittered crops | 87.8% (−10.8) |
| **plates that give >1 province under jitter** | **100 / 142 = 70%** |
| **mean distinct provinces per plate** | **2.87** (max 10) |

**CONFIRMED — this is the flicker.** With a perfectly-framed box the classifier is
98.6% right, but **move the box a little and 70% of plates flip to 2–3 different
provinces** — exactly the `Mondul Kiri / Oddar Meanchey / Kampong Cham / Koh Kong`
flicker seen live. On live video the detector's box naturally wobbles every frame, so
each frame is a slightly different crop → a different province.

**Root cause (a real DL finding):** the province classifier was trained on **clean,
tightly-framed crops** (`build_province_dataset.py`), so it never learned to be
*invariant to framing*. It's accurate but fragile: it keys on exact crop position.
This is a **crop-framing domain gap**, distinct from the image-quality one (which
Step 2 ruled out).

**We did NOT need live capture to prove this** — the jitter test is a controlled proxy
and it reproduces the flicker from clean data. Live crops would only add confirmation.

---

## Step 3 (revised) — Retrain the province classifier to be FRAMING-INVARIANT ✅ DONE 2026-08-07 — BIG WIN
- **Type:** training (~20–40 min; ResNet18, small dataset). New candidate weights file.
- **What:** retrain with aggressive **framing augmentation** — `RandomResizedCrop`,
  random translate/scale/pad — so the model classifies the province correctly
  regardless of exactly where the detector's box lands.
- **Validation metric (already built):** re-run the **Step 2b jitter test**. Success =
  the "70% of plates flip" and "2.87 distinct provinces/plate" numbers drop sharply,
  while base accuracy stays ~97%.
- **Promote only if** base test accuracy holds AND jitter-stability improves.
- [x] **done — promoted to deployed (hashes verified equal).**

**RESULT — framing augmentation crushed the flicker while accuracy held:**

| metric | BEFORE | AFTER | change |
|--------|--------|-------|--------|
| base (well-framed) accuracy | 98.6% | 98.6% | held |
| clean test accuracy | 95.24% | **95.24%** | held |
| flip rate under box jitter | 11.5% | **0.5%** | **23× better** |
| **plates flickering (>1 province)** | **70%** (100/142) | **6%** (9/142) | **−64 pts** |
| mean distinct provinces / plate | 2.87 (max 10) | **1.08** (max 3) | near-eliminated |

30 epochs, ImageNet-pretrained, best val 96.79%. This is the model-level cure for the
live province flicker — consecutive frames of one plate now give a stable province,
with no loss on well-framed crops. Validates the whole discipline: measure → find the
real cause (framing, not blur, not raw accuracy) → fix the model → re-measure.

**Tooling built along the way (2026-08-07):**
- `train_province_classifier.py` gained framing augmentation (`RandomResizedCrop` +
  `RandomAffine`, default ON) and an `--out` flag so the candidate never overwrites
  the deployed model.
- `scripts/tools/test_province_stability.py` is the before/after metric (the Step 2b
  jitter test, reusable on any weights).

**BASELINE — current deployed model (the "before" to beat):**
| metric | deployed |
|--------|----------|
| base (well-framed) accuracy | 98.6% |
| flip rate under box jitter | 11.5% |
| **plates flickering (>1 province)** | **70%** |
| mean distinct provinces / plate | 2.87 |

**Commands to run (training ~15–30 min on the RTX 3050):**
```
# 1) train the framing-invariant candidate (deployed model stays safe)
python scripts/recognition/train_province_classifier.py \
    --out models/recognition/province_classifier_framing.pth --epochs 30 --pretrained

# 2) measure the candidate's stability (compare to the baseline table above)
python scripts/tools/test_province_stability.py \
    --weights models/recognition/province_classifier_framing.pth \
    --config  models/recognition/province_classifier_framing_config.json

# 3) if flicker dropped AND base accuracy held (~97%+), promote it:
cp models/recognition/province_classifier_framing.pth models/recognition/province_classifier_best.pth
cp models/recognition/province_classifier_framing_config.json models/recognition/province_classifier_config.json
```

---

## Step 4 — Make the plate DETECTOR rotation-invariant (detect at any angle)
User goal: "I want my model to detect the plate at every angle." Same class of fix as
Step 3 — the model only knew what it was trained on (here, upright plates only).

- **Type:** training (YOLOv10, 2054 train images). New candidate weights file.
- **What:** retrain with **rotation augmentation** (`degrees=180`, `flipud=0.5`) so the
  image and its boxes are randomly rotated to any angle + flipped during training.
  `train.py` gained `--degrees`, `--flipud`, `--name`, `--out` (default 0.0 = old
  behaviour; `--out` keeps `best.pt` intact).
- **Validation metric (built):** `scripts/tools/test_detector_rotation.py`.

**BASELINE — current `best.pt` (the "before"), 60 test frames:**
| angle | 0° | ±10° | ±20° | ±30° | ±45° | **±90°** | 180° |
|-------|----|----|----|----|----|----|----|
| detect% | 98 | 97–98 | 90–95 | 37–52 | 7–12 | **2** | 37 |

Worst-angle rate **2%**. Goal: keep 0° high while lifting the worst angle far above 2%.

**Commands (fine-tune from best.pt; ~30–60 min):**
```
# 1) train the rotation-robust candidate (best.pt stays safe)
python scripts/detection/train.py --model models/detection/best.pt \
    --degrees 180 --flipud 0.5 --epochs 60 \
    --name plate_detector_rot --out models/detection/best_rot.pt

# 2) measure rotation robustness (compare to the baseline above)
python scripts/tools/test_detector_rotation.py --weights models/detection/best_rot.pt

# 3) if the worst-angle rate is much higher AND 0deg stayed high, promote it:
cp models/detection/best_rot.pt models/detection/best.pt
```
- [x] **done — trained, then ROLLED BACK. Key lesson below.**

**OUTCOME (2026-08-07): the rotation detector works but breaks the pipeline — rolled back.**

Training crashed at epoch 26 (flaky Windows cuDNN error) but the 25-epoch checkpoint
was already rotation-robust: worst-angle detection **2% → 82%**. It was promoted, then
tested end-to-end, and the result was clear:

| on the 149 UPRIGHT test frames | original detector | rotation detector |
|--------------------------------|-------------------|-------------------|
| detection rate | **95.97%** | 85.9% (−10 pp) |
| composed exact-match | **77.55%** | 66.7% |

And feeding the number CRNN a **rotated crop** returns garbage:
`3E-6306` upright → `3E-6306` ✓; rotated 90° → `2--25` ✗; rotated 180° → `9069-36` ✗.

**The lesson — you can't make ONE model in a pipeline robust in isolation.** We taught
the *detector* to find plates at any angle, but the *readers* (number CRNN, province
classifier) were only ever trained on UPRIGHT crops. So a rotated plate is detected,
then read as garbage. The rotation detector also *hurt* the common upright case (−10 pp
detection). Net: it buys nothing end-to-end and costs accuracy → **rolled back to the
original detector** (`runs/detect/plate_detector_week2/weights/best.pt`), restoring
95.97% / 77.55%.

**What TRUE any-angle reading would require (a much bigger project, not worth it for an
upright gate):** rotation-robust detector (done) → estimate the plate's angle (or an
oriented-bounding-box detector) → **de-rotate the crop to upright** → then the existing
readers work. Simply augmenting the readers with rotated crops is worse, because reading
sideways text with CTC is fundamentally hard; de-rotation is the right approach. For a
real gate (plates upright ±20°, already handled), none of this is needed.

---

## Step 5 — TRUE end-to-end any-angle reading (detect AND read rotated plates correctly)
User goal: "detect the rotated plate with the correct result and high accuracy on real
test." This is the whole-pipeline version Step 4's lesson pointed to.

**Approach — orientation search at inference (NO new training):** use the fact that the
readers are excellent on UPRIGHT crops and produce garbage otherwise, and that we have a
plate-format grammar to tell good reads from garbage. So try the frame at several
orientations and let the readers pick the right one:
```
for angle in [0, 90, 180, 270]:
    rframe = rotate(frame, angle)
    for det in detector.detect(rframe):          # original upright detector, 95.97%
        number, conf = crnn.read(det.crop)
        score = conf * (1.0 if plate_format.is_valid(number) else penalty)
        keep the (number, angle, det) with the best score
return the best-scoring read   # = the orientation where the plate read cleanly
```
Whichever rotation makes the plate upright is the one where the CRNN reads with high
confidence AND a valid format — so that read wins. The readers' native ±20° tolerance
covers the angles between the four cardinal steps, giving near-full-circle coverage.

- **Type:** inference logic. **No training.** Uses the current best models.
- **Cost:** ~4× detection/read per frame (accuracy over speed — the user's choice).
- **Validation (to build):** `test_orientation_search.py` — rotate the 149 test frames
  by 0/90/180/270 and measure NUMBER accuracy, naive vs orientation-search. Success =
  high accuracy at ALL four orientations (naive collapses at 90/180/270).
- **Discipline:** prove it on the test set FIRST; only then wire it into the pipeline
  behind a config flag.
- [x] **proven — works. Integrating.**

**PROOF (2026-08-07, 149 test frames rotated to each orientation, `number_best.pt`):**

| camera rotation | NAIVE (today's pipeline) | ORIENTATION SEARCH |
|-----------------|--------------------------|--------------------|
| 0° | 77.9% | 76.5% |
| 90° | **0.0%** | **76.5%** |
| 180° | **0.0%** | **76.5%** |
| 270° | **0.0%** | **76.5%** |
| **mean** | **19.5%** | **76.5%** |

The pipeline goes from **blind at every rotation** to **reading at full upright accuracy
at all of them**, no training. Cost: 0° dips 77.9→76.5 (occasionally a wrong orientation
yields a valid-format garbage read) and ~4× detection/read per frame. Tool:
`scripts/tools/test_orientation_search.py`. Integrated into `ALPRSystem` behind
`gate.orientation_search` (default off, since a normal gate is upright).

**Honest scope note:** a real gate sees plates roughly upright (±20°), which the current
model already handles — full any-angle robustness is a project/demo goal. It may cost a
little peak precision on upright plates (rotated boxes are looser at train time); the
rotation meter's 0° row will show if that regressed. If the number-line detector
(`number_best.pt`) also needs any-angle robustness for the full pipeline, repeat this
with its own dataset.
