> **SUPERSEDED (2026-09-12).** Written 2026-08-19, before `check_stn.py` showed (2026-08-21) that the STN predicts a near-identity transform and contributes nothing; the upside-down ability comes from the `--rotate180` augmentation. Current truth: `docs/PRESENTATION_GUIDE.md` §3 and `metrics/experiment_log.csv`.

# HANDOFF — Upside-down / rotated plate reading (for the next AI window)

> **Read this first.** The previous chat ran out of tokens. This file is a complete,
> self-contained brief so a fresh AI (no memory of the prior chat) can continue the
> work. It covers the whole journey, the current state, the exact files/commands, and
> the two remaining tasks (#1 and #2). Companion docs:
> `docs/MODEL_IMPROVEMENT_PLAN.md` (province/detector fixes) and
> `docs/IMPROVEMENT_PLAN_V2.md` (pipeline work).
>
> Written: 2026-08-19.

---

## The goal (what the user wants)
Make the deep-learning **models** read a plate even when it is **upside-down (180°)** —
by *training the models*, NOT by a hardcoded frame-rotation wrapper. This is a Deep
Learning course project; the value is in the model + honest measurement, not business
logic. **The user explicitly rejected the inference wrapper (`orientation_search`)** —
do not push it; keep the focus on the models.

## Hard-won lessons (do NOT repeat these dead ends)
1. **Naive flip augmentation on the CRNN fails.** Just flipping crops with `--rotate180`
   dropped upright reading 80.5% → 62% while upside-down only reached 12%. The CRNN is a
   horizontal CTC reader; upside-down text is flipped + reversed, which it can't learn
   well and it wrecks the normal case. **Refuted, measured.**
2. **Rotation-training the DETECTOR hurts upright.** `best_rot.pt` (trained `--degrees 180`)
   detected rotated plates but dropped upright detection 95.97% → 85.9% AND the readers
   still couldn't read the rotated crops → garbage. Rolled back. **Refuted, measured.**
3. **The `orientation_search` wrapper WORKS technically** (76% at all angles, upright
   preserved) but the user does not want it. It is OFF (`gate.orientation_search: false`)
   and should stay off. It exists in `alpr_system.py::_orient_upright` if ever needed.

## What WORKED — the current approach (Way 1 + Way 2)
**Way 1 — Spatial Transformer Network (STN):** a small learnable "straightening layer"
INSIDE the CRNN (`src/recognition/crnn_model.py::STN`, integrated in `CRNN(use_stn=True)`).
It learns to rotate a crop upright by itself from the CTC loss — initialised to identity
so it starts as a no-op and never hurts upright. `load_crnn` **auto-detects** the STN from
the weights (any key starting `stn.`), so the reader/measurement need no flags.

**Way 2 — lots of realistic synthetic data:** generate thousands of synthetic plates;
`--rotate180 0.5` flips ~half at train time, giving the STN far more "flipped → correct
text" examples than the 578 real crops alone.

### Measured results (held-out 149 real test crops, `test_rotation_reading.py`)
| model | upright | upside-down |
|-------|---------|-------------|
| deployed `crnn_finetuned.pth` | 80.5% | 0% |
| naive flip (dead end) | 62% ❌ | 12% |
| STN, little data (`crnn_stn.pth`) | 81% ✅ | 10% |
| **STN + Way 2 synthetic (`crnn_stn2.pth`)** | **79.2%** ✅ | **55%** ✅ |

**`crnn_stn2.pth` is the current best. It reads upside-down 55% with upright intact.**
It is trained and saved (user has it on Google Drive `ALPR/trained/`). It is **NOT yet
promoted** — the deployed model is still `crnn_finetuned.pth`.

---

## THE TWO REMAINING TASKS

### ✅ #1 — Better-matched synthetic data  (CODE DONE 2026-08-19, needs a train run)
**Goal:** close the synthetic→real gap so the STN transfers better to real plates.
Target: upside-down 55% → ~65%, upright stays ~79%.

**What was already implemented (this is done, just needs to be trained):**
`scripts/recognition/generate_synthetic.py` was upgraded (`render_plate` +
`_realistic_augment`): dark-blue plate ink, blue border + underline like real
Cambodian plates, and photo-realistic degradation — mild **perspective** (cv2),
uneven **lighting gradient**, blur, sensor **noise**, and **JPEG** artefacts. Verified
it renders. Colab-safe fonts already added (`/usr/share/fonts/.../DejaVuSans-Bold.ttf`).

**How to run #1 (Colab — the T4 is ~4× the user's RTX 3050 laptop):**
1. Rebuild the bundle so Colab gets the new generator code:
   `python scripts/tools/make_colab_bundle.py`  → `alpr_colab_bundle.zip` (~417 MB)
2. User uploads it to Google Drive folder `ALPR` (REPLACING the old zip — critical).
3. Run `notebooks/colab_train.ipynb` (already set up: generate synthetic → train STN → measure).
   The key training command inside it:
   ```
   python scripts/recognition/finetune_crnn.py --stn --rotate180 0.5 --synth-n 16000 \
       --out models/recognition/crnn_stn3.pth --epochs 80
   ```
   (Use a NEW name like `crnn_stn3.pth` so `crnn_stn2.pth` stays as the comparison.)
4. Measure: `python scripts/tools/test_rotation_reading.py --weights models/recognition/crnn_stn3.pth`
5. **Decision rule the user gave:** if #1 lifts upside-down above 55% → keep it. Then do #2.
   If #1 does not help → still keep the better of the two, then do #2. Either way #2 follows.

**Possible further #1 improvement if still short:** the biggest remaining synthetic gap is
the FONT — real Cambodian plate digits have a specific style that DejaVu/Liberation don't
match. If a real Cambodian-plate TTF font can be obtained, add its path to
`FONT_CANDIDATES` in the generator — likely the single largest remaining lever for #1.

### ⬜ #2 — More REAL upside-down examples  (NOT started — the strongest lever)
**Goal:** give the STN real flipped signal, not just synthetic. Target: toward 70%+.
Currently only 578 real crops exist (in `data/crnn_crops/train/`, labelled in
`data/crnn_crops/real_labels.csv`), flipped at train time by `--rotate180`.

**How to do #2:**
- The training ALREADY flips real crops (via `--rotate180`), so #2 is about **more real
  crops**, not more flipping. Two sub-options:
  - **(a) Label more real plates** using the existing active-learning tools:
    `scripts/recognition/harvest_active.py` (see `docs/IMPROVEMENT_PLAN_V2.md` Phase 5 —
    the harvest/label/merge workflow is built). There are ~350 more labelable crops in
    the unlabelled pool. Label them → they get flipped in training too.
  - **(b) Ask the user to photograph more plates** (real data collection) — strongest but
    manual. Guide in `docs/DATA_COLLECTION_GUIDE.md`.
- After adding real labels, re-run the same STN + synthetic Colab training and measure.
- **Guards (mandatory):** never train on the 149-frame test split; `harvest_active.py`
  already hard-excludes `data/crnn_crops/test/`. Verify with `--check-mix`.

---

## KEY FILES (all paths from project root)
| file | role |
|------|------|
| `src/recognition/crnn_model.py` | `STN` class + `CRNN(use_stn=...)` + `load_crnn` auto-detect |
| `scripts/recognition/finetune_crnn.py` | CRNN training. Flags: `--stn`, `--rotate180 P`, `--synth-n N`, `--out PATH` |
| `scripts/recognition/generate_synthetic.py` | synthetic plates (upgraded for #1) |
| `scripts/tools/test_rotation_reading.py` | THE metric: upright vs upside-down on 149 real test crops |
| `scripts/tools/make_colab_bundle.py` | builds `alpr_colab_bundle.zip` for Colab |
| `notebooks/colab_train.ipynb` | Colab notebook: generate synth → train STN → measure → save to Drive |
| `models/recognition/crnn_finetuned.pth` | DEPLOYED reader (upright-only, the baseline) |
| `models/recognition/crnn_stn2.pth` | current best rotation reader (79% up / 55% down) — on user's Drive |

## HOW TO PROMOTE a better reader (when one beats crnn_stn2.pth and upright holds ~79%)
The live reader auto-loads the STN, so promotion is just pointing the config at the file:
```yaml
# configs/system_config.yaml
crnn_weights: "models/recognition/crnn_stn3.pth"   # or whichever won
```
Then restart the dashboard. Do NOT overwrite `crnn_finetuned.pth`; keep it as a fallback.
**Always measure on the 149 test set before promoting — never trust training loss.**

## HONEST CEILING (tell the user this, don't over-promise)
Upside-down likely tops out around **65–75%** — 180° text is inherently ambiguous and
won't fully match upright (79%). 55% is already a defensible result. Frame #1/#2 as
"pushing a good result higher," not "solving it."

## ENVIRONMENT NOTES
- Windows laptop (RTX 3050, 4 GB). Project is Windows-only by design (DEV-001).
- Colab (Linux, T4) is for **training only**; bring weights back to Windows.
- venv python: `.venv/Scripts/python.exe`. On Colab it's plain `python`.
- The user prefers **simple, plain-language** explanations and **honest measured** results
  (they will call out any unmeasured assumption). Always measure before claiming.

## IMMEDIATE NEXT ACTION for the new AI
#1's code is done. The next concrete step is to **rebuild the bundle** (`make_colab_bundle.py`)
and have the user re-upload + run `notebooks/colab_train.ipynb` to train + measure #1
(`crnn_stn3.pth`). Then follow the decision rule above and proceed to #2.
