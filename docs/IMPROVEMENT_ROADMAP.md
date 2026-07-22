# Cambodian ALPR — Improvement Roadmap

> Companion to `PROJECT_OVERVIEW.md`, `docs/HANDOFF.md`, and `docs/SRS_ALIGNMENT_PLAN.md`.
> Written for whoever picks this project up next — human or AI — to know what to
> do, in what order, and why.
>
> Last updated: 2026-07-17.
>
> **Successor:** Phase 1–3 below are done. Work that comes after them lives in
> `docs/IMPROVEMENT_PLAN_V2.md`, which is organised as sequential phases with a
> measurement gate between each one. Start there.

---

## How this is prioritized

Two axes: **impact on the metric that matters** (real-world "gate opens for a
correctly registered car" rate) and **effort to ship**. The evidence for
ranking comes straight from the project's own numbers — see the "why" note
under each item.

**Constraints that apply to every item below** (carried over from the project
overview, repeated here so this doc is self-contained):
- Must run on a 4 GB VRAM laptop GPU — stay edge-deployable.
- Fail-safe default-deny gate behavior is a hard safety requirement — never weaken it.
- `models/detection/best.pt` and `models/recognition/crnn_best.pth` stay read-only baselines.
- Windows-only by design (DEV-001) — don't reintroduce a Linux dependency.
- Never train on the human-verified test split.

### Definition of done (the finish line)
Current SRS targets are **number-only** and already met (CER ≤ 15%, word-acc ≥ 70%
→ 10.21% / 72.48%). This roadmap adds a **composed-plate** goal, because that is
what actually decides the gate:

> **Target: composed-plate exact-match accuracy ≥ 85% on the human-labeled real
> test set, with a false-accept rate of 0** (no unregistered plate ever opens the
> gate). Set the real baseline with item 1.1 before tuning anything.

---

## Phase 1 — High impact, low effort (do these first)

> **Ordering note:** 1.1 (the benchmark) comes before 1.2 (constrained decoding)
> on purpose. The benchmark is the *instrument* that proves whether constrained
> decoding helps — and, critically, whether it introduces false accepts. Build
> the measurement first, then change the thing it measures.

### 1.1 Composed-plate (province + number) end-to-end benchmark  ← do this first  ✅ DONE (2026-07-17)
**Status:** Implemented as `scripts/system/benchmark_composed.py`; runs the real
integrated pipeline on the 149 human-labeled test frames in an isolated temp
DB/photo dir (real `plates.db`/`photos/` untouched). Results in
`metrics/composed_benchmark.json`.

**Baseline measured (2026-07-17):**
| Metric | Value |
|--------|-------|
| Detection rate | 95.97% (143/149) |
| **Number end-to-end accuracy** | **67.79% (101/149)** — same 101 correct as the recorded 70.6%, but over all 149 frames (stricter denominator incl. 6 non-detections) |
| Number CER (end-to-end) | 15.15% |
| Number failures | 42 (length-wrong 23, not-detected 6) |
| **False-accept rate** | **0.00%** (0 opens on 137 wrong-neighbour plates — fail-safe holds) |
| Composed exact-match | 65.87% **ESTIMATE only** |

**Finding / data gap:** true composed-plate exact-match is **not measurable yet**
— there is **no province ground truth aligned to the 149 test frames** (province
crops use different filenames; 0 join). Composed accuracy is currently the
independence estimate `number_acc × 0.9718`. To measure it for real, supply
province labels for the test frames and run with
`--province-gt data/crnn_crops/province_test_labels.csv` (columns:
`image,province_class`). **This is a new prerequisite for the Definition of Done.**

The **0% false-accept baseline is the bar item 1.2 must not regress.**

**Problem:** All current accuracy numbers (CER, word-acc, 70.6% end-to-end)
are number-only. There's no metric for "did the system get the *whole* plate
right," which is what actually determines a correct gate decision, since
`compose_plate()` needs both province and number correct.

**Proposed change:** Extend `scripts/system/srs_acceptance_test.py` (or a new
script in `scripts/tools/`) to run the full pipeline on the labeled real test
set and report:
- number-only accuracy (existing),
- province-only accuracy,
- **composed-plate exact-match accuracy** (the new headline metric),
- **false-accept rate** — how often an *unregistered* plate produces `ENTRY_ALLOWED`
  (must be 0); test this explicitly with unregistered plates that are near-neighbors
  of registered ones,
- a confusion breakdown of *where* composed failures come from (province wrong vs.
  number wrong vs. both).

**Effort:** Low (existing test set, existing components — just new
aggregation/reporting code). No training, no new data.
**Expected impact:** Turns section 8's "known blind spot" into a real number,
sets the baseline for the Definition of Done, and is the safety net for 1.2.

---

### 1.2 Closed-set / whitelist-constrained decoding  ✅ DONE (2026-07-17)
**Status:** Implemented as a **safe, non-opening** review path. Exact matching
(the only thing that may open the gate) is untouched; a new
`PlateDatabase.nearest_registered()` finds the closest ACTIVE plate within
`match_max_distance` edits, and `ALPRSystem.process_frame` routes a *confident,
non-exact, near* read to **REVIEW_REQUIRED** with a `suggested_plate`
("did-you-mean") — it never auto-opens. Config flags:
`gate.constrained_matching` (default true), `gate.match_max_distance` (default 1).
Files: `src/utils/database.py`, `src/core/alpr_system.py`,
`configs/system_config.yaml`. Suspended/expired plates are excluded from
suggestions (verified).

**Measured with the 1.1 benchmark (149 real frames):**
| Result | Value |
|--------|-------|
| False-accept harness | **0 opens** (unchanged — 1.2 adds zero false-accept risk) |
| Confident auto-open (exact) | 100/143 (69.93%) |
| **1.2 recovered DENY → REVIEW** | **17** legit plates (1-char misreads) now flagged for a human instead of silently denied — the concrete win |
| Low-confidence → REVIEW (REC-005) | 12 |
| Still hard DENY | 13 |

**New finding (pre-existing, NOT caused by 1.2):** the benchmark's intruder probe
found **1** confident read that exact-matches a *different* registered plate (a
cross-plate misread) when all 143 plates are registered simultaneously. This is a
worst-case property of **exact matching + a large whitelist** — 1.2 neither causes
nor fixes it. Mitigations belong elsewhere: item **2.2 province ↔ number
cross-validation** would catch most (the province would differ), and a realistic
small whitelist makes it far less likely. Logged here as a candidate for 2.2.

**Problem:** The CRNN has ~10% CER, but the gate decision does an *exact*
string match against the whitelist. One misread character out of ~7 turns a
legitimate car into `ENTRY_DENIED`. This is very likely the single biggest gap
between the CRNN's real word-accuracy (72.48%) and what a user actually
experiences at the gate.

**Proposed change:** At decode time, instead of (or alongside) free greedy CTC
decoding, score the CRNN's per-timestep output against each string in the
whitelist (CTC forward-score, or greedy-decode + edit distance against the
whitelist set) and accept the best-scoring match if it clears a threshold;
otherwise fall through to `REVIEW_REQUIRED`. Since the whitelist is small and
known, this is cheap to compute per frame.

**⚠️ Safety risk — false accepts (read this before implementing):**
Snapping the CRNN output to the *nearest* whitelist string can convert a
should-be-DENIED plate into an ALLOWED one. Example: an unregistered car reads
as `3E-6690`; the whitelist has `3E-6694` (edit distance 1); naive constrained
decoding "corrects" it and opens the gate for the **wrong car**. In a default-deny
gate a false accept is far worse than a false reject. Mitigations, all required:
- The threshold tuning *is* the whole risk — measure the **false-accept rate**
  (item 1.1) on near-neighbor unregistered plates, not just the acceptance rate.
- Prefer routing a *corrected* read (one that didn't already exact-match) to
  **`REVIEW_REQUIRED`** rather than `ENTRY_ALLOWED` — you gain recall without
  silently opening on a guess.
- Fail-safe still holds: no whitelist match above threshold → gate stays closed.

**Where it lives:** `src/recognition/crnn_reader.py` (decoding step) and
`src/core/alpr_system.py` (`_match_number` / decision stage) — **no model
retraining needed.**

**Effort:** Low (decoding-logic change, no new data/training).
**Expected impact:** Potentially the highest ROI item in this whole roadmap —
touches the metric that matters directly, using data you already have, *provided
the false-accept rate stays 0*.

---

### 1.3 Active-learning loop from evidence photos
**Problem:** The real-labeled training set (473 crops) grew through manual,
somewhat ad hoc labeling. Meanwhile, every low-confidence read and every
`REVIEW_REQUIRED` event is already being logged with an evidence photo — a
free source of exactly the hard cases the model needs.

**Proposed change:** A small script that pulls all `REVIEW_REQUIRED` and
low-`crnn_conf` entries from `plates.db` + `photos/` over a time window, surfaces
them for quick human labeling (reuse the existing montage-transcription
workflow from `scripts/recognition/crop_numbers.py`), and appends only the
confirmed-correct labels to the real training set before the next fine-tune.

**⚠️ Data-hygiene guards (required or your metrics become dishonest):**
- **No test-set leakage.** You are mining `photos/` and `plates.db` for hard
  cases; if any overlap with the 149 human-labeled test crops, they leak into
  training and CER becomes meaningless. Hard-exclude anything on a `/test/` path.
- **No selection bias.** Fine-tuning *only* on low-confidence/REVIEW cases skews
  the model toward hard examples and can degrade easy-case accuracy. Mix the
  mined hard cases **back into** the existing labeled set rather than training on
  them alone.

**Effort:** Low–medium (mostly gluing existing pieces together).
**Requires training:** yes — a CRNN fine-tune after new labels are added.
**Expected impact:** Cheaper and more targeted than blind data collection —
you're mining the exact distribution of images the model currently fails on.

---

## Phase 2 — High impact, medium effort

### 2.1 Targeted data collection for hard conditions
**Why now:** The CER curve (94.89% → 25.93% → 20.32% → 10.21% as real labels
grew 0 → 143 → 324 → 473) hasn't plateaued — more real data is still the
lever with the clearest track record in this project's own history. But
undirected collection is inefficient; conditions like night, motion blur,
steep angles, and dirty/worn plates are almost certainly under-represented in
the current 473 crops.

**Proposed change:** Deliberately capture and label a batch specifically
covering those conditions (a checklist of 4–6 target conditions, ~50–100
crops each), rather than adding more "easy" daylight captures.

**Effort:** Medium (capture time + labeling time, no new code).
**Requires training:** yes — a CRNN fine-tune on the expanded set.
**Expected impact:** Directly extends the proven data→CER trend into the
conditions most likely to occur at a real gate.

### 2.2 Province ↔ number cross-validation  ✅ DONE (2026-07-17)
**Status:** Implemented as a **non-opening** consistency flag. Two signals, both
config-gated: (a) **weak pairing** — the matched number box's scale-invariant
overlap with the province box (`align = overlap / smaller-box`) is below
`gate.number_alignment_min`, i.e. likely mis-paired in a multi-plate frame; and
(b) **uncertain province** — a Khmer prefix is being composed from a classifier
below `gate.province_confidence_min`. A confident, non-whitelisted, non-near read
that trips either signal goes to **REVIEW_REQUIRED** instead of a silent
`ENTRY_DENIED`. It **never downgrades a confirmed `ENTRY_ALLOWED`** (exact match
already confirms the province), so it adds no delay for registered cars and no
false-accept risk. Files: `src/core/alpr_system.py` (`_match_number` now returns
pairing quality), `configs/system_config.yaml`.

**Tuning (done against the 1.1 benchmark):** the first metric (`overlap/number-width`)
over-triggered badly — 44 flags at 27% precision — because the number line is
normally *wider* than the Khmer province line. Switching to the scale-invariant
`align` metric and threshold **0.20** nearly doubled precision:
| Metric | Value |
|--------|-------|
| Reads flagged → REVIEW | 22/143 |
| Flag on a WRONG number (catch) | 10 |
| Flag on a CORRECT number (noise) | 12 |
| Flag precision | 45.45% |
| Dominant reason | weak-number-alignment (21), uncertain-province (1) |

**Addresses the 1.2 cross-plate finding:** a number misread as another plate's
number will usually pair with a *different* province, which 2.2's signals (and the
composed-text whitelist) help surface. **Honest limitation:** precision is moderate
and the signal targets province/pairing, so full validation needs province ground
truth for the test frames (same gap flagged in 1.1). The noise is low-harm — 2.2
only converts DENYs of *unregistered* reads into REVIEWs; registered cars are
unaffected. Set `gate.consistency_check: false` to disable.

### 2.2 Province ↔ number cross-validation — original notes
**Why:** The two branches (province classifier, number CRNN) currently run
independently with no consistency check. A plate where the province is
unreadable but the number reads confidently — or vice versa — currently
produces a possibly-wrong composed plate with no internal red flag.

**Proposed change:** Add a soft consistency signal: e.g., flag composed reads
where province-classifier confidence and CRNN confidence diverge sharply, or
where the province and number boxes fail the expected geometric relationship
(`_match_number`'s "is-below" check) by a wide margin. Route flagged reads to
`REVIEW_REQUIRED` rather than a silent pass-through.

**Effort:** Medium (new logic in the decision stage; needs some tuning against
the 1.1 benchmark to avoid over-triggering). **No training needed.**
**Expected impact:** Reduces confidently-wrong composed plates, which the new
1.1 benchmark will make visible for the first time.

---

## Phase 3 — Medium impact, worth doing eventually

### 3.1 Model export & quantization (ONNX / TensorRT)  ✅ DONE — ONNX (2026-07-17)
**Status:** All four models export to ONNX via `scripts/tools/export_onnx.py`, each
verified against its PyTorch original:
| Model | ONNX | Parity (max|torch−onnx|) |
|-------|------|--------------------------|
| YOLO plate detector | `models/onnx/best.onnx` | ORT runs OK (Ultralytics layout) |
| YOLO number detector | `models/onnx/number_best.onnx` | ORT runs OK |
| CRNN number reader | `models/onnx/crnn_finetuned.onnx` | 1.14e-05 |
| Province classifier | `models/onnx/province_classifier.onnx` | 1.19e-06 |

**One model change (behaviour-preserving):** the CRNN's height-collapse used
`adaptive_avg_pool2d(., (1, w))`, which ONNX can't export with a dynamic size.
Replaced with `conv.mean(dim=2, keepdim=True)` — **mathematically identical**.
Verified the live pipeline is unchanged (number acc still 67.79%, CER 15.15%,
FAR 0) after the swap. New deps: `onnx`, `onnxruntime` (CPU) added to the venv.

**Not done (deferred):** TensorRT/quantization and swapping the runtime to ONNX
Runtime. Export is the enabling step; actually running the pipeline on ORT is a
separate task worth doing only when targeting non-laptop edge hardware. Current
latency (~51 ms / 19.6 FPS) has headroom, so there's no bottleneck to fix now.
**Impact:** Deployment portability, not accuracy. No training.

### 3.2 Lightweight experiment tracking  ✅ DONE (2026-07-17)
**Status:** Implemented as `scripts/tools/experiment_log.py` — one append-only CSV
`metrics/experiment_log.csv` (columns: `timestamp, git_commit, component, metric,
value, split, notes`). No new deps. Usage:
- `python scripts/tools/experiment_log.py --show` — print the log.
- `python scripts/tools/experiment_log.py --component crnn --metric cer --value 0.1021 --split real-test --notes "..."` — log one row.
- `python scripts/tools/experiment_log.py --backfill` — seed the documented history once (idempotent).
- Importable `log_metric(...)` for scripts; **`benchmark_composed.py` now auto-appends** its headline metrics (number acc, CER, detection rate, FAR, 2.2 precision) with the current commit on every run.

The history was backfilled from `docs/HANDOFF.md` (the CER curve 0.9489 → 0.1021,
detector mAPs, province acc, latency), so the log opens with the baseline story
instead of empty. Every future run stamps the commit, so a regression like the
DET-005 padding one can't be silently rediscovered.

**Original intent (met):** metrics were scattered across `results/`, `metrics/`,
`runs/`, and prose in `HANDOFF.md`; now there is one diffable, greppable,
Excel-openable source of truth.

---

## Phase 4 — Lower priority / explicitly deferred

- **Admin panel authentication** — real gap, but acceptable for a LAN demo;
  flag clearly before any deployment beyond the current scope.
- **Merging the two YOLO detectors into one multi-class model** — the data in
  section 6 shows detection (mAP50 0.9664 / 0.943) is *not* where accuracy is
  being lost; this would be a latency/maintenance cleanup, not an accuracy
  win. Worth doing only after Phase 1–2 land.
- **Full-Khmer CRNN (undoing DEV-002)** — explicitly out of scope per the
  approved deviation; the classifier approach is working (97.18% acc) and
  reopening this would be effort spent away from the actual bottleneck (the
  number CRNN).

---

## Does this roadmap require collecting and training new data?

Short answer: **some of it does, some of it doesn't.** Split by whether a GPU
training run is involved:

| Item | New data? | Retrain a model? |
|------|-----------|------------------|
| 1.1 Composed-plate benchmark | No | No — pure measurement |
| 1.2 Constrained decoding | No | No — decode-time logic only |
| 1.3 Active-learning loop | Yes (mined from your own logs) | **Yes — CRNN fine-tune** |
| 2.1 Targeted data collection | Yes (new captures) | **Yes — CRNN fine-tune** |
| 2.2 Cross-validation | No | No — decision logic only |
| 3.1 ONNX/TensorRT export | No | No — converts existing weights |
| 3.2 Experiment tracking | No | No |

So **all of Phase 1 (the highest-value work) needs no new training at all.**
Training only re-enters the picture at 1.3 and 2.1, when you fine-tune the CRNN
on newly labeled real crops. See the next section for where that training runs.

---

## When to use Google Colab (and when NOT to)

Colab has exactly **one job for this project: a free/bigger GPU for *training*.**
It does nothing for any part of the system that doesn't train. So the decision is
never "should I connect Colab" — it's "does this task involve a training run?"

**You do NOT need Colab for:**
- **All of Phase 1** (1.1 benchmark, 1.2 constrained decoding) — pure code, no GPU training.
- 2.2 cross-validation, 3.1 ONNX export, 3.2 tracking — no training.
- Running the system (detection/recognition **inference** runs fine on the 4 GB laptop GPU).

**Colab becomes *optional* for:**
- **1.3 and 2.1** — the only items that fine-tune the CRNN on newly labeled real data.

**Why it's optional, not required:** the laptop **RTX 3050 (4 GB VRAM) already
trained every model in this project to target** (the CRNN reached 10.21% CER on
this machine). For the ~500–1,000-crop fine-tunes in this roadmap, the laptop is
enough — it's already done exactly that.

**When Colab is actually worth the hassle:** only if a *future* training run
outgrows the laptop — e.g. after 2.1 collects a much larger set (~2,000+ crops),
or you want a bigger batch size / faster epochs, or you don't want to tie up the
laptop for hours. Colab then gives a **free T4 (16 GB VRAM, ~4× the laptop)**.

**Colab's costs / gotchas (why it isn't free lunch):**
- You must **upload the dataset** and **download the new `.pth` weights** back into
  `models/recognition/` afterward.
- Sessions **time out** (~90 min idle, ~12 h max) — long runs need checkpointing.
- Colab is **Linux**; this project is Windows by design (DEV-001). Keep Colab use
  to *training only* — never move the runtime/inference pipeline there.
- **Train big, deploy small:** the model still has to run on the 4 GB laptop at the
  gate. Don't let Colab's 16 GB tempt you into a model that won't fit back home.

**Rule of thumb:** start Phase 1 with no Colab. Reach for Colab **only** when a
CRNN fine-tune (1.3 / 2.1) is too slow or runs out of memory on the RTX 3050 —
and even then, only for the training step, bringing the weights back to Windows.

---

## Suggested documentation updates alongside this work

- `docs/HANDOFF.md` — add a "Phase 1/2/3" status table mirroring this roadmap
  so the acceptance suite and this plan stay in sync.
- `docs/srs.md` / `docs/SRS_ALIGNMENT_PLAN.md` — once 1.1 (composed-plate
  benchmark) exists, promote composed-plate exact-match to a named SRS
  acceptance metric rather than leaving it implicit.
- `docs/SRS_DEVIATION_LOG.md` — no new deviations expected from Phase 1–2;
  flag here immediately if 2.1's data collection or 3.1's export work implies
  any new deviation from the original SRS.

---

## One-line summary if you only read this far

**Do 1.1 (composed-plate benchmark) then 1.2 (constrained decoding) first** —
neither needs new data or training. Build the benchmark before the change so you
can prove 1.2 helps *and* never opens the gate for the wrong car. Everything
after that either targets a gap 1.1 makes measurable, or the data lever (1.3/2.1)
your own CER curve already proves works.
