# Cambodian ALPR — Improvement Plan V2 (phase-by-phase)

> Successor to `docs/IMPROVEMENT_ROADMAP.md`, whose Phase 1–3 items are now done
> (1.1 benchmark, 1.2 constrained matching, 2.2 cross-validation, 3.1 ONNX,
> 3.2 experiment tracking). This document covers what comes **after** that.
>
> Written: 2026-07-22. Branch at time of writing: `fix/detect`.
>
> **How to use this doc:** do **one phase at a time, in order**. Every phase ends
> with a *Gate* — a measurement you must run and record before starting the next
> phase. Do not batch phases together; the whole point is that each change is
> individually attributable in `metrics/experiment_log.csv`.

---

## Constraints that apply to every phase

Carried over unchanged from `IMPROVEMENT_ROADMAP.md` — repeated so this doc is
self-contained:

- Must run on a 4 GB VRAM laptop GPU — stay edge-deployable.
- **Fail-safe default-deny is a hard safety requirement.** No phase may increase
  the false-accept rate above 0. Every phase's Gate re-checks it.
- `models/detection/best.pt` and `models/recognition/crnn_best.pth` stay
  read-only baselines.
- Windows-only by design (DEV-001).
- **Never train on the human-verified test split** (the 149 frames).
- Every behaviour change goes behind a config flag in
  `configs/system_config.yaml`, defaulting to the **old** behaviour until its
  Gate passes. This makes every phase revertible with one line.

---

## Findings this plan is built on (measured 2026-07-22)

These are facts read out of the repo/DB, not estimates. They are the evidence
each phase points back to.

| # | Finding | Source |
|---|---------|--------|
| A | Province ground truth **already exists** (149 rows) but is the classifier's own pre-fill — 0 rows carry a note, 126/149 sit at `pred_conf == 1.0`. Never human-verified. *(→ resolved in Phase 0: 7 were wrong)* | `data/crnn_crops/province_test_labels.csv` |
| B | The benchmark has **never been run with `--province-gt`** — `composed_measured_on: 0`. Composed accuracy is still an estimate. *(→ resolved in Phase 0: 68.71% measured)* | `metrics/composed_benchmark.json` |
| C | The most recent benchmark ran on **n=40**, not 149. The 85% / CER 6.69% figures are not comparable to 67.79% / 15.15%. *(→ resolved in Phase 0; the n=40 run was noise)* | `metrics/experiment_log.csv` |
| D | **~800 reads passed the 0.70 confidence gate in a plate format that does not exist.** See table below. | `plates.db`, `data/crnn_crops/real_labels.csv` |
| E | Confidence is badly clumped: 5,272 of 7,341 reads (72%) fall in the 0.9–1.0 bucket, so the 0.70 threshold barely discriminates. | `plates.db` |
| F | One car in view produces ~40 frames whose digits are stable but whose prefix flickers (`1J/1H/1M/1A/1N-7776`). The pipeline keeps **one** frame and discards the rest. | working tree `photos/`, `alpr_system.py:522` |
| G | The mined-data pool is now **7,341 audit reads / 6,122 evidence photos** — 1,675 REVIEW_REQUIRED, 844 below conf 0.1. | `plates.db`, `photos/` |

### Finding D in detail — the format gap

Ground truth format across 622 human labels:

| pattern | count | share |
|---------|-------|-------|
| `DL-DDDD` | 345 | 55.5% |
| `D-DDDD`  | 248 | 39.9% |
| `DLL-DDDD` | 20 | 3.2% |
| everything else | 9 | 1.4% |

Confident live reads (`crnn_confidence >= 0.70`, n=5,952):

| pattern | count | legal? |
|---------|-------|--------|
| `DL-DDDD` | 4,847 | ✅ |
| `DL-DDD` | 306 | ❌ |
| `D-DDDD` | 302 | ✅ |
| `DL-DD` | 96 | ❌ |
| `DLL-DDDD` | 70 | ✅ |
| `D-DDD` | 36 | ❌ |
| `DDDDD` | 34 | ❌ |
| `DDDDDD` | 32 | ❌ |
| `L-DDDD` | 29 | ❌ |
| `DL-D-DD` | 23 | ❌ |

(`D` = digit, `L` = Latin letter.) Roughly 800 confident reads are in an
impossible format — this is the same population as the benchmark's
"length-wrong 23" failures, and it is detectable without touching the model.

---

## Phase overview

| Phase | Name | Effort | New data? | Retrain? | Unlocks |
|-------|------|--------|-----------|----------|---------|
| **0** | Restore the instrument ✅ **DONE 2026-07-22** | ~30 min | No | No | Every later measurement |
| **1** | Plate-format validation | Low | No | No | Kills the dominant failure mode |
| **2** | Confidence recalibration | Low | No | No | Makes the REC-005 gate mean something |
| **3** | Visit-level benchmark | Medium | No | No | Makes Phase 4 measurable |
| **4** | Multi-frame read fusion | Medium | No | No | Biggest training-free accuracy gain |
| **5** | Active-learning harvest | Medium | Yes (mined) | **Yes** | Extends the proven CER curve |
| **6** | Deployment hardening | Low–Medium | No | No | Removes the "not for real deployment" asterisk |

**Phases 0–4 require no new data and no GPU training.** Training re-enters only
at Phase 5.

---

# Phase 0 — Restore the instrument

**Do this before writing any code.** Every later phase is judged against the
benchmark, and the benchmark is currently reporting on the wrong sample size
(finding C) with an unmeasured headline metric (findings A, B).

### Why
You cannot attribute a gain to Phase 1 if the baseline it is compared against
was measured on 40 frames and the composed metric is an arithmetic estimate.

### Steps

1. **Verify the province pre-fills.** The CSV was generated by the classifier
   grading itself — that is not ground truth. Open the contact sheets and check
   each plate's printed **English** province name against `province_latin`:

   ```
   python scripts/tools/label_province_test.py --stats
   ```

   Sheets are at `results/province_sheets/sheet_*.png`, legend at
   `results/province_sheets/LEGEND.txt`.

   Prioritise by confidence — the 23 rows with `pred_conf < 1.0` (8 of them
   below 0.9, lowest 0.201) are where errors concentrate. Correct
   `province_class` in the CSV and write a short reason in the `note` column so
   a future reader can tell verified rows from untouched pre-fills.

   > Minimum bar to proceed: **every row with `pred_conf < 0.9` has been looked
   > at by a human and carries a note.** Ideally all 149.

2. **Re-run the full benchmark with province ground truth** — no `--limit`:

   ```
   python scripts/system/benchmark_composed.py --province-gt data/crnn_crops/province_test_labels.csv
   ```

3. **Record the baseline.** The run auto-appends to
   `metrics/experiment_log.csv`. Copy the headline numbers into the table below
   so this doc carries its own baseline.

### ✅ DONE (2026-07-22) — baseline established

**Measured baseline, commit `dd108fc`:**

| Metric | n=149 baseline | Notes |
|--------|----------------|-------|
| Detection rate | **95.97%** (143/149) | 6 frames detect nothing |
| Number e2e accuracy | **67.79%** (101/149) | |
| Number CER | **15.15%** | failures 42 (length-wrong 23, not-detected 6) |
| Province accuracy | **95.24%** (140/147) | **first real measurement** |
| **Composed exact-match** | **68.71%** (101/147) | **first real measurement** — the headline |
| False-accept rate | **0.00%** | 0 opens on 137 wrong-neighbour plates |
| 2.2 flag precision | 45.45% (22 flagged) | the n=40 run's 20% was sample noise, as suspected |

Gap to the Definition of Done (composed ≥ 85%): **16.3 points.**

#### What the verification actually found

The pre-fill was **not** trustworthy, and `pred_conf` turned out to be a **bad proxy
for label risk** — the plan's original "check rows below 0.9" rule would have missed
most of the errors, several of which sat at `pred_conf = 0.999`.

The real risk population is **structural**, and it is found by re-running the
detector rather than by reading the confidence column:

| Risk class | Frames | Why the pre-fill is unreliable |
|------------|--------|-------------------------------|
| **0 detections** | 6 | `label_province_test.py` falls back to `crop = frame`, so the classifier graded a whole street scene, not a plate |
| **>1 detection** | 7 | the tool classifies `dets[0]`, which is not necessarily the plate whose number is in `real_labels.csv` |
| single detection | 136 | pre-fill sound in principle |

All 149 were then verified by eye against the **printed English province name**,
using enlarged per-plate crops. Result: **7 confirmed wrong, 2 unverifiable, 140 confirmed correct.**

**Corrections applied:**

| Frame | Pre-fill said | Truth | Evidence |
|-------|---------------|-------|----------|
| `10004b2e62…` | Siem_Reap | **Oudor_Meanchey** | plate prints ODDAR MEANCHEY |
| `22107033…` | Preah_Sihanouk | **Other** | green STATE plate `2-8585` |
| `22642443…` | Svay_Rieng | **Other** | green STATE plate `2-1492` |
| `27003377…` | Battambang | **Other** | green STATE plate `3-0047` |
| `IMG_4415…` | Kandal | **Svay_Rieng** | plate prints SVAY RIENG |
| `IMG_4291…` | Takeo | **Siem_Reap** | labelled plate `1AE-4647` prints SIEM REAP; Takeo was the *other* bike in frame |
| `IMG_4305…` | Preah_Sihanouk | **Phnom_Penh** | labelled plate `-7495` prints PHNOM PENH; Preah Sihanouk was the *other* bike |

**2 rows marked `UNVERIFIABLE`** and excluded from the measured metric (scoring
against a label nobody could read would silently corrupt the headline):
`IMG_2741…` (plate *has* a Khmer province line, so `Other` is wrong, but the
province is illegible at 640×640) and `IMG_1108…` (pre-fill classified the wrong
plate; the labelled plate is illegible).

#### Root cause — and a correction to an early hypothesis

The three green STATE plates initially looked like a systematic classifier blind
spot. **They are not.** Roughly 20 other green STATE/RCAF/POLICE plates in the set
are labelled `Other` correctly. All three failures are **zero-detection frames**,
where the classifier received the entire photo instead of a plate crop.

So the mechanism is one bug, not two: **detection failure → whole-frame fallback →
arbitrary province.** The remaining 4 errors are all the **multi-plate `dets[0]`
mismatch.** Both are properties of the *labelling tool*, not of the province
classifier — which is why the classifier's 97.18% test accuracy never surfaced them.

#### Finding: province errors do not stack with number errors

Composed exact-match (101/147) equals number-correct (101), which is only possible
if **every frame with a correct number also had a correct province.** All 7 province
errors land on frames whose number was already wrong. Practical consequence: the
province branch is *not* currently costing composed accuracy — **the number CRNN is
the whole gap.** This is direct evidence for the Phase 1/2/4 ordering, and it means
province work stays correctly deferred.

#### Changes made to the harness
- `load_province_gt()` now skips `UNVERIFIABLE`-noted rows and reports the count,
  so an unreadable frame can never be silently scored.
- The benchmark now logs `composed_exact_match` and `province_accuracy` to
  `metrics/experiment_log.csv` — **only when actually measured**, never the estimate.

### Gate — PASSED
- [x] `composed_kind` reads `measured`, not `ESTIMATE`
- [x] `n_frames` is 149
- [x] `false_accept_rate` is 0.0
- [x] Baseline table filled in

### Carry-forward for later phases
- The 6 zero-detection frames are a **detector** gap (all green/state or low-contrast
  plates). They cap composed accuracy at 95.97% no matter how good the CRNN gets.
  Worth a Phase 6.3 data-collection target.
- `label_province_test.py` still has both bugs (whole-frame fallback, `dets[0]`
  assumption). Fix before it is ever used to label a new split.

---

# Phase 1 — Plate-format validation

### Why
Finding D. About 800 reads cleared the confidence gate in a format no Cambodian
plate uses. These are not marginal reads the gate is meant to let through —
they are structurally impossible, and the system currently treats them as
confident. This is the single cheapest accuracy *and* safety win available: a
malformed string can no longer accidentally exact-match a registered plate.

### What to build

**New file `src/recognition/plate_format.py`** — the grammar in one place:

- The legal patterns derived from ground truth: `D L - D D D D`,
  `D - D D D D`, `D L L - D D D D`.
- `is_valid(number: str) -> bool`
- `normalise(number: str) -> str` — strip spaces, upper-case, insert the dash
  where the pattern implies one (`1M7776` → `1M-7776`). The charset in
  `models/recognition/charset.txt` already includes both `-` and space, so the
  CRNN emits both forms.
- Keep the pattern list **data-driven and documented** — write it as a small
  table with the observed counts, so when a new plate series appears the fix is
  one line, not a code hunt.

**Two levels of enforcement — ship level 1 first, measure, then decide on level 2:**

| Level | Change | Where | Risk |
|-------|--------|-------|------|
| 1 — *reject* | If `is_valid()` fails, scale confidence to 0 so REC-005 routes it to `REVIEW_REQUIRED` | `src/recognition/crnn_reader.py:68-86`, alongside the existing `length_factor` | None — strictly fewer opens |
| 2 — *repair* | Constrained CTC beam search that only emits grammar-conforming strings, so `1M-777` decodes to `1M-7776` instead of being thrown away | `crnn_reader.py` decode step | Can *create* a valid-looking wrong plate — see below |

**Level 2 safety note.** Repairing a read to the nearest legal format can turn an
unreadable plate into a confident wrong one. Mitigation: a repaired read must
never be treated as an exact whitelist hit on its own — route it the same way
ROADMAP 1.2 routes near-matches, i.e. to `REVIEW_REQUIRED`. Only ship level 2 if
the Gate below shows false-accept rate still 0.

### Config

```yaml
gate:
  format_validation: false      # Phase 1 level 1 — flip to true after the gate passes
  format_repair: false          # Phase 1 level 2 — only after level 1 is proven
```

Default **false** so `main` behaviour is unchanged until measured.

### Gate — do not start Phase 2 until
- [ ] Benchmark re-run at n=149 with `format_validation: true`
- [ ] `false_accept_rate` still 0.00%
- [ ] `number_failures.length-wrong` has dropped (the target this phase exists for)
- [ ] Number e2e accuracy has **not** regressed — if it has, the grammar is too
      tight; widen the pattern table, don't disable the feature
- [ ] Result logged to `metrics/experiment_log.csv` with a note naming this phase

---

# Phase 2 — Confidence recalibration

### Why
Finding E. `crnn_reader.py:80` computes confidence as the **mean** of the
per-timestep max probability. But an exact whitelist match dies on a **single**
wrong character — and a mean over 7 characters hides one bad one. The
consequence is visible in the distribution: 72% of all reads land in the 0.9+
bucket, so the 0.70 threshold separates almost nothing.

### What to change

In `src/recognition/crnn_reader.py::_infer`:

- Compute confidence from the **weakest character** (`min` over the non-blank
  timesteps), or a geometric mean, instead of the arithmetic mean.
- Return the **per-character confidences** alongside the text, so the dashboard
  and the audit row can show *which* character is doubtful. This is what makes
  a human review fast instead of a guess.
- Keep the existing `length_factor` — it is doing a different job (penalising
  implausibly short reads) and Phase 1 does not replace it.

**Free safety property worth stating in the commit message:** `min ≤ mean`
always, so every confidence can only go **down**. Strictly fewer auto-opens,
therefore this phase provably cannot introduce a false accept.

### Expect the threshold to need re-tuning
Confidences drop across the board, so `crnn_confidence_threshold: 0.70` will
suddenly be much stricter and throughput will fall. Sweep the threshold against
the Phase 0 benchmark and pick the value that maximises correct auto-opens while
keeping false accepts at 0. **Record the sweep** — a table of threshold vs.
(auto-open, review, deny, false-accept) belongs in the experiment log.

### Config

```yaml
gate:
  confidence_mode: "mean"       # "mean" (current) | "min" | "geometric"
  crnn_confidence_threshold: 0.70   # re-tune after switching mode
```

### Gate — do not start Phase 3 until
- [ ] Threshold sweep table recorded
- [ ] `false_accept_rate` 0.00% at the chosen threshold
- [ ] Correct auto-opens ≥ the Phase 1 number (this phase should improve the
      *separation*, not cost throughput — if it costs throughput, the threshold
      is wrong, not the metric)
- [ ] Per-character confidences visible in the dashboard

---

# Phase 3 — Visit-level benchmark

### Why
`scripts/system/benchmark_composed.py` grades 149 **independent still frames**.
The real gate does not see stills — it sees a car for ~40 frames and makes
**one** decision. That mismatch means the current harness structurally cannot
measure Phase 4. Build the instrument before the change it measures — the same
ordering discipline that made ROADMAP 1.1 precede 1.2.

### Why this is cheap
You already have the dataset. `photos/` holds 6,122 evidence frames named
`plate_YYYYMMDD_HHMMSS_mmm_<TEXT>.jpg` — millisecond timestamps. Consecutive
frames of one car group into a visit by time gap, using the same rule the live
pipeline already uses (`logging.dedup_gap_sec`, default 3.0 s). No new capture,
no labelling of images — only a plate label per *visit*, which is far less work
than per frame.

### What to build

Add a `--visits` mode to the benchmark (or a sibling
`scripts/system/benchmark_visits.py`):

1. Group `photos/` by timestamp gap into visits.
2. For each visit, establish one ground-truth plate (label the visit once —
   most visits are already obvious from the majority of their filenames, but a
   human must confirm, since those filenames are the model's own output).
3. Replay each visit's frames through the pipeline and score **one decision per
   visit**: correct-open / wrong-open / review / deny.
4. Report the same safety metric as always — false accepts must be 0.

**Data-hygiene guard, non-negotiable:** exclude any visit overlapping the 149
human-labeled test frames. Phase 5 mines the same pool for *training*, and if
the two sets blur, both CER and this benchmark become meaningless.

### Gate — do not start Phase 4 until
- [ ] Visit grouping produces a sane visit count (spot-check: a visit should be
      tens of frames, not one, and not hundreds)
- [ ] Visit-level baseline recorded for the **current** best-single-frame logic —
      this is the number Phase 4 must beat
- [ ] False-accept rate reported at visit level and equal to 0

---

# Phase 4 — Multi-frame read fusion

### Why
Finding F. `_dedup_persist` ([`alpr_system.py:522-555`](../src/core/alpr_system.py#L522-L555))
collapses a visit into one audit row and keeps the **single highest-confidence
frame**, discarding the other ~39. Your own working tree shows why that is a bad
bet: across one car's frames the digits `7776` are rock-stable while the prefix
flickers across `1J / 1H / 1M / 1A / 1N`. Best-single-confidence gambles the
gate decision on one frame; a vote uses all of them.

This is very likely worth more than the next CRNN fine-tune, and it costs no new
data.

### What to build

A `VisitAggregator` (new class, kept out of `ALPRSystem` so it is unit-testable):

- Accumulate **per-position character votes** across the visit, weighted by the
  per-character probability that Phase 2 now exposes.
- Accumulate **province-class votes** the same way.
- Emit the consensus plate for the visit, plus a consensus confidence.
- Require **k agreeing frames** before `ENTRY_ALLOWED`.

Then rewire `_dedup_persist` to upgrade its single audit row from the
aggregator's consensus rather than from whichever frame scored highest.

### Second benefit — this fixes the cross-plate false open
The 1.2 benchmark found 1 confident read that exact-matched a *different*
registered plate. A single-frame fluke cannot open a gate that requires k frames
to agree. This is the mitigation that item was waiting for, and it arrives for
free with the accuracy work.

### Interaction with earlier phases
- Depends on **Phase 2** for per-character probabilities (a vote weighted by a
  whole-string mean is much weaker).
- Depends on **Phase 3** to be measurable at all.
- Composes with **Phase 1**: vote first, then validate the consensus against the
  grammar.

### Config

```yaml
gate:
  visit_fusion: false           # flip on after the Phase 3 baseline exists
  visit_min_agree: 3            # k — frames that must agree before an open
```

### Gate — do not start Phase 5 until
- [ ] Visit-level accuracy beats the Phase 3 baseline
- [ ] `intruder_false_open` is 0 (this is the phase that should finally kill it)
- [ ] Frame-level benchmark not regressed
- [ ] Latency still inside SRS PERF (< 500 ms end-to-end, > 15 FPS) — the
      aggregator runs per frame, so measure it, don't assume it

---

# Phase 5 — Active-learning harvest

> **This is the first phase that needs a GPU training run.** Everything above is
> code and measurement only.

### Why
Finding G. ROADMAP 1.3 was written when this pool was hypothetical; it now holds
**7,341 audit reads and 6,122 evidence photos**, including 1,675
`REVIEW_REQUIRED` and 844 reads below confidence 0.1 — precisely the
distribution the model currently fails on. And the CER curve has not plateaued:

| real labels | 0 | 143 | 324 | 473 |
|-------------|---|-----|-----|-----|
| CER | 94.89% | 25.93% | 20.32% | **10.21%** |

More real data remains the lever with the clearest track record in this
project's own history.

### What to build
A harvest script that:
1. Pulls `REVIEW_REQUIRED` and low-`crnn_confidence` rows from `plates.db` with
   their `photo_path`.
2. **Dedupes by visit** — reuse Phase 3's grouping, or you will label the same
   car 40 times and skew the set toward whichever cars idled longest.
3. Surfaces them through the existing montage-transcription workflow
   (`scripts/recognition/crop_numbers.py`, `scripts/tools/make_montage.py`).
4. Appends **only confirmed-correct** labels to the real training set.

### Two guards, both mandatory
- **No test-set leakage.** Hard-exclude anything overlapping the 149
  human-labeled test crops. Without this, CER becomes a meaningless number and
  every measurement in this document is retroactively void.
- **No selection bias.** Fine-tuning *only* on hard cases skews the model and can
  degrade easy-case accuracy. Mix mined hard cases **back into** the existing 473
  labeled crops; do not train on the mined set alone.

### Then fine-tune
`scripts/recognition/finetune_crnn.py`, on the expanded set. Per the existing
roadmap analysis, the RTX 3050 (4 GB) is sufficient for a fine-tune at this scale
— it already produced the 10.21% CER result. Reach for Colab only if the set
grows past ~2,000 crops, and even then bring the weights back to Windows
(DEV-001). **Train big, deploy small: the model still has to run at the gate.**

### Gate — do not start Phase 6 until
- [ ] New CER recorded against the **unchanged** 149-frame test set
- [ ] Both frame-level and visit-level benchmarks re-run
- [ ] False-accept rate still 0
- [ ] Leakage check documented — state explicitly how test overlap was excluded

---

# Phase 6 — Deployment hardening

Deferred deliberately until the accuracy work lands, because none of it changes
the metric that decides whether the gate opens.

### 6.1 Admin panel authentication
`scripts/system/admin_web.py` currently has none. Acceptable for a LAN demo,
**not** acceptable for anything beyond it. A session cookie plus a single admin
password is roughly an hour's work and removes the standing asterisk.

### 6.2 Live camera validation
`camera_source` currently points at `http://10.45.245.88:8080/video` (an
uncommitted local change — decide whether that belongs in the committed config
or in a local override). The RTSP/reconnect code in `src/utils/rtsp_reader.py`
is written but has never been through the 2-hour stability run. Run it, and
watch `logs/alerts.log` and the `system_metrics` table for disconnect and
latency alerts.

### 6.3 Hard-condition data collection
Night, motion blur, steep angles, dirty plates — under-represented in the
current 473 crops, which are mostly easy daylight shots. Deliberately capture
4–6 target conditions, ~50–100 crops each, rather than more easy daylight.
Needs a further CRNN fine-tune, so treat it as a repeat of Phase 5 with a
different data source.

### 6.4 Retune the 2.2 consistency check
Flag precision swung 45% → 20% between runs — but the 20% figure came from the
n=40 run (finding C), so it is sample noise, not a regression. Retune
`gate.number_alignment_min` against the Phase 0 n=149 baseline **with province
ground truth**, which is the validation 2.2 was always missing.

---

## Explicitly still deferred

Unchanged from `IMPROVEMENT_ROADMAP.md` Phase 4 — listed so nobody re-litigates
them mid-plan:

- **Merging the two YOLO detectors** — detection (mAP50 0.9664 / 0.943) is not
  where accuracy is lost. Latency/maintenance cleanup only.
- **Full-Khmer CRNN (undoing DEV-002)** — the classifier approach works (97.18%);
  reopening it spends effort away from the real bottleneck.
- **TensorRT / quantisation, ONNX runtime swap** — export is done; at ~51 ms /
  19.6 FPS there is no bottleneck to fix.
- **Crop padding (DET-005)** — stays at 0.0; padding regressed accuracy
  70.6% → 46.2%, logged as DEV-004.

---

## One-line summary

**Phase 0 first — the instrument is broken (n=40, composed unmeasured) and every
later number depends on it.** Then Phase 1 (format) and Phase 2 (confidence),
both cheap, both provably unable to add false-accept risk. Then Phase 3 before
Phase 4, because a still-frame benchmark cannot see a multi-frame fix. Training
only re-enters at Phase 5.
