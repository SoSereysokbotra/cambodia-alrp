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
| **1** | Plate-format validation ✅ **DONE 2026-07-23** | Low | No | No | Kills the dominant failure mode |
| **2** | Confidence recalibration ✅ **DONE 2026-07-23 — hypothesis refuted, default kept** | Low | No | No | Makes the REC-005 gate mean something |
| **3** | Visit-level benchmark ✅ **DONE 2026-07-23** | Medium | No | No | Makes Phase 4 measurable |
| **4** | ~~Multi-frame read fusion~~ ❌ **CANCELLED — refuted by Phase 3** | Medium | No | No | ~~Biggest training-free accuracy gain~~ — ceiling is 2.1 pp; both implementations lose |
| **5** | Active-learning harvest ✅ **TOOLING DONE 2026-07-23** — fine-tune ran, candidate not shipped | Medium | Yes (mined) | **Yes** | Extends the proven CER curve; the only lever left |
| **6** | Deployment hardening — 6.1 auth ✅ **DONE**, 6.4 2.2-retune ✅ **DONE (disabled)**; 6.2 live camera + 6.3 hard-condition capture remain | Low–Medium | No | No | Removes the "not for real deployment" asterisk |

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

### ✅ DONE (2026-07-23) — level 1 shipped, level 2 rejected on evidence

Implemented as `src/recognition/plate_format.py` (grammar + self-check), wired into
`CRNNReader._infer` behind `gate.format_validation`, now **`true`** by default.
`benchmark_composed.py` gained `--format-validation on|off` and `--dump` so the A/B
is repeatable without editing the committed config.

**Grammar coverage — the "is it too tight?" check, run before shipping:**

| Set | Accepted |
|-----|----------|
| 622 ground-truth labels | **619 (99.52%)** |
| rejected | `6667`, `-7495`, `10-1152` — all three are known **labelling artefacts** (plate partly out of frame, so the label is a fragment), not grammar gaps |
| 5,952 confident live reads | 5,226 legal / **726 impossible (12.2%)** |

The vanity rule earns its place: 6 real CAMBODIA-series plates (`COVI19`,
`HENGHENG`, `HYWAZA9`, `ELDC865`, `SELA GTR`) would otherwise be pushed to REVIEW,
and since every impossible pattern observed is digit-led, allowing vanity costs
**zero** detection power.

**A/B on the 149-frame test set (`--format-validation off` vs `on`):**

| Metric | OFF | ON | Δ |
|--------|-----|----|----|
| Number e2e accuracy | 67.79% | 67.79% | — |
| Composed exact-match | 68.71% | 68.71% | — |
| **False-accept rate** | **0.00%** | **0.00%** | — ✅ |
| confident auto-open (exact) | 100 | **100** | **0 — no correct open lost** |
| low-confidence → REVIEW | 12 | 25 | +13 |
| 1.2 recovered DENY→REVIEW | 17 | 12 | −5 |
| **still hard DENY** | **13** | **5** | **−8** |
| intruder false auto-open | 1 | 1 | — |

**The win: 8 confidently-wrong reads that were being silently `ENTRY_DENIED` are
now `REVIEW_REQUIRED`, at zero cost to correct auto-opens.** For a gate, a wrong
denial is a stranded driver; a review is a human glance.

Per-frame analysis confirms the mechanism: the grammar catches **23 of the 42 wrong
reads (55%)**, and rejects exactly **1** correct read — `6667`, one of the three
labelling artefacts, which was already below the confidence gate, so it cost nothing
at the gate.

**Known minor regression (safety-neutral):** validation runs before the ROADMAP 1.2
near-match logic, so 5 reads that used to reach REVIEW *with* a `suggested_plate`
("did-you-mean") now reach REVIEW as low-confidence *without* one. Same gate outcome
(closed) and same destination (a human), but slightly less context for the reviewer.
Fixable by computing the suggestion even for format-rejected reads; deferred as
low-value.

### Level 2 (repair) — evaluated and NOT shipped

`normalise()` exists and is unit-tested, but is deliberately **not** wired into any
gate path. Measured on the 42 wrong reads, dash-insertion repair would change only
**2** strings: `27389 → 2-7389` (correct) and `---7 → -7` (still wrong). **Net +1
frame out of 149 (+0.67 pp), against the risk of inventing a plate string.**

The plan's fuller level 2 — a grammar-constrained CTC beam search — has a ceiling of
the 23 invalid-format wrong reads (~15 pp) but carries exactly the risk the plan
flagged: forcing a read into a legal shape can manufacture a *confident wrong plate*.
**Recommendation: defer it behind Phase 4.** Multi-frame voting attacks the same 23
length-wrong reads by aggregating real evidence across frames rather than guessing
within one frame — strictly safer for the same target. Revisit level 2 only if
Phase 4 leaves those 23 unresolved.

### Gate — PASSED (with one criterion corrected)
- [x] Benchmark re-run at n=149 with `format_validation: true`
- [x] `false_accept_rate` still 0.00%
- [x] Number e2e accuracy has **not** regressed (67.79% → 67.79%)
- [x] Result logged to `metrics/experiment_log.csv`
- [x] ~~`number_failures.length-wrong` has dropped~~ — **this criterion was
      mis-specified when the plan was written.** Level 1 only changes *confidence*,
      never the decoded string, so it cannot move `length-wrong` (still 23) by
      construction. That is a **level-2 (repair) criterion**, and it is the metric
      Phase 4 should be judged on. The correct level-1 criterion — and the one that
      passed — is *"wrong reads diverted from silent DENY to REVIEW, at no cost to
      correct auto-opens"*: **8 diverted, 0 lost.**

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

### ✅ DONE (2026-07-23) — hypothesis REFUTED, default unchanged

**The Phase 2 hypothesis was wrong, and the measurement says so clearly.** All three
statistics are implemented and switchable (`gate.confidence_mode`), but the default
stays **`mean`**, because `min` is *worse* and no alternative beats it by more than
noise.

**Separability (AUC = P(confidence of a correct read > confidence of a wrong read)),
n_correct=101, n_wrong=42, Hanley–McNeil SE:**

| Mode | format OFF | format ON |
|------|-----------|-----------|
| **mean** (legacy) | 0.8191 ± 0.0349 | 0.8455 ± 0.0320 |
| **min** (the hypothesis) | **0.7930** ± 0.0375 | 0.8471 ± 0.0318 |
| geometric | 0.8309 ± 0.0336 | 0.8545 ± 0.0309 |

1. **`min` is *worse* than `mean` on the raw signal** (0.7930 vs 0.8191). Taking the
   minimum over ~7 characters is noisy: one unlucky low-probability timestep on an
   otherwise-correct read drags the whole score down, and the mean averages exactly
   that noise out. The hypothesis is not merely unsupported, it is backwards.
2. **Every difference is within noise.** `geometric − mean` = +0.0090 against an SE of
   ±0.031 — about 0.25 SE. Changing a shipped default on that would be chasing noise,
   which is the thing the experiment log exists to prevent.
3. **Phase 1 improved separability for every mode** (+0.026 for mean). That is an
   independent second confirmation that Phase 1 was worth shipping.

**Threshold sweep** (detected frames, at each threshold: correct auto-opens / false
opens / review / deny). It shows `min` needs its threshold re-tuned from 0.70 to
~0.20–0.30 just to *match* `mean` — and then it is exactly equivalent:

| Mode @ threshold | open_ok | false open | review | deny |
|---|---|---|---|---|
| mean @ 0.70 | 100 | 1 | 37 | 5 |
| min @ 0.70 | **63** | 1 | 76 | 3 |
| min @ 0.30 | 100 | 1 | 37 | 5 |
| geometric @ 0.70 | 100 | 1 | 38 | 4 |

### Why no confidence statistic can fix the remaining false open

The one intruder false-open is `3E-8990` read for ground truth `3A-8990`. Its
per-character confidences:

```
 3      E      -      8      9      9      0
0.764  0.9985  0.9994 0.9993 0.9996 0.9996 0.9993
  ^weakest  ^THE WRONG CHARACTER
```

**The model is 99.85% confident in the character it got wrong**, while the weakest
character (`3`, 0.76) is correct. This is not an uncertainty problem — it is a
*wrong-but-certain* problem, and no aggregate or per-character confidence can
separate it. Only evidence from outside the single frame can.

**This is the strongest evidence yet for the Phase 4 ordering**, and it explains why
Phase 2 had to be measured before Phase 4 rather than assumed.

### What Phase 2 did deliver
- **Per-character confidences** (`char_confidences`, `weakest_char` in the
  `process_frame` result) via `CRNNReader.read_detailed()` — `read()` is unchanged,
  so no caller broke.
- **Tuning infrastructure:** `--confidence-mode`, `--threshold` on the benchmark.
- Two negative results recorded so nobody re-runs this experiment.

**Deliberately NOT done — dashboard "weakest character" highlight.** The plan called
for it, but the evidence above shows the weakest-character pointer aims at the
*correct* character while the wrong one scores 0.999. Shipping it as a prominent
review hint would actively mislead an operator. The data stays available in the
result dict for the audit trail and admin panel; the OpenCV overlay is dropped on
purpose, not overlooked.

### Gate — PASSED (as a decision to keep the default)
- [x] Threshold sweep table recorded
- [x] `false_accept_rate` 0.00% at the chosen threshold
- [x] Correct auto-opens ≥ Phase 1 (100 = 100, default unchanged)
- [x] ~~Per-character confidences visible in the dashboard~~ — data-layer only; see above

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

### ✅ DONE (2026-07-23) — and it **refutes Phase 4**

Implemented as `scripts/system/benchmark_visits.py` (`--build` / `--score`), with
`metrics/visits_manifest.json`, human labels in `metrics/visits_labels.csv`, and
results in `metrics/visit_benchmark.json`.

**Design decision that made this cheap:** the per-frame CRNN read is already in each
photo's *filename*, and `plate_reads.photo_path` joins it to `crnn_confidence`. So
the harness reconstructs visits from recorded reads and never re-runs inference —
it scores in seconds. This also dodges a trap: the saved photos are **annotated**
(boxes drawn on them), so replaying the image files through the detector would not
be a faithful reconstruction.

**Data hygiene — the contamination was real, not hypothetical.** 664 of 6,122 photos
(10.8%) are **640×640**, the Roboflow export size, i.e. replays of the annotated
dataset through the pipeline. These are excluded; live captures are 1920×1080 /
720×480 / 1080×1080. Keeping them would have leaked the 149-frame test set into
this benchmark.

**Results — 140 labelled visits, 6 distinct vehicles:**

| Strategy | per-visit (micro) | per-plate (macro) |
|----------|------------------|-------------------|
| **best_conf (current deployed)** | **75.71%** (106/140) | **61.56%** |
| majority | 73.57% (103/140) | 56.25% |
| char_vote | 73.57% (103/140) | 56.25% |

**Both fusion strategies are WORSE than the behaviour they were meant to replace.**

### The oracle bound — why this result is implementation-independent

| Plate | visits | **ORACLE** | best_conf | majority | char_vote |
|-------|--------|-----------|-----------|----------|-----------|
| 1A-4249 | 10 | **0** | 0 | 0 | 0 |
| 1M-7776 | 110 | 95 | **92** | 91 | 91 |
| 2-8554 | 5 | **0** | 0 | 0 | 0 |
| 2A-2265 | 2 | 2 | 2 | 2 | 2 |
| 3A-1311 | 7 | 6 | **6** | 5 | 5 |
| 4A-1483 | 6 | 6 | **6** | 5 | 5 |
| **TOTAL** | **140** | **109** | **106** | 103 | 103 |

ORACLE = at least one frame in the visit read the plate correctly. It is the ceiling
for *any* strategy that selects or fuses among observed reads.

1. **`best_conf` already achieves 106 of a 109 ceiling — 97% of what is attainable.**
   The entire remaining headroom for Phase 4 is **3 visits (2.1 pp)**, not the large
   win the plan assumed.
2. **31 of 140 visits (22%) are unreachable by any fusion** — for `1A-4249` and
   `2-8554` the model is *never* right in *any* frame. You cannot vote your way to an
   answer nobody proposed. That is a model failure, not a fusion failure.
3. Because the oracle is a property of the observed reads, this bound holds for a
   smarter implementation too — including one weighted by the per-character
   confidences Phase 2 added. (The offline harness cannot simulate per-character
   weighting, since historical logs only stored whole-read confidence; but the
   ceiling caps that variant at the same 109.)

### Why the Phase 4 intuition was wrong

The plan argued fusion "uses all 40 frames instead of betting the decision on one."
That reasoning treats the frames as independent samples of equal quality. They are
not: a visit is typically **one good view plus many blurry ones**, and the confidence
score already identifies the good one. Averaging the bad frames in *dilutes* the
signal — which is exactly the pattern in the table, where voting loses a visit on
`1M-7776`, `3A-1311` and `4A-1483` alike.

### Honest limits of this corpus
- **Only 6 distinct vehicles**, and one (`1M-7776`) is 110 of 140 visits — hence the
  macro column, which is the number to quote.
- 27 visits were **excluded as `SKIP`**: the camera was pointed at a *phone screen*
  displaying a plate, not a vehicle. Not a gate visit.
- The corpus comes from development testing, not a deployed gate, so absolute values
  should not be read as production accuracy. **The oracle argument does not depend on
  any of this** — it is the robust part of the finding.

### Gate — PASSED
- [x] Visit grouping sane (167 visits ≥5 frames; median 21, max 246 frames)
- [x] Visit-level baseline recorded for current best-single-frame logic: **75.71% micro / 61.56% macro**
- [x] False-accept rate: not applicable at visit level here — the corpus has no
      whitelist context (6 known vehicles, no intruder set). Frame-level FAR stays 0.00%.

---

# Phase 4 — Multi-frame read fusion  ❌ **DO NOT BUILD** (refuted by Phase 3, 2026-07-23)

> **Status: cancelled on evidence.** Phase 3 measured this phase's premise before it
> was built — which is exactly why the plan ordered them this way — and the premise
> does not hold.
>
> | | |
> |---|---|
> | Claimed here | "the biggest free accuracy gain available", "likely worth more than the next CRNN fine-tune" |
> | **Measured** | **Maximum possible gain: 3 visits (2.1 pp).** Both natural implementations score *worse* than the current logic (73.57% vs 75.71%). |
>
> The current `best_conf` selector already reaches **106 of the 109-visit oracle
> ceiling (97%)**, and **22% of visits contain no correct read in any frame**, so no
> fusion can reach them. See the Phase 3 table above for the per-plate breakdown and
> for why the "use all 40 frames" intuition fails (a visit is one good view plus many
> blurry ones; confidence already finds the good one).
>
> **Where the effort should go instead:** the binding constraint is the model, not
> the aggregation — 31 of 140 visits are wrong in *every* frame. That is **Phase 5**
> (more real training data), which the CER curve already shows working.
>
> **The one piece worth keeping:** requiring *k* agreeing frames before
> `ENTRY_ALLOWED` is a *safety* mechanism, not an accuracy one, and it was never
> tested by the accuracy benchmark above. It remains the only identified mitigation
> for the confidently-wrong false open documented in Phase 2 (`3E-8990` read for
> `3A-8990` with the wrong character at 0.9985). Consider it on its own merits as a
> small safety change — not as the accuracy project described below.
>
> The original reasoning is preserved below for the record.

### Why (original — premise now refuted)
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

### ✅ TOOLING DONE (2026-07-23) — but the plan pointed at the wrong pool

Implemented as `scripts/recognition/harvest_active.py`
(`--scan` / `--sheets` / `--merge --apply` / `--check-mix`).

**The pool this phase was written against does not support a fine-tune.** The plan
said to mine `photos/` + `plates.db` ("7,341 audit reads, 6,122 evidence photos").
Phase 3 measured that pool:

- after visit de-duplication it holds **~6 distinct vehicles** (one is 110 of 140 visits),
- 226 visits are the camera pointed at a **phone screen**, not a vehicle,
- and decisively: **none of those 6 vehicles appear in the 149-frame test set**, so
  any gain from labelling them would be **unmeasurable**.

It is development testing, not gate traffic. Labelling it would add hundreds of
near-duplicates of the same few plates.

**The pool that does support a fine-tune was already on disk:**

| Split | crops | labelled | unlabelled |
|-------|-------|----------|-----------|
| train | 1803 | 473 | **1330** |
| valid | 643 | 0 | **643** |
| test | 436 | 149 | 287 — **never touched** |

**1,973 unlabelled real number-crops** from the diverse Roboflow dataset, already
extracted. This is the same kind of data whose labelling drove the CER curve
94.89% → 25.93% → 20.32% → 10.21% (0 → 143 → 324 → 473 labels), which has never
plateaued.

### Active-learning ranking — and an unplanned synergy with Phase 1

Ranking the 1,973 unlabelled crops by how much the model is struggling:

| Priority | Reason | Count | Share |
|---|---|---|---|
| 1 | **format-invalid** (Phase 1 grammar proves the read is wrong) | **604** | 30.6% |
| 2 | low-confidence (below the REC-005 gate) | **11** | 0.6% |
| 3 | mid-confidence | 159 | 8.1% |
| — | model-confident + valid | 1199 | 60.8% |

**604 format-invalid versus 11 low-confidence.** Confidence — the classic
active-learning signal — finds almost nothing, because (as Phase 2 established) this
model's errors are *confident*, not uncertain. **Phase 1's grammar, built for the
gate, turned out to be the selection signal Phase 5 needed.** That synergy was not
planned.

### Labelling batch actually produced
84 top-priority crops transcribed by eye → **48 labels (57% yield)**, all 48 passing
the Phase 1 grammar as a self-check. Merged via `--merge --apply`:
**473 → 521 train labels (+10.1%)**, 670 total.

The 36 rejects are informative: many priority-1 crops are format-invalid because the
**crop itself is bad** — a radiator grille, motion blur, or in two cases an *"ANCHOR"
beer advertisement* — not because a readable plate was misread. So the 615 headline
overstates the usable yield; expect ~57%, i.e. **~350 usable labels** in the queue.

**Split breakdown (matters more than it looks):** the harvest draws from *both*
train and valid, so the 48 labels landed as **35 train + 13 valid**. But
`finetune_crnn.py` selects rows with `--match "/train/"`, so **only the 35 train
labels were actually used for training**:

| | before | after | used by the fine-tune |
|---|---|---|---|
| train | 473 | **508** (+7.4%) | yes |
| valid | 0 | 13 | **no** |
| test | 149 | 149 | never |

So the effective training-set growth was **+7.4%**, not the +10.1% that the raw
"+48 labels" figure suggests. Worth remembering when planning the next batch: about
a quarter of harvested labels land outside the training split.

### Honest expectation for this fine-tune
+7.4% labels is a much smaller step than the historical ones (324 → 473 was +46%
and bought −10 pp CER). A gain this size may well be **within run-to-run variance**.
The run is being done to *measure* that, not because a large gain is expected — and
it writes to `models/recognition/crnn_ft_phase5.pth` so the deployed
`crnn_finetuned.pth` stays intact for an A/B. (`finetune_crnn.py` gained an `--out`
flag for this; it previously always overwrote the deployed weights.)

### Fine-tune result (2026-07-23) — candidate NOT shipped

50 epochs, ~1 h on the RTX 3050. Internal val CER reached **7.05%** (from a 91.59%
synthetic baseline) — but the internal val split is carved from the *train* labels
and is not comparable across runs. The decisive number is the **unchanged 149-frame
test set**:

| Metric | Deployed `crnn_finetuned.pth` | Candidate `crnn_ft_phase5.pth` |
|--------|------------------------------|-------------------------------|
| CER (isolated) | 10.21% | **10.11%** |
| **Word accuracy (isolated)** | **72.48%** | **70.47%** |
| Number e2e accuracy | **67.79%** (101/149) | 66.44% (99/149) |
| Number CER (e2e) | 15.15% | **14.84%** |
| **Composed exact-match** | **68.71%** | **67.35%** |
| Confident auto-open | **100**/143 | 99/143 |
| False-accept rate | 0.00% | 0.00% |

**CER improved fractionally; every decision-relevant metric got slightly worse.**
CER is a character metric, but the gate needs an *exact* match — so word accuracy and
composed exact-match are what count, and both fell.

**Verdict: keep the deployed weights.** The candidate stays on disk as
`models/recognition/crnn_ft_phase5.pth` for reference; it is not referenced by any
config.

**Read this honestly:** a 2-frame difference on n=149, from one seed, is *within
run-to-run variance*. The correct claim is **not** "the new model is worse" — it is
"**there is no evidence it is better**", and swapping a deployed model requires
evidence of improvement. The prediction made before the run — that a single-digit
percentage increase in labels would land inside the noise — is what happened.

**What this quantifies:** the labelling effort actually required. The historical
steps that moved CER were **+46%** label increases (324 → 473 bought −10 pp). This
was **+7.4%** and bought nothing. Reaching the next real improvement needs on the
order of **+200–350 train labels**, which is most of the usable harvest queue (~350
after the 57% yield rate, and remembering ~a quarter land in valid rather than
train) — roughly **8–10 more labelling batches** of the size done here.

### Labelling progress (cumulative — do NOT retrain until the threshold)

| Batch | Crops reviewed | Labels kept | Yield | Train labels after |
|-------|---------------|-------------|-------|--------------------|
| 1 | 84 | 48 (35 train + 13 valid) | 57% | 508 |
| 2 | 112 (queue 84–195) | 79 (3 dropped as unrepresentable) | 73% | 566 (+93, +19.7%) |
| 3 | 84 (queue 196–279) | 59 (1 dropped) | 71% | 606 (+133, +28.1%) |
| 4 | 84 (queue 280–363) | 53 (1 dropped) | 64% | 642 (+169, +35.7%) |
| 5 | 84 (queue 364–447) | 48 (3 dropped) | 61% | **679 (+206 vs 473, +43.6%)** |

**Threshold cleared at batch 5.** Cumulative growth is now **+206 train labels
(+43.6%)** — on par with the historical **324 → 473 (+46%)** step that bought −10 pp
CER, which is exactly why a fine-tune is finally worth running. A second fine-tune
(`crnn_ft_phase5b.pth`) is running on the 679-label set; result recorded below when it
lands. The deployed weights stay untouched until it clears the same A/B bar Phase 5's
first candidate failed.

**Do not fine-tune between batches.** Phase 5 established that a step this size lands
in run-to-run noise; retraining after every batch just reproduces that null result and
burns an hour each time. This campaign labelled 5 batches (287 kept of 448 reviewed,
64% mean yield) and trained **once** at the end.

**Recurring drop reason across the campaign:** ~8% of otherwise-legible plates are
**vanity/special plates with a `.` separator** (`JJ.88`, `DR.GLOW`, `T.N.9889`,
`HSR.9898`, `BNR.1616`, `GUD.LCUK`, `SELA.GTR`, `KEO.NARA`, `MPA.3689`, `M.SAM99`…).
The CRNN charset has no dot, so these are unrepresentable and were dropped rather than
mislabelled with a dash. Letter-only vanity plates (`SAMNANG9`, `WENVANNA`, `LUCKY`,
`SINGLE`, `KANHA585`, `KUNTHY33`, `VANNRIYA`, `AMATAK99`) were kept — they pass the
grammar and the charset represents them. **If these dotted plates matter at the target
gate, the fix is a charset+grammar change, not more labelling** — noting it here as a
scoped follow-up.

**Batch-2 note — the CRNN charset can't represent dots.** Three legible plates were
*dropped*, not labelled: `JJ.88` and `DR.GLOW` (dot separator — `.` is not in
`CHARSET`, so a label would train a wrong mapping) and a bare `168` (ambiguous
3-digit). `SAMNANG9` and other letter-only vanity plates were kept — they are in the
charset and pass the grammar. Yield rose 57% → 73% because batch 2 reached into the
mid-confidence band, where crops are less often pure garbage than the priority-1 tail.

### Gate — PASSED as a decision not to ship
- [x] New CER recorded against the **unchanged** 149-frame test set (10.11%)
- [x] Frame-level benchmark re-run on the candidate; visit-level unaffected
      (it replays recorded reads from the deployed model, not the candidate)
- [x] False-accept rate still 0.00% for both
- [x] **Leakage check documented.** `data/crnn_crops/test/` is excluded
      unconditionally by `HARVEST_SPLITS`; `--merge` refuses any test-split row and
      `--check-mix` re-verifies (0 leaked). Separately, the 640×640 Roboflow replays
      were excluded from the Phase 3 visit corpus for the same reason.
- [x] **Selection-bias guard.** Hard cases are 6.7% of the merged train set (`--check-mix`
      warns above 50%), so they are mixed into the existing labels rather than
      replacing them.

---

# Phase 6 — Deployment hardening

Deferred deliberately until the accuracy work lands, because none of it changes
the metric that decides whether the gate opens.

### 6.1 Admin panel authentication ✅ **DONE 2026-07-23**

`scripts/system/admin_web.py` had **no authentication at all** — anyone on the LAN
could add, suspend or delete whitelist plates, i.e. decide who the gate opens for.

**Implemented (stdlib only, no new dependency):**
- **PBKDF2-HMAC-SHA256**, 200k iterations, per-install random salt. Only the hash is
  stored, in `configs/admin_auth.json` (**added to `.gitignore`**). Plaintext is never
  written to disk. Set it with `--set-password` (prompts, or reads
  `ALPR_ADMIN_PASSWORD`); minimum 8 characters.
- **Server-side sessions** — a `secrets.token_urlsafe(32)` in an `HttpOnly`,
  `SameSite=Strict` cookie, 8-hour expiry. Nothing sensitive is stored in the browser,
  and `SameSite=Strict` blocks cross-site POSTs to the mutating routes.
- **Constant-time verification** (`hmac.compare_digest`) so timing can't leak the hash.
- **Brute-force lockout** — 5 failures per client IP, then 5 minutes locked, and a
  *correct* password is still refused while locked.
- **Fails closed on the mutating routes**: an unauthenticated POST gets a flat `401`
  rather than a redirect, so a script can't be bounced into a login page and retried.
- **Refuses to start without a password.** `--no-auth` exists for a localhost demo but
  **refuses to bind to a non-loopback address**, so the panel cannot be exposed
  unauthenticated by mistake.
- Defensive headers on every response: `X-Frame-Options: DENY` (a framed DELETE button
  is trivial clickjacking), `X-Content-Type-Options`, `Referrer-Policy`, `Cache-Control: no-store`.

**Verified (9 checks, all passing):**

| # | Check | Result |
|---|-------|--------|
| 1 | start with no password | refuses, prints instructions |
| 2 | `--no-auth` with `--host 0.0.0.0` | refuses |
| 3 | `GET /` with no session | login page |
| 4 | **`POST /delete` with no session** | **401** |
| 5 | `POST /login` wrong password | 401 |
| 6 | `POST /login` correct | 303 + `HttpOnly; SameSite=Strict` cookie |
| 7 | `GET /` with session | dashboard |
| 8 | `GET /logout` | session invalidated |
| 9 | 6 wrong logins, then the *correct* one | 429 — lockout holds |

**Before first use:** run `python scripts/system/admin_web.py --set-password`. No
password is currently set (the one used for testing was deleted), so the panel will
refuse to start until you set one.

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

### 6.4 Retune the 2.2 consistency check ✅ **DONE 2026-07-23 — turned OFF**

Retuned against the Phase 0 baseline **with province ground truth** — the validation
2.2 was always missing. The conclusion is that it cannot be usefully tuned, because
the signal carries no information.

**1. The flag is no better than chance.** Measured over the 117 confident reads that
have province ground truth (the population where 2.2 actually decides DENY vs REVIEW):

| | |
|---|---|
| base rate of wrong reads | 17/117 = **14.5%** |
| precision of the 2.2 flag | 2/14 = **14.3%** |
| **lift over random selection** | **0.98×** |

Identical against number-only and composed correctness.

**2. `align` is binary, not continuous — there is no threshold to tune.**

| align value | reads |
|---|---|
| 0.0 (no overlap at all) | 17 |
| < 0.25 | 4 |
| ≥ 0.75 (contained) | 122 |

Nothing falls between 0.25 and 0.75, so **every threshold from 0.05 to 0.75 produces
exactly 14 flags and identical behaviour.** The swept parameter never mattered.

**3. The province branch is effectively dead.** `province_confidence < 0.55` fires on
**1 of 143** reads (minimum observed 0.399).

**4. The reported 45.45% precision was an artefact of the instrument.**
`process_frame` checks REC-005 *before* the 2.2 branch, so a low-confidence read goes
to REVIEW regardless and the flag changes nothing for it. The benchmark was counting
those: 22 flagged / 45.45% precision over all detected reads, versus **15 flagged /
20.0%** over the confident reads where 2.2 decides the outcome. `benchmark_composed.py`
now counts only the confident population.

**Decision: `gate.consistency_check: false`.** Cost of leaving it on was ~10% of
confident reads (12 per 117) — unregistered vehicles sent for a human REVIEW instead
of a clean DENY, for no benefit. Accuracy and safety metrics are byte-identical with
it off (composed 68.71%, FAR 0.00%, 100 auto-opens); only the spurious flags
disappear (22 → 0). One line to revert; the thresholds are kept in the config.

**Correction to a claim in `IMPROVEMENT_ROADMAP.md`:** that doc states 2.2
"addresses the 1.2 cross-plate finding". **It cannot.** `process_frame` checks
`is_reg` *before* the consistency branch, so a read that exact-matches a different
registered plate is ALLOWED and never reaches 2.2. It never was a false-open
mitigation. (The `k`-agreeing-frames idea salvaged from Phase 4 remains the only
identified one.)

**Caveat:** n=14 flags is small, so the 0.98× lift has a wide interval. But the point
estimate sits exactly on the base rate, no threshold changes the outcome, and the
underlying signal is binary — three independent reasons not to keep it.

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

---

## Where things actually landed (2026-07-23)

Phases 0–3 are done. Of the three code changes this plan proposed, **one shipped,
two were refuted by their own gates** — which is the plan working as intended.

| Phase | Outcome |
|-------|---------|
| 0 Restore the instrument | ✅ Composed exact-match **68.71%** measured for the first time; 7 of 149 province labels were wrong |
| 1 Format validation | ✅ **Shipped.** 8 silent DENYs → REVIEW, 0 correct opens lost, FAR still 0 |
| 2 Confidence recalibration | ❌ **Refuted.** `min` is *worse* than `mean`; all modes within noise. Default unchanged |
| 3 Visit benchmark | ✅ Built — and it refuted Phase 4 |
| 4 Multi-frame fusion | ❌ **Cancelled.** Ceiling 2.1 pp; both implementations lose to current logic |
| 5 Active-learning harvest | ✅ Tooling built + 48 labels merged. Fine-tune ran: **no evidence of improvement, candidate not shipped.** Quantified the real cost: **+200–350 labels needed**, not +48 |

**The consistent signal across all four phases: the CRNN itself is the binding
constraint, and no amount of decode-time or aggregation-time cleverness moves it.**

- Phase 0: every province error landed on a frame whose number was already wrong —
  the number branch is the whole composed gap.
- Phase 2: the false open is *confidently* wrong (0.9985 on the wrong character), so
  no confidence statistic can catch it.
- Phase 3: 22% of visits are wrong in *every single frame*, so no fusion can fix them.

All three point the same way: **Phase 5 (more real training data) is the only
remaining lever**, and the project's own CER curve (94.89% → 10.21% as labels grew
0 → 473, never plateauing) is the evidence it works. Phase 6.3's hard-condition
collection matters for the same reason — `1A-4249` and `2-8554` fail in every frame
because nothing like them is in the training set.
