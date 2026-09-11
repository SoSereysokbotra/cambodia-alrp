# Presentation Guide — Cambodian ALPR

> **Purpose:** everything you need to defend this project in front of your teacher.
> Every number below was produced by a script in this repo that anyone can re-run.
>
> **The core principle: don't ask them to believe you — run it in front of them.**
>
> Last measured: 2026-08-24.

---

## 1. The demo — run these live, in this order

Always from the project root, inside the venv.

### Command 1 — "My test set is honest"

```
.venv\Scripts\python.exe scripts\tools\check_leakage.py --audit
```

**What appears:** 52 of the 149 test crops (34.9%) share a plate number with the
training data, plus 4 byte-identical duplicate images.

**What to say:**
> "Before trusting any accuracy number, I audited my own test set. A third of it
> was contaminated — the same physical plate appeared in both training and
> testing under different filenames. My path-based guard never caught it because
> the filenames differ. I found it, removed the contaminated training crops, and
> re-measured everything."

**Why it lands:** most student projects — and the published Cambodian ALPR paper
this work is compared against — never run this check at all.

---

### Command 2 — "Here is my real accuracy"

```
.venv\Scripts\python.exe scripts\tools\score_pipeline.py
```

**What appears:**

```
  scenes scored        : 149
  detector found       : 143   (missed 6)
  NUMBER correct       :  83.9%   (125/149)
  PROVINCE correct     :  94.0%   (140/149)
  BOTH correct         :  82.6%   (123/149)
```

…preceded by every single mistake, listed with its filename, ground truth, and
what the model actually read.

**What to say:**
> "This runs the complete pipeline — detector, number reader, province
> classifier — over 149 photos with a ground-truth answer key. It prints every
> error by name. A detector miss counts as a failure, so nothing is hidden by
> scoring the stages separately."

**If asked about the 83.9%:** break it down (see §3) — it is **89.7%** on
standard provincial plates. Seven of the eighteen number errors are on novelty
plates whose ground truth is text like `HENGHENG` or `COVI19`.

---

### Command 3 — "The rotation training worked"

```
.venv\Scripts\python.exe scripts\tools\test_rotation_reading.py --weights models\recognition\crnn_finetuned.pth
.venv\Scripts\python.exe scripts\tools\test_rotation_reading.py --weights models\recognition\crnn_stn5.pth
```

**What appears:**

| model | upright | upside-down |
|---|---|---|
| `crnn_finetuned.pth` (original) | 80.5% | **0.0%** |
| `crnn_stn5.pth` (this work) | 87.2% | **51.7%** |

**What to say:**
> "Same script, same 149 held-out crops, back to back. The original model reads
> zero upside-down plates. Mine reads just over half — and upright accuracy went
> up, not down."

---

### Command 4 — "I test my own claims, including the ones that failed"

```
.venv\Scripts\python.exe scripts\tools\check_stn.py --weights models\recognition\crnn_stn5.pth
```

**What appears:** a large `FAIL`.

**What to say:**
> "My design included a Spatial Transformer — a layer meant to rotate the plate
> upright before reading. I wrote a tool to verify it actually does that. It
> doesn't. It predicts a near-identity transform whether the input is upright or
> flipped, so it never learned to rotate anything. The rotation ability came
> entirely from the training augmentation, not from that layer. I am reporting
> the component as a negative result rather than claiming it as the mechanism."

**Why it lands:** this is the single most credible thing in the presentation.
Anyone can show a success. Showing a tool you built that proves your own idea
failed is what separates measurement from marketing.

---

### Command 5 (optional) — the visual demo

```
.venv\Scripts\python.exe main.py demo --limit 20
```

Shows the system working on real photos, with boxes and text.

> **Never demo by pointing a camera at a laptop screen.** Photographing a monitor
> adds moiré patterns, glare and colour shift that the models never trained on.
> It makes a working system look broken and tells you nothing. Feed image files
> directly, or point the camera at a real plate on a real vehicle.

---

## 2. Headline results

### End-to-end, upright (149 labelled scenes, full pipeline)

| stage | result |
|---|---|
| detector found the plate | 143 / 149 (96.0%) |
| number correct | 83.9% (125/149) |
| province correct | 94.0% (140/149) |
| both correct | 82.6% (123/149) |

### Number reading by plate type

| plate type | accuracy |
|---|---|
| **standard provincial plates** | **89.7%** (87/97) |
| novelty / police / custom ("other") | 82.6% (38/46) |

7 of the 18 number errors are on plates whose ground truth is not a valid
Cambodian plate shape at all (`HENGHENG`, `COVI19`, `SELAGTR`, `HYWAZA9`…).

### Number reader development (149 held-out crops)

| model | change | upright | upside-down |
|---|---|---|---|
| `crnn_finetuned` | baseline, deployed at project start | 80.5% | 0.0% |
| naive flip augmentation | flip crops, no STN | 62% | 12% |
| `crnn_stn2` | STN + 20k synthetic | 79.2% | 55.0% |
| `crnn_stn4` | + photo-realistic synthetic + 199 new labels | 85.9% | 69.1% ⚠ |
| **`crnn_stn5`** | **trained on de-leaked data** | **87.2%** | **51.7%** |
| `crnn_stn6` | supervised STN (failed) | 77.9% | 37.6% |

⚠ `crnn_stn4`'s 69.1% was inflated by test-set leakage. The honest figure is
`crnn_stn5`'s 51.7%.

### Province classifier

| model | upright | upside-down |
|---|---|---|
| `province_classifier_best.pth` (deployed) | **96.1%** | 19.4% |
| `province_classifier_rot2.pth` (rotation-trained) | 92.4% | **93.3%** |

---

## 3. The five findings worth presenting

### Finding 1 — Test-set leakage inflated results by 17 points

52 of 149 test crops (34.9%) shared a plate number with training data, and 4
were byte-identical duplicates. Scoring `crnn_stn4` on each bucket separately:

| bucket | n | upright | upside-down |
|---|---|---|---|
| CLEAN (plate never in training) | 97 | 80.4% | **56.7%** |
| LEAKED (plate also in training) | 52 | 96.2% | **92.3%** |

A 35.6-point gap on the same model is the signature of memorisation, not
reading. After retraining on de-leaked data the gap fell to 9.2 points.

**This is a documented, field-wide problem, not a beginner error.** Cite:

- Laroca et al. report **46.9%** duplicate contamination in AOLP-A, **67.6%** in
  AOLP-B, **19.1%** in CCPD. Your 34.9% is inside that range.
- The ICT Express survey: contaminated splits "may measure in-domain fitting or
  memorization of repeated visual evidence rather than transferable recognition
  ability."
- Published de-duplication cost 2.7–5.2 points. **Yours cost 17.4** — worth
  stating; rotated reading appears to depend on memorisation more than upright
  reading does.

### Finding 2 — The Spatial Transformer never worked

The STN predicts, for **both** upright and flipped input:

```
theta = [[+1.09,  0.00, +0.05]        a 180° flip would be   [[-1, 0, 0]
         [ 0.00, +1.05, -0.01]]                               [ 0,-1, 0]]
```

Near-identity, and effectively identical for the two orientations — it cannot
even distinguish them. The upside-down ability came from the `--rotate180`
training augmentation alone.

Attempting to fix it by supervising the transform directly (`crnn_stn6`) failed
and cost accuracy: going from identity (+1 diagonal) to a flip (−1 diagonal)
must pass through zero, where the matrix is singular and the image collapses to
a point. Gradient descent will not cross that wall — the diagonal fell from
+1.09 to +0.55 and stalled without ever going negative.

### Finding 3 — The detector is the real bottleneck, not the reader

| scenes | detector found | number correct | province correct |
|---|---|---|---|
| upright | 143/149 (96%) | 83.9% | 94.0% |
| **upside-down** | **44/149 (30%)** | 16.8% | 3.4% |

Of the 44 inverted plates the detector *did* find, the reader got 25 right
(56.8%) — matching its 51.7% crop-level score. **The reader works; the detector
rarely gives it anything to read.** Measuring on pre-cut crops hides this
entirely.

### Finding 4 — ImageNet transfer learning gave nothing

Identical data, epochs and test split; only the transfer strategy changed:

| run | initial weights | trainable params | upright accuracy |
|---|---|---|---|
| A from scratch | random | 11,189,850 | 94.2% |
| **B feature extraction** | ImageNet | **13,338 (0.12%)** | **51.3%** |
| C fine-tuning | ImageNet | 11,189,850 | 94.5% |

Fine-tuning beat from-scratch by 0.3 points — about 2 crops out of 567, i.e.
noise. Feature extraction collapsed, and its loss curve shows **underfitting**,
not overfitting: it plateaued around 39% validation by epoch 15.

**Conclusion:** ImageNet features are built for natural photographs and do not
separate Khmer script. Domain-specific training is justified for this task.

*Caveat to state:* run B used the same aggressive augmentation as the others.
Frozen features have less capacity to absorb heavy augmentation, so B might
improve with lighter settings — though not by 43 points.

### Finding 5 — The rotation "ceiling" hypothesis was wrong

The intuitive explanation for imperfect upside-down reading is that characters
like 6 and 9 are ambiguous when rotated. **Measured, this is false:**

| plates containing 6 or 9 | upside-down accuracy |
|---|---|
| yes (n=89) | **53.9%** |
| no (n=60) | 48.3% |

Plates *with* 6 or 9 score slightly **better**. The real failure pattern is
different: of 54 crops that read perfectly upright but fail when flipped, 20
(37%) drop a character entirely, and the rest are single-character substitutions
(`4→2` six times, `1→0` five times). Only 18 of 149 crops are unreadable in both
orientations, so the addressable headroom runs to ~87%, not ~52%.

---

## 4. Likely questions and honest answers

**"Why only 149 test images?"**
> That is the test split for the number reader — the only set with a complete
> answer key. Training uses 716 real crops plus 16,000 synthetic. The detector
> trains on 2,054 scenes, the province classifier on 2,515 crops.

**"How do I know you didn't train on the test set?"**
> Two ways. `finetune_crnn.py` raises an exception if any `/test/` row reaches
> training — proof by construction, not by promise. And `check_leakage.py` goes
> further and checks for the same plate *number* appearing in both splits, which
> a path check cannot catch.

**"Your accuracy is lower than the published paper's."**
> The numbers are not comparable — different datasets, different difficulty. That
> paper also used EasyOCR off the shelf, and its own conclusion names this as its
> key limitation, listing domain-specific fine-tuning as future work. This project
> trained a domain-specific recogniser, which is that future work.

**"Why is upside-down only 51.7%?"**
> Because that is what it measures. The reader could reach ~87% if every crop it
> reads upright were also read flipped; the current failures are mostly single
> characters, not garbage. End-to-end, though, detection is the binding
> constraint at 30%.

**"Is the improvement statistically significant?"**
> Tested with McNemar's exact test on paired outcomes over the same crops.
> `crnn_stn2 → crnn_stn4` upside-down on clean crops: 4 regressions, 16 fixes,
> **p = 0.0118, significant**. `crnn_stn4 → crnn_stn5`: p = 0.13, **not**
> significant — those two are statistically indistinguishable and I do not claim
> otherwise.

**"Why not just rotate the image at inference time?"**
> That works (~76%) and is implemented, but it was deliberately rejected. The
> objective was for the *model* to learn rotation, not for surrounding logic to
> hide the problem. It stays disabled (`gate.orientation_search: false`).

---

## 5. Datasets

| model | train | validation | test |
|---|---|---|---|
| detector (`best.pt`) | 2,054 scenes | 741 | 504 |
| number reader (CRNN) | 716 real + 16,000 synthetic | 157 | **149** |
| province classifier | 2,515 crops | 529 | 567 |

**Limitation to state up front:** the data is a public Roboflow dataset, not
purpose-collected. It was not designed to be balanced, and it contained the
duplicate contamination described in Finding 1. There are no tilt-angle labels,
so skew robustness cannot be measured.

---

## 6. Reproducibility checklist

| claim | command that proves it |
|---|---|
| test set is clean | `scripts\tools\check_leakage.py --audit` |
| end-to-end accuracy | `scripts\tools\score_pipeline.py` |
| upside-down reading | `scripts\tools\test_rotation_reading.py --weights <model>` |
| province, both orientations | `scripts\tools\test_province_rotation.py --weights <model>` |
| STN does nothing | `scripts\tools\check_stn.py --weights <model>` |
| history of every measurement | `metrics\experiment_log.csv` |
| every individual error | `results\pipeline_mistakes.csv` |

---

## 7. References

1. O. Bol, B. Kem, S. Sin, "Automatic Cambodian License Plate Recognition Using
   Deep Learning With Robust Skew Handling," *ISHE 2025*, AIJR Proceedings
   Vol. 8, Issue 1, pp. 22–30, 2026. doi:10.21467/proceedings.8.1.4

2. D. Ning, D. S. Han, "Deep learning for license plate recognition: A
   comprehensive survey of datasets, methods, and future directions,"
   *ICT Express*, Elsevier, 2026.

3. R. Laroca, V. Estevam, A. S. B. Jr., R. Minetto, D. Menotti, "Do we train on
   test data? The impact of near-duplicates on license plate recognition,"
   *IJCNN 2023*, pp. 1–8. doi:10.1109/IJCNN54540.2023.10191584

---

## 8. The closing line

> "The strongest result in this project is not the accuracy. It is that I found
> a 34.9% contamination in my own test set, corrected it, watched my headline
> number fall from 69.1% to 51.7%, and reported the lower number. I also built a
> tool that proved my own Spatial Transformer does nothing, and I am reporting
> that as a negative result. Every number in this presentation comes from a
> script you can run yourself."
