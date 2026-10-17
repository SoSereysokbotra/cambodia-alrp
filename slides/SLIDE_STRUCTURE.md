# Slide structure — 10-minute presentation (rubric §6.2 order)

Build the deck from this file (PowerPoint / Google Slides), export to
`slides/presentation.pdf`. 14 main slides + 5 appendix. Target **9:30**.

Every number below is measured and lives in `results/province_study/` or
`metrics/experiment_log.csv`. Figures referenced are in
`results/province_study/figures/`.

Legend per slide: **TITLE** · ON-SLIDE TEXT (what the audience reads) · FIGURE ·
SAY (what you say out loud; do not put this on the slide) · time.

---

## Slide 1 — Title  (0:20)

**Khmer Province Classification for Cambodian License Plates**
*A comparison of four deep-learning approaches*

- Your full name · Bachelor of Software Engineering, Year 2
- Deep Learning — Final Project · Lecturer: Mr. Soklong HIM · 2026
- GitHub: `sobotra/<repo-name>`

FIGURE: one real plate photo (e.g. a test scene from `data/annotated/test/images/`) — crop it so the three lines are visible.

SAY: "I built a plate-reading gate system; the part I'll examine today is one model inside it: how to read the Khmer province line, and which training approach works best."

---

## Slide 2 — Problem & motivation  (0:45)

**Why the province, and why classification?**

- A Cambodian plate has 3 lines: **Khmer province · number · English province**
- `2A-0243` exists once *per province* → a gate must read the province too
- Khmer script: complex glyph shaping, little data → not a good fit for character-by-character OCR
- So: **whole-image classification** — 128×128 crop of the Khmer line → 1 of 26 classes (25 provinces + *other*)
- Research question: **on a small, imbalanced Khmer-script dataset, which architecture and training strategy is best — and why?**

FIGURE: diagram: plate photo → detector box on the top line → 128×128 crop → "ភ្នំពេញ (Phnom Penh)".

SAY: Explain the three lines; stress that the number alone does not identify the car.

---

## Slide 3 — Dataset  (0:45)

**Data: Plate_v4 (Roboflow, CC BY 4.0) — public, not self-collected**

- 3 299 Cambodian plate photos → province-line crops, 128×128
- Split (identical for every approach): **2 515 train / 529 val / 567 test**
- 26 classes, **heavily imbalanced**: *other* 453, Phnom Penh 254 … Mondul Kiri 18, Ratanakiri 20
- Known problem, measured: **13.4 %** of test crops share a source photo with train (split was made at crop level) → every result also reported on the **clean subset (n = 491)**

FIGURE: bar chart of training crops per class (make from the README table, or `data/province_crops/manifest.json`).

SAY: "I audited my own test set. The leak did not inflate the numbers — clean-subset accuracy is actually higher — but I report it."

---

## Slide 4 — Pipeline & preprocessing  (0:30)

**Same preprocessing and augmentation for all four approaches**

- Test-time: resize 128×128 → ImageNet mean/std normalisation
- Train-time augmentation (all runs): RandomResizedCrop (0.6–1.0) · RandomAffine ±6°, shift 12 % · ColorJitter 0.2
- 40 epochs · batch 32 · Adam · StepLR ×0.5 / 10 epochs · **seed 42** (Python, NumPy, PyTorch, cuDNN deterministic)
- Model selection on **validation** only; test touched once per run

FIGURE: 3–4 example augmented crops side by side (optional).

SAY: "Nothing differs between approaches except what the rubric calls the approach."

---

## Slide 5 — The four approaches  (1:00)

**Two dimensions: training strategy (A/B/C) and architecture (D)**

| run | architecture | initial weights | what trains | trainable params |
|---|---|---|---|---|
| **A** | ResNet18 | random | everything | 11.19 M |
| **B** | ResNet18 | ImageNet | **only the final layer** (frozen backbone) | 13 338 (0.12 %) |
| **C** | ResNet18 | ImageNet | everything, low LR (fine-tuning) | 11.19 M |
| **D** | small 4-block CNN | random | everything | 0.40 M |

FIGURE: simple block diagram — ResNet18 with a lock icon on the backbone for B; the 4-block CNN for D.

SAY: "A vs C asks whether ImageNet weights help. B asks whether ImageNet features alone are enough. D asks whether we even need 11 million parameters."

---

## Slide 6 — Experimental setup  (0:30)

**Metrics, tuning, hardware**

- Metrics: **top-1 accuracy** (primary), **macro-F1** (imbalance), per-class accuracy, confusion matrix — all on the same 567 crops
- Hyperparameter grid on approach C: lr {1e-4, 3e-4, 1e-3} × weight decay {0, 1e-4} = 6 runs
- Hardware: Google Colab **Tesla T4**, ≈ 5–6 min per run; 10 runs ≈ 1 hour total
- Every run saves `history.csv`, `run.json`, `test_predictions.csv` → all figures rebuilt by one script

SAY: one sentence each; don't linger.

---

## Slide 7 — Results table  (1:00)

**Test results — same 567 crops**

| run | strategy | best epoch | val acc | **test acc** | clean subset | macro-F1 | time |
|---|---|---|---|---|---|---|---|
| **A** | ResNet18 scratch | 38 | 93.8 % | **93.3 %** | 94.7 % | 92.9 % | 6.1 min |
| **B** | ResNet18 frozen | 39 | 40.1 % | **42.0 %** | 43.2 % | 37.9 % | 5.3 min |
| **C** | ResNet18 fine-tune | 31 | 92.6 % | **94.0 %** | 95.3 % | 92.4 % | 5.9 min |
| **D** | small CNN scratch | 33 | 48.4 % | **46.6 %** | 47.7 % | 40.7 % | 5.3 min |

- A ≈ C (McNemar p = 0.63 — statistically tied)
- B and D fail for **different reasons** (next slides)

FIGURE: `fig1_test_accuracy.png` on the right half.

SAY: "Two clear winners, two clear failures. The interesting part is *why*."

---

## Slide 8 — Learning curves  (1:00)

**Convergence, over-fitting, under-fitting**

- **C** reaches 85 % val by epoch 5; **A** needs ~20 epochs → ImageNet buys **speed**, not a better end point
- **A, C:** train 97–99 % vs val 92 % — mild over-fitting, bounded by augmentation; best-val epoch is kept
- **B, D:** train *and* val stay low (47 % / 59 % train) → **under-fitting**, still rising slowly at epoch 40

FIGURE: `fig2_learning_curves.png` full width.

SAY: Point at the green curve (C) shooting up, the blue (A) catching up, the orange (B) and red (D) flat.

---

## Slide 9 — Hyperparameter tuning  (0:45)

**Approach C: learning rate × weight decay (selected on validation)**

| lr | wd = 0 | wd = 1e-4 |
|---|---|---|
| 1e-4 | 92.6 % *(= run C)* | 93.2 % |
| 3e-4 | 95.3 % | 94.9 % |
| **1e-3** | **96.6 % ← selected** | 96.0 % |

- The textbook "low LR for fine-tuning" was the **worst** cell: 40 epochs × StepLR at 1e-4 under-trains
- Weight decay slightly hurts (−0.4 to −0.6) — augmentation already regularises
- **Final model: lr 1e-3, wd 0 → 96.5 % test accuracy** (+2.5 over untuned C)
- lr 1e-4 / wd 0 reproduces run C to the digit → seeding verified

FIGURE: `fig6_tuning_heatmap.png`.

SAY: "lr 3e-4 has a higher *test* score — 97.0 — but lower validation. I pick on validation, so I report 96.5."

---

## Slide 10 — Error analysis  (1:00)

**Where the 34 errors of C come from**

- **8 of 34 (24 %) are one pair: Banteay Meanchey ↔ Oudor Meanchey** — on plates both are abbreviated `ប.មានជ័យ` / `ឧ.មានជ័យ`; only the first glyph differs
- Other confusions share a word: Battambang ↔ Pursat, Kampong Chhnang ↔ Kampong Thom, Kampong Speu ↔ Kampong Thom
- Most remaining failures: motion blur, ~20-px text, JPEG blocks — unreadable to a human too
- Class size matters less than expected: ≤ 50-crop classes 90.0 % vs 94.2 %; Mondul Kiri (18 crops) 5/5 correct

FIGURE: `fig5_failures_C_resnet18_finetune.png` (left) + `fig4_per_class_C_resnet18_finetune.png` (right, or appendix).

SAY: "Visual similarity of the *name*, not the amount of data, drives the errors."

---

## Slide 11 — Why the winners won (course concepts)  (1:00)

**Explaining the ranking**

- **Frozen backbone collapses (42 %) → inductive bias / domain shift.** ImageNet filters encode natural-image textures and parts; Khmer stroke topology needs the *low-level filters* to change. A 13 k-parameter head cannot fix that.
- **Fine-tune ≈ scratch (94 vs 93 %) → data size.** 2 515 images are enough to learn glyph features from random init; pretraining is a good initialisation, not new knowledge.
- **Small CNN under-fits (47 %) → capacity.** 0.4 M params / 4 blocks cannot separate 26 near-identical Khmer words at 128 px; depth + residual connections matter.
- **Tuning (+2.5) → optimisation.** Full fine-tuning is just training; it wants a training-size LR.

FIGURE: none, or a small 2×2 "concept → evidence" grid.

SAY: This slide is the "referring to the concepts covered in the course" requirement — name each concept explicitly.

---

## Slide 12 — Limitations  (0:30)

**What this study cannot claim**

- Public scraped dataset, not purpose-collected; no tilt/lighting labels
- 13.4 % near-duplicate rate in the test split (reported; clean-subset scores given)
- 567 test crops → roughly **± 2 points** at 95 % (binomial); A vs C is *not* separable
- One seed per configuration (variance not measured)
- Only two architecture families; ResNet18 only for the strategy comparison

SAY: Say it before they ask.

---

## Slide 13 — Conclusion & future work  (0:30)

**Takeaways**

1. For Khmer glyph crops, **train the whole network** — frozen ImageNet features do not transfer
2. Pretraining buys **4× faster convergence**, not accuracy, at this data size
3. **Capacity matters**: 0.4 M params is not enough; 11 M is
4. The biggest single gain came from **tuning the LR**, not from the architecture

**Next, in priority order:** re-split by source photo and photograph the 5 smallest provinces · deploy the rotation-trained variant (93 % upside-down, measured) · a third family (ConvNeXt-T / small ViT) at the same budget

SAY: end on the closing line: "Every number here comes from a script you can run."

---

## Slide 14 — Thank you / Q&A  (0:10)

- Repo link · `results/province_study/` · `README.md`
- "Laptop is open — I can run any experiment or show any line."

---

# Appendix (only if asked)

## A1 — The ALPR system this model lives in
Pipeline diagram from README §8; four models; end-to-end on 149 real scenes: plate found 143/149, **number 83.9 %, province 94.0 %**, both 82.6 %; zero false accepts on 137 near-miss plates, one cross-plate misread in 143.

## A2 — Confusion matrix
`fig3_confusion_C_resnet18_finetune.png`.

## A3 — Test-set leakage in the CRNN split (the other model)
`check_leakage.py`: 34.9 % of CRNN test crops shared a plate with training; headline upside-down score fell 69.1 → 51.7 after cleaning. Cite Laroca et al. 2023 (46.9 % / 67.6 % in AOLP; 19.1 % in CCPD).

## A4 — A negative result: the Spatial Transformer
`check_stn.py`: the STN in the deployed CRNN predicts a near-identity transform for upright *and* flipped input; upside-down reading (0 → 51.7 %) came from augmentation. Reported as a failed component.

## A5 — Reproducibility
Commands from README §7; `metrics/experiment_log.csv` (every measurement tagged with its git commit); seeds; `last.pth` + `--resume`.

---

# Timing check

| slides | minutes |
|---|---|
| 1–2 problem | 1:05 |
| 3–4 data & pipeline | 1:15 |
| 5–6 approaches & setup | 1:30 |
| 7–9 results & tuning | 2:45 |
| 10–11 errors & why | 2:00 |
| 12–14 limits, conclusion, thanks | 1:10 |
| **total** | **9:45** |

Rehearse twice with a timer. If over: cut slide 4 to 15 s and merge slide 6 into 5.
