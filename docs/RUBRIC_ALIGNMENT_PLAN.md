# Rubric Alignment Plan — turning the ALPR system into a gradeable comparison study

> Source of truth for what to do next. Tick boxes as you go. Written 2026-09-12
> against `DL_Final_Project_Instruction-HSL.pdf` (Mr. Soklong HIM, 50 % of grade).
> Topic is already approved. Every gap listed here was verified in the repo, not assumed.

---

## 0. The framing decision (read this first)

The rubric grades a **comparison of 2–3 distinct deep-learning approaches on the
same problem and the same test set**. The ALPR gate system is *not* that — it is
four models for four different sub-tasks.

**The graded study will be:**

> **"Khmer province classification on Cambodian plate crops: which training
> strategy and architecture works best, and why?"**

| approach | architecture | training strategy | rubric dimension |
|---|---|---|---|
| **A** | ResNet18 | from scratch (random init) | B |
| **B** | ResNet18 | ImageNet-pretrained, **frozen backbone** (linear probe) | B |
| **C** | ResNet18 | ImageNet-pretrained, **full fine-tune** | B |
| **D** (recommended, cheap) | **small CNN** (3–4 conv blocks) | from scratch | A |

A/B/C is literally the rubric's own image-classification example. D adds the
architecture dimension so the study spans both A and B of §4.

- **Dataset:** `data/province_crops/` — 26 classes, 2 515 / 529 / 567 (train/val/test),
  derived from Plate_v4 (Roboflow, CC BY 4.0). Same split for every approach.
  Class imbalance is real (train: Phnom Penh 453 crops vs Mondul Kiri 18) — this
  becomes part of the error analysis, not something to hide.
- **Metrics:** top-1 accuracy (primary), macro-F1 (because of imbalance),
  per-class accuracy, confusion matrix. Same test-time preprocessing for all
  (Resize 128 + ImageNet normalise, no trim).
- **The ALPR system** (YOLO detectors, CRNN, gate logic) becomes *motivation +
  application context*: 1 slide up front, a demo if time allows, appendix for Q&A.
- **The leakage audit, STN negative result and CRNN rotation work** stay in the
  story as "rigour" evidence — but as supporting material, not the main study.

**Why not compare the CRNN checkpoints instead?** `crnn_finetuned` vs
`crnn_stn2/4/5` differ by augmentation and data; §4 explicitly says that does not
count as a distinct approach. The STN is an architecture change, but it was
proven to be a no-op. Don't build the grade on it.

---

## 1. Make the training script rubric-compliant

File: `scripts/recognition/train_province_classifier.py`

- [ ] **Seeds** — add `--seed 42` and at the top of `main()`:
      `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`,
      `torch.backends.cudnn.deterministic = True`, `cudnn.benchmark = False`.
      Also seed the DataLoader (`generator=torch.Generator().manual_seed(seed)`).
- [ ] **`--arch {resnet18, small_cnn}`** — add a `SmallCNN` class (e.g. 4 × [Conv3×3 → BN → ReLU → MaxPool], 32→64→128→256 channels, global-avg-pool, Linear(256, 26)). Put it in `src/recognition/province_classifier.py` next to `build_resnet18` so inference can load it too.
- [ ] **`--weight-decay`** (default 0) passed to Adam — this is the "one regularization choice" the rubric requires you to tune.
- [ ] **Per-epoch CSV log** — write `results/province_study/<run>/history.csv` with columns
      `epoch, train_loss, train_acc, val_loss, val_acc, lr, epoch_sec`. (Currently only prints.) Compute **train accuracy** and **val loss** — both are missing today.
- [ ] **Run metadata JSON** — `results/province_study/<run>/run.json`: arch, strategy, seed, lr, weight_decay, epochs, batch, **trainable params**, **total params**, **training wall-time**, **GPU name** (`torch.cuda.get_device_name(0)`), best epoch, best val acc, **test acc**, **test macro-F1**.
- [ ] **Test predictions dump** — `results/province_study/<run>/test_predictions.csv`
      (`image, true_class, pred_class, confidence`) so confusion matrices and error galleries can be built offline without re-running.
- [ ] **Checkpoint resume** — save `last.pth` every epoch (model + optimizer + epoch) and add `--resume` to continue after a Colab disconnect. (`best.pth` is already saved.)
- [ ] Keep the **eval transform identical** for all runs (already true — do not add trim here).
- [ ] Use `--no-framing-aug` for the study? **Decision: no.** Use the *same* train augmentation for all four runs (framing aug ON, as deployed) so augmentation is not a confound. State this in the README.

**Definition of done:** one command runs one approach end-to-end and leaves
`history.csv`, `run.json`, `test_predictions.csv`, `best.pth`, `last.pth` in its
results folder. Test locally with `--epochs 1` before going to Colab.

---

## 2. Run the four approaches (Colab T4, or local RTX 3050)

Notebook: `notebooks/colab_transfer_learning.ipynb` (already exists — update it to
the new flags and **keep its outputs saved** this time).

Fixed for all runs: `--epochs 40 --batch 32 --seed 42`, same split, same augmentation.

- [ ] A `--arch resnet18` (no `--pretrained`)                          `--lr 1e-3`
- [ ] B `--arch resnet18 --pretrained --freeze`                       `--lr 1e-3`
- [ ] C `--arch resnet18 --pretrained`                                `--lr 1e-4`
- [ ] D `--arch small_cnn`                                            `--lr 1e-3`
- [ ] Copy each run folder back into `results/province_study/` and **commit it**
      (CSV/JSON/PNG are small; `.pth` files are 35–45 MB — under the 50 MB rule, so they *may* be committed, or upload to Drive and put the link in the README).
- [ ] Append one row per run to `metrics/experiment_log.csv` (`component=province_study`).
- [ ] Re-run `make_colab_bundle.py` first so Colab gets the new script.

Expected budget: ResNet18 on 2.5 k × 128 px images ≈ 5–10 min per 40-epoch run on a T4. Whole phase < 1 h.

---

## 3. Hyperparameter tuning on the best approach (expected: C)

- [ ] Grid: `lr ∈ {1e-4, 3e-4, 1e-3}` × `weight_decay ∈ {0, 1e-4}` = 6 runs, 40 epochs, seed 42. (Optional: +2 runs with `--seed 1, 2` on the winner to show variance.)
- [ ] Select by **validation** accuracy, then report test accuracy of the selected config once. Never select on test.
- [ ] Write `results/province_study/tuning.csv` (one row per run) and a short paragraph: what you tried, what won, what you observed (e.g. "1e-3 with pretrained weights diverged / over-fit by epoch 12").
- [ ] If the tuned C beats the deployed `province_classifier_best.pth` on the 567 test crops, promote it (copy + config JSON) and re-run `score_pipeline.py` — otherwise leave deployment alone. Log either way.

---

## 4. Figures (rubric: at least two comparison figures)

New script: `scripts/tools/make_study_figures.py` → `results/province_study/figures/`

- [ ] `fig1_test_accuracy_bar.png` — A/B/C/D test accuracy + macro-F1, error bars if seeds were repeated.
- [ ] `fig2_learning_curves.png` — overlaid train vs val **loss** and **accuracy** per approach (2×2 grid or two panels). This is where you *show* under-fitting (B plateaus) vs over-fitting (A: train ≫ val).
- [ ] `fig3_confusion_matrix_C.png` — normalised, 26×26, for the winner.
- [ ] `fig4_per_class_accuracy.png` — bar chart sorted by train-set size, so imbalance ↔ accuracy is visible.
- [ ] `fig5_failure_gallery.png` — 12–16 worst misclassified test crops with true/pred labels.
- [ ] `fig6_tuning_heatmap.png` — lr × weight_decay → val accuracy.
- [ ] Every figure: axis labels, units, legend, title, readable at slide size. Same colour per approach across all figures.

---

## 5. Error analysis and discussion (write once, reuse in README + slides)

- [ ] Per-class table from `test_predictions.csv`: which provinces fail, and are they the low-data classes (Mondul Kiri 18, Ratanakiri 20, Kep 42)? Quantify the correlation.
- [ ] Look at the failure gallery: name the causes (visually similar Khmer names, blur, box catching the number line, "other" class absorbing province plates).
- [ ] Explain the ranking with course concepts:
      - B fails → ImageNet features are built for natural-image textures; Khmer glyph shapes need re-learned low-level filters (inductive bias / domain shift).
      - C ≈ A → 2.5 k images are enough to learn from scratch; pretraining mainly speeds convergence (show epochs-to-90 % from the curves).
      - D vs A → capacity: does a 0.3 M-param CNN match an 11 M-param ResNet on 128 px crops?
- [ ] **Limitations** paragraph: public dataset not purpose-collected; 34.9 % near-duplicate rate found in the *CRNN* split — state whether the province split was audited the same way (**do this audit**: same plate number in train and test province crops? `check_leakage.py` logic applied to `province_crops`); class imbalance; 567-image test set → ±3–4 pt CI; no tilt labels.
- [ ] **Future work**, prioritised: collect real photos for the 5 smallest classes; rotation-trained classifier (`rot2`, already measured 93 % upside-down) into the pipeline; a ViT / ConvNeXt as a third architecture.

---

## 6. README rewrite (rubric §6.1 — every bullet is checked by the marker)

Rewrite `README.md` in this order. Keep the current system content but move it *below* the study.

- [ ] Title + **your name** + course + lecturer.
- [ ] Problem statement: **input → output** ("128×128 RGB crop of the Khmer province line → one of 26 classes"), problem type (multi-class image classification), why it matters (ALPR context, 2 sentences).
- [ ] Dataset: source + **license** (Plate_v4, CC BY 4.0), how crops were produced (`build_province_dataset.py`), counts per split, **class-distribution table or figure** (from `manifest.json`), known noise/bias (imbalance, near-duplicates, "other" class). Fix the false sentence "photographed and labelled for this project".
- [ ] Approaches compared: the A/B/C/D table with params, training time, hardware.
- [ ] **Results table** (single table, same test set, same metrics) + the two headline figures embedded.
- [ ] Hyperparameter tuning summary + link to `tuning.csv`.
- [ ] Error analysis + limitations (short version, link to full).
- [ ] **How to run every experiment** — one command per approach + the figure script + `score_pipeline.py` for the system.
- [ ] Weights: Drive links (or note they are committed).
- [ ] Citations: Plate_v4, Ultralytics YOLOv10, torchvision ResNet18, `third_party/yolov10`, the three papers in PRESENTATION_GUIDE §7.
- [ ] **AI-use disclosure** (mandatory): tools used (Claude via Claude Code), scope (code drafting/refactoring, docs, debugging, label transcription of *training* crops via montage sheets, analysis write-ups), verification (all numbers re-run from scripts; test labels human-verified; you can explain/modify the code). Be specific and honest — the repo's docs make the AI involvement obvious, and the rubric says fabrication/undisclosed use = zero.
- [ ] Keep the ALPR system section, but update the stale end-to-end table to today's `score_pipeline.py` numbers (83.9 / 94.0 / 82.6) and fix `metrics/week2_metrics.json` by re-running `scripts/detection/evaluate.py`.

---

## 7. Slides (rubric §6.2 — 10–20 slides, 10 minutes, fixed order)

Create `slides/` and export **PDF** (plus PPTX if you have it). Suggested 14 slides:

- [ ] 1 Title / name
- [ ] 2 Problem & motivation — ALPR gate context, the 3-line plate, input→output
- [ ] 3 Dataset — source, license, split, class-distribution chart
- [ ] 4 Pipeline — preprocessing, augmentation (same for all), test-time transform
- [ ] 5 Approaches A/B/C/D — one diagram, params table, why these four
- [ ] 6 Experimental setup — metrics, seeds, hardware, epochs, tuning grid, runtimes
- [ ] 7 Results table
- [ ] 8 Fig: test accuracy bar
- [ ] 9 Fig: overlaid learning curves + over/under-fitting reading
- [ ] 10 Fig: confusion matrix + per-class vs data size
- [ ] 11 Error analysis — failure gallery, causes
- [ ] 12 Discussion — why C (or A) won, with course concepts; why B collapsed
- [ ] 13 Limitations (incl. the leakage story — 1 line, it's your credibility slide)
- [ ] 14 Conclusion & prioritised future work
- [ ] Appendix: ALPR system architecture, end-to-end 83.9 %, STN negative result, CRNN rotation 0 → 51.7 %, McNemar, live demo command
- [ ] Rehearse to **≤ 9:30**. Every figure labelled. No slide with more than one table.

---

## 8. Repository hygiene

- [ ] `results/province_study/` committed with CSV/JSON/PNG (`.gitignore` currently ignores nothing there — verify after adding).
- [ ] `requirements.txt` — add `scikit-learn` (macro-F1, confusion matrix) and `pandas` if the figure script uses them; pin versions (`pip freeze` the ones you import).
- [ ] Remove or move clutter that a marker will open: `alpr_colab_bundle.zip` (151 MB, in root — gitignored? check), `yolo26n.pt`, `scripts/tools/make_label_sheet.py.bak`, stale `models/recognition/*.bak.pth` (git-ignored anyway).
- [ ] Mark stale docs at the top with one line: `PROJECT_OVERVIEW.md`, `HANDOFF.md`, `HANDOFF_ROTATION_UPSIDEDOWN.md` → "superseded by README / PRESENTATION_GUIDE (2026-09-12)".
- [ ] **Commit after every phase** with a meaningful message. 10 of your 23 commits are on one day; from now on, small regular commits.
- [ ] Final check: fresh clone → `pip install -r requirements.txt` → run approach C for 1 epoch → figure script runs. If that works, "anyone can reproduce" is true.

---

## 9. Q&A preparation

- [ ] Be able to open and explain, line by line: `SmallCNN`, the ResNet head swap, the freeze loop, the training loop, the eval transform, `CTCDecoder._collapse`, `process_frame` stage 4.
- [ ] Prepare answers for the 16 questions in the previous review (CI on 149 / 567 images; why B fails; what the STN was; leakage; cross-plate false open; "has it opened a real gate?" → no).
- [ ] Have `score_pipeline.py`, `check_leakage.py --audit`, `check_stn.py` ready to run live — they are your proof.

---

## Order of work and rough effort

| phase | effort | blocks |
|---|---|---|
| 1 script changes + 1-epoch local test | ½ day | everything |
| 2 four runs on Colab | ½ day (mostly waiting) | 3, 4, 5 |
| 3 tuning grid | ½ day (waiting) | 4, 5 |
| 4 figures script | ½ day | 6, 7 |
| 5 error analysis + province leakage audit | ½ day | 6, 7 |
| 6 README | ½ day | — |
| 7 slides + rehearsal | 1 day | — |
| 8 hygiene + fresh-clone test | ¼ day | — |

≈ 4–5 working days. Phases 1–3 are the ones that move the 50 % criterion; do them first even if time runs short on the rest.

**Deadline fields (fill in):** final repo submission Week 14 = ________ ; presentation Week 15 = ________ .
