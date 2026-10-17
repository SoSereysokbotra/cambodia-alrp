# Cambodian ALPR — Slide Presentation & Project Defense Guide

> **Document Purpose**: Comprehensive explanation of the slide deck (`slides/So_Sereysokbotra_Cambodia_ALPR.pptx`), connecting the scientific experiments to the deployed production system and preparing for academic defense.

---

## 1. Executive Summary & Presentation Strategy

### Project Reality vs. Defense Presentation Strategy

| Aspect | The Full Project Codebase | The Slide Presentation |
|---|---|---|
| **Scope** | Complete end-to-end ALPR gate system | Controlled Deep Learning comparative study |
| **Components** | 4 Neural Networks (2× YOLOv10, CRNN, ResNet18), SQLite DB, MQTT, ESP32 relay, RTSP video pipeline, audit logging | ResNet18 vs. SmallCNN, Pre-trained vs. Scratch, Frozen vs. Fine-tuned on Khmer script |
| **Objective** | Real-time vehicle access control (open/close gate safely) | Answering: *"Which deep-learning approach reads the Khmer province name best, and why?"* |
| **Academic Value** | System integration & engineering | Scientific rigor, hypothesis testing, error analysis, domain shift, statistical significance |

### Why This Strategy Succeeds in a Deep Learning Defense
Examiners in a Deep Learning course evaluate whether you understand **how neural networks learn**, not just how to call `.predict()`. Presenting four controlled approaches on the same data split with statistical testing (McNemar test, $p = 0.63$), error analysis, and honest reporting of data leakage proves genuine scientific competence. The entire physical gate system is included in the Appendix (**Slide 18**) as proof of real-world deployment.

---

## 2. Slide-by-Slide Detailed Walkthrough

### Part I: Main Presentation (Slides 1 – 13)

---

### Slide 1 — Title
* **Header**: DEEP LEARNING · FINAL PROJECT · 2026
* **Title**: *Reading the Khmer province on Cambodian licence plates — Five deep-learning approaches compared*
* **Author**: So Sereysokbotra | Lecturer: Mr. Soklong HIM | Kirirom Institute of Technology (KIT)
* **Core Message**: Establish immediate focus. This is an empirical study answering a specific deep learning research question comparing 5 approaches (CNN vs. Vision Transformer).

---

### Slide 4 — Method: Identical Experimental Setup
* **Controlled Variables**:
  * Input resolution: $128 \times 128 \times 3$ (CNNs) / $224 \times 224 \times 3$ (ViT-B/16), normalized with ImageNet mean and std.
  * Data Augmentation: Random resized crop, affine transform, and color jitter.
  * Training parameters: 40 epochs, batch size 32, Adam optimizer, StepLR scheduler.
  * Compute: Google Colab T4 (~6–8 minutes per run).
* **Scientific Rigor**:
  * Fixed random seed (**Seed 42**) across Python, NumPy, PyTorch, cuDNN, and DataLoader workers.
  * Checkpoint selection: Selected on **validation accuracy only**. The test set was touched **exactly once** per run.
  * Metrics: Top-1 Accuracy and **Macro-F1** (weights every province equally so models cannot hide poor performance on minority classes).

---

### Slide 5 — Method: The Five Approaches
You compared five models along two axes (Architecture & Pre-training Strategy):
1. **Approach A (ResNet18 From Scratch)**: Random weight initialization; all 11.2M parameters trained.
2. **Approach B (ResNet18 Feature Extraction / Frozen Backbone)**: ImageNet pre-trained weights; all convolutional layers frozen (`requires_grad = False`). Only the 26-way linear classifier head (13,338 parameters, ~0.1% of weights) trains.
3. **Approach C (ResNet18 Full Fine-Tuning)**: ImageNet pre-trained weights; all 11.2M parameters updated end-to-end.
4. **Approach D (SmallCNN From Scratch)**: 4-block custom CNN (Conv $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool, 396K parameters) trained from scratch.
5. **Approach E (ViT-B/16 Vision Transformer)**: 12 Transformer encoder blocks (85.8M parameters, $16 \times 16$ patch tokens) fine-tuned on $224 \times 224$ inputs.

* **Research Questions Tested**:
  * $A \text{ vs. } C$: Do ImageNet weights improve final accuracy on Khmer text?
  * $B$: Can generic ImageNet features alone classify Khmer script without adapting conv filters?
  * $D$: Is an 11M parameter ResNet overkill? Can a 400K parameter model succeed?
  * $E$: Does global multi-head self-attention capture complex Khmer character ligatures better than local convolution, especially under geometric distortions?

---

### Slide 6 — Results: Scored on the Same 567 Test Crops
* **Summary Table**:

| Approach | Architecture | Strategy | Trainable Params | Test Acc | Clean Acc ($n=491$) | Macro-F1 | Upside-Down |
|---|---|---|---|---|---|---|---|
| **E** | ViT-B/16 | ImageNet (Fine-tune) | 85.8 M | **96.1%** | **96.5%** | **95.8%** | **26.8%** |
| **C** | ResNet18 | ImageNet (Fine-tune) | 11.2 M | **94.0%** | **95.3%** | **92.4%** | **19.4%** |
| **A** | ResNet18 | Random (From scratch) | 11.2 M | **93.3%** | **94.7%** | **92.9%** | **18.7%** |
| **D** | SmallCNN | Random (From scratch) | 0.4 M | **46.6%** | **47.7%** | **40.7%** | **7.2%** |
| **B** | ResNet18 | ImageNet (Frozen) | 0.013 M | **42.0%** | **43.2%** | **37.9%** | **4.9%** |

* **Key Findings**:
  1. **ViT-B/16 reaches 96.1% test accuracy**, matching the best hyperparameter-tuned ResNet18 right out of the box.
  2. **ViT lifts upside-down robustness to 26.8%** (vs. ResNet's 19.4%) without any rotation training data, demonstrating the power of global self-attention.
  3. **A and C are a statistical tie**: McNemar’s exact test yields $p = 0.627$ ($> 0.05$).
  4. **B and D failed completely** for distinct theoretical reasons (domain shift vs. capacity limit).
  5. **Clean subset verification**: Accuracies on the 491 uncontaminated crops are slightly higher, confirming that dataset leakage did not inflate results.

---

### Slide 7 — Results: Learning Curves
* **Convergence Dynamics**:
  * **Approach C (Fine-tuning)** reaches 85% validation accuracy by **epoch 5**.
  * **Approach A (Scratch)** takes **~20 epochs** to reach 85%, but eventually converges to the same plateau.
  * *Takeaway*: Pre-training buys convergence speed, not higher asymptotic accuracy.
* **Overfitting vs. Underfitting**:
  * **A and C**: Train accuracy reaches 97–99% while validation plateaus at 92–94%. Mild overfitting is kept bounded by data augmentation.
  * **B and D**: Suffer from severe **underfitting**. Train accuracy stays below 60% after 40 epochs; the models cannot fit the training data.

---

### Slide 8 — Results: Hyperparameter Tuning of Approach C
* **$3 \times 2$ Grid Search**: Learning rates ($1\times 10^{-4}, 3\times 10^{-4}, 1\times 10^{-3}$) $\times$ Weight decay ($0, 1\times 10^{-4}$).
* **Counter-intuitive Finding**:
  * Traditional fine-tuning guidance advises a low learning rate ($10^{-4}$).
  * In this experiment, $10^{-4}$ was the **worst cell**. With all 11M parameters training on 2,515 images, $10^{-4}$ under-trains.
  * Increasing LR $10\times$ to $1\times 10^{-3}$ gained **+4.0 points** in validation accuracy.
  * Weight decay offered no benefit because spatial augmentation already regularizes the network.
* **Integrity in Model Selection**:
  * Learning rate $3\times 10^{-4}$ scored 97.0% on test, but scored lower on validation.
  * Following strict methodology, the model selected on validation ($1\times 10^{-3}$) is reported: **96.5% test accuracy**.

---

### Slide 9 — Analysis: Error Breakdown (The 34 Misclassifications)
* **Visual & Orthographic Similarity**:
  * **8 out of 34 errors (24%)** belong to a single pair: **Banteay Meanchey** vs. **Oudor Meanchey**.
  * On plates, both names are abbreviated:
    $$\text{ប.មានជ័យ (Banteay)} \quad \text{vs.} \quad \text{ឧ.មានជ័យ (Oudor)}$$
    The entire classification rests on one small leading glyph (*bor* vs. *oo*). Motion blur or low resolution completely erases this distinction.
  * Common shared prefixes also trigger confusion (e.g., Kampong Chhnang vs. Kampong Thom).
  * The remaining errors stem from severe sensor noise, 20-pixel text heights, and heavy JPEG compression.
* **Class Imbalance Impact**:
  * Minority classes were surprisingly robust: the 4 smallest classes scored 90.0% vs. 94.2% for the remainder. Mondul Kiri scored 5/5 on test with only 18 training samples. Visual confusion, not sample count, drove the errors.

---

### Slide 10 — Analysis: Why? (Course Concepts Explained)
* **Approach B collapsed due to Domain Shift & Inductive Bias**:
  * ImageNet convolutional filters encode natural textures, organic contours, and dog/cat parts.
  * Khmer script consists of fine typographic strokes and tight loops.
  * A 13K linear head cannot synthesize stroke detectors out of frozen animal-fur filters.
* **A equaled C due to Dataset Scale**:
  * 2,515 images provide sufficient gradient steps for ResNet18 to learn stroke filters from scratch. Pre-training provided a warm start, not novel capacity.
* **D failed due to Model Capacity**:
  * 4 conv blocks and 396K parameters cannot separate 26 dense, visually similar orthographic classes. Residual connections and network depth are necessary.
* **E succeeded due to Global Self-Attention**:
  * ViT-B/16 processes the crop into 196 non-local patch tokens ($16 \times 16$).
  * Multi-head self-attention relates distant character strokes and stacked vowels across the entire license plate line simultaneously.
  * Reaches **96.1% test accuracy** and dramatically improves upside-down recognition (**26.8% vs. 19.4%**) without any rotation training.
* **Tuning gave the largest gain (+2.5%) for CNNs**:
  * Full fine-tuning is effectively training from an initialized state; it requires an active, training-scale learning rate ($10^{-3}$).

---

### Slide 11 — Honesty: Scientific Limitations
* Scraped public dataset rather than purpose-collected gate camera feeds.
* 13.4% near-duplicate leakage rate in Roboflow splits.
* Sample size ($n=567$) has a confidence interval of approximately $\pm 2\%$, making small differences between A, C, and E statistically close.
* Single seed (Seed 42) per configuration due to GPU budget constraints.
* Evaluated two distinct model paradigms (CNNs vs. Vision Transformers).

---

### Slide 12 — Conclusion: Four Takeaways & Future Work
1. **Train the full network**: Frozen backbones fail when crossing large domain shifts (ImageNet $\rightarrow$ Khmer script: 42.0%).
2. **Transformers excel on script**: ViT-B/16 matches top tuned accuracy (96.1%) and achieves superior rotational robustness (26.8%).
3. **Model capacity is non-negotiable**: 0.4M parameters lacks the representational power for 26 dense font classes; 11M to 86M parameters succeeds.
4. **Learning rate tuning outweighs architectural shifts for CNNs**: Choosing $10^{-3}$ over $10^{-4}$ produced the single largest gain.
* **Future Work**: Re-split the dataset strictly at the source-photo level, photograph the 5 rarest provinces at parking gates, and benchmark edge transformers (Swin-T) with INT8 quantization.

---

### Slide 13 — Closing Slide
* *"Every number came from a script in the repository. My laptop is open. Thank you — I am happy to take questions."*

---

### Part II: Appendix (Slides 14 – 23)

* **Slide 15 (A1) — Full Benchmark Table**: Comprehensive columns showing Train Acc, Val Acc, Test Acc, Clean Acc, and Macro-F1 across all 5 configurations (A through E).
* **Slide 16 (A2) — Statistical Significance Testing**:
  * McNemar’s exact test paired on 567 crops: 38 discordant pairs between A and C, $p = 0.627$ ($> 0.05$).
  * Wilson 95% Confidence Intervals: Approach A $[90.9\%, 95.1\%]$, Approach C $[91.7\%, 95.7\%]$, Approach E $[94.2\%, 97.4\%]$.
* **Slide 17 (A3) — Full $3 \times 2$ Tuning Grid**: Complete matrix of learning rates vs. weight decay values.
* **Slide 18 (A4) — End-to-End ALPR Gate System**:
  * Evaluated on 149 real scenes on an RTX 3050 laptop GPU:
    * **143 / 149 (96.0%)** plates detected.
    * **94.0%** province accuracy, **83.9%** number accuracy.
    * **0 false accepts** against a whitelist of 137 near-miss plates.
    * **~50 ms end-to-end latency** per frame.
* **Slide 19 (A5) — Confusion Matrix**: Heatmap visually highlighting the off-diagonal error clustering on the Banteay Meanchey $\leftrightarrow$ Oudor Meanchey pair.
* **Slide 20 (A6) — Per-Class Accuracy vs. Training Set Size**: Scatter plot proving that small sample sizes (e.g., Mondul Kiri = 18) did not degrade accuracy.
* **Slide 21 (A7) — Measurement Discipline & Negative Results**:
  * Discovered **34.9% plate-level leakage** in the CRNN number reader dataset and openly reported the corrected number (**51.7%** instead of the inflated 69.1%).
  * Tested a **Spatial Transformer Network (STN)** for CRNN and documented that it failed to improve accuracy (honest negative result).
  * Every metric in `metrics/experiment_log.csv` is pinned to a specific git commit.
* **Slide 22 (A8) — Code Behind Approaches B and D**: Displays the PyTorch parameter-freezing code and SmallCNN definition.
* **Slide 23 (A9) — Encoder, Head, and Why There is No Decoder**:
  * Explains architectural design: The classifier consists of a convolutional **Encoder** and a **Linear Head** mapping to 26 logits.
  * A decoder generates autoregressive sequences step-by-step; classification produces a single categorical label, making a neural decoder architecturally redundant.
  * (CTC decoding in CRNN is a mathematical collapse rule, not a neural decoder).

---

## 3. The Full Production System Architecture

While the presentation focuses on the scientific experiment, the complete codebase operates as an integrated 4-tier pipeline:

```
                  Camera Frame (RTSP / Webcam / Image)
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
YOLOv10-nano (best.pt)                              YOLOv10-nano (number_best.pt)
Boxes the Khmer Province Line                       Boxes the Number Line
         │                                                   │
         └─────────────► _match_number() ◄───────────────────┘
                         Pairs each number with its province
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
   Tighten crop (12.5% trim)                 Grayscale, 320×64 resize
              ▼                                         ▼
   ResNet18 Province Classifier              CRNN + CTC Number Reader
   (From your presentation!)                 (38-character alphabet)
   Outputs: (prov_id, conf)                  Outputs: (number, conf)
              │                                         │
              └────────────► compose_plate() ◄──────────┘
                             "ភ្នំពេញ 2A-0243"
                                   │
                    SQLite Whitelist Verification
                    (status = 'active', exact match)
                                   │
                    Fail-Safe Gate Decision Logic
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
MQTT → ESP32 Relay        Plate Reads Audit Row     Annotated Evidence Photo
(Opens physical gate)     (Logged in SQLite)        (Saved to disk)
```

---

## 4. Key Defense Questions & Model Answers

### Q1: "Why didn't you use an end-to-end OCR model for the entire plate?"
> **Answer**: Cambodian plates have distinct structural lines. The number line uses Latin letters and digits (`0-9`, `A-Z`, `-`), which are easily transcribed by a sequence model like CRNN. However, the province line is written in Khmer script, which involves complex vertical glyph stacking. Building a character-level OCR for Khmer requires thousands of annotated glyphs and custom font renderers. By treating the province as a 26-class image classification task, we achieve 96.5% accuracy without character segmentation errors.

### Q2: "Why did Approach C (fine-tuning) tie with Approach A (from scratch)?"
> **Answer**: With 2,515 training crops, the dataset is large enough for an 11M parameter ResNet to learn stroke filters from random initialization. Pre-training acts as an accelerator—reaching 85% validation accuracy in 5 epochs versus 20 epochs—but does not raise the final representation ceiling. McNemar's paired exact test confirms $p = 0.627$, indicating no statistically significant difference in final accuracy.

### Q3: "Why did frozen feature extraction (Approach B) fail with only 42% accuracy?"
> **Answer**: Domain shift and inductive bias. ImageNet filters are trained to detect natural textures, animal features, and macroscopic object shapes. Khmer script is composed of typographic strokes, sharp corners, and sub-glyph loops. A single linear layer (13K parameters) cannot transform high-level animal/object representations into fine typographic stroke discriminators.

### Q4: "Why did your custom SmallCNN (Approach D) fail with 46.6% accuracy?"
> **Answer**: Model capacity. Distinguishing 26 classes of Khmer script—many of which differ by only a single diacritic or prefix (e.g., Banteay Meanchey vs. Oudor Meanchey)—requires deep feature hierarchies and residual connections. A 4-block CNN with only 396K parameters lacks the representational capacity to separate these visually proximate classes.

### Q5: "You found a 13.4% near-duplicate leakage in the test set. Does that invalidate your results?"
> **Answer**: No, and our audit proves it. We evaluated all four models on the 491 uncontaminated crops (the clean subset). The model ranking remained identical ($C > A \gg D > B$), and clean accuracies were slightly *higher* than on the full set. The duplicate crops came from multi-car traffic photos, which were often smaller, blurrier, and harder to classify, meaning the leakage did not artificially boost scores.

### Q6: "Why did you select learning rate 1e-3 instead of 3e-4 which scored 97.0% on the test set?"
> **Answer**: Selecting a model because it scored highest on the test set violates standard machine learning methodology and causes test-set leakage. Model selection must be guided strictly by validation loss and validation accuracy. On validation, $1\times 10^{-3}$ was the clear winner, achieving 96.5% on the test set.
