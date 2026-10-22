# Cambodian ALPR — Presentation Script

> **Slide Deck Reference:** [`So_Sereysokbotra_Cambodia_ALPR.pptx`](file:///d:/Year2/Semester2/Deep%20Learning/Cambodian%20ALPR%20Project/slides/So_Sereysokbotra_Cambodia_ALPR.pptx) & [`So_Sereysokbotra_Cambodia_ALPR.pdf`](file:///d:/Year2/Semester2/Deep%20Learning/Cambodian%20ALPR%20Project/slides/So_Sereysokbotra_Cambodia_ALPR.pdf)  
> **Speaker:** So Sereysokbotra  
> **Institution:** Kirirom Institute of Technology (KIT)  
> **Target Duration:** ~9:15 speaking time at 140 words per minute (~45 seconds buffer under 10-minute limit)  
> **In-Room Delivery:** Use **Presenter View (Alt + F5)** in PowerPoint to see these notes directly alongside each slide.

---

## Script Guide
- **Bold** = key numbers, metrics, or phrases to land on with vocal emphasis.
- `[Brackets]` = stage directions, pointing gestures, or screen transitions (do not read aloud).

---

## Slide 1 — Title & Introduction  *(0:25)*

Good morning, teachers and colleagues. My name is **So Sereysokbotra**.

My project addresses Automated License Plate Recognition for Cambodia. While I built a complete end-to-end system that detects a plate at a barrier gate and decides whether to grant access, today I want to focus on **one central model inside that system**, and answer one question: **Which deep learning approach recognizes Khmer province names best, and why?** To answer this, I empirically benchmarked **five deep-learning approaches** along two fundamental dimensions.

---

## Slide 2 — The Real-World Access Control Problem  *(0:45)*

`[Point to the use-case cards on the slide]`  
Across Cambodia, automated security gates, residential Boreys, smart parking garages, and expressway toll plazas require rapid, hands-free vehicle entry.

Commercial off-the-shelf ALPR systems fail in Cambodia due to one critical reality: **the plate number alone is not globally unique**. A registration number like `2A-0243` repeats across almost every one of Cambodia’s 26 provinces. If a security barrier reads only the Latin characters and digits, an unauthorized vehicle from another province could falsely trigger gate entry.

Furthermore, standard Western OCR engines cannot read the top line because **Khmer script has complex vertical stacked ligatures** that break traditional character-by-character segmentation.

`[Point to the 128x128 crop diagram]`  
Therefore, I frame Khmer province recognition as a **whole-image classification problem**: a 128-by-128 pixel crop in, and one of **26 discrete province classes** out. Our research question is: on a small, real-world imbalanced Khmer dataset, **which deep-learning approach reads this line best, and why?**

---

## Slide 3 — End-to-End System Design Flow  *(0:45)*

`[Walk across the 5 pipeline stages from left to right]`  
Here is how our complete automated access control system operates:

1. **Camera Capture**: A road or gate camera captures the incoming vehicle.
2. **Plate Detection**: A **YOLOv10** detector locates the license plate bounding box in **15 milliseconds**, resilient to angle, distance, and lighting shifts.
3. **Dual Parallel Recognition**: The cropped plate enters two specialized recognition branches simultaneously:
   - The top Khmer province line is classified by our **26-class deep neural network** — the focus of today's study.
   - The bottom line of alphanumeric characters is parsed by a **CRNN with CTC decoding**.  
   *Running both models in parallel keeps overall recognition under 35 milliseconds.*
4. **Whitelist & Decision Logic**: The two predictions form a canonical string such as `Takeo 1L-4082`, which is verified against an **SQLite database whitelist** with strict confidence scoring.
5. **Gate Control**: On 149 real gate-camera test scenes, the full system achieves **zero false accepts** with a total latency of **50 milliseconds per frame**.

---

## Slide 4 — Dataset & Audit Discipline  *(0:40)*

The data is derived from **Plate_v4**, an open Cambodian license plate dataset on Roboflow under CC-BY license. **I did not hand-collect the raw images myself.**

By cropping the Khmer province strip from each plate, we obtain:
- **2,515 training crops**
- **529 validation crops**
- **567 test crops**  
*Every single model in this benchmark is trained and evaluated on this exact split.*

`[Emphasize]` Rather than assuming dataset perfection, I audited the split. The original split was performed per crop rather than per vehicle photo. Because a single photo can capture multiple vehicles, **13.4% of test crops share a source photo with the training set**. I measured this overlap, I report it transparently, and every model is evaluated both on the full test set and on the 100% leak-free clean subset.

---

## Slide 5 — Class Distribution & 25:1 Imbalance  *(0:45)*

`[Point to the distribution table]`  
Here is the distribution across all 26 classes.

The dataset exhibits severe class imbalance: Phnom Penh and "Other" have over 400 crops, while remote provinces like Mondul Kiri have only **18 total crops** — an extreme **25-to-1 ratio**.

`[Point to the red rows at the bottom]`  
The bottom four provinces — Koh Kong, Kep, Ratanakiri, and Mondul Kiri — have 50 or fewer total examples.

Because of this long-tail distribution, overall accuracy alone is misleading: a naive model could score high by simply ignoring rare provinces. Therefore, I track **macro-F1**, which weights all 26 classes equally and severely penalizes failures on minority provinces.

---

## Slide 6 — Experimental Setup & Controlled Variables  *(0:40)*

To ensure scientific rigor, every variable except the model approach itself is held strictly constant.

- **Preprocessing & Augmentation**: Identical 128×128 resolution, normalization, and modest data augmentation (random rotation ±7°, color jitter, perspective warp).
- **Optimization**: 40 epochs, batch size 32, Adam optimizer.
- **Reproducibility**: **Random seed 42** is fixed across Python, NumPy, PyTorch, and DataLoader workers.
- **Discipline**: The best checkpoint is chosen purely by **validation accuracy**; the **test set is evaluated exactly once** per run.
- **Environment**: All experiments ran on a Google Colab Tesla T4 GPU, averaging approximately **6 minutes** per training run.

---

## Slide 7 — Five Deep-Learning Approaches Compared  *(0:50)*

I compared five distinct approaches across two axes: **transfer learning strategy** and **architectural paradigm**:

1. **Approach A (ResNet18 from scratch)**: Random initialization; all 11.2 million parameters are learned directly on our Khmer crops.
2. **Approach B (ResNet18 linear probe / frozen backbone)**: Pretrained ImageNet weights; backbone frozen, training **only the 13,000 parameters** in the final linear classifier.
3. **Approach C (ResNet18 full fine-tuning)**: Pretrained ImageNet initialization; all layers updated at a low learning rate.
4. **Approach D (Custom 4-block CNN from scratch)**: A lightweight convolutional model with **400,000 parameters**, testing whether heavy parameter depth is necessary.
5. **Approach E (Vision Transformer — ViT-B/16)**: An **86-million-parameter transformer** using multi-head self-attention over 16×16 non-overlapping image patches.

*A vs. C evaluates pretraining benefit; B evaluates frozen feature transferability; D evaluates parameter efficiency; and E evaluates attention mechanisms against convolutions.*

---

## Slide 8 — Empirical Results on the Test Set  *(0:55)*

`[Point to the results table]`  
Here are the benchmark results evaluated on the 567 test crops:

- Full fine-tuning (**Approach C**) achieves **94.0% accuracy** (89.5% macro-F1).
- Training from scratch (**Approach A**) achieves **93.3% accuracy** (87.7% macro-F1).
- The frozen backbone (**Approach B**) **collapses to 42.0% accuracy**.
- The lightweight 400K CNN (**Approach D**) stalls at **46.6% accuracy**.
- The Vision Transformer (**Approach E**) achieves the highest raw performance at **96.1% accuracy** (92.4% macro-F1), while maintaining **26.8% accuracy on upside-down plates** without any rotation training.

`[Crucial statistical point]`  
Are A and C different? I conducted **McNemar’s exact test** on paired predictions and obtained **p = 0.63**. Statistically, **A and C are tied**.

On the clean subset without multi-car overlap, the exact ranking holds and accuracies are slightly higher. This confirms that test leakage did not artificially inflate scores — duplicate multi-car crops are actually more distant and blurred.

---

## Slide 9 — Convergence & Learning Curves  *(0:50)*

`[Point to the green curve (C) and blue curve (A)]`  
Comparing the training dynamics reveals how these models learn:

- **Approach C (Fine-tuning)** reaches **85% validation accuracy in just 5 epochs**.
- **Approach A (Scratch)** eventually reaches the same accuracy, but requires **20 epochs**.  
*Conclusion: Pretrained weights provide convergence speed, not a higher performance ceiling, when dataset size is sufficient.*

Both A and C show slight overfitting (training accuracy 98%, validation 92%), controlled by data augmentation.

`[Point to orange (B) and red (D)]`  
Approaches B and D exhibit classic **underfitting**, not overfitting. Their training curves flatten below 60%. These models lack either the adaptable filters (B) or the representational capacity (D) to fit Khmer script.

---

## Slide 10 — Systematic Hyperparameter Tuning  *(0:50)*

`[Point to the 3×2 grid]`  
Next, I performed a grid search on Approach C across 3 learning rates ($10^{-4}, 3\times 10^{-4}, 10^{-3}$) and 2 weight decay levels ($0, 10^{-4}$).

- Conventional transfer learning textbooks recommend using a very low learning rate like $10^{-4}$. In our grid, **$10^{-4}$ was the worst performing cell**.
- Increasing the learning rate tenfold to **$1e-3$ improved validation accuracy by 4.0 percentage points**. Because all layers are unfrozen on a small dataset, lower learning rates under-train the network within 40 epochs.
- Adding weight decay slightly degraded performance across all columns because geometric data augmentation already provides effective regularization.

Our selected model ($lr = 10^{-3}$, weight decay $= 0$) achieves **96.5% test accuracy**.

---

## Slide 11 — Error Analysis & Failure Modes  *(1:00)*

`[Point to the error confusion card]`  
Our fine-tuned model produces only **34 errors out of 567 test samples**.

- **24% of all errors (8 samples) stem from a single province pair: Banteay Meanchey vs. Oudor Meanchey.**  
  On Cambodian plates, both names are abbreviated as *Bor-Meanchey* and *Oo-Meanchey*. The **entire classification hinges on a single initial character**, which disappears under low resolution or motion blur.
- Similar errors occur between provinces sharing root words, such as *Kampong Chhnang* vs. *Kampong Thom*.
- `[Point to the visual gallery]` The remaining misclassifications occur on extreme samples: severe motion blur, night-time glare, or heavy occlusion.

`[Highlight key takeaway]`  
Class size is **not** the primary error driver. The four smallest classes achieve **90.0% accuracy** versus 94.2% for common classes, and Mondul Kiri achieved 5 out of 5 correct on test. **Visual stroke ambiguity, not sample scarcity, causes the errors.**

---

## Slide 12 — Theoretical Explanations  *(0:55)*

Why did the models rank this way? We can explain each result using core deep learning concepts:

1. **Why B collapses (Inductive Bias & Domain Shift)**: ImageNet filters encode natural textures, edges, and animal contours. Khmer text consists of stroke geometries and stacked vowels. A frozen 13,000-parameter head cannot alter low-level spatial feature extractors.
2. **Why A equals C (Sample Efficiency)**: 2,500 domain-specific crops provide sufficient gradient signal for ResNet18 to learn specialized character kernels from scratch.
3. **Why D fails (Network Capacity)**: 400,000 parameters lack the representational depth needed to discriminate 26 visually dense Khmer ligatures.
4. **Why E succeeds (Global Self-Attention)**: ViT-B/16 captures long-range dependencies across the entire word shape, allowing it to recognize global letter patterns and tolerate spatial inversions.
5. **Why tuning mattered most**: Full fine-tuning is effectively full training; it requires an active learning rate ($10^{-3}$) to converge.

---

## Slide 13 — Honest Limitations & Threats to Validity  *(0:25)*

In the spirit of honest scientific evaluation, this study has known limitations:

1. **Dataset Origin**: The dataset is web-scraped and open-source, not captured under fully controlled real-world gate conditions.
2. **Test Overlap**: 13.4% of crops originate from multi-car images present in training, although clean subset evaluation shows no artificial inflation.
3. **Statistical Margin**: With 567 test samples, the 95% confidence interval is approximately **±2.0%**, which is why A and C are statistically indistinguishable.
4. **Hardware Scope**: Evaluations were conducted on a single random seed across two model families (CNN vs. ViT).

---

## Slide 14 — Conclusions & Future Work  *(0:35)*

Four actionable conclusions emerge from this study:

1. **Train the full network**: Frozen ImageNet backbones fail on non-Latin scripts (42.0% accuracy). Low-level features must adapt.
2. **Vision Transformers excel at script**: ViT-B/16 achieves 96.1% accuracy and superior rotational invariance (26.8% upside-down).
3. **Model capacity is required**: 400K parameters is insufficient; 11M to 86M parameters provides necessary expressive power.
4. **Tuning beats architecture**: For CNNs, optimizing learning rate yielded a larger gain (+2.5%) than switching from scratch to pretraining.

**Immediate Next Steps**: Re-split the dataset strictly by source vehicle, collect targeted crops for the 5 smallest provinces, and evaluate hybrid mobile architectures like MobileNetV4 and Swin Transformer.

---

## Slide 15 — Close & Q&A  *(0:15)*

Every metric, table, and figure presented today is generated from reproducible scripts in the project repository. My code and models are open for inspection.

Thank you for your time. I am now open to your questions.

---

## Appendix Slides (Reference Guide)

| Slide | Title | When to Use During Q&A |
|:---|:---|:---|
| **A1** | Full End-to-End ALPR System | When asked: *"How does this perform on full video frames or barrier gates?"* |
| **A2** | Complete 26×26 Confusion Matrix | When asked: *"Which specific provinces get confused most often?"* |
| **A3** | Per-Class Accuracy vs Dataset Size | When asked: *"Did the model fail on small classes like Kep or Mondul Kiri?"* |
| **A4** | Measurement Discipline & Audit | When asked: *"How did you measure and handle the 13.4% duplicate leakage?"* |
| **A5** | Code & Run Reproducibility | When asked: *"Can we re-run this exact code or verify your weights?"* |

---

## Anticipated Questions & Model Answers

### Q1: "Why did you use only ResNet18 for the transfer learning comparison?"
> **Answer:**  
> "To isolate the transfer learning strategy as our sole independent variable. If I changed the network architecture at the same time, any difference in performance could come from the architecture rather than the transfer learning method. Approach D (400K CNN) and Approach E (ViT-B/16) were introduced specifically to handle the architectural comparison."

### Q2: "Isn't 567 test images too small to make definitive claims?"
> **Answer:**  
> "Yes, and that is why I report the exact 95% confidence interval of ±2.0% and conducted McNemar's exact test ($p = 0.63$). Rather than over-claiming that fine-tuning beat scratch, I honestly conclude that ResNet18 scratch and fine-tuning are statistically tied on this test size."

### Q3: "Why did you tune Approach C instead of Approach A?"
> **Answer:**  
> "While A and C achieve comparable final accuracy, C converges four times faster — reaching 85% validation accuracy in just 5 epochs compared to 20 epochs for A. That computational efficiency made the 6-run hyperparameter grid search significantly faster and less resource-intensive."

### Q4: "How do you know the test set wasn't accidentally seen during training?"
> **Answer:**  
> "Through three strict controls: first, the directory split is fixed on disk; second, the PyTorch training dataloader only ever loads from the `train/` directory; and third, I wrote an automated audit script that checked image hashes and detected the multi-car photo overlap."

### Q5: "Why did you report Macro-F1 alongside Accuracy?"
> **Answer:**  
> "Because of the 25-to-1 class imbalance. In a dataset where Phnom Penh has over 400 crops and Mondul Kiri has only 18, a model could ignore the rare provinces and still achieve high accuracy. Macro-F1 treats all 26 classes with equal importance, ensuring rare provinces are not overlooked."

### Q6: "Why did Approach B and Approach D both fail at ~45%?"
> **Answer:**  
> "They failed for completely different theoretical reasons. Approach B has plenty of parameter capacity (11.2M), but its convolutional filters were frozen on ImageNet textures that do not match Khmer script strokes. Approach D had trainable filters, but its 400K parameters lacked the capacity and depth needed to separate 26 dense, ligature-heavy classes."

### Q7: "What would you do differently if you started over?"
> **Answer:**  
> "I would perform the train/val/test split by source photo rather than by cropped bounding box from day one. That single step would have completely eliminated the 13.4% multi-car image overlap."

### Q8: "Can your system handle upside-down (inverted) license plates?"
> **Answer:**  
> "Yes! In our baseline benchmark on Slide 8, models without rotation training only achieve 19.4% (ResNet) and 26.8% (ViT). However, in my project code, I specifically trained a dedicated rotation model using `--rotate180` data augmentation (`province_classifier_rot2.pth`), and it successfully achieved **93.3% accuracy on upside-down plates**, while our CRNN number reader achieves **51.7%**."

### Q9: "Why did you use 128×128 resolution for CNNs and 224×224 for ViT?"
> **Answer:**  
> "Two simple reasons:  
> 1. For CNNs, real Khmer plate crops are naturally small (~120×40 px). Using 128×128 preserves sharp stroke detail, cuts computation by 67%, and runs in under 4 ms for real-time barrier gates.  
> 2. For ViT, it is an architectural requirement: pretrained Vision Transformers divide images into a 14×14 grid of 16×16 patches, which natively requires 224×224 to match their pretrained positional embeddings."

### Q10: "How many layers / hidden layers are in your CNN?"
> **Answer:**  
> "Our main model is ResNet-18, which has 18 layers with weights: 17 convolutional feature extraction layers and 1 fully connected output layer for the 26 classes (~11.2 million parameters). For comparison, our small baseline CNN (Approach D) has 4 convolutional layers and 1 output layer (~400k parameters)."
