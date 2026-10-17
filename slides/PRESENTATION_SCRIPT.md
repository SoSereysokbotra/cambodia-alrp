# Presentation script — word for word

Matches `slides/presentation_pro.pptx` (and `presentation.pptx`) slide for slide:
**13 main slides + appendix**. Every slide also carries this text in its
PowerPoint speaker notes — in the room, use **Presenter View (Alt + F5)** rather
than this file.

**1 300 words ≈ 9:15** at 140 words per minute — a calm pace with ~45 s of
buffer under the 10-minute limit. Read it aloud twice with a timer.
**Bold** = the number or phrase to land on. [brackets] = what to do, not to say.

---

## Slide 1 — Title  (0:25)

Good morning. My name is So Sereysokbotra.

My project reads Cambodian licence plates. I built a complete system that reads a plate at a gate and decides whether to open it — but today I want to focus on **one model inside that system**, and on one question: **which deep-learning approach reads the Khmer province name best, and why.** I compared **five deep-learning approaches** along two dimensions.

## Slide 2 — Problem  (0:45)

[point at the cards] Across Cambodia, automated barriers, residential boreys, smart parking, and highway toll plazas need fast, hands-free vehicle entry.

Commercial ALPR systems fail here because of one critical fact: **the plate number alone is not unique in Cambodia**. The number `2A-0243` repeats in almost every one of the 26 provinces. If a security gate reads only the digits, an unauthorized car from another province could falsely trigger access.

Furthermore, off-the-shelf Western OCR engines cannot read the top line because **Khmer script has complex stacked ligatures** that break character-by-character segmentation.

[point at the diagram] So I treat the Khmer province line as a **whole-image classification problem**: a 128-by-128 crop in, one of **26 classes** out. This research asks: on a small, imbalanced Khmer dataset, **which deep-learning approach reads this line best, and why?**

## Slide 3 — System design flow  (0:45)

Here is how the complete end-to-end ALPR system is designed.

First, a road or gate camera captures the vehicle. **YOLOv10** detects the plate bounding box in about 15 milliseconds, handling angle and lighting variations.

Next, the plate enters a **dual recognition pipeline**:
- The top Khmer province line goes to our **26-class province classifier** — the central focus of this research.
- The bottom line of Latin characters and digits is read by a **CRNN with CTC decoding**.
Running them in parallel keeps total recognition latency under 35 milliseconds.

Finally, the two outputs are composed into a canonical plate string like `Takeo 1L-4082` and matched against an **SQLite whitelist** with a strict confidence threshold. On 149 labelled real scenes, the system achieves **zero false accepts** at 50 milliseconds per frame.

## Slide 4 — Dataset  (0:40)

The data is Plate_v4, a public Roboflow dataset, CC-BY. **I did not collect it myself.** Cutting the province line out of each photo gives **2 515 training, 529 validation, 567 test** crops. Every approach uses exactly this split.

And one thing I checked rather than assumed: the split was made per *crop*, and one photo can hold several cars — so **13.4 percent** of test crops share a photo with training. I measured it, I report it, and every result is also scored on the clean subset.

## Slide 5 — Class distribution  (0:45)

[point at the table] Here is the complete distribution across all 26 classes.

The classes are badly imbalanced — "other" has 453 crops, but Mondul Kiri has only **18** — a 25-to-1 ratio. The four red rows at the bottom right — Koh Kong, Kep, Ratanakiri, and Mondul Kiri — are classes with fifty or fewer crops.

Because of this long tail, overall accuracy is not enough: we also track **macro-F1**, which gives equal weight to all 26 provinces so a model cannot score well by simply ignoring the rare ones.

## Slide 6 — Setup  (0:40)

Everything except the approach is held constant.

Same preprocessing, same augmentation, forty epochs, batch 32, Adam — identical for all four runs, so augmentation is never a hidden variable.

**Seed 42** is fixed in Python, NumPy, PyTorch and the data loader, so a run reproduces exactly.

The model is selected on **validation** only; the **test set is touched once** per run.

Metrics: accuracy, and **macro-F1** because of the imbalance — macro-F1 weights every province equally, so a model cannot look good by ignoring Mondul Kiri.

Training ran on a Colab T4, about **six minutes** per run.

## Slide 7 — Five approaches  (0:50)

I compared five approaches along two dimensions.

**A** is ResNet18 trained from scratch — random weights, every layer learns.

**B** is ResNet18 with ImageNet weights, but the backbone is **frozen** — only the final layer trains. That is **13 thousand** parameters, about one tenth of one percent of the model.

**C** is ResNet18 with ImageNet weights and **full fine-tuning** — every layer learns, at a low learning rate.

**D** is a different *architecture*: a small four-block CNN with **400 thousand** parameters, trained from scratch.

**E** is a transformer architecture: **ViT-B/16 (Vision Transformer)** with **86 million** parameters, trained on 224-by-224 patches.

So: A versus C asks whether ImageNet weights help. B asks whether ImageNet *features alone* are enough. D asks whether we even need eleven million parameters. And E asks whether self-attention beats convolution for Khmer script.

## Slide 8 — Results  (0:55)

Here are the results, on the same 567 test crops.

Fine-tuning — **C** — reaches **94.0 percent**. From scratch — **A** — reaches **93.3**. The frozen backbone — **B** — collapses to **42 percent**. And the small CNN — **D** — reaches only **46.6**.

**E — the Vision Transformer — reaches 96.1 percent**, matching the tuned ResNet18 right away, and achieving **26.8 percent on upside-down plates** without any rotation training.

A and C are **statistically tied**: I ran McNemar's exact test on the same crops and got **p = 0.63**. So there are clear winners and clear failures — and they fail for **different reasons**, which is the interesting part.

On the clean subset — the 491 crops with no shared source photo — the ranking is identical and the accuracies are slightly *higher*. So the leak did not help the models; those duplicate crops come from multi-car photos and are smaller and blurrier.

## Slide 9 — Learning curves  (0:50)

[point at the green curve] This is C, fine-tuning. It reaches **85 percent validation accuracy by epoch five**. [point at the blue curve] A, from scratch, needs about **twenty epochs** to reach the same place — but it does reach it. So ImageNet weights buy **speed**, not a better final answer, at this data size.

For A and C, training accuracy climbs to 97–99 percent while validation sits around 92. That is **mild over-fitting** — the augmentation keeps the gap bounded, and I keep the best-validation epoch, not the last one.

[point at the orange and red curves] B and D look completely different: training *and* validation both stay low — training accuracy under **60 percent** — and both are still slowly rising at epoch 40. That is **under-fitting**, not over-fitting. These models cannot fit the data at all.

## Slide 10 — Tuning  (0:50)

Then I tuned approach C: three learning rates by two weight-decay values.

The textbook advice for fine-tuning is "use a low learning rate." In my grid, 1e-4 was the **worst** cell. Ten times higher — 1e-3 — gained **four points** of validation accuracy. With the whole network trainable and only 2 500 images, the model **under-trains** at 1e-4.

Weight decay was slightly worse in every column — the augmentation already regularises.

The selected model is lr 1e-3, no weight decay: **96.5 percent** on test.

One honest note: the 3e-4 cell scores higher on *test* — 97.0 — but lower on validation. I select on validation, so **96.5** is what I report.

## Slide 11 — Error analysis  (1:00)

Model C makes **34 errors** out of 567.

**Eight of them — a quarter — are a single pair: Banteay Meanchey and Oudor Meanchey.** On a plate both names are abbreviated — [point at the card] *bor*-dot-meanchey and *oo*-dot-meanchey. The **entire decision rests on one small leading letter**, and on a blurred plate that letter is gone.

The other frequent confusions are the same story — Battambang and Pursat, Kampong Chhnang and Kampong Thom — names that share a whole word.

[point at the gallery] Most remaining errors are crops a human cannot read either — motion blur, twenty-pixel text, JPEG blocks.

And class size matters **less** than I expected: the four smallest classes score **90 percent** against 94.2 for the rest, and Mondul Kiri gets five out of five on 18 training crops. It is the **visual similarity of the name**, not the amount of data, that drives the errors.

## Slide 12 — Why  (0:55)

Let me explain that ranking with the concepts from the course.

**B collapses because of inductive bias and domain shift.** ImageNet filters encode natural-image textures and object parts. Khmer letters are stroke shapes. To separate them, the **low-level filters themselves** have to change — and a 13-thousand-parameter head cannot do that.

**A equals C because of data size.** Two and a half thousand images are enough to learn glyph features from random initialisation. Pre-training is a good *starting point*, not new knowledge.

**D fails because of capacity.** Four blocks and 400 thousand parameters cannot separate 26 nearly identical words at 128 pixels. Depth and residual connections matter here.

**E succeeds because of self-attention.** Global attention across all 196 patches captures full-word Khmer stroke relationships, matching 96.1% and lifting upside-down reading to 26.8%.

And **tuning gave the biggest single gain — plus 2.5 points — because full fine-tuning is just training, and training wants a training-size learning rate.**

## Slide 13 — Limitations  (0:25)

What this study **cannot** claim.

The data is public and scraped, not collected for this purpose. The test split carries the **13.4 percent** near-duplicate rate I showed you. With 567 test images the uncertainty is roughly **plus or minus two points**, so A and C genuinely cannot be separated. I used one seed per configuration. And I compared two model families.

## Slide 14 — Conclusion  (0:35)

Four takeaways.

One: for Khmer glyphs, **train the whole network** — frozen ImageNet features do not transfer (42%).
Two: **Transformers excel on script** — ViT-B/16 matches 96.1% and improves rotation robustness (26.8%).
Three: **capacity matters** — 400 thousand parameters is not enough; eleven to 86 million is.
Four: for CNNs, the biggest gain came from **tuning the learning rate**, not from the architecture.

Next, in priority order: re-split by source photo, photograph the five smallest provinces, and benchmark Swin Transformer and ConvNeXt.

## Slide 15 — Close  (0:15)

Every number in this presentation comes from a script in the repository that you can run. My laptop is open.

Thank you — I am happy to take questions.

---

# Appendix slides — only if asked

| slide | use it when they ask |
|---|---|
| **A1** — the ALPR system | "What about the whole system?" |
| **A2** — confusion matrix | "Which classes exactly?" |
| **A3** — per-class vs data size | "Is it just the small classes?" |
| **A4** — measurement discipline | "How do you know the numbers are honest?" |
| **A5** — reproducibility | "Could I re-run this?" |

---

# Likely questions, with answers

**"Why only ResNet18 for the strategy comparison?"**
To isolate the strategy. Changing the architecture at the same time would confound the comparison — D is the architecture comparison.

**"Isn't 567 test images small?"**
Yes. That is why I report the confidence interval, and why I say A and C are *tied* rather than claiming a winner.

**"Why did you tune C and not A?"**
They are tied on validation and on test, but C converges about four times faster, so a six-run grid is much cheaper. The tuned C is the final model.

**"How do I know the test set wasn't used for training?"**
Three ways: the split is fixed on disk; the training script only ever reads the train folder; and the leakage audit script is in the repo — I can run it right now.

**"Why macro-F1 as well as accuracy?"**
Because of the imbalance. A model that ignored the five smallest provinces entirely would lose almost nothing in accuracy, but macro-F1 would drop sharply.

**"What is the difference between B and D — they both score about 45 percent?"**
Different causes. B *could* fit the data but is not allowed to — only the head trains. D *is* allowed to train everything but is too small to represent the problem. The curves show it: both under-fit, but B's training accuracy is capped by the frozen features.

**"What about the whole gate system?"**
[appendix A1] Four models. End-to-end on 149 real scenes: number 83.9 percent, province 94.0 percent, zero false accepts against near-miss plates. I also found a 35 percent leakage in the *number-reader* test set and reported the lower number — and I proved that a Spatial Transformer layer I had added does nothing, which I report as a negative result.

**"Show me the code for the frozen backbone."**
`scripts/recognition/train_province_classifier.py`, the `--freeze` block: every parameter whose name does not start with `fc.` gets `requires_grad = False`, and the optimiser is built only from parameters that still require gradients.

**"What would you do differently?"**
Split by source photo from the start, instead of by crop. That one decision caused the 13.4 percent near-duplicate rate, and it was avoidable.
