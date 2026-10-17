# Khmer Province Classification — Presentation Study Summary

## 1. The Project in One Big Picture

The project asks:

> **Which deep-learning architecture and training strategy works best for recognizing the Cambodian province written on a license plate?**

The input is a **128×128 RGB image crop** containing the Khmer province name on a Cambodian license plate.

The output is one of **26 classes**:

- 25 Cambodian provinces
- `other` for police, military, and state plates that do not contain a province

The main research question is:

> **For a small and imbalanced Khmer-script dataset, which architecture and training strategy works best, and why?**

---

# 2. What Problem Are You Solving?

A Cambodian license plate contains important information such as:

```text
Khmer province name
        ↓
    2A-0243
        ↓
English province name
```

The **number alone is not enough** to identify the vehicle because the same number can exist in different provinces.

Therefore, the system needs to recognize both:

```text
Province → "Phnom Penh"
Number   → "2A-0243"
```

For the province part, the project treats the task as **whole-image classification** instead of character-by-character sequence recognition.

### Presentation Question

**Q: What is the task?**

**Answer:**

> It is a 26-class image classification problem. The model receives a 128×128 RGB crop of the Khmer province name and predicts which province it belongs to.

---

# 3. Dataset

The project uses the **Plate_v4 dataset from Roboflow Universe**.

The original dataset contains 3,299 Cambodian plate photos. Province-line crops were created from these images.

### Final Dataset

| Dataset | Images |
|---|---:|
| Train | 2,515 |
| Validation | 529 |
| Test | 567 |
| **Total** | **3,611** |

There are:

> **25 provinces + Other = 26 classes**

## Main Dataset Problem: Imbalance

The classes do not have the same number of samples.

For example:

```text
Other          → 453 training images
Phnom Penh     → 254
...
Mondul Kiri    → 18
Ratanakiri     → 20
```

So some provinces have hundreds of images while some have fewer than 20.

### Presentation Question

**Q: What is the main problem with your dataset?**

**Answer:**

> The dataset is imbalanced. Some classes have hundreds of samples, while the smallest classes have fewer than 20 training samples.

---

# 4. The Four Approaches Compared

This is the **most important part of the research**.

The project compares four approaches:

```text
                 4 Experiments
                      │
        ┌─────────────┴─────────────┐
        │                           │
     ResNet18                    Small CNN
        │                           │
    ┌───┼───┐                       D
    │   │   │
    A   B   C
```

---

## A — ResNet18 From Scratch

```text
Random weights
      ↓
ResNet18
      ↓
Train everything
```

The model starts with random weights and learns everything from the dataset.

- Trainable parameters: **11,189,850**
- Test accuracy: **93.3%**

---

## B — ResNet18 + ImageNet, Frozen Backbone

```text
ImageNet pretrained ResNet18
            ↓
       Freeze backbone
            ↓
       Train final layer
```

Only the final classification layer is trained.

- Trainable parameters: **13,338**
- Test accuracy: **42.0%**

This approach performed very poorly.

---

## C — ResNet18 + ImageNet + Full Fine-Tuning

```text
ImageNet pretrained ResNet18
            ↓
     Train entire network
            ↓
     Adapt to Khmer plates
```

The model starts with ImageNet knowledge but allows **all layers to learn**.

Initial experiment:

> **94.0% test accuracy**

After learning-rate tuning:

> **96.5% test accuracy**

This became the **final model**.

---

## D — Small CNN From Scratch

The project also tested a much smaller CNN:

```text
32 → 64 → 128 → 256
```

It has approximately **396,058 parameters**, compared with more than 11 million for ResNet18.

Test accuracy:

> **46.6%**

The purpose was to determine whether a much smaller network would be sufficient.

---

# 5. Most Important Results

Remember this table:

| Approach | Method | Test Accuracy |
|---|---|---:|
| **A** | ResNet18 from scratch | **93.3%** |
| **B** | ImageNet + frozen backbone | **42.0%** |
| **C** | ImageNet + full fine-tuning | **94.0%** |
| **D** | Small CNN from scratch | **46.6%** |
| **C tuned** | ImageNet + fine-tuning, LR 1e-3 | **96.5%** ⭐ |

### Main Result

> **ResNet18 with ImageNet initialization and full fine-tuning performed best, reaching 96.5% test accuracy after hyperparameter tuning.**

---

# 6. Why Did Approach B Fail?

Approach B uses:

```text
ImageNet ResNet18
      ↓
Frozen backbone
      ↓
Only final layer learns
```

The problem is that ImageNet contains natural images, while this project needs to recognize:

```text
Khmer characters
Khmer strokes
Khmer glyph shapes
```

The ImageNet features are not specialized for Khmer script.

Because the backbone is frozen, the model cannot adapt those features to the new domain.

Result:

> **42.0% accuracy**

### Presentation Question

**Q: Why did the frozen ImageNet model perform so badly?**

**Answer:**

> ImageNet features were learned from natural images, while our task requires recognizing Khmer glyph shapes. Because the backbone was frozen, the model could not adapt its features to the Khmer-script domain.

---

# 7. Why Did Approach C Work?

Approach C also starts from ImageNet weights.

However, unlike B, the whole network can learn:

```text
ImageNet knowledge
       ↓
    ResNet18
       ↓
Fine-tune all layers
       ↓
Learn Khmer-specific features
```

This gives the model:

1. Useful pretrained features
2. The ability to adapt those features to Khmer license plates

The initial result was:

> **94.0% test accuracy**

After tuning:

> **96.5% test accuracy**

### Important Finding

ResNet18 from scratch was already strong:

> **93.3%**

Therefore, ImageNet pretraining did **not** dramatically increase the final performance.

Instead, it mainly helped the model **converge faster**.

### Presentation Question

**Q: Why use transfer learning if training from scratch already achieves 93.3%?**

**Answer:**

> ImageNet pretraining mainly improved convergence speed. The fine-tuned model reached high validation accuracy much earlier, although the final performance was similar to training ResNet18 from scratch.

---

# 8. Why Did Approach D Fail?

The Small CNN has only about:

> **396K parameters**

while ResNet18 has:

> **11.19M parameters**

The Small CNN reached only:

> **46.6% test accuracy**

The important concept is:

## Underfitting

The model was not powerful enough to learn the differences between 26 visually similar Khmer province names.

Its training accuracy was only about:

> **58.9%**

This means it could not even learn the training data well.

### Presentation Question

**Q: Why did the Small CNN fail?**

**Answer:**

> The Small CNN underfit because its capacity was too low to distinguish 26 visually similar Khmer province names. ResNet18's deeper architecture and residual connections provided more capacity.

---

# 9. Hyperparameter Tuning

After choosing Approach C, the project tested different:

### Learning rates

```text
1e-4
3e-4
1e-3
```

### Weight decay

```text
0
1e-4
```

The best validation-selected configuration was:

```text
Learning rate = 1e-3
Weight decay = 0
```

Result:

> **96.5% test accuracy**

## Important Finding

The conventional low learning rate:

```text
1e-4
```

was actually too small.

The fully trainable network learned too slowly and under-trained.

The tuning showed:

```text
1e-4 → too slow
3e-4 → better
1e-3 → best selected model
```

### Presentation Question

**Q: What did your hyperparameter tuning discover?**

**Answer:**

> A learning rate of 1e-3 worked best. The lower learning rate of 1e-4 under-trained the fully fine-tuned network.

---

# 10. Error Analysis

The final C model had:

```text
567 test images
       ↓
34 errors
```

The biggest confusion was:

```text
Banteay Meanchey
        ↕
Oudor Meanchey
```

These two province names are visually similar in Khmer.

This pair caused:

> **8 of the 34 errors = 24%**

Other common errors involved visually similar province names such as:

- Battambang ↔ Pursat
- Kampong Chhnang ↔ Kampong Thom
- Kampong Speu ↔ Kampong Thom

## Image Quality Problems

Many remaining errors came from images that are difficult even for humans:

- Motion blur
- Very small text
- JPEG artifacts
- Poor image quality

### Important Finding

Class size matters, but it was **not the biggest problem**.

The visual similarity of the province name was a major source of errors.

### Presentation Question

**Q: What causes most classification errors?**

**Answer:**

> The main causes are visually similar Khmer province names and poor image quality such as blur, small text, and JPEG artifacts.

---

# 11. Dataset Leakage

There was a potential **near-duplicate leakage** problem.

The split was performed at the crop level.

One original photo could contain multiple license plates.

Therefore, the same source photo could appear in both training and test data:

```text
Same source photo
      │
      ├── Crop → Training
      │
      └── Crop → Test
```

There were:

> **76 / 567 test crops = 13.4%**

However, the project also evaluated a clean subset that removed these potentially related test crops.

Interestingly:

> The clean-subset accuracy was slightly HIGHER.

Therefore, the leakage did **not** inflate the reported result.

### Presentation Question

**Q: Did dataset leakage affect your result?**

**Answer:**

> There was 13.4% near-duplicate leakage because the split was performed at crop level. We evaluated both the full test set and a clean subset. The clean subset actually performed slightly better, so the leakage did not inflate our score.

---

# 12. What Is the Complete ALPR System?

The province classifier is only **one component** of the complete ALPR system.

The full system looks like:

```text
                 Camera
                   │
                   ▼
          YOLOv10 Plate Detector
                   │
          ┌────────┴────────┐
          │                 │
    Province crop       Number crop
          │                 │
          ▼                 ▼
      ResNet18           CRNN + CTC
          │                 │
          ▼                 ▼
     "ភ្នំពេញ"            "2A-0243"
          │                 │
          └────────┬────────┘
                   ▼
          "ភ្នំពេញ 2A-0243"
                   │
                   ▼
           SQLite Whitelist
                   │
                   ▼
             Gate Decision
                   │
                   ▼
                 ESP32
```

The README clearly separates the **graded research study** from the application:

```text
§1–§7 → Main graded classifier comparison

§8     → Application using the classifier
```

---

# 13. Complete ALPR Results

The complete system contains several models:

| Component | Result |
|---|---:|
| Plate detector — YOLOv10-n | **96.6% mAP50** |
| Number detector — YOLOv10-n | **95.7% mAP50** |
| Number reader — CRNN + CTC | **87.2%** upright |
| Province classifier — ResNet18 | **96.1%** deployed preprocessing |
| Number + province both correct | **82.6%** |

The end-to-end system was tested on **149 labelled full scenes**.

Important results:

- Plate found: **143/149**
- Number correct: **83.9%**
- Province correct: **94.0%**
- Both correct: **82.6%**
- Latency: approximately **50 ms/frame** on RTX 3050
- Zero gate-opens against the whitelist of 137 near-miss plates

---

# 14. Gate Decision Logic

The system is designed to be **fail-safe**.

The decision priority is:

```text
Emergency stop?
      ↓
REVIEW_REQUIRED

Reader confidence < 0.70?
      ↓
REVIEW_REQUIRED

Exact active whitelist match?
      ↓
ENTRY_ALLOWED

Within one edit of registered plate?
      ↓
REVIEW_REQUIRED

Otherwise
      ↓
ENTRY_DENIED
```

The key principle is:

> **When the system is uncertain, it does not open the gate.**

---

# 15. The Most Important Things to Memorize

If you only memorize **10 things**, memorize these:

### 1. Problem

> 26-class classification of Cambodian license-plate province names.

### 2. Input

> 128×128 RGB crop containing the Khmer province name.

### 3. Dataset

> 2,515 train / 529 validation / 567 test.

### 4. Main challenge

> Small and imbalanced dataset + visually similar Khmer province names.

### 5. Models

> ResNet18 from scratch, ResNet18 frozen ImageNet, ResNet18 fine-tuned ImageNet, and Small CNN.

### 6. Best model

> **ResNet18 + ImageNet + full fine-tuning.**

### 7. Best result

> **96.5% test accuracy** after learning-rate tuning.

### 8. Why frozen ResNet failed

> ImageNet features were not suitable for Khmer glyph recognition, and the frozen backbone could not adapt.

### 9. Why Small CNN failed

> It had insufficient capacity and underfit the 26 classes.

### 10. Main conclusion

> **A sufficiently large ResNet18 with full fine-tuning works best for this small Khmer-script classification problem. ImageNet mainly helps convergence, while the network still needs to adapt its features to the Khmer domain.**

---

# 16. The Whole Project as One Story

Do not think of the README as hundreds of lines.

Think of it as this:

```text
PROBLEM
  ↓
Recognize Cambodian province from Khmer plate text
  ↓
DATA
  ↓
Small + imbalanced dataset
  ↓
RESEARCH QUESTION
  ↓
Which model/training strategy works best?
  ↓
EXPERIMENT
  ↓
A: ResNet18 scratch       → 93.3%
B: ResNet18 frozen        → 42.0% ❌
C: ResNet18 fine-tuned    → 94.0%
D: Small CNN              → 46.6% ❌
  ↓
HYPERPARAMETER TUNING
  ↓
C + Learning Rate 1e-3
  ↓
96.5% ⭐
  ↓
ERROR ANALYSIS
  ↓
Similar Khmer names + poor image quality
  ↓
CONCLUSION
  ↓
Full fine-tuned ResNet18 is the best approach
  ↓
APPLICATION
  ↓
Use the classifier inside the complete ALPR
gate-control system
```

---

# 17. A Short Presentation Answer

If your teacher says:

> **"Explain your project."**

You can answer:

> "My project focuses on Cambodian license-plate province classification. The task is a 26-class image classification problem where the input is a 128×128 crop of the Khmer province name and the output is the province.
>
> The dataset contains 2,515 training images, 529 validation images, and 567 test images, and it is imbalanced.
>
> I compared four approaches: ResNet18 from scratch, ResNet18 with a frozen ImageNet backbone, ResNet18 with full fine-tuning, and a small CNN.
>
> ResNet18 with a frozen backbone performed poorly at 42% because ImageNet features were not suitable for Khmer glyphs and could not adapt. The small CNN also performed poorly at 46.6% because it underfit.
>
> ResNet18 from scratch achieved 93.3%, while full fine-tuning achieved 94.0%. After learning-rate tuning, the best model achieved 96.5% test accuracy.
>
> The main conclusion is that full fine-tuning of ResNet18 is the best approach for this dataset, and ImageNet pretraining mainly improves convergence speed. The main remaining errors come from visually similar Khmer province names and poor image quality."

---

## Final Mental Model

Remember:

```text
             WHAT?
               │
     Province classification
               │
             DATA
               │
      Small + imbalanced
               │
          WHAT DID I TRY?
               │
       ┌───────┼────────┐
       │       │        │
    Scratch  Frozen  Fine-tune
       │       │        │
     93.3%   42.0%    94.0%
                        │
                    Tune LR
                        │
                      96.5%
                        │
                    WHY BEST?
                        │
             Can adapt to Khmer
             + enough capacity
```

**If you understand this diagram, you understand the main research story of the README.**
