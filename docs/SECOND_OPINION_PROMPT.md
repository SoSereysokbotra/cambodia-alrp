# Second-opinion prompt — is the rotation result good enough?

Copy everything below the line into a fresh AI chat. It is self-contained.

---

I need a critical second opinion on a deep-learning result. Please be skeptical
rather than encouraging — I want to know if I should stop or keep pushing, and I
would rather hear "this is not significant" than false reassurance.

## Context

University Deep Learning course project: an automatic license plate reader (ALPR)
for Cambodian plates. The graded value is in the model and honest measurement, not
in surrounding business logic.

**The specific goal:** make the model read a plate correctly even when the plate is
upside-down (rotated 180 degrees). Hard constraint: this must be solved by TRAINING
the model. An inference-time wrapper that just tries the frame at several rotations
was built, measured at ~76%, and deliberately rejected — it is switched off and is
not an acceptable answer.

## Architecture

- CRNN (CNN + BiLSTM + CTC) reading the plate number as a text sequence.
- A Spatial Transformer Network (STN) "straightening layer" inserted inside the CRNN.
  It has 22 parameters, initialised to the identity transform so it starts as a no-op,
  and learns an affine correction purely from the CTC loss.
- Charset: `0-9 A-Z -` (37 symbols, no space).

## Measurement protocol

- Metric: exact string match of the full plate number (not character accuracy).
- Test set: 149 real plate crops, held out, never trained on. A guard in the training
  script raises an exception if any `test/` row reaches the training data.
- Each test crop is read twice: as-is (upright), and rotated 180 degrees (upside-down).
- The exact same script and the exact same 149 crops were used for every row below.

## Measured results

| model | what changed | upright | upside-down |
|---|---|---|---|
| `crnn_finetuned` | deployed baseline, no rotation training | 80.5% (120/149) | 0.0% (0/149) |
| naive flip aug | just flipped crops, no STN | 62% | 12% |
| `crnn_stn` | STN added, small synthetic set | 81% | 10% |
| `crnn_stn2` | STN + 20k synthetic plates | 79.2% (118/149) | 55.0% (82/149) |
| `crnn_stn3` | + photo-realistic synthetic renderer | 86.6% (129/149) | 65.1% (97/149) |
| `crnn_stn4` | + 199 newly hand-labelled real crops | 85.9% (128/149) | 69.1% (103/149) |

`crnn_stn4` is the current best.

## Training recipe for the latest run (`crnn_stn4`)

- 674 real train crops (oversampled x6) + 167 more real crops (x6) + 16,000 synthetic
  = 21,046 samples per epoch.
- `--rotate180 0.5`: each training crop is flipped 180 degrees with probability 0.5.
- 80 epochs, lr 1e-4, full fine-tune from a CRNN base, Tesla T4.
- Best real-validation CER 3.11% (from 93.91% before fine-tuning). Validation is 118
  real crops held out from train — separate from the 149-crop test set.
- The photo-realistic synthetic generator adds perspective warp, uneven lighting,
  blur, sensor noise and JPEG artefacts to rendered plates.

## Approaches already tried and refuted (with measurements)

1. Naive 180-degree flip augmentation on the CRNN without an STN: upright collapsed
   80.5% -> 62%, upside-down only reached 12%.
2. Training the YOLO **detector** with 180-degree rotation: rotated plates were
   detected, but upright detection fell 95.97% -> 85.9%, and the reader still could
   not read the rotated crops. Rolled back.
3. Inference-time rotation search: works (~76% at all angles) but rejected on the
   grounds above.

## Resources still available

- ~1,486 unlabelled real plate crops on disk. An active-learning scan ranked them;
  the last batch of 199 labels took about an hour of manual work.
- The synthetic generator can produce unlimited plates, but its font is generic
  (Liberation Sans Bold) and does not match the real Cambodian plate typeface.
- Training runs on a free Colab T4 and take about one hour per experiment.

## What I want from you

1. **Is the +4.0 point upside-down gain (65.1% -> 69.1%, i.e. 97 -> 103 of 149 crops)
   statistically meaningful, or is it noise?** These are paired measurements on the
   same 149 crops. Please name the right test (McNemar?), state what confidence
   interval applies at n=149, and tell me plainly whether I am over-reading 6 crops.
   Also: is n=149 simply too small to resolve differences of this size, and if so
   what would I need?

2. **Is 85.9% upright / 69.1% upside-down a defensible place to stop** for an
   undergraduate course project, or does it look unfinished? Note upright accuracy
   is essentially unchanged from the 86.6% run (a 1-crop difference), so the model
   is not trading away the normal case.

3. **If it is worth pushing, rank the remaining levers by expected gain per hour:**
   - label another ~200 real crops (the previous 199 bought ~+4 points)
   - obtain a real Cambodian plate TTF font to close the synthetic/real font gap
   - increase rotation augmentation probability above 0.5
   - train longer / larger backbone / different STN design (e.g. full affine or TPS
     instead of the current constrained transform)
   - something I have not listed

4. **Is there a fundamental ceiling here?** My working assumption is that 180-degree
   text is inherently more ambiguous than upright (6/9, 2/Z, M/W etc. under rotation)
   so upside-down will never match upright accuracy. Is that reasoning sound, and is
   ~70% roughly where it should top out, or should a well-trained STN get much closer
   to the upright number?

5. **Do you see a methodological flaw** in the measurement or the training setup that
   would make these numbers untrustworthy?

Please be concrete and quantitative. If you think I am fooling myself somewhere, say so.
