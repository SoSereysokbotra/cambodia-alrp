# PROJECT HANDOFF — Cambodian ALPR (paste this into a new chat)

You are continuing an in-progress **Cambodian Automatic License Plate Recognition
(ALPR)** project. Read this whole file first, then continue from "WHAT'S NEXT".

---

## 1. WHAT THE PROJECT IS
A gate-automation ALPR for Cambodian plates. Pipeline:
`camera → detect plate → read text → check whitelist (SQLite) → open/deny gate (MQTT→ESP32) → log`.
It is driven by an SRS (`docs/srs.md`) and DB schema (`docs/database.md`). We are
bringing a working prototype into SRS conformance via `docs/SRS_ALIGNMENT_PLAN.md`.

**Cambodian plates have 3 lines:** Khmer province (top), the NUMBER (middle,
e.g. `2A-0243`), English province (bottom). This 3-line structure is central to
everything below.

---

## 2. ENVIRONMENT (Windows 11 — kept, NOT Ubuntu; documented deviation)
- Project root: `d:\Year2\Semester2\Deep Learning\Cambodian ALPR Project`
- **ALWAYS activate the venv first**, or you get a CPU-only torch:
  `.\.venv\Scripts\activate`  (prompt must show `(.venv)`)
- GPU: NVIDIA RTX 3050 Laptop (4 GB). Python 3.10, torch 2.5.1+cu121.
- When running via a bash/tool shell, call the venv python directly:
  `./.venv/Scripts/python.exe ...`

---

## 3. ARCHITECTURE — TWO DETECTORS + TWO READERS (important!)
The single most important thing to understand:

- **Plate/province detector** `models/detection/best.pt` (YOLOv10, 1 class):
  detects the plate region. Trained on Plate_v4 whose boxes are the **Khmer
  province line**. mAP50 **0.9664**.
- **Province classifier** `models/recognition/province_classifier_best.pth`
  (ResNet18, 26 classes = 25 provinces + "other"): reads WHICH province from the
  crop. Test acc **97.18%**. Class→Khmer map + `compose_plate()` in
  `src/recognition/province_map.py`. (Fixes ImageFolder lexicographic ordering via
  a saved `idx_to_class` in its config json — do NOT assume folder idx == class id.)
- **Number detector** `models/detection/number_best.pt` (YOLOv10, 1 class
  `plate_number`): finds the NUMBER line specifically. mAP50 **0.943**. We had to
  build this because best.pt only finds the province line, NOT the number.
- **CRNN number reader** `models/recognition/crnn_finetuned.pth` (CRNN+CTC,
  charset = digits+Latin+`- ` only, NO Khmer): reads the number string.
  `src/recognition/crnn_model.py`, `crnn_reader.py` (`read(crop)->(text, conf)`).

**Final intended per-plate flow:**
`best.pt → province classifier → provinceKhmer` AND
`number_best.pt → fine-tuned CRNN → number`, then
`compose_plate(province_id, number) → "ភ្នំពេញ 1AB-2345" → db.is_registered → gate`.

---

## 4. CURRENT STATE (what's done)
- Weeks 1–12 complete: detection, synthetic CRNN, full pipeline, system test,
  latency profiler, demo, ESP32 firmware (`hardware/esp32_gate_controller/`).
- SRS alignment: **Phase 1** (confidence gate / REVIEW_REQUIRED, threshold 0.70,
  REC-005) DONE; **Phase 2** (DB schema matches `database.md`: registered_plates,
  plate_reads, system_metrics; `log_read` has crnn_confidence etc.) DONE;
  **Phase 3** (province classifier + `compose_plate`) DONE (97.18%).
- **Phase 4 (real CRNN) DONE** — fine-tuned on 473 real train labels: test CER
  **10.21%**, word-acc **72.48%** (both SRS targets met). `crnn_finetuned.pth`.
- **INTEGRATION (two-detector flow) DONE** — `src/core/alpr_system.py` now runs
  `number_best.pt` → fine-tuned CRNN for the number AND `best.pt` → province
  classifier for the province, pairs them via `_match_number()`, and composes
  `"provinceKhmer number"`. Config points at `crnn_finetuned.pth` +
  `number_weights`. First real **ENTRY_ALLOWED** demonstrated on `ភ្នំពេញ 3E-6694`
  (enrolled a correctly-read real plate; un-enrolled real plates → ENTRY_DENIED).
- **Phase 7 (manual override + E-stop) DONE** — `run_video` keys: `o` →
  `manual_override()` (opens gate, logs MANUAL_OVERRIDE), `e` → `emergency_stop()`
  (toggles fail-safe: gate held closed, reads can't open it), `q` → quit. Verified:
  all actions < 5 ms, logged; registered plate under E-stop stays closed.
- **Phase 9 (health metrics + alerts) DONE** — `log_metrics_sample()` writes
  fps/latency/GPU-MB/cpu/rtsp/uptime to `system_metrics` every N frames; `_alert()`
  → `logs/alerts.log` for high latency (>500ms), stream disconnect (>15s), GPU OOM,
  DB error. Verified. **Full two-detector pipeline latency: avg 51 ms (~19.6 FPS)**
  — well under the 500 ms / 15 FPS SRS targets. `configs` gained a `health:` block.
- **Phase 12 (SRS acceptance suite) DONE** — `scripts/system/srs_acceptance_test.py`
  rewritten to measure LIVE (detector mAP from saved artifacts w/ provenance; CER,
  latency, FPS, zero-false-accept, fail-safe, audit completeness, manual override
  driven live). Result: **16/16 requirements PASS** → `metrics/srs_acceptance.json`.
  (Task 12.1 config-key renaming to SRS Appendix C names still minor-pending; folds
  into Phase 5.)
- **Phase 10 (dashboard GUI) TODO** — deferred to build interactively with the user.
- **Phase 5 (live RTSP) TODO** — deferred until a phone stream is available to validate.

### Metrics so far
| Model | Metric |
|-------|--------|
| YOLOv10 plate detector | mAP50 0.9664 |
| Province classifier (26-cls) | test acc 97.18% |
| Number detector | mAP50 0.943 |
| CRNN on synthetic | CER 0% |
| CRNN on REAL — baseline (synthetic only) | **CER 94.89%**, word-acc 0% |
| CRNN on REAL — fine-tuned on 143 train | **CER 25.93%**, word-acc 47.65% |
| CRNN on REAL — fine-tuned on 324 train | **CER 20.32%**, word-acc 51.68% |
| CRNN on REAL — fine-tuned on 473 train | **CER 10.21%**, word-acc 72.48% ✅ |
| Full two-detector pipeline on real test imgs | **number-correct end-to-end 70.6%** (101/143) |

Target: CER < 15%, word-acc > 70% (SRS REC-001) — **MET** (10.21% / 72.48%).

---

## 5. EXACTLY WHERE WE ARE IN PHASE 4
- Number crops staged at `data/crnn_crops/{train,valid,test}/` (2,882 clean number
  crops from `scripts/recognition/crop_numbers.py`).
- Real labels at `data/crnn_crops/real_labels.csv`: **149 TEST (human-labelled)**
  + **324 TRAIN (AI-labelled via montages)** = 473 rows.
- A re-fine-tune on the 324-label train set was **running in the background** when
  this handoff was written. **FIRST ACTION in the new chat:** confirm it finished
  and measure it:
  ```
  ./.venv/Scripts/python.exe scripts/recognition/evaluate_crnn_on_real.py --split test --weights models/recognition/crnn_finetuned.pth --tag finetuned
  ```
  Compare to the 25.93% / 47.65% from the 143-label run.

### Decision after measuring
- If CER still > ~15%: label ~200 more train crops (NO user work needed — do it
  yourself): `make_montage.py --split train --count 200 --offset <next>` → read the
  montages (Read tool shows images) → transcribe → write CSV → `import_label_csv.py`
  → re-run `finetune_crnn.py`. (Test set stays human — never train on it.)
- Otherwise proceed to integration (next section).

---

## 6. WHAT'S NEXT (the plan, in order)
1. **Measure the 324-label fine-tune** (above). Maybe one more label+finetune loop.
2. **INTEGRATION (the milestone):** rewire `src/core/alpr_system.py` `process_frame`
   to the two-detector flow: run `number_best.pt` to crop the number → fine-tuned
   CRNN reads it; run province classifier for the province; `compose_plate()`; then
   whitelist + gate. Point config at `crnn_finetuned.pth`. Add `number_weights` to
   `configs/system_config.yaml`. Then honest end-to-end demo on REAL images
   (`scripts/system/run_demo.py`) — first real `ENTRY_ALLOWED`.
3. **Remaining SRS phases** (`docs/SRS_ALIGNMENT_PLAN.md`): Phase 5 live smartphone
   RTSP, Phase 7 manual override + E-stop (CRITICAL), Phases 9–10 health metrics +
   dashboard, Phase 12 `scripts/system/srs_acceptance_test.py` all green.
4. Update `docs/SRS_DEVIATION_LOG.md`, save final metrics, commit to git.

---

## 7. KEY GOTCHAS / LESSONS (do not relearn these the hard way)
- **venv**: not activating it → CPU torch → training fails. Use `./.venv/Scripts/python.exe`.
- **Windows console cp1252**: printing ✓/Khmer crashes unless
  `sys.stdout.reconfigure(encoding="utf-8")` (already in scripts) and file writes use
  `encoding="utf-8"`.
- **Git-bash path mangling (MSYS)**: an arg like `/train/` becomes `C:/Program Files/Git/train/`.
  Prefix commands with `MSYS_NO_PATHCONV=1` when passing `--match "/train/"` etc.
  (Only affects the tool shell; the user's PowerShell is fine.)
- **cuDNN crash on this GPU**: training scripts set `cudnn.benchmark=False` and retry
  with `amp=False` on a cuDNN error.
- **ImageFolder ordering**: folders sort lexicographically ('0','1','10',..,'2'),
  so the province classifier saves/uses `idx_to_class`. Don't bypass it.
- **READ-ONLY models**: never overwrite `models/detection/best.pt` or
  `models/recognition/crnn_best.pth` (synthetic). New outputs get new names
  (`number_best.pt`, `crnn_finetuned.pth`).
- **Test labels are human, train labels are AI** — this is deliberate; never train
  on the 149 test rows (`/test/` paths) or the CER is dishonest.
- **Long/GPU commands**: run in background; output is buffered through pipes so
  `results.csv` / saved weights are better progress signals than stdout.

---

## 8. USEFUL SCRIPTS (all under scripts/<group>/, run from project root)
- `scripts/recognition/crop_numbers.py` — re-crop number lines with number_best.pt
- `scripts/tools/make_montage.py` + `Read` the montages — bulk label train crops
- `scripts/tools/import_label_csv.py <csv>` — merge labels into real_labels.csv
- `scripts/recognition/finetune_crnn.py` — fine-tune CRNN (full, lr 1e-4, synth+real mix, aug)
- `scripts/recognition/evaluate_crnn_on_real.py --split test [--weights .. --tag ..]`
- `scripts/system/system_test.py` — 6-component readiness check
- Project structure: `docs/PROJECT_STRUCTURE.md`

**Dataset credit:** detection = Plate_v4 (`taki-dk0de`, Roboflow, CC BY 4.0);
number boxes = user-annotated (`cambodian-plate-number`, Roboflow).
```
