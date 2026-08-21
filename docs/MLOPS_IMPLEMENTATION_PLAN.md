# Pragmatic MLOps Implementation Plan

> **Status:** proposal — no code written yet.
> **Drafted against:** commit `c3158c8`, branch `refactor/data-argumentation`, 2026-08-21.
> **Supersedes:** the "Integrate MLOps" and "Pragmatic MLOps Integration Plan (Revised)" drafts.

This document is the implementation spec. Everything below was checked against the
actual code — file paths, function signatures and line numbers are real, not assumed.

---

## 1. Why this ordering

The previous draft led with DVC. That is not where the damage is.

**Measured facts about this repo:**

| Fact | Evidence |
|---|---|
| Solo developer, 12 commits | `git shortlog -sn --all`, `git rev-list --count HEAD` |
| Training runs on Colab, not locally | `colab_train.ipynb` mounts Drive, unzips a 151 MB bundle |
| A local GPU *does* exist | RTX 3050 Laptop, `torch.cuda.is_available() == True` |
| ~20,900 images in `data/` | `find data -type f -name '*.jpg' ...` |
| 12 `.pth` candidates in `models/recognition/` | `ls models/recognition` |
| An experiment log already exists | `metrics/experiment_log.csv`, 206 rows, with a `git_commit` column |
| **That log has a 13-day hole** | last row is commit `57937e9` (Aug 8). `crnn_stn2.pth`, `crnn_stn4.pth`, `crnn_stn5.pth` (Aug 19–20) are unlogged |

The last row is the problem. Three models were trained and evaluated, and the SRS
evidence chain records none of them. No storage tool fixes that — it is a missing
function call. So **Phase A comes first**, and it is the only phase that is strictly
necessary.

### Corrections carried over from review

Two errors in the previous draft are fixed here rather than repeated:

1. **`dvc.yaml` is not part of this plan.** `dvc.yaml` is the *pipeline* file — the
   thing deliberately cancelled in §6. Tracking `models/` produces `.dvc/`,
   `.dvcignore`, and `models.dvc`. Nothing more.
2. **`finetune_crnn.py` does not write `experiment_log.csv` today.** It imports `csv`
   only to *read* label files (`read_csv`, line 85). The real writers are
   `scripts/system/benchmark_composed.py` and `scripts/tools/export_onnx.py`.
   Any verification step saying "confirm the CSV is still updated after training"
   would pass vacuously. Phase A is what makes that step meaningful.

---

## 2. Phase A — close the audit-trail gap  *(required)*

**Goal:** every model evaluation appends a row to `metrics/experiment_log.csv`,
automatically, tagged with the commit that produced it.

The helper already exists and needs no changes:

```python
# scripts/tools/experiment_log.py:57
log_metric(component, metric, value, split="", notes="", commit=None, csv_path=LOG_CSV)
```

### A.1 — `scripts/tools/check_stn.py`

At the end of `main()`, after the `rotates` / `keeps` / `looks_upright` booleans are
computed (the PASS/FAIL block at the bottom of the file), add a `--log` flag that
appends four rows:

| component | metric | value | split | notes |
|---|---|---|---|---|
| `crnn_stn` | `theta_flipped_a00` | `dn_a[:,0,0].mean()` | `real-test` | weights filename |
| `crnn_stn` | `theta_upright_a00` | `up_a[:,0,0].mean()` | `real-test` | weights filename |
| `crnn_stn` | `stn_flip_fraction` | `frac` | `real-test` | fraction of crops individually flipped |
| `crnn_stn` | `stn_verdict` | `1` PASS / `0` FAIL | `real-test` | weights filename |

Logging the raw theta values matters more than the verdict — the docstring already
records that `crnn_stn2/4/5` produced a near-identity transform, and that number is
the evidence behind the write-up claim that the STN was decorative.

### A.2 — `scripts/tools/check_leakage.py`

`audit()` (line 66) already prints a per-bucket table of upright / upside-down
accuracy (lines 134–143). Add the same `--log` flag and append, per bucket
(`clean`, `leaked`, `ALL`):

| component | metric | split | notes |
|---|---|---|---|
| `crnn` | `word_accuracy_upright` | `real-test-{bucket}` | weights filename, `n=` |
| `crnn` | `word_accuracy_upsidedown` | `real-test-{bucket}` | weights filename, `n=` |

Also log the contamination level once per audit:

| component | metric | value | split |
|---|---|---|---|
| `dataset` | `test_leaked_crop_fraction` | `len(leaked_crops)/len(by['test'])` | `real-test` |

This is the number the docstring pins at 34.9%. It belongs in the log, not only in a
docstring, because it is the reason the headline `crnn_stn4` score dropped from 69.1%
to a true 56.7%.

### A.3 — `scripts/recognition/finetune_crnn.py`

One call at the end of `main()`, after the summary print at line 365:

```python
log_metric("crnn", "best_val_cer", best_cer, split="real-val",
           notes=f"{out_path.name}, {args.epochs}ep, stn={args.stn}, "
                 f"stn_supervise={args.stn_supervise}")
```

Training-time val CER, clearly marked `real-val` so it is never confused with a
held-out test number. Test numbers keep coming from the evaluation scripts.

### A.4 — Backfill the three missing models

Re-run the audits against the existing weights and let the new `--log` flags write the
history. These are inference-only passes and run fine on the RTX 3050:

```bash
python scripts/tools/check_stn.py     --weights models/recognition/crnn_stn2.pth --log
python scripts/tools/check_stn.py     --weights models/recognition/crnn_stn4.pth --log
python scripts/tools/check_stn.py     --weights models/recognition/crnn_stn5.pth --log
python scripts/tools/check_leakage.py --audit --weights models/recognition/crnn_stn2.pth --log
python scripts/tools/check_leakage.py --audit --weights models/recognition/crnn_stn4.pth --log
python scripts/tools/check_leakage.py --audit --weights models/recognition/crnn_stn5.pth --log
```

Rows land under the *current* commit, not the original one. Note that in the `notes`
field — e.g. `[backfill 2026-08-21] trained ~2026-08-19` — so the log does not
silently claim these were measured at `c3158c8`.

**Deliverable:** `--log` on two scripts, one `log_metric` call in a third, ~30 lines
total, plus roughly a dozen backfilled rows. No new dependencies.

---

## 3. Phase B — DVC for `models/` only

**Goal:** offsite backup plus a git-committed hash of exactly which weights produced a
given result.

### B.1 — What this buys, and what it does not

Be honest about the benefit, because it determines how much effort is worth spending.

**It does give you:** a content hash in Git tying a commit to exact weight files, and a
real offsite copy that is not the same Google Drive folder the bundle already uses.

**It does not give you:** an end to `crnn_stn5_final_v2.pth` naming. `dvc add models/`
creates a *single* `models.dvc` for the whole directory, and DVC's workflow — delete
old files, `git checkout` to restore — conflicts with how this project actually works.
`metrics/` shows candidates benchmarked side by side (`crnn_real_baseline`,
`_finetuned`, `_augmented`, `_candidate679`, `_deployed`), which requires several
`.pth` files present at once. All 12 will stay on disk. Naming discipline is a
convention decision, not a tooling one.

### B.2 — Remote: Cloudflare R2, not an external drive

This is the answer to the open question, and it follows from where weights are born.

Weights are produced **on Colab**. `colab_train.ipynb:215-218` copies the result to
`/content/drive/MyDrive/ALPR/trained/`. So:

```
external drive:  Colab -> Google Drive -> laptop -> (manual) dvc push -> drive
                 3 copies, 2 sync systems, DVC engages only after a manual download

R2:              Colab -> dvc push -> R2
                 laptop pulls when needed; the Drive hop disappears
```

An external drive is unreachable from Colab, which turns DVC into a manual backup
script — which is what `gdrive_storage.py` already is. **Use R2.** (S3 or B2 are
equivalent; R2 has no egress fee, which matters when pulling ~430 MB.)

### B.3 — Steps

1. `pip install "dvc[s3]"` — R2 speaks the S3 API. Add to a new
   `requirements-mlops.txt` rather than mixing it into `requirements-gdrive.txt`.
2. `dvc init` — creates `.dvc/`, `.dvcignore`. Commit both.
3. Configure the remote, keeping the secret out of Git:
   ```bash
   dvc remote add -d r2 s3://alpr-models
   dvc remote modify r2 endpointurl https://<account-id>.r2.cloudflarestorage.com
   dvc remote modify --local r2 access_key_id     <key>
   dvc remote modify --local r2 secret_access_key <secret>
   ```
   `--local` writes to `.dvc/config.local`, which DVC gitignores automatically.
   **Verify** that file is untracked before the first commit.
4. **Remove `models/` from `.gitignore`.** It is currently ignored under the
   `--- Model weights (large binaries) ---` block. DVC writes its own
   `models/.gitignore`, so weights stay out of Git either way — but leaving the
   top-level rule in place makes `dvc add` fail confusingly. Leave the
   `*.pt` / `*.pth` / `*.onnx` extension rules and the root-level `yolov10*.pt`
   rules alone; they still protect the project root.
5. `dvc add models/`, then `git add models.dvc models/.gitignore .gitignore`.
6. `dvc push`.

### B.4 — Colab side

Add a cell to `colab_train.ipynb` that installs DVC, injects the R2 credentials from
Colab secrets (`google.colab.userdata`, **not** hardcoded), and pushes after training.
Keep the existing `shutil.copy` to `/content/drive/MyDrive/ALPR/trained/` for now —
two independent paths during the trial period is the point, not redundancy to remove.

### B.5 — Explicitly not touched

`src/utils/gdrive_storage.py` (720 lines) and `data/` stay exactly as they are. No
deletion, no deprecation, no edits. Revisit only after Phase B has survived a full
train → evaluate → ship cycle. Deleting working sync code before its replacement is
proven is how a dataset is lost near a deadline.

---

## 4. Phase C — WandB live curves

**Goal:** when a Colab run dies at epoch 40, the curves survive.

Scope is strictly real-time visualisation. `experiment_log.csv` remains the permanent
SRS audit trail; WandB is never the system of record.

### C.1 — `scripts/recognition/finetune_crnn.py`

- Add `--wandb` (off by default, so nothing changes for existing invocations) and
  `--wandb-project` (default `cambodian-alpr-crnn`).
- Init after args are parsed (~line 247), passing `vars(args)` as config so the run
  records `--epochs`, `--lr`, `--stn`, `--stn-supervise`, `--real-oversample`.
- Log inside the epoch loop, right after the existing print at line 357:
  ```python
  wandb.log({"epoch": epoch, "train_loss": run / max(n, 1),
             "stn_aux": aux_run / max(n, 1), "val_cer": vcer,
             "val_word_acc": vacc, "lr": scheduler.get_last_lr()[0]})
  ```
- Guard every call so a missing package or absent API key degrades to a no-op print
  rather than killing a training run mid-Colab-session.

### C.2 — Colab

API key via `google.colab.userdata`, never committed. `wandb/` is already in
`.gitignore`.

---

## 5. Verification plan

Corrected — the previous draft's CSV check tested something that never happened.

### Phase A
- `python scripts/tools/experiment_log.py --show | tail -20` shows the new STN and
  leakage rows.
- Row count rises from 206 by the number of backfilled measurements.
- Every backfilled row carries `[backfill …]` in `notes`.
- Re-running an audit with `--log` twice appends twice — the log is append-only by
  design and must not be made idempotent.

### Phase B
- `git status` shows `.dvc/config.local` **untracked**. Blocking check: if the R2
  secret is staged, stop and rotate the key.
- `dvc push`, then `dvc status --cloud` reports everything up to date.
- Move `models/` aside, `dvc pull`, and confirm all 12 `.pth` plus `charset.txt` and
  `province_classifier_config.json` return with matching sizes.
- **End-to-end gate:** `python scripts/system/system_test.py` still passes against
  DVC-restored weights. The SRS suite reports 16/16 today; it must still report 16/16.
- Confirm `.gitignore` still blocks `alpr_colab_bundle.zip` and root-level `yolo*.pt`.

### Phase C
- A 2-epoch run with `--wandb` produces a live chart with `val_cer` and `train_loss`.
- The same run **without** `--wandb` behaves exactly as before — the regression check
  that the flag is genuinely optional.
- With `wandb` uninstalled, `--wandb` prints a warning and trains normally.

---

## 6. Explicitly out of scope

| Item | Reason |
|---|---|
| `dvc.yaml` / `dvc repro` / `dvc exp run` | Heavy compute is on Colab via the zip bundle. A local pipeline would define stages that never execute. |
| DVC on `data/` | ~20,900 files. High API-call cost, low benefit — the raw set is stable and already synced. |
| Google Drive as the DVC remote | `dvc-gdrive` is a separate, poorly-maintained plugin; Google's OAuth restrictions and per-file rate limits make it hostile at this file count. |
| Removing `src/utils/gdrive_storage.py` | Works today. Not touched until Phase B survives a full cycle. |
| Replacing `experiment_log.csv` with WandB | The CSV is the SRS evidence chain: greppable, diffable, offline, in Git. WandB is a hosted convenience. |
| MLflow | Solves the same problem as the CSV, which already exists and already works. |

---

## 7. Effort and sequencing

| Phase | Effort | Risk | Necessary? |
|---|---|---|---|
| A — audit trail | ~1 hour | none — additive `--log` flags, default off | **Yes.** SRS evidence is broken now. |
| B — DVC + R2 | ~2–3 hours | low; needs an R2 account and key hygiene | Nice to have |
| C — WandB | ~30 min | none — opt-in flag | Nice to have |

**Do Phase A even if B and C are dropped.** It is the only part addressing something
currently broken, it adds no dependency, and it costs an hour.

Phases B and C are independent of each other; either can be skipped or reordered.

---

## 8. Rollback

- **A:** `git revert`. The CSV is append-only — delete the appended rows by hand if needed.
- **B:** `dvc destroy` removes `.dvc/` and `models.dvc`. Restore `models/` to
  `.gitignore`. Weight files on disk are untouched throughout.
- **C:** drop the `--wandb` flag; default-off means nothing else changes.
