#!/usr/bin/env python3
"""
scripts/recognition/harvest_active.py
=====================================
IMPROVEMENT_PLAN_V2 Phase 5 — active-learning harvest.

What changed vs the plan
------------------------
The plan pointed this phase at `photos/` + `plates.db` ("7,341 audit reads, 6,122
evidence photos"). Phase 3 measured that pool and it does **not** support a
fine-tune: after visit de-duplication it contains only **~6 distinct vehicles**
(one of them 110 of 140 visits) plus 226 "unreadable" visits where the camera was
pointed at a phone screen. It is development testing, not gate traffic. Labelling
it would add hundreds of near-duplicates of the same few plates, and — decisively —
**none of those vehicles appear in the 149-frame test set, so any gain would be
unmeasurable.**

The pool that *does* support a fine-tune was already on disk:

    data/crnn_crops/train   1803 crops    473 labelled   1330 UNLABELLED
    data/crnn_crops/valid    643 crops      0 labelled    643 UNLABELLED
    data/crnn_crops/test     436 crops    149 labelled    287 (NEVER TOUCH)

**1,973 unlabelled real number-crops** from the diverse Roboflow dataset, already
extracted. The project's CER curve (94.89% -> 25.93% -> 20.32% -> 10.21% as labels
grew 0 -> 143 -> 324 -> 473) is the evidence that labelling more of exactly this
kind of data works, and it has never plateaued.

How this is *active* learning
-----------------------------
Rather than labelling 1,973 crops in arbitrary order, rank them by how much the
model is struggling, so a labelling session buys the most accuracy per crop:

  priority 1  FORMAT-INVALID  — the Phase 1 grammar proves the read is wrong
                                (no real Cambodian plate has that shape)
  priority 2  LOW CONFIDENCE  — below the REC-005 gate threshold
  priority 3  MID CONFIDENCE  — plausible but not certain
  (skipped)   HIGH CONFIDENCE + valid format — the model already handles these

Guards (both mandatory — see the plan's Phase 5 section)
--------------------------------------------------------
* **No test leakage.** `data/crnn_crops/test/` is excluded unconditionally; the
  script refuses to emit any crop from it. Without this, CER becomes meaningless
  and every measurement in this project is retroactively void.
* **No selection bias.** This tool selects HARD cases on purpose. Training on them
  ALONE would skew the model and degrade easy-case accuracy. The emitted labels are
  meant to be *appended to* `real_labels.csv` and trained together with the existing
  473 — never used as a standalone training set. `--check-mix` verifies that.

Usage
-----
    python scripts/recognition/harvest_active.py --scan          # rank the pool
    python scripts/recognition/harvest_active.py --sheets 60     # montages to label
    python scripts/recognition/harvest_active.py --check-mix     # verify guards
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "recognition"))

CROPS = PROJECT_ROOT / "data" / "crnn_crops"
LABELS = CROPS / "real_labels.csv"
QUEUE = PROJECT_ROOT / "metrics" / "harvest_queue.csv"
SHEETS = PROJECT_ROOT / "results" / "harvest_sheets"
CONFIG = PROJECT_ROOT / "configs" / "system_config.yaml"

# Splits this tool is ALLOWED to read. 'test' is absent by design, not by omission.
HARVEST_SPLITS = ("train", "valid")
FORBIDDEN_SPLIT = "test"


def labelled_names() -> set[str]:
    if not LABELS.exists():
        return set()
    with open(LABELS, encoding="utf-8") as f:
        return {Path(r["image_path"]).name for r in csv.DictReader(f)}


def unlabelled_crops() -> list[Path]:
    """Every crop in the harvestable splits that has no label yet."""
    have = labelled_names()
    out = []
    for split in HARVEST_SPLITS:
        for p in sorted((CROPS / split).glob("*.jpg")):
            if p.name not in have:
                out.append(p)
    return out


def scan(limit: int | None) -> None:
    import cv2
    import yaml
    from crnn_reader import CRNNReader
    from plate_format import is_valid, signature

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    gate = cfg.get("gate", {})
    thr = float(gate.get("crnn_confidence_threshold", 0.70))
    reader = CRNNReader(PROJECT_ROOT / cfg["crnn_weights"],
                        PROJECT_ROOT / cfg["charset_path"],
                        format_validation=False,   # we want the RAW read + our own verdict
                        confidence_mode=str(gate.get("confidence_mode", "mean")))

    crops = unlabelled_crops()
    if limit:
        crops = crops[:limit]
    print(f"[scan] {len(crops)} unlabelled crops in {'/'.join(HARVEST_SPLITS)} "
          f"(test split excluded by design)")

    rows = []
    for i, p in enumerate(crops, 1):
        img = cv2.imread(str(p))
        if img is None:
            continue
        text, conf = reader.read(img)
        ok = is_valid(text)
        if not ok:
            prio, why = 1, "format-invalid"
        elif conf < thr:
            prio, why = 2, "low-confidence"
        elif conf < 0.90:
            prio, why = 3, "mid-confidence"
        else:
            prio, why = 4, "model-confident"
        rows.append({"image_path": f"data/crnn_crops/{p.parent.name}/{p.name}",
                     "split": p.parent.name, "pred_text": text,
                     "confidence": round(conf, 4), "format_ok": int(ok),
                     "signature": signature(text), "priority": prio, "reason": why,
                     "plate_text": ""})
        if i % 250 == 0:
            print(f"  scanned {i}/{len(crops)} ...")

    rows.sort(key=lambda r: (r["priority"], r["confidence"]))
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with open(QUEUE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    c = Counter(r["reason"] for r in rows)
    print("\n" + "=" * 58)
    print(" ACTIVE-LEARNING HARVEST QUEUE (PLAN_V2 Phase 5)")
    print("=" * 58)
    for why in ("format-invalid", "low-confidence", "mid-confidence", "model-confident"):
        n = c.get(why, 0)
        print(f"  {why:<18} {n:>5}  ({100*n/max(1,len(rows)):>5.1f}%)")
    print("-" * 58)
    worth = c.get("format-invalid", 0) + c.get("low-confidence", 0)
    print(f"  HIGH-VALUE to label  {worth:>5}  (priority 1-2)")
    print("=" * 58)
    print(f"\n[scan] wrote {QUEUE.relative_to(PROJECT_ROOT)}")
    print("       Fill `plate_text` for the top rows, then append them to")
    print("       data/crnn_crops/real_labels.csv and re-run the fine-tune.")


def sheets(n: int, offset: int = 0) -> None:
    """Montages of the highest-priority crops, for transcription.

    `offset` skips the first N queue rows, so successive labelling batches render
    the *next* chunk instead of re-rendering ones already transcribed. The cell
    label shows the absolute queue index, which is what the labeller keys on.
    """
    import cv2
    import numpy as np
    if not QUEUE.exists():
        print("[X] no queue — run --scan first")
        sys.exit(1)
    all_rows = list(csv.DictReader(open(QUEUE, encoding="utf-8")))
    rows = all_rows[offset:offset + n]
    SHEETS.mkdir(parents=True, exist_ok=True)
    CW, CH, COLS, PER = 460, 150, 2, 14
    cells = []
    for i, r in enumerate(rows):
        p = PROJECT_ROOT / r["image_path"]
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        sc = min(CW / w, (CH - 26) / h)
        img = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))),
                         interpolation=cv2.INTER_CUBIC)
        cell = np.full((CH, CW, 3), 35, np.uint8)
        cell[24:24 + img.shape[0], :img.shape[1]] = img
        cv2.putText(cell, f"#{offset + i}  pred={r['pred_text']}  c={r['confidence']}",
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cells.append(cell)
    for s in range(0, len(cells), PER):
        page = cells[s:s + PER]
        rn = (len(page) + COLS - 1) // COLS
        sheet = np.full((rn * (CH + 5) + 5, COLS * (CW + 5) + 5, 3), 20, np.uint8)
        for k, cl in enumerate(page):
            rr, cc = divmod(k, COLS)
            sheet[5 + rr * (CH + 5):5 + rr * (CH + 5) + CH,
                  5 + cc * (CW + 5):5 + cc * (CW + 5) + CW] = cl
        cv2.imwrite(str(SHEETS / f"h{offset:04d}_{s // PER:02d}.png"), sheet)
    print(f"[sheets] wrote {(len(cells) + PER - 1) // PER} sheets -> "
          f"{SHEETS.relative_to(PROJECT_ROOT)}")


def merge(dry_run: bool = True) -> None:
    """Append filled `plate_text` rows from the queue into real_labels.csv.

    Refuses to write anything from the test split, refuses duplicates, and refuses
    labels that fail the Phase 1 grammar (a transcription typo is far more likely
    than a genuinely new plate shape — and a bad label is worse than no label).
    """
    from plate_format import is_valid

    if not QUEUE.exists():
        print("[X] no queue — run --scan first")
        sys.exit(1)
    rows = list(csv.DictReader(open(QUEUE, encoding="utf-8")))
    have = labelled_names()
    new, skipped = [], {"empty": 0, "duplicate": 0, "bad_format": 0, "test_split": 0}
    for r in rows:
        text = (r.get("plate_text") or "").strip().upper()
        path = r["image_path"].replace("\\", "/")
        if not text:
            skipped["empty"] += 1
            continue
        if f"/{FORBIDDEN_SPLIT}/" in path:
            skipped["test_split"] += 1        # must never happen; belt and braces
            continue
        if Path(path).name in have:
            skipped["duplicate"] += 1
            continue
        if not is_valid(text):
            skipped["bad_format"] += 1
            print(f"  [!] rejected {text!r} ({path}) — fails the plate grammar")
            continue
        new.append({"image_path": path, "plate_text": text})

    print(f"[merge] {len(new)} new labels ready | skipped: {skipped}")
    if not new:
        return
    if dry_run:
        print("[merge] DRY RUN — re-run with --apply to write real_labels.csv")
        return
    with open(LABELS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "plate_text"])
        w.writerows(new)
    total = len(labelled_names())
    print(f"[merge] appended to {LABELS.relative_to(PROJECT_ROOT)} "
          f"-> {total} labels total")


def check_mix() -> None:
    """Verify the two mandatory guards before anyone starts a fine-tune."""
    ok = True
    if QUEUE.exists():
        rows = list(csv.DictReader(open(QUEUE, encoding="utf-8")))
        leaked = [r for r in rows if f"/{FORBIDDEN_SPLIT}/" in r["image_path"].replace("\\", "/")]
        print(f"[guard] test-split leakage : {len(leaked)} rows "
              f"{'OK' if not leaked else '*** LEAK ***'}")
        ok &= not leaked
        filled = [r for r in rows if (r.get("plate_text") or "").strip()]
        print(f"[guard] newly labelled     : {len(filled)}")
        existing = len(labelled_names())
        if filled:
            frac = len(filled) / max(1, existing + len(filled))
            print(f"[guard] hard-case fraction : {frac:.1%} of the merged train set")
            if frac > 0.5:
                print("        *** WARNING: hard cases would dominate. Mix with the")
                print("        existing labels or the model will skew to hard examples.")
                ok = False
    else:
        print("[guard] no queue yet — run --scan first")
    print(f"\n[guard] {'PASS' if ok else 'ATTENTION NEEDED'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="active-learning harvest (PLAN_V2 Phase 5)")
    ap.add_argument("--scan", action="store_true", help="rank unlabelled crops by informativeness")
    ap.add_argument("--limit", type=int, default=None, help="scan only the first N crops")
    ap.add_argument("--sheets", type=int, metavar="N", help="write montages for the top N crops")
    ap.add_argument("--offset", type=int, default=0, help="skip the first N queue rows (with --sheets)")
    ap.add_argument("--check-mix", action="store_true", help="verify leakage/bias guards")
    ap.add_argument("--merge", action="store_true",
                    help="append filled labels to real_labels.csv (dry run by default)")
    ap.add_argument("--apply", action="store_true", help="with --merge, actually write")
    args = ap.parse_args()
    if args.scan:
        scan(args.limit)
    elif args.sheets:
        sheets(args.sheets, args.offset)
    elif args.merge:
        merge(dry_run=not args.apply)
    elif args.check_mix:
        check_mix()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
