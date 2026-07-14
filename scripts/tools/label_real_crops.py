#!/usr/bin/env python3
"""
scripts/tools/label_real_crops.py
=================================
Manual transcription tool — type the NUMBER on each real plate crop so we get
labelled real data to fine-tune the CRNN (Phase 4).

Everything happens in ONE window (no console typing). The crop is enlarged and
contrast-enhanced so it's easy to read; you type directly into the window.

Controls (type into the window):
    0-9  A-Z  -  space   ... build the number
    BACKSPACE            ... delete last character
    ENTER                ... SAVE this label + go to next
    TAB                  ... SKIP (illegible / bad crop)
    ESC                  ... SAVE progress + quit (resume later)

Output (append, resumable): data/crnn_crops/real_labels.csv  (image_path,plate_text)

Run:
    python scripts/tools/label_real_crops.py --split test --limit 150
    python scripts/tools/label_real_crops.py --split train        # continue later
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
CROPS_ROOT = PROJECT_ROOT / "data" / "crnn_crops"
OUT_CSV = CROPS_ROOT / "real_labels.csv"
ALLOWED = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ- ")
IMG_EXTS = {".jpg", ".jpeg", ".png"}

DISP_W = 820          # display width for the crop
BAR_H = 130           # bottom bar height for the text + hints


def load_done(csv_path: Path) -> set[str]:
    done = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["image_path"])
    return done


def build_canvas(cv2, np, crop_gray, typed, idx, total, saved):
    """Enlarged + contrast-enhanced crop with a text/hints bar underneath."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enh = clahe.apply(crop_gray)
    h, w = enh.shape[:2]
    disp_h = int(h * (DISP_W / w))
    big = cv2.resize(enh, (DISP_W, disp_h), interpolation=cv2.INTER_CUBIC)
    big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)

    canvas = np.full((disp_h + BAR_H, DISP_W, 3), 40, dtype=np.uint8)
    canvas[:disp_h] = big
    y0 = disp_h
    cv2.putText(canvas, f"[{idx}/{total}]  saved:{saved}", (12, y0 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(canvas, f"> {typed}_", (12, y0 + 68),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(canvas, "ENTER=save  TAB=skip  BKSP=del  ESC=quit",
                (12, y0 + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "valid", "test", "all"])
    ap.add_argument("--limit", type=int, default=150, help="Max NEW crops this session.")
    args = ap.parse_args()

    import cv2
    import numpy as np

    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
    crops = []
    for s in splits:
        d = CROPS_ROOT / s
        if d.is_dir():
            crops += sorted(str(p) for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not crops:
        print(f"[X] no crops under {CROPS_ROOT}. Run crop_numbers.py first.")
        sys.exit(1)

    done = load_done(OUT_CSV)
    todo = [c for c in crops
            if Path(c).relative_to(PROJECT_ROOT).as_posix() not in done][:args.limit]
    if not todo:
        print("Nothing new to label for this split (all done). "
              "Increase --limit or pick another split.")
        return

    print(f"already labelled: {len(done)} | to do this session: {len(todo)}")
    print("A window will open — type into it. ENTER=save, TAB=skip, ESC=quit.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUT_CSV.exists()
    win = "Label plate number"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    saved = skipped = 0
    quit_flag = False

    with open(OUT_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["image_path", "plate_text"])

        for i, path in enumerate(todo, 1):
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            typed = ""
            while True:
                cv2.imshow(win, build_canvas(cv2, np, gray, typed, i, len(todo), saved))
                k = cv2.waitKey(0) & 0xFF
                if k == 27:                       # ESC -> quit
                    quit_flag = True
                    break
                elif k in (13, 10):               # ENTER -> save
                    if typed.strip():
                        rel = Path(path).relative_to(PROJECT_ROOT).as_posix()
                        writer.writerow([rel, typed.strip()])
                        f.flush()
                        saved += 1
                    break
                elif k == 9:                      # TAB -> skip
                    skipped += 1
                    break
                elif k == 8:                      # BACKSPACE
                    typed = typed[:-1]
                elif k != 255:
                    ch = chr(k).upper()
                    if ch in ALLOWED:
                        typed += ch
            if quit_flag:
                break

    cv2.destroyAllWindows()
    print("-" * 50)
    print(f"saved {saved} new labels, skipped {skipped}. total now: {len(done) + saved}")
    print(f"CSV -> {OUT_CSV}")
    print("\nNext: python scripts/recognition/evaluate_crnn_on_real.py --split test")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted — progress saved.")
    except Exception as exc:
        print(f"[X] labelling failed: {exc}")
        sys.exit(1)
