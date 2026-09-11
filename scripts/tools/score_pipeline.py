#!/usr/bin/env python3
"""
scripts/tools/score_pipeline.py
===============================
Score the WHOLE pipeline against ground truth — the honest end-to-end number.

`main.py demo` shows you what the system reads, but not whether it is RIGHT.
This runs the same pipeline over the 149 labelled test scenes and compares both
outputs to the answer key in data/crnn_crops/province_test_labels.csv:

    detector -> number crop  -> CRNN            vs gt_number
    detector -> province crop -> classifier     vs province_class

    python scripts/tools/score_pipeline.py
    python scripts/tools/score_pipeline.py --limit 30 --show-all
    python scripts/tools/score_pipeline.py --trim 0.0     # compare settings

Every mistake is written to results/pipeline_mistakes.csv so you can look at the
actual images rather than guess.

These are FULL SCENES, fed straight in — no camera, no screen. Photographing a
monitor adds moire and glare the models never trained on and tells you nothing
about real performance.
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "recognition"))

KEY = PROJECT_ROOT / "data" / "crnn_crops" / "province_test_labels.csv"
IMAGES = PROJECT_ROOT / "data" / "annotated" / "test" / "images"
OUT = PROJECT_ROOT / "results" / "pipeline_mistakes.csv"


def norm(s):
    return (s or "").upper().replace(" ", "").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N (0 = all)")
    ap.add_argument("--trim", type=float, default=None,
                    help="override gate.province_crop_trim (to compare settings)")
    ap.add_argument("--show-all", action="store_true",
                    help="print every image, not just the mistakes")
    ap.add_argument("--province-weights", type=Path, default=None,
                    help="override gate/province_weights, to compare candidates")
    ap.add_argument("--rotate180", action="store_true",
                    help="turn each SCENE upside-down first. This is the honest "
                         "end-to-end rotation test: the DETECTOR has to find a "
                         "flipped plate too, which reading pre-cut crops never "
                         "checks. The detector was never rotation-trained, so "
                         "expect it to be the bottleneck, not the reader.")
    args = ap.parse_args()

    import cv2
    import yaml
    from detection.detector import PlateDetector
    from crnn_reader import CRNNReader
    from province_classifier import ProvinceClassifier
    from province_map import PROVINCE_LATIN
    from core.alpr_system import ALPRSystem

    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "system_config.yaml").read_text("utf-8"))
    gate = cfg.get("gate", {})
    trim = args.trim if args.trim is not None else float(gate.get("province_crop_trim", 0.125))
    pad = float(cfg.get("detection", {}).get("crop_padding", 0.10))

    det = PlateDetector(PROJECT_ROOT / cfg["yolo_weights"],
                        conf=gate.get("yolo_confidence_threshold", 0.5))
    num_det = None
    nw = PROJECT_ROOT / cfg.get("number_weights", "")
    if cfg.get("number_weights") and nw.exists():
        num_det = PlateDetector(nw, conf=gate.get("number_confidence_threshold", 0.4))
    reader = CRNNReader(PROJECT_ROOT / cfg["crnn_weights"],
                        PROJECT_ROOT / cfg["charset_path"],
                        format_validation=gate.get("format_validation", False),
                        confidence_mode=gate.get("confidence_mode", "mean"))
    prov_w = args.province_weights or PROJECT_ROOT / cfg.get(
        "province_weights", "models/recognition/province_classifier_best.pth")
    clf = ProvinceClassifier(prov_w)

    def prov_name(i):
        return PROVINCE_LATIN[i] if 0 <= i < len(PROVINCE_LATIN) else "other"

    rows = list(csv.DictReader(open(KEY, encoding="utf-8")))
    if args.limit:
        rows = rows[:args.limit]

    print("=" * 74)
    print(f" PIPELINE SCORE vs GROUND TRUTH   ({len(rows)} labelled scenes)")
    print(f" reader={Path(cfg['crnn_weights']).name}  province={Path(prov_w).name}"
          f"  trim={trim}"
          + ("   [SCENES ROTATED 180]" if args.rotate180 else ""))
    print("=" * 74)

    n = miss = num_ok = prov_ok = both_ok = 0
    mistakes = []
    for r in rows:
        img = cv2.imread(str(IMAGES / r["image"]))
        if img is None:
            continue
        if args.rotate180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        n += 1
        gt_num, gt_prov = norm(r["gt_number"]), int(r["province_class"])

        dets = det.detect(img, pad=pad)
        if not dets:
            miss += 1
            mistakes.append([r["image"], gt_num, "(no detection)",
                             prov_name(gt_prov), "(no detection)"])
            if args.show_all:
                print(f"  MISS  {r['image'][:34]:<34} detector found nothing")
            continue

        d = dets[0]
        crop = d["crop"]
        if num_det is not None:
            nd = num_det.detect(img, pad=pad)
            if nd:
                crop_num = nd[0]["crop"]
            else:
                crop_num = d["crop"]
        else:
            crop_num = d["crop"]

        read, _ = reader.read(crop_num)
        pid, pconf = clf.predict(ALPRSystem._tighten_crop(crop, trim))

        ngood = norm(read) == gt_num
        pgood = pid == gt_prov
        num_ok += ngood
        prov_ok += pgood
        both_ok += (ngood and pgood)

        if not (ngood and pgood):
            mistakes.append([r["image"], gt_num, read or "(empty)",
                             prov_name(gt_prov), prov_name(pid)])
        if args.show_all or not (ngood and pgood):
            mark = "ok  " if (ngood and pgood) else "WRONG"
            print(f"  {mark} {r['image'][:30]:<30} "
                  f"num {gt_num:<9}-> {norm(read) or '(empty)':<9} {'Y' if ngood else 'N'} | "
                  f"prov {prov_name(gt_prov)[:14]:<14}-> {prov_name(pid)[:14]:<14} "
                  f"{'Y' if pgood else 'N'}")

    print("=" * 74)
    print(f"  scenes scored        : {n}")
    print(f"  detector found       : {n - miss}   (missed {miss})")
    print(f"  NUMBER correct       : {100*num_ok/max(n,1):5.1f}%   ({num_ok}/{n})")
    print(f"  PROVINCE correct     : {100*prov_ok/max(n,1):5.1f}%   ({prov_ok}/{n})")
    print(f"  BOTH correct         : {100*both_ok/max(n,1):5.1f}%   ({both_ok}/{n})")
    print("=" * 74)

    if args.rotate180:
        print("  NOTE: the detector was never trained on rotated scenes, so a low")
        print("  score here is a DETECTION limit, not a reading limit. Compare the")
        print("  'detector found' line against the upright run.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "gt_number", "read_number", "gt_province", "read_province"])
        w.writerows(mistakes)
    print(f"  {len(mistakes)} mistakes -> {OUT.relative_to(PROJECT_ROOT)}")
    print("  Open that CSV and look at the named images to see WHY they failed.")


if __name__ == "__main__":
    main()
