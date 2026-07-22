#!/usr/bin/env python3
"""
scripts/system/benchmark_composed.py
=====================================
IMPROVEMENT_ROADMAP item 1.1 — composed-plate end-to-end benchmark.

Runs the REAL integrated pipeline (ALPRSystem) on the human-labeled real test
frames and reports the metrics the roadmap asks for:

  * detection rate                         (did YOLO find a plate at all?)
  * number end-to-end exact-match accuracy (pred number == ground-truth number)
  * number CER                             (character error rate)
  * number-failure breakdown               (length errors, top confusions)
  * composed-plate exact-match accuracy    (province + number) -- see note below
  * false-accept rate                      (a whitelist of 1-edit NEIGHBOURS of the
                                            reads must NEVER open the gate)

Ground-truth note
-----------------
`data/crnn_crops/real_labels.csv` gives ground-truth NUMBERS for the 149 test
frames, but there is currently **no province ground truth aligned to those
frames**. So true composed exact-match cannot be *measured* yet -- it is reported
as an ESTIMATE (number_acc x province_test_acc) until province labels exist.
Provide them via --province-gt <csv> (columns: image,province_class) to measure
it for real.

Side effects are isolated: a throwaway temp DB / photo dir is used, so your real
`plates.db` and `photos/` are never touched.

Run:
    python scripts/system/benchmark_composed.py
    python scripts/system/benchmark_composed.py --limit 40      # quick sweep
    python scripts/system/benchmark_composed.py --province-gt data/crnn_crops/province_test_labels.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CONFIG = PROJECT_ROOT / "configs" / "system_config.yaml"
REAL_CSV = PROJECT_ROOT / "data" / "crnn_crops" / "real_labels.csv"
TEST_IMAGES = PROJECT_ROOT / "data" / "annotated" / "test" / "images"
METRICS_OUT = PROJECT_ROOT / "metrics" / "composed_benchmark.json"
PROVINCE_TEST_ACC = 0.9718        # classifier test accuracy (from HANDOFF metrics)


# --------------------------------------------------------------------------- #
# small text helpers
# --------------------------------------------------------------------------- #
def levenshtein(a: str, b: str) -> int:
    """Edit distance (used for CER and for building near-neighbour plates)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm_num(s: str) -> str:
    """Normalise a plate number for comparison: upper, strip spaces."""
    return (s or "").upper().replace(" ", "").strip()


def one_edit_neighbour(num: str) -> str:
    """A deterministic 1-character-different, still-plausible neighbour of a
    plate number (used to stress-test false accepts). Swaps one digit."""
    chars = list(num)
    for i, c in enumerate(chars):
        if c.isdigit():
            chars[i] = str((int(c) + 1) % 10)     # 9->0, keeps it a digit
            return "".join(chars)
    # no digit -> bump the first alpha
    for i, c in enumerate(chars):
        if c.isalpha():
            chars[i] = "A" if c.upper() != "A" else "B"
            return "".join(chars)
    return num + "0"


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_test_rows(limit: int | None) -> list[dict]:
    """Human-labeled TEST rows joined to their full-frame image path."""
    rows = []
    with open(REAL_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if "crnn_crops/test/" not in r["image_path"].replace("\\", "/"):
                continue
            fn = Path(r["image_path"]).name
            img = TEST_IMAGES / fn
            if not img.exists():
                continue
            rows.append({"file": fn, "image": img, "gt_number": norm_num(r["plate_text"])})
    rows.sort(key=lambda x: x["file"])
    if limit:
        rows = rows[:limit]
    return rows


def load_province_gt(path: Path | None) -> tuple[dict[str, int], int]:
    """Optional province ground truth: ({image_filename: province_class}, n_skipped).

    Rows whose `note` starts with UNVERIFIABLE are SKIPPED, not trusted: these are
    frames a human could not read the province from (too low-resolution, or the
    pre-fill classified a different plate in a multi-plate frame). Scoring against
    a label nobody could verify would silently corrupt the headline metric, so they
    are excluded from the measured composed accuracy and reported separately.
    """
    if not path or not Path(path).exists():
        return {}, 0
    gt, skipped = {}, 0
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = Path(r.get("image") or r.get("image_path") or "").name
            if (r.get("note") or "").strip().upper().startswith("UNVERIFIABLE"):
                skipped += 1
                continue
            try:
                gt[key] = int(r["province_class"])
            except (KeyError, ValueError):
                continue
    return gt, skipped


# --------------------------------------------------------------------------- #
# isolated ALPRSystem (temp DB + temp photo/output dir)
# --------------------------------------------------------------------------- #
def make_isolated_system(tmp: Path):
    """Instantiate ALPRSystem against a temp config so the real plates.db and
    photos/ are never written to. Returns the ALPRSystem."""
    import yaml
    from core.alpr_system import ALPRSystem

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["db_path"] = str(tmp / "bench.db")
    cfg.setdefault("output", {})
    cfg["output"]["photo_dir"] = str(tmp / "photos")
    cfg["output"]["output_dir"] = str(tmp / "out")
    cfg["output"]["save_annotated"] = False
    cfg.setdefault("logging", {})["log_dir"] = str(tmp / "logs")
    cfg["camera_source"] = str(TEST_IMAGES)           # not used, but keep valid
    tmp_cfg = tmp / "bench_config.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return ALPRSystem(str(tmp_cfg))


def best_plate(res: dict) -> dict | None:
    """Pick the highest-detection-confidence plate from a process_frame result."""
    plates = res.get("plates") or []
    if not plates:
        return None
    return max(plates, key=lambda p: p.get("confidence", 0.0))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only first N test frames")
    ap.add_argument("--province-gt", type=str, default=None,
                    help="CSV of province ground truth (columns: image,province_class)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    rows = load_test_rows(args.limit)
    if not rows:
        print("[X] no test rows found — check data/crnn_crops/real_labels.csv "
              "and data/annotated/test/images/")
        sys.exit(1)
    prov_gt, prov_skipped = load_province_gt(
        Path(args.province_gt) if args.province_gt else None)

    import cv2
    tmp = Path(tempfile.mkdtemp(prefix="alpr_bench_"))
    print(f"[bench] {len(rows)} real test frames | isolated temp: {tmp}")
    print("[bench] loading models (first run takes a few seconds)...\n")

    try:
        system = make_isolated_system(tmp)
        conf_threshold = system.crnn_conf_threshold      # REC-005 gate (0.70)
        from recognition.province_map import compose_plate

        records = []
        detected = 0
        for i, row in enumerate(rows, 1):
            frame = cv2.imread(str(row["image"]))
            if frame is None:
                continue
            res = system.process_frame(frame)
            pl = best_plate(res)
            if pl is None:
                records.append({**row, "detected": False, "pred_number": "",
                                "pred_prov": None, "plate_text": ""})
                continue
            detected += 1
            records.append({
                "file": row["file"],
                "gt_number": row["gt_number"],
                "detected": True,
                "pred_number": norm_num(pl["number"]),
                "pred_prov": pl.get("province_id"),
                "plate_text": pl.get("plate_text", ""),
                "crnn_conf": pl.get("crnn_confidence", 0.0),
                "action": pl.get("action", ""),
                "consistency_reasons": pl.get("consistency_reasons", []),
            })
            if i % 25 == 0:
                print(f"  processed {i}/{len(rows)} ...")

        # ---- number end-to-end accuracy + CER ---------------------------- #
        n = len(records)
        num_correct = sum(1 for r in records if r["detected"]
                          and r["pred_number"] == r["gt_number"])
        tot_chars = sum(len(r["gt_number"]) for r in records)
        tot_edits = sum(levenshtein(r["pred_number"], r["gt_number"]) for r in records)
        cer = tot_edits / max(1, tot_chars)
        num_acc = num_correct / max(1, n)
        det_rate = detected / max(1, n)

        # ---- number-failure breakdown ------------------------------------ #
        wrong = [r for r in records if r["detected"] and r["pred_number"] != r["gt_number"]]
        len_mismatch = sum(1 for r in wrong if len(r["pred_number"]) != len(r["gt_number"]))
        no_detect = sum(1 for r in records if not r["detected"])

        # ---- composed-plate accuracy (measured if GT given, else estimate) #
        if prov_gt:
            composed_ok = composed_total = 0
            prov_ok = 0
            for r in records:
                if r["file"] not in prov_gt:
                    continue
                composed_total += 1
                gt_comp = norm_num(compose_plate(prov_gt[r["file"]], r["gt_number"]))
                pred_comp = norm_num(r["plate_text"])
                composed_ok += (gt_comp == pred_comp)
                prov_ok += (r["pred_prov"] == prov_gt[r["file"]])
            composed_acc = composed_ok / max(1, composed_total)
            prov_acc = prov_ok / max(1, composed_total)
            composed_kind = "measured"
        else:
            composed_acc = num_acc * PROVINCE_TEST_ACC     # independence estimate
            prov_acc = PROVINCE_TEST_ACC
            composed_total = 0
            composed_kind = "ESTIMATE (no province GT for these frames)"

        # ---- false-accept safety harness --------------------------------- #
        # Whitelist = 1-edit NEIGHBOURS of the (correct) ground-truth numbers.
        # None of these plates are actually present, so a correct + fail-safe
        # system must open the gate ZERO times. This is the baseline that item
        # 1.2 (constrained decoding) must not regress.
        neighbours = {one_edit_neighbour(r["gt_number"]) for r in records if r["gt_number"]}
        neighbours = {nb for nb in neighbours
                      if nb not in {r["gt_number"] for r in records}}
        system.database  # keep ref; register neighbours as authorised
        for nb in neighbours:
            # register as a bare number (matches how these 'other'-province reads compose)
            system.database.add_plate(nb, "FALSE_ACCEPT_PROBE", "car", "bench neighbour")
        false_accepts = 0
        for r in records:
            if not r["detected"]:
                continue
            # would THIS read open a gate whose whitelist is only wrong neighbours?
            if system.database.is_registered(r["plate_text"]) or \
               system.database.is_registered(r["pred_number"]):
                # only counts as a FALSE accept if the true plate isn't in the list
                if r["gt_number"] not in neighbours:
                    false_accepts += 1
        far = false_accepts / max(1, detected)

        system.close()

        # ---- gate simulation: EXACT vs 1.2 CONSTRAINED matching ---------- #
        # Province GT is unavailable, so this simulates at the NUMBER level:
        # every detected legit plate is "registered" under its ground-truth
        # number, and we compare how the gate responds. This isolates exactly
        # what item 1.2 changes (matching the read against the whitelist).
        # Faithful to the real gate: a read only auto-opens if it is CONFIDENT
        # (crnn_conf >= threshold, REC-005) AND an exact whitelist hit.
        det_recs = [r for r in records if r["detected"]]
        authorized = {r["gt_number"] for r in det_recs if r["gt_number"]}
        exact_open = recovered_review = wrong_review = hard_deny = low_conf_review = 0
        for r in det_recs:
            if r.get("crnn_conf", 0.0) < conf_threshold:
                low_conf_review += 1          # REC-005: not confident -> REVIEW
                continue
            if r["pred_number"] == r["gt_number"]:
                exact_open += 1
                continue
            d, nearest = min((levenshtein(r["pred_number"], a), a) for a in authorized)
            if d <= 1:
                if nearest == r["gt_number"]:
                    recovered_review += 1     # correct suggestion (was silent DENY)
                else:
                    wrong_review += 1         # different plate -> REVIEW, still no open
            else:
                hard_deny += 1
        # intruder false-open probe, faithful to the real gate: a CONFIDENT read
        # that EXACT-matches a DIFFERENT registered plate would wrongly open.
        # (Independent of 1.2 constrained matching, which never auto-opens.)
        intruder_false_open = 0
        for r in det_recs:
            if r.get("crnn_conf", 0.0) < conf_threshold:
                continue
            if r["pred_number"] in (authorized - {r["gt_number"]}):
                intruder_false_open += 1
        nd = max(1, len(det_recs))

        # ---- 2.2 consistency-flag measurement ---------------------------- #
        # The pipeline ran with an EMPTY whitelist, so every confident read that
        # 2.2 flags shows up as a REVIEW it converted from a would-be DENY. Cross-
        # tab the flag against number correctness: a flag on a WRONG read is a good
        # catch; a flag on a CORRECT read is an over-trigger (lost throughput).
        flagged = [r for r in det_recs if r.get("consistency_reasons")]
        flag_wrong = sum(1 for r in flagged if r["pred_number"] != r["gt_number"])
        flag_right = len(flagged) - flag_wrong
        reason_counts: dict[str, int] = {}
        for r in flagged:
            for why in r["consistency_reasons"]:
                reason_counts[why] = reason_counts.get(why, 0) + 1
        flag_precision = flag_wrong / max(1, len(flagged))

        # ---- report ------------------------------------------------------ #
        def pct(x): return f"{x*100:.2f}%"
        print("\n" + "=" * 62)
        print(" COMPOSED-PLATE BENCHMARK  (roadmap 1.1)")
        print("=" * 62)
        print(f" test frames              : {n}")
        print(f" detection rate           : {pct(det_rate)}  ({detected}/{n})")
        print("-" * 62)
        print(f" NUMBER end-to-end acc    : {pct(num_acc)}  ({num_correct}/{n})")
        print(f" NUMBER CER               : {pct(cer)}")
        print(f"   failures               : {len(wrong)}  "
              f"(length-wrong {len_mismatch}, not-detected {no_detect})")
        print("-" * 62)
        print(f" province acc             : {pct(prov_acc)}")
        print(f" COMPOSED exact-match     : {pct(composed_acc)}   [{composed_kind}]")
        if prov_gt:
            print(f"   measured on            : {composed_total} frames"
                  + (f"  ({prov_skipped} excluded as UNVERIFIABLE)" if prov_skipped else ""))
        if not prov_gt:
            print("   -> to MEASURE this, pass --province-gt <csv> with province")
            print("      labels for the test frames (columns: image,province_class)")
        print("-" * 62)
        print(f" FALSE-ACCEPT harness     : {false_accepts} opens on "
              f"{len(neighbours)} wrong-neighbour plates  -> FAR {pct(far)}")
        print(f"   {'PASS (fail-safe holds)' if false_accepts == 0 else '*** FAIL — gate opened for a wrong plate ***'}")
        print("-" * 62)
        print(" GATE SIMULATION (number-level, confidence-faithful)")
        print(f"   confident auto-open (exact)   : {exact_open}/{len(det_recs)}  ({pct(exact_open/nd)})")
        print(f"   low-confidence -> REVIEW      : {low_conf_review}  (REC-005)")
        print(f"   1.2 recovered DENY->REVIEW    : {recovered_review}  "
              f"(legit plate, 1-char misread, now flagged not denied)")
        print(f"   1.2 other near-match -> REVIEW: {wrong_review}")
        print(f"   still hard DENY               : {hard_deny}")
        print(f"   intruder false auto-open      : {intruder_false_open}  "
              f"{'PASS (confidence gate blocks it)' if intruder_false_open == 0 else '*** pre-existing exact-match risk — see note ***'}")
        print("   note: 1.2 constrained matching NEVER auto-opens — recovered reads")
        print("         become REVIEW_REQUIRED, so it adds ZERO false-accept risk.")
        print("-" * 62)
        print(" 2.2 CONSISTENCY FLAGS (province<->number), whitelist empty")
        print(f"   reads flagged -> REVIEW       : {len(flagged)}/{len(det_recs)}")
        print(f"     on a WRONG number (catch)   : {flag_wrong}")
        print(f"     on a CORRECT number (noise) : {flag_right}")
        print(f"   flag precision (wrong|flagged): {pct(flag_precision)}")
        if reason_counts:
            print(f"   reasons                      : "
                  + ", ".join(f"{k}={v}" for k, v in sorted(reason_counts.items())))
        print("   note: 2.2 never downgrades a confirmed ALLOW; it only converts")
        print("         suspect DENYs into REVIEWs. Full tuning needs province GT.")
        print("=" * 62)

        METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
        METRICS_OUT.write_text(json.dumps({
            "n_frames": n,
            "detection_rate": det_rate,
            "number_accuracy": num_acc,
            "number_cer": cer,
            "number_failures": len(wrong),
            "province_accuracy": prov_acc,
            "composed_accuracy": composed_acc,
            "composed_kind": composed_kind,
            "composed_measured_on": composed_total,
            "false_accepts": false_accepts,
            "false_accept_rate": far,
            "neighbour_whitelist_size": len(neighbours),
            "gate_sim": {
                "detected": len(det_recs),
                "exact_open": exact_open,
                "low_conf_review": low_conf_review,
                "recovered_review": recovered_review,
                "other_near_review": wrong_review,
                "hard_deny": hard_deny,
                "intruder_false_open": intruder_false_open,
            },
            "consistency_2_2": {
                "flagged": len(flagged),
                "flag_on_wrong": flag_wrong,
                "flag_on_correct": flag_right,
                "flag_precision": flag_precision,
                "reasons": reason_counts,
            },
        }, indent=2), encoding="utf-8")
        print(f"\n[bench] wrote {METRICS_OUT.relative_to(PROJECT_ROOT)}")

        # ROADMAP 3.2: append headline metrics to the experiment log (with commit)
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools"))
            from experiment_log import log_metric
            note = f"benchmark_composed.py (n={n})"
            log_metric("pipeline", "number_e2e_accuracy", round(num_acc, 4),
                       split="real-test", notes=note)
            log_metric("pipeline", "number_cer", round(cer, 4),
                       split="real-test", notes=note)
            log_metric("pipeline", "detection_rate", round(det_rate, 4),
                       split="real-test", notes=note)
            # the roadmap-V2 headline metric: only logged when actually MEASURED
            # against province ground truth, never the independence estimate.
            if prov_gt:
                log_metric("pipeline", "composed_exact_match", round(composed_acc, 4),
                           split="real-test",
                           notes=f"measured on {composed_total} frames"
                                 + (f", {prov_skipped} unverifiable excluded"
                                    if prov_skipped else ""))
                log_metric("pipeline", "province_accuracy", round(prov_acc, 4),
                           split="real-test", notes=f"measured, n={composed_total}")
            log_metric("pipeline", "false_accept_rate", round(far, 4),
                       split="real-test", notes="neighbour harness")
            log_metric("pipeline", "consistency_flag_precision",
                       round(flag_precision, 4), split="real-test",
                       notes=f"2.2, {len(flagged)} flagged")
            print("[bench] appended headline metrics to metrics/experiment_log.csv")
        except Exception as exc:
            print(f"[bench] (experiment-log append skipped: {exc})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
