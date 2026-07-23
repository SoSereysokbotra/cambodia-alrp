#!/usr/bin/env python3
"""
scripts/system/benchmark_visits.py
==================================
IMPROVEMENT_PLAN_V2 Phase 3 — the VISIT-level benchmark.

Why this exists
---------------
`benchmark_composed.py` grades 149 independent still frames. The real gate never
sees a still: it sees one car for ~40 frames and makes ONE decision. That mismatch
means the frame-level harness structurally cannot measure a multi-frame fix, so it
must exist before Phase 4 (read fusion) is built — the same "build the instrument
first" discipline that made ROADMAP 1.1 precede 1.2.

Where the data comes from (no new capture, no re-inference)
-----------------------------------------------------------
`photos/` holds one annotated evidence frame per processed frame, named
`plate_YYYYMMDD_HHMMSS_mmm_<TEXT>.jpg` — so **the per-frame CRNN read is in the
filename**, and `plates.db.plate_reads.photo_path` joins each frame to its
`crnn_confidence`. That is everything a fusion strategy needs, which means this
benchmark runs in seconds and never re-runs inference.

It also avoids a trap: the saved photos are ANNOTATED (detection boxes drawn on
them), so replaying the image files through the detector would NOT be a faithful
reconstruction. Reading the recorded text+confidence instead sidesteps that.

Consequence: this harness measures the NUMBER-FUSION decision only. It cannot
measure detection or province. That is the right scope — Phase 0 established that
the number branch is the entire composed-accuracy gap.

Data hygiene (see the honest-limits section in the docs)
--------------------------------------------------------
* **640x640 frames are excluded.** That is the Roboflow export resolution, i.e.
  replays of the annotated train/test images through the pipeline. 664 of 6,122
  photos (10.8%) are 640x640, and keeping them would leak the 149-frame test set
  into a benchmark. Live captures are 1920x1080 / 720x480 / 1080x1080.
* Ground truth is HUMAN-supplied per visit (`--labels`), never the model's own
  majority read — scoring a fusion strategy against a consensus label would let
  voting win by construction.

Usage
-----
    python scripts/system/benchmark_visits.py --build      # manifest + label sheets
    python scripts/system/benchmark_visits.py --score      # score the strategies
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "recognition"))

PHOTOS = PROJECT_ROOT / "photos"
DB = PROJECT_ROOT / "plates.db"
MANIFEST = PROJECT_ROOT / "metrics" / "visits_manifest.json"
LABELS = PROJECT_ROOT / "metrics" / "visits_labels.csv"
SHEETS = PROJECT_ROOT / "results" / "visit_sheets"
OUT = PROJECT_ROOT / "metrics" / "visit_benchmark.json"

# Roboflow export size — replays of the annotated dataset, NOT live gate captures.
EXCLUDED_SIZE = (640, 640)
# Live de-duplication rule, mirrored from configs/system_config.yaml `logging:`
DEDUP_GAP_SEC = 3.0
DEDUP_MERGE_EDITS = 2
MIN_FRAMES = 5          # a "visit" worth scoring; 1-2 frame blips are not visits

NAME_RE = re.compile(r"^plate_(\d{8}_\d{6}_\d{3})_(.*)\.jpg$")


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------- #
# loading + grouping
# --------------------------------------------------------------------------- #
def load_frames() -> tuple[list[dict], dict]:
    """Every live evidence frame as {t, text, conf, file}, plus exclusion stats."""
    from PIL import Image

    conf_by_name: dict[str, float] = {}
    if DB.exists():
        con = sqlite3.connect(DB)
        for path, conf in con.execute(
                "select photo_path, crnn_confidence from plate_reads "
                "where photo_path is not null and photo_path != ''"):
            conf_by_name[Path(path).name] = float(conf or 0.0)
        con.close()

    frames, n_excluded, n_unparsed, n_noconf = [], 0, 0, 0
    for p in PHOTOS.glob("plate_*.jpg"):
        m = NAME_RE.match(p.name)
        if not m:
            n_unparsed += 1
            continue
        try:
            with Image.open(p) as im:
                size = im.size
        except Exception:
            n_unparsed += 1
            continue
        if size == EXCLUDED_SIZE:
            n_excluded += 1           # Roboflow replay -> test-set hygiene
            continue
        if p.name not in conf_by_name:
            n_noconf += 1
        frames.append({
            "t": datetime.strptime(m.group(1), "%Y%m%d_%H%M%S_%f"),
            "text": m.group(2),
            "conf": conf_by_name.get(p.name, 0.0),
            "file": p.name,
        })
    frames.sort(key=lambda x: x["t"])
    return frames, {"excluded_640": n_excluded, "unparsed": n_unparsed,
                    "no_confidence": n_noconf}


def group_visits(frames: list[dict]) -> list[list[dict]]:
    """Split the frame stream into visits using the LIVE de-duplication rule:
    a frame continues the current visit if it is within DEDUP_GAP_SEC of the
    previous one AND within DEDUP_MERGE_EDITS of it. Mirroring the deployed rule
    keeps the benchmark faithful to how the system actually batches a car."""
    if not frames:
        return []
    visits, cur = [], [frames[0]]
    for a, b in zip(frames, frames[1:]):
        gap = (b["t"] - a["t"]).total_seconds()
        if gap <= DEDUP_GAP_SEC and levenshtein(a["text"], b["text"]) <= DEDUP_MERGE_EDITS:
            cur.append(b)
        else:
            visits.append(cur)
            cur = [b]
    visits.append(cur)
    return visits


# --------------------------------------------------------------------------- #
# fusion strategies — what Phase 4 will be judged against
# --------------------------------------------------------------------------- #
def strat_best_conf(visit: list[dict]) -> str:
    """CURRENT deployed behaviour: keep the single highest-confidence frame and
    discard the rest (`ALPRSystem._dedup_persist`). The baseline to beat."""
    return max(visit, key=lambda f: f["conf"])["text"]


def strat_majority(visit: list[dict]) -> str:
    """Most common exact string across the visit (confidence-weighted ties)."""
    score: dict[str, float] = defaultdict(float)
    for f in visit:
        score[f["text"]] += 1.0
    best = max(score.items(), key=lambda kv: (kv[1], sum(
        g["conf"] for g in visit if g["text"] == kv[0])))
    return best[0]


def strat_char_vote(visit: list[dict]) -> str:
    """Per-position, confidence-weighted character vote.

    Length is decided first by a weighted vote (a plate has one true length), then
    each position is voted on using only the frames of that length. This is the
    cheap stand-in for the Phase 4 aggregator: it uses every frame's evidence
    instead of betting the decision on one frame.
    """
    if not visit:
        return ""
    by_len: dict[int, float] = defaultdict(float)
    for f in visit:
        by_len[len(f["text"])] += max(f["conf"], 1e-6)
    target = max(by_len.items(), key=lambda kv: kv[1])[0]
    pool = [f for f in visit if len(f["text"]) == target]
    if not pool:
        return strat_best_conf(visit)
    out = []
    for i in range(target):
        votes: dict[str, float] = defaultdict(float)
        for f in pool:
            votes[f["text"][i]] += max(f["conf"], 1e-6)
        out.append(max(votes.items(), key=lambda kv: kv[1])[0])
    return "".join(out)


STRATEGIES = {
    "best_conf (current)": strat_best_conf,
    "majority": strat_majority,
    "char_vote": strat_char_vote,
}


# --------------------------------------------------------------------------- #
# build / score
# --------------------------------------------------------------------------- #
def build() -> None:
    frames, stats = load_frames()
    visits = [v for v in group_visits(frames) if len(v) >= MIN_FRAMES]
    print(f"[build] live frames {len(frames)} "
          f"(excluded {stats['excluded_640']} Roboflow 640x640 replays)")
    print(f"[build] visits with >= {MIN_FRAMES} frames: {len(visits)}")

    records = []
    for v in visits:
        cnt = Counter(f["text"] for f in v)
        records.append({
            "visit_id": v[0]["file"],
            "start": v[0]["t"].isoformat(),
            "n_frames": len(v),
            "modal_read": cnt.most_common(1)[0][0],
            "distinct_reads": len(cnt),
            "frames": [{"file": f["file"], "text": f["text"],
                        "conf": round(f["conf"], 4)} for f in v],
        })
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[build] wrote {MANIFEST.relative_to(PROJECT_ROOT)}")

    if not LABELS.exists():
        with open(LABELS, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["visit_id", "gt_plate", "note"])
            for r in records:
                w.writerow([r["visit_id"], "", ""])
        print(f"[build] wrote EMPTY label template {LABELS.relative_to(PROJECT_ROOT)}")
        print("        Fill gt_plate by LOOKING at a frame of each visit — never")
        print("        paste the modal read, or voting wins by construction.")
    else:
        print(f"[build] {LABELS.relative_to(PROJECT_ROOT)} exists — left untouched")


def load_labels() -> dict[str, str]:
    if not LABELS.exists():
        return {}
    out = {}
    with open(LABELS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt = (r.get("gt_plate") or "").strip().upper()
            if gt and not gt.startswith("SKIP"):
                out[r["visit_id"]] = gt
    return out


def score() -> None:
    if not MANIFEST.exists():
        print("[X] no manifest — run --build first")
        sys.exit(1)
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))
    labels = load_labels()
    if not labels:
        print(f"[X] no labels in {LABELS.relative_to(PROJECT_ROOT)} — "
              "fill gt_plate for at least some visits")
        sys.exit(1)

    scored = [r for r in records if r["visit_id"] in labels]
    print(f"[score] {len(scored)} labelled visits of {len(records)}")

    # Plate-level macro average corrects the corpus imbalance: one vehicle
    # dominates the visit count, so a plain per-visit mean would mostly report
    # performance on that single car.
    results = {}
    for name, fn in STRATEGIES.items():
        per_visit = []
        by_plate: dict[str, list[int]] = defaultdict(list)
        for r in scored:
            gt = labels[r["visit_id"]]
            pred = fn([{"text": f["text"], "conf": f["conf"]} for f in r["frames"]])
            ok = int(pred.upper() == gt)
            per_visit.append(ok)
            by_plate[gt].append(ok)
        micro = sum(per_visit) / max(1, len(per_visit))
        macro = (sum(sum(v) / len(v) for v in by_plate.values()) / len(by_plate)
                 if by_plate else 0.0)
        results[name] = {"micro": micro, "macro": macro,
                         "correct": sum(per_visit), "n": len(per_visit),
                         "n_plates": len(by_plate)}

    n_plates = next(iter(results.values()))["n_plates"]
    print("\n" + "=" * 68)
    print(" VISIT-LEVEL BENCHMARK  (PLAN_V2 Phase 3)")
    print("=" * 68)
    print(f" labelled visits : {len(scored)}   distinct plates : {n_plates}")
    print("-" * 68)
    print(f" {'strategy':<22} {'per-visit':>12} {'per-plate':>12}")
    print(f" {'':<22} {'(micro)':>12} {'(macro)':>12}")
    for name, r in results.items():
        print(f" {name:<22} {r['micro']*100:>11.2f}% {r['macro']*100:>11.2f}%"
              f"   ({r['correct']}/{r['n']})")
    print("-" * 68)
    print(" macro = mean over DISTINCT plates. Prefer it: the visit corpus is")
    print(" dominated by one vehicle, so micro mostly reports that one car.")
    print("=" * 68)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "labelled_visits": len(scored), "total_visits": len(records),
        "distinct_plates": n_plates, "strategies": results,
    }, indent=2), encoding="utf-8")
    print(f"\n[score] wrote {OUT.relative_to(PROJECT_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="visit-level benchmark (PLAN_V2 Phase 3)")
    ap.add_argument("--build", action="store_true",
                    help="group frames into visits, write manifest + label template")
    ap.add_argument("--score", action="store_true",
                    help="score fusion strategies against the human labels")
    args = ap.parse_args()
    if args.build:
        build()
    elif args.score:
        score()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
