#!/usr/bin/env python3
"""
scripts/tools/make_label_sheet.py
=================================
Generate a self-contained HTML page for labelling number crops in your BROWSER
(no terminal / no OpenCV window). Each crop is enlarged + contrast-enhanced with
a text box under it; a "Download CSV" button saves your labels.

Two sources of crops
--------------------
1. A whole split (the original behaviour):
       python scripts/tools/make_label_sheet.py --split test --limit 150

2. The active-learning queue from harvest_active.py --scan (recommended — these
   are the crops that actually teach the model something):
       python scripts/tools/make_label_sheet.py --queue --best-first --limit 200

   `--best-first` reverses the queue so the READABLE crops come first. The queue
   is ranked hardest-first, and "hardest" is dominated by broken crops (motion
   blur, grille mesh), so labelling from the front wastes time.

Workflow:
    1. run one of the commands above
    2. open the printed .html file in your browser
    3. type the number under each crop (leave blank for illegible — a guess is
       worse than no label)
    4. click "Download CSV"  ->  saves to your Downloads folder
    5. python scripts/tools/import_label_csv.py <that downloaded csv>

Already-labelled crops (in data/crnn_crops/real_labels.csv) are excluded, so you
can generate a fresh sheet to continue. The test split is never emitted in
--queue mode.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
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
QUEUE = PROJECT_ROOT / "metrics" / "harvest_queue.csv"
RESULTS = PROJECT_ROOT / "results"
IMG_EXTS = {".jpg", ".jpeg", ".png"}
FORBIDDEN_SPLIT = "test"   # never label the held-out measurement set


def load_done() -> set[str]:
    done = set()
    if OUT_CSV.exists():
        with open(OUT_CSV, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["image_path"].replace("\\", "/"))
    return done


def encode_crop(cv2, np, path: str) -> str:
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return ""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enh = clahe.apply(gray)
    enh = cv2.resize(enh, (640, 128), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".png", enh)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def from_queue(done: set[str], best_first: bool, offset: int, limit: int):
    """Rows from the active-learning queue: (abs_path, pred_text, confidence)."""
    if not QUEUE.exists():
        print(f"[X] no queue at {QUEUE.relative_to(PROJECT_ROOT)} — run:")
        print("    python scripts/recognition/harvest_active.py --scan")
        sys.exit(1)
    rows = list(csv.DictReader(open(QUEUE, encoding="utf-8")))
    rows = [r for r in rows
            if f"/{FORBIDDEN_SPLIT}/" not in r["image_path"].replace("\\", "/")]
    if best_first:
        rows = rows[::-1]
    todo = []
    for r in rows:
        rel = r["image_path"].replace("\\", "/")
        if rel in done:
            continue
        p = PROJECT_ROOT / rel
        if p.exists():
            todo.append((str(p), r.get("pred_text", "") or "", r.get("confidence", "") or ""))
    return todo[offset:offset + limit]


def from_split(done: set[str], split: str, offset: int, limit: int):
    splits = ["train", "valid", "test"] if split == "all" else [split]
    crops = []
    for s in splits:
        d = CROPS_ROOT / s
        if d.is_dir():
            crops += sorted(str(p) for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not crops:
        print(f"[X] no crops under {CROPS_ROOT}. Run crop_numbers.py first.")
        sys.exit(1)
    todo = [c for c in crops
            if Path(c).relative_to(PROJECT_ROOT).as_posix() not in done]
    return [(c, "", "") for c in todo[offset:offset + limit]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "valid", "test", "all"])
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--offset", type=int, default=0, help="skip the first N rows")
    ap.add_argument("--queue", action="store_true",
                    help="take crops from metrics/harvest_queue.csv instead of a split")
    ap.add_argument("--best-first", action="store_true",
                    help="with --queue: readable crops first (reverses the ranking)")
    args = ap.parse_args()

    import cv2
    import numpy as np

    done = load_done()
    if args.queue:
        todo = from_queue(done, args.best_first, args.offset, args.limit)
        tag = "queue"
        heading = "active-learning queue" + (" (readable first)" if args.best_first else "")
    else:
        todo = from_split(done, args.split, args.offset, args.limit)
        tag = args.split
        heading = f"split: {args.split}"

    if not todo:
        print("Nothing new to label. Increase --limit or --offset, or re-run --scan.")
        return

    print(f"already labelled: {len(done)} | generating sheet for: {len(todo)} crops")

    cards = []
    for i, (path, pred, conf) in enumerate(todo):
        b64 = encode_crop(cv2, np, path)
        if not b64:
            continue
        rel = html.escape(Path(path).relative_to(PROJECT_ROOT).as_posix())
        pred_u = html.escape((pred or "").strip().upper())
        if pred_u:
            try:
                cf = f"{float(conf):.0%}"
            except (TypeError, ValueError):
                cf = "?"
            hint = (f'<div class="hint" data-pred="{pred_u}" onclick="useHint(this)">'
                    f'model guess: <b>{pred_u}</b> <span class="c">({cf} sure)</span>'
                    f' &mdash; click to use</div>')
        else:
            hint = '<div class="hint none">model read nothing</div>'
        cards.append(f'''
        <div class="card">
          <div class="idx">#{args.offset + i}</div>
          <img src="data:image/png;base64,{b64}" alt="crop">
          {hint}
          <input type="text" data-path="{rel}" autocomplete="off"
                 spellcheck="false" placeholder="type number (blank = skip)">
        </div>''')

    page = f'''<!doctype html><html><head><meta charset="utf-8">
<title>Label number crops ({tag})</title>
<style>
  body{{font-family:system-ui,Arial,sans-serif;background:#111;color:#eee;margin:0;padding:16px}}
  header{{position:sticky;top:0;background:#1a1a1a;padding:12px 16px;border-radius:8px;
         display:flex;gap:16px;align-items:center;z-index:10;box-shadow:0 2px 8px #0008;flex-wrap:wrap}}
  header b{{font-size:18px}} .count{{color:#6f6;font-variant-numeric:tabular-nums}}
  button{{background:#2d7;color:#012;border:0;padding:10px 18px;border-radius:6px;
          font-weight:700;cursor:pointer;font-size:15px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px;margin-top:16px}}
  .card{{background:#1c1c1c;border-radius:8px;padding:10px;text-align:center}}
  .idx{{color:#888;font-size:12px;text-align:left}}
  .card img{{width:100%;image-rendering:auto;border-radius:4px;background:#000}}
  .hint{{font-size:12px;color:#9ad;margin-top:6px;cursor:pointer;user-select:none}}
  .hint:hover{{color:#cfe}} .hint .c{{color:#888}} .hint.none{{color:#666;cursor:default}}
  .card input{{width:92%;margin-top:6px;padding:9px;font-size:18px;text-align:center;
              text-transform:uppercase;border-radius:5px;border:1px solid #444;background:#222;color:#6f6}}
  .card input:focus{{outline:2px solid #2d7}}
</style></head><body>
<header>
  <b>Label plate numbers &mdash; {heading}</b>
  <span class="count"><span id="filled">0</span>/{len(cards)} filled</span>
  <button onclick="dl()">Download CSV</button>
  <span style="color:#999">Type what you SEE. Blank = skip. TAB moves down.
  Check every model guess against the picture before using it.</span>
</header>
<div class="grid">{''.join(cards)}</div>
<script>
  const inputs = [...document.querySelectorAll('input')];
  const filled = document.getElementById('filled');
  function upd(){{ filled.textContent = inputs.filter(i=>i.value.trim()).length; }}
  inputs.forEach(i=>i.addEventListener('input',upd));
  function useHint(el){{
    const inp = el.parentElement.querySelector('input');
    inp.value = el.dataset.pred; inp.focus(); upd();
  }}
  function dl(){{
    let rows=[['image_path','plate_text']];
    inputs.forEach(i=>{{ const v=i.value.trim().toUpperCase();
      if(v) rows.push([i.dataset.path, v]); }});
    if(rows.length<2){{ alert('No labels typed yet.'); return; }}
    const csv=rows.map(r=>r.map(c=>'"'+c.replace(/"/g,'""')+'"').join(',')).join('\\n');
    const blob=new Blob([csv],{{type:'text/csv'}});
    const a=document.createElement('a');
    a.href=URL.createObjectURL(blob); a.download='label_sheet_{tag}.csv'; a.click();
  }}
  window.addEventListener('beforeunload',e=>{{
    if(inputs.some(i=>i.value.trim())){{ e.preventDefault(); e.returnValue=''; }}
  }});
</script></body></html>'''

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"label_sheet_{tag}.html"
    out.write_text(page, encoding="utf-8")
    print("-" * 60)
    print(f"[OK] sheet -> {out}")
    print(f"\nOpen it in your browser:\n    start {out}")
    print("Fill the boxes, click 'Download CSV', then:")
    print("    python scripts/tools/import_label_csv.py <downloaded csv path>")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] failed: {exc}")
        sys.exit(1)
