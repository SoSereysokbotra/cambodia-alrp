#!/usr/bin/env python3
"""
scripts/tools/make_label_sheet.py
=================================
Generate a self-contained HTML page for labelling number crops in your BROWSER
(no terminal / no OpenCV window). Each crop is enlarged + contrast-enhanced with
a text box under it; a "Download CSV" button saves your labels.

Workflow:
    1. python scripts/tools/make_label_sheet.py --split test --limit 150
    2. open the printed .html file in your browser
    3. type the number under each crop (leave blank / click SKIP for illegible)
    4. click "Download CSV"  ->  saves label_sheet_<split>.csv to Downloads
    5. python scripts/tools/import_label_csv.py <that downloaded csv>

Already-labelled crops (in data/crnn_crops/real_labels.csv) are excluded, so you
can generate a fresh sheet to continue.
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
RESULTS = PROJECT_ROOT / "results"
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def load_done() -> set[str]:
    done = set()
    if OUT_CSV.exists():
        with open(OUT_CSV, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["image_path"])
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "valid", "test", "all"])
    ap.add_argument("--limit", type=int, default=150)
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

    done = load_done()
    todo = [c for c in crops
            if Path(c).relative_to(PROJECT_ROOT).as_posix() not in done][:args.limit]
    if not todo:
        print("Nothing new to label (all done for this split). Increase --limit.")
        return

    print(f"already labelled: {len(done)} | generating sheet for: {len(todo)} crops")

    cards = []
    for i, path in enumerate(todo):
        b64 = encode_crop(cv2, np, path)
        if not b64:
            continue
        rel = html.escape(Path(path).relative_to(PROJECT_ROOT).as_posix())
        cards.append(f'''
        <div class="card">
          <div class="idx">#{i + 1}</div>
          <img src="data:image/png;base64,{b64}" alt="crop">
          <input type="text" data-path="{rel}" autocomplete="off"
                 spellcheck="false" placeholder="type number (blank = skip)">
        </div>''')

    page = f'''<!doctype html><html><head><meta charset="utf-8">
<title>Label number crops ({args.split})</title>
<style>
  body{{font-family:system-ui,Arial,sans-serif;background:#111;color:#eee;margin:0;padding:16px}}
  header{{position:sticky;top:0;background:#1a1a1a;padding:12px 16px;border-radius:8px;
         display:flex;gap:16px;align-items:center;z-index:10;box-shadow:0 2px 8px #0008}}
  header b{{font-size:18px}} .count{{color:#6f6}}
  button{{background:#2d7;color:#012;border:0;padding:10px 18px;border-radius:6px;
          font-weight:700;cursor:pointer;font-size:15px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px;margin-top:16px}}
  .card{{background:#1c1c1c;border-radius:8px;padding:10px;text-align:center}}
  .idx{{color:#888;font-size:12px;text-align:left}}
  .card img{{width:100%;image-rendering:auto;border-radius:4px;background:#000}}
  .card input{{width:92%;margin-top:8px;padding:9px;font-size:18px;text-align:center;
              text-transform:uppercase;border-radius:5px;border:1px solid #444;background:#222;color:#6f6}}
  .card input:focus{{outline:2px solid #2d7}}
</style></head><body>
<header>
  <b>Label plate numbers — {args.split}</b>
  <span class="count"><span id="filled">0</span>/{len(cards)} filled</span>
  <button onclick="dl()">Download CSV</button>
  <span style="color:#999">Type the number; leave blank to skip. TAB moves down.</span>
</header>
<div class="grid">{''.join(cards)}</div>
<script>
  const inputs = [...document.querySelectorAll('input')];
  const filled = document.getElementById('filled');
  function upd(){{ filled.textContent = inputs.filter(i=>i.value.trim()).length; }}
  inputs.forEach(i=>i.addEventListener('input',upd));
  function dl(){{
    let rows=[['image_path','plate_text']];
    inputs.forEach(i=>{{ const v=i.value.trim().toUpperCase();
      if(v) rows.push([i.dataset.path, v]); }});
    if(rows.length<2){{ alert('No labels typed yet.'); return; }}
    const csv=rows.map(r=>r.map(c=>'"'+c.replace(/"/g,'""')+'"').join(',')).join('\\n');
    const blob=new Blob([csv],{{type:'text/csv'}});
    const a=document.createElement('a');
    a.href=URL.createObjectURL(blob); a.download='label_sheet_{args.split}.csv'; a.click();
  }}
</script></body></html>'''

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"label_sheet_{args.split}.html"
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
