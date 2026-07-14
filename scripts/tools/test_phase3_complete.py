#!/usr/bin/env python3
"""
scripts/tools/test_phase3_complete.py
=====================================
Verify Phase 3 (province classifier) end to end:

  [1] dataset stratification — every class present in train/val/test
  [2] classifier test accuracy >= 95%
  [3] composition + DB query — every registered demo plate composes and matches

Run:
    python scripts/tools/test_phase3_complete.py
"""

from __future__ import annotations

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

DATA_DIR = PROJECT_ROOT / "data" / "province_crops"
MODEL = PROJECT_ROOT / "models" / "recognition" / "province_classifier_best.pth"
DB_PATH = PROJECT_ROOT / "plates.db"
ACC_TARGET = 0.95

passed = failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    passed += ok
    failed += (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def test_stratification() -> None:
    print("\n[1] Dataset stratification")
    manifest = DATA_DIR / "manifest.json"
    if not manifest.exists():
        check("manifest exists", False, "run build_province_dataset.py")
        return
    m = json.loads(manifest.read_text(encoding="utf-8"))
    per = m["per_class"]
    # every class that has ANY samples should appear in all three splits
    missing = []
    for c in range(m["n_classes"]):
        total = sum(int(per[s][str(c)]) for s in ("train", "val", "test"))
        if total == 0:
            continue
        if any(int(per[s][str(c)]) == 0 for s in ("train", "val", "test")):
            missing.append(c)
    check("all populated classes span train/val/test", not missing,
          f"gaps in classes {missing}" if missing else
          f"{m['total_crops']} crops across {m['n_classes']} classes")


def test_accuracy() -> None:
    print("\n[2] Classifier accuracy")
    if not MODEL.exists():
        check("classifier trained", False, "run train_province_classifier.py")
        return
    try:
        import torch
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        from province_classifier import ProvinceClassifier
        clf = ProvinceClassifier(MODEL)
        tf = transforms.Compose([
            transforms.Resize((clf.img_size, clf.img_size)),
            transforms.ToTensor(), transforms.Normalize(clf.mean, clf.std)])
        ds = datasets.ImageFolder(str(DATA_DIR / "test"), transform=tf)
        dl = DataLoader(ds, batch_size=64, shuffle=False)
        correct = total = 0
        with torch.no_grad():
            for x, y in dl:
                x = x.to(clf.device)
                pred = clf.model(x).argmax(1).cpu()
                correct += (pred == y).sum().item(); total += y.numel()
        acc = correct / max(total, 1)
        check(f"test accuracy >= {ACC_TARGET*100:.0f}%", acc >= ACC_TARGET,
              f"{acc*100:.2f}% on {total} crops")
    except Exception as exc:
        check("classifier accuracy", False, str(exc))


def test_composition() -> None:
    print("\n[3] Composition + DB query (registered demo plates)")
    import sqlite3
    from province_map import compose_plate, PROVINCE_KHMER
    khmer_to_id = {v: k for k, v in PROVINCE_KHMER.items() if v}

    con = sqlite3.connect(str(DB_PATH))
    rows = [r[0] for r in con.execute(
        "SELECT plate_text FROM registered_plates ORDER BY id")]
    con.close()

    # only the "provinceKhmer number" style entries (contain a space + Khmer)
    prov_plates = [p for p in rows if " " in p and any(ch > "ក" for ch in p)]
    if not prov_plates:
        check("registered province plates found", False, "none in DB")
        return

    con = sqlite3.connect(str(DB_PATH))
    all_ok = True
    for full in prov_plates:
        prov, _, number = full.partition(" ")
        pid = khmer_to_id.get(prov)
        if pid is None:
            all_ok = False
            print(f"      [!] province not in map: {prov}")
            continue
        composed = compose_plate(pid, number)
        is_reg = con.execute(
            "SELECT 1 FROM registered_plates WHERE plate_text = ? "
            "AND status='active' LIMIT 1", (composed,)).fetchone() is not None
        ok = (composed == full) and is_reg
        all_ok = all_ok and ok
        print(f"      {'OK ' if ok else 'X  '} {full}  ->  compose={composed}  reg={is_reg}")
    con.close()
    check("all province plates compose + query correctly", all_ok)


def main() -> None:
    print("=" * 60)
    print(" PHASE 3 VERIFICATION — PROVINCE CLASSIFIER")
    print("=" * 60)
    test_stratification()
    test_accuracy()
    test_composition()
    print("-" * 60)
    print(f" RESULT: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] test failed to run: {exc}")
        sys.exit(1)
