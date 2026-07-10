#!/usr/bin/env python3
"""
scripts/view_database_week3.py
==============================
Inspect plates.db anytime — registered whitelist, recent audit log, and stats.

Run:
    python scripts/view_database_week3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.database import PlateDatabase  # noqa: E402

DB_PATH = PROJECT_ROOT / "plates.db"


def print_registered(db: PlateDatabase) -> None:
    rows = db.get_all_registered()
    print("\n== REGISTERED PLATES (whitelist) ==")
    print(f"{'ID':<3} | {'PLATE':<20} | {'OWNER':<15} | {'VEHICLE':<15} | STATUS")
    print("-" * 72)
    if not rows:
        print("  (none)")
    for r in rows:
        print(f"{r['id']:<3} | {str(r['plate_text']):<20} | "
              f"{str(r['owner_name'] or ''):<15} | "
              f"{str(r['vehicle_type'] or ''):<15} | {r['status']}")


def print_reads(db: PlateDatabase, limit: int = 20) -> None:
    rows = db.get_recent_reads(limit=limit)
    print(f"\n== PLATE READS (audit log, last {limit}) ==")
    print(f"{'ID':<4} | {'TIME':<19} | {'PLATE':<18} | {'REG':<3} | "
          f"{'ACTION':<13} | CONF")
    print("-" * 78)
    if not rows:
        print("  (none)")
    for r in rows:
        print(f"{r['id']:<4} | {str(r.get('timestamp','')):<19} | "
              f"{str(r['detected_plate'])[:18]:<18} | "
              f"{('yes' if r['is_registered'] else 'no'):<3} | "
              f"{str(r['action']):<13} | {r['yolo_confidence']:.2f}")


def print_stats(db: PlateDatabase) -> None:
    s = db.get_stats()
    print("\n== STATISTICS ==")
    print(f"  Total registered : {s['total_registered']}")
    print(f"  Total reads      : {s['total_reads']}")
    print(f"  Allowed today    : {s['allowed_today']}")
    print(f"  Denied today     : {s['denied_today']}")


def main() -> None:
    print("=" * 78)
    print(f" DATABASE VIEW — {DB_PATH}")
    print("=" * 78)
    if not DB_PATH.exists():
        print("[X] plates.db not found. Run scripts/setup_database_week3.py first.")
        sys.exit(1)

    db = PlateDatabase(DB_PATH)
    print_registered(db)
    print_reads(db, limit=20)
    print_stats(db)
    db.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] view_database failed: {exc}")
        sys.exit(1)
