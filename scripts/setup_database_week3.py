#!/usr/bin/env python3
"""
scripts/setup_database_week3.py
===============================
STEP 1 of Week 3 — create plates.db and register 8 test vehicles.

Run:
    python scripts/setup_database_week3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Khmer text + checkmarks -> force UTF-8 console output on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Make src/ importable regardless of where the script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.database import PlateDatabase  # noqa: E402

DB_PATH = PROJECT_ROOT / "plates.db"

# (plate_text, owner_name, vehicle_type) — real Khmer province names.
TEST_PLATES = [
    ("ភ្នំពេញ 1AB-2345", "Sokhem Ouch",   "Honda Civic"),
    ("ភ្នំពេញ 2CD-6789", "Bopha Mara",    "Toyota Camry"),
    ("កណ្តាល 3EF-0123",  "Chan Rith",     "Mitsubishi"),
    ("សៀមរាប 4GH-4567",  "Mey Sophea",    "Lexus RX"),
    ("បាត់ដំបង 5IJ-8901", "Nary Sophel",   "Honda Accord"),
    ("កំពត 6KL-2345",    "Dara Piseth",   "Suzuki"),
    ("ព្រៃវែង 7MN-6789", "Kosal Mony",    "Toyota Vios"),
    ("ក្រចេះ 8OP-0123",  "Sreyleak Pov",  "Hyundai Tucson"),
]


def main() -> None:
    print("=" * 70)
    print(" WEEK 3 — STEP 1: DATABASE SETUP")
    print("=" * 70)

    db = PlateDatabase(DB_PATH)

    added, skipped = 0, 0
    for plate_text, owner, vehicle in TEST_PLATES:
        if db.add_plate(plate_text, owner, vehicle):
            added += 1
        else:
            skipped += 1  # already existed (safe to re-run)

    print(f"\nAdded {added} new plate(s), skipped {skipped} existing.\n")

    # Confirmation table of everything currently registered.
    rows = db.get_all_registered()
    print(f"{'ID':<3} | {'PLATE':<20} | {'OWNER':<15} | {'VEHICLE':<15} | STATUS")
    print("-" * 70)
    for r in rows:
        print(f"{r['id']:<3} | {r['plate_text']:<20} | "
              f"{(r['owner_name'] or ''):<15} | "
              f"{(r['vehicle_type'] or ''):<15} | {r['status']}")
    print("-" * 70)

    print(f"\n✓ Database ready at {DB_PATH} ({len(rows)} registered plates)")
    db.close()
    print("\nNext: python scripts/alpr_pipeline_week3.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # never crash silently
        print(f"[X] setup_database failed: {exc}")
        sys.exit(1)
