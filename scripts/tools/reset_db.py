#!/usr/bin/env python3
"""
scripts/tools/reset_db.py
=========================
Wipe the ALPR database back to zero for a clean test run.

By default it BACKS UP plates.db first (reversible), clears every data table
(whitelist, reads, metrics, parking sessions), resets the id counters so the next
read is id=1, and compacts the file. The photos/ folder is left alone unless you
pass --photos.

Usage:
    python scripts/tools/reset_db.py                 # backup + wipe tables
    python scripts/tools/reset_db.py --photos        # also delete evidence photos
    python scripts/tools/reset_db.py --no-backup     # skip the backup (careful)
    python scripts/tools/reset_db.py --yes           # don't ask for confirmation
"""
from __future__ import annotations

import argparse
import datetime
import shutil
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
DB = PROJECT_ROOT / "plates.db"
PHOTOS = PROJECT_ROOT / "photos"


def main() -> None:
    ap = argparse.ArgumentParser(description="reset the ALPR database to zero")
    ap.add_argument("--photos", action="store_true",
                    help="also delete the evidence JPGs in photos/")
    ap.add_argument("--no-backup", action="store_true",
                    help="do NOT copy plates.db to a timestamped backup first")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    if not DB.exists():
        print(f"[X] no database at {DB}")
        sys.exit(1)

    con = sqlite3.connect(DB)
    tables = [r[0] for r in con.execute(
        "select name from sqlite_master where type='table' and name != 'sqlite_sequence'")]
    counts = {t: con.execute(f"select count(*) from {t}").fetchone()[0] for t in tables}
    n_photos = len(list(PHOTOS.glob("plate_*.jpg"))) if PHOTOS.is_dir() else 0

    print("Will CLEAR:")
    for t, n in counts.items():
        print(f"  {t:<20} {n:>7} rows")
    if args.photos:
        print(f"  photos/ (*.jpg)      {n_photos:>7} files")
    print()

    if not args.yes:
        if input("Type 'yes' to wipe to zero: ").strip().lower() != "yes":
            print("aborted — nothing changed.")
            con.close()
            return

    if not args.no_backup:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = DB.with_name(f"plates.db.bak_{stamp}")
        shutil.copy2(DB, bak)
        print(f"[backup] {DB.name} -> {bak.name}   (restore: cp {bak.name} plates.db)")

    for t in tables:
        con.execute(f"DELETE FROM {t}")
    con.execute("DELETE FROM sqlite_sequence")     # ids restart at 1
    con.commit()
    con.execute("VACUUM")
    con.close()
    print("[db] all tables cleared, id counters reset.")

    if args.photos and PHOTOS.is_dir():
        removed = 0
        for p in PHOTOS.glob("plate_*.jpg"):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
        print(f"[photos] deleted {removed} evidence JPGs.")

    print("\n[done] database is at zero. The whitelist is empty, so the gate will")
    print("       DENY every car until you enroll one (menu 4 / admin panel).")


if __name__ == "__main__":
    main()
