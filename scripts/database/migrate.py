#!/usr/bin/env python3
"""
scripts/migrate_db_week13.py
============================
Migrate plates.db to the docs/database.md schema WITHOUT losing data.

Steps:
    1. Back up plates.db -> plates_backup_YYYYMMDD_HHMMSS.db
    2. Read existing registered_plates rows
    3. Drop old registered_plates + plate_reads
    4. Recreate all tables in the new schema (via PlateDatabase)
    5. Re-insert the registered plates
    6. Verify counts

Run:
    python scripts/migrate_db_week13.py
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src"))
DB_PATH = PROJECT_ROOT / "plates.db"


def main() -> None:
    if not DB_PATH.exists():
        print(f"[i] {DB_PATH} does not exist yet — nothing to migrate.")
        print("    Run scripts/setup_database_week3.py to create a fresh DB.")
        return

    # 1. backup
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = PROJECT_ROOT / f"plates_backup_{ts}.db"
    shutil.copy2(DB_PATH, backup)
    print(f"[1/6] backup -> {backup.name}")

    # 2. read existing registered plates (tolerate old column set)
    raw = sqlite3.connect(str(DB_PATH))
    raw.row_factory = sqlite3.Row
    try:
        rows = raw.execute("SELECT * FROM registered_plates ORDER BY id").fetchall()
    except sqlite3.Error:
        rows = []
    existing = [dict(r) for r in rows]
    print(f"[2/6] read {len(existing)} registered plate(s)")

    # 3. drop old tables
    raw.execute("DROP TABLE IF EXISTS registered_plates")
    raw.execute("DROP TABLE IF EXISTS plate_reads")
    # keep system_metrics if it exists; new schema will create if missing
    raw.commit()
    raw.close()
    print("[3/6] dropped old registered_plates + plate_reads")

    # 4. recreate in new schema
    from utils.database import PlateDatabase
    db = PlateDatabase(DB_PATH)
    print("[4/6] recreated tables in docs/database.md schema")

    # 5. re-insert registered plates
    restored = 0
    for r in existing:
        ok = db.add_plate(
            plate_text=r.get("plate_text"),
            owner_name=r.get("owner_name") or "unknown",
            vehicle_type=r.get("vehicle_type") or "",
            notes=r.get("notes"),
        )
        # restore non-active status if it was suspended/expired
        if ok and r.get("status") and r["status"] != "active":
            db.conn.execute(
                "UPDATE registered_plates SET status = ? WHERE plate_text = ?",
                (r["status"], r["plate_text"]))
            db.conn.commit()
        restored += int(ok)
    print(f"[5/6] restored {restored} registered plate(s)")

    # 6. verify
    stats = db.get_stats()
    print(f"[6/6] verify -> registered={stats['total_registered']} "
          f"reads={stats['total_reads']}")
    # confirm new columns exist
    cols = [c[1] for c in db.conn.execute("PRAGMA table_info(plate_reads)").fetchall()]
    need = {"crnn_confidence", "plate_text", "location"}
    print(f"      plate_reads columns present: {need.issubset(set(cols))}")
    db.close()

    print("\nMigration complete. Old DB preserved at:", backup.name)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] migration failed: {exc}")
        sys.exit(1)
