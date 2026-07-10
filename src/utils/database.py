"""
src/utils/database.py
=====================
SQLite database module for the Cambodian ALPR system.

Two tables:
  * registered_plates  — the whitelist (who is allowed through the gate)
  * plate_reads        — the audit log (every detection, allowed or denied)

SQLite stores TEXT as UTF-8, so Khmer plate text (e.g. "ភ្នំពេញ 1AB-2345")
is handled natively.

Example
-------
    from utils.database import PlateDatabase
    db = PlateDatabase("plates.db")
    db.add_plate("ភ្នំពេញ 1AB-2345", "Sokhem Ouch", "Honda Civic")
    db.is_registered("ភ្នំពេញ 1AB-2345")          # -> True
    db.log_read("PLATE_1_DETECTED", 0.92, False, "ENTRY_DENIED")
    db.close()
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class PlateDatabase:
    """Thin, safe wrapper over the SQLite whitelist + audit log."""

    def __init__(self, db_path: str | Path = "plates.db") -> None:
        self.db_path = str(db_path)
        # check_same_thread=False keeps it usable from simple scripts/threads.
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # rows behave like dicts
        self._create_tables()

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def _create_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS registered_plates (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text      TEXT UNIQUE NOT NULL,
                owner_name      TEXT,
                vehicle_type    TEXT,
                registered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status          TEXT DEFAULT 'active'
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS plate_reads (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                detected_plate  TEXT,
                yolo_confidence REAL,
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_registered   INTEGER,
                action          TEXT,
                photo_path      TEXT,
                notes           TEXT
            )
            """
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Whitelist operations
    # ------------------------------------------------------------------ #
    def add_plate(self, plate_text: str, owner_name: str = "",
                  vehicle_type: str = "") -> bool:
        """Register a plate. Returns False if it already exists / on error."""
        try:
            self.conn.execute(
                "INSERT INTO registered_plates (plate_text, owner_name, vehicle_type) "
                "VALUES (?, ?, ?)",
                (plate_text, owner_name, vehicle_type),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            # UNIQUE constraint — plate already registered.
            return False
        except sqlite3.Error as exc:
            print(f"[PlateDatabase] add_plate error: {exc}")
            return False

    def is_registered(self, plate_text: str) -> bool:
        """True if the plate exists AND is 'active'."""
        try:
            row = self.conn.execute(
                "SELECT 1 FROM registered_plates "
                "WHERE plate_text = ? AND status = 'active' LIMIT 1",
                (plate_text,),
            ).fetchone()
            return row is not None
        except sqlite3.Error as exc:
            print(f"[PlateDatabase] is_registered error: {exc}")
            return False

    # ------------------------------------------------------------------ #
    # Audit log
    # ------------------------------------------------------------------ #
    def log_read(self, detected_plate: str, yolo_conf: float,
                 is_registered: bool, action: str,
                 photo_path: str | None = None, notes: str | None = None) -> bool:
        """Append one detection event to the audit log."""
        try:
            self.conn.execute(
                "INSERT INTO plate_reads "
                "(detected_plate, yolo_confidence, is_registered, action, "
                " photo_path, notes) VALUES (?, ?, ?, ?, ?, ?)",
                (detected_plate, float(yolo_conf), 1 if is_registered else 0,
                 action, photo_path, notes),
            )
            self.conn.commit()
            return True
        except sqlite3.Error as exc:
            print(f"[PlateDatabase] log_read error: {exc}")
            return False

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def get_recent_reads(self, limit: int = 10) -> list[dict]:
        try:
            rows = self.conn.execute(
                "SELECT * FROM plate_reads ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            print(f"[PlateDatabase] get_recent_reads error: {exc}")
            return []

    def get_all_registered(self) -> list[dict]:
        try:
            rows = self.conn.execute(
                "SELECT * FROM registered_plates ORDER BY id"
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            print(f"[PlateDatabase] get_all_registered error: {exc}")
            return []

    def get_stats(self) -> dict:
        """Summary counters. 'today' uses local calendar date."""
        stats = {"total_registered": 0, "total_reads": 0,
                 "allowed_today": 0, "denied_today": 0}
        try:
            cur = self.conn.cursor()
            stats["total_registered"] = cur.execute(
                "SELECT COUNT(*) FROM registered_plates").fetchone()[0]
            stats["total_reads"] = cur.execute(
                "SELECT COUNT(*) FROM plate_reads").fetchone()[0]
            # Compare local dates (timestamps are stored UTC by SQLite).
            stats["allowed_today"] = cur.execute(
                "SELECT COUNT(*) FROM plate_reads "
                "WHERE action = 'ENTRY_ALLOWED' "
                "AND DATE(timestamp, 'localtime') = DATE('now', 'localtime')"
            ).fetchone()[0]
            stats["denied_today"] = cur.execute(
                "SELECT COUNT(*) FROM plate_reads "
                "WHERE action = 'ENTRY_DENIED' "
                "AND DATE(timestamp, 'localtime') = DATE('now', 'localtime')"
            ).fetchone()[0]
        except sqlite3.Error as exc:
            print(f"[PlateDatabase] get_stats error: {exc}")
        return stats

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        try:
            self.conn.close()
        except sqlite3.Error:
            pass
