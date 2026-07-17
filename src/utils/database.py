"""
src/utils/database.py
=====================
SQLite database module for the Cambodian ALPR system.

Schema conforms to docs/database.md:
  * registered_plates  — the whitelist (who is allowed through the gate)
  * plate_reads        — the audit log (every detection + decision)
  * system_metrics     — periodic health/performance samples

SQLite stores TEXT as UTF-8, so Khmer plate text (e.g. "ភ្នំពេញ 1AB-2345")
is handled natively.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

VALID_ACTIONS = (
    "ENTRY_ALLOWED", "ENTRY_DENIED", "REVIEW_REQUIRED", "MANUAL_OVERRIDE", "ERROR",
)


class PlateDatabase:
    """Whitelist + audit log + metrics, matching docs/database.md."""

    def __init__(self, db_path: str | Path = "plates.db") -> None:
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    # ------------------------------------------------------------------ #
    # Schema (exactly as docs/database.md)
    # ------------------------------------------------------------------ #
    def _create_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS registered_plates (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text      TEXT UNIQUE NOT NULL,
                owner_name      TEXT NOT NULL,
                vehicle_type    TEXT,
                registered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status          TEXT DEFAULT 'active'
                                CHECK (status IN ('active', 'suspended', 'expired')),
                notes           TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS plate_reads (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text      TEXT,
                detected_plate  TEXT NOT NULL,
                yolo_confidence REAL CHECK (yolo_confidence >= 0.0 AND yolo_confidence <= 1.0),
                crnn_confidence REAL CHECK (crnn_confidence >= 0.0 AND crnn_confidence <= 1.0),
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                location        TEXT DEFAULT 'Main Gate',
                action          TEXT CHECK (action IN
                                ('ENTRY_ALLOWED','ENTRY_DENIED','REVIEW_REQUIRED',
                                 'MANUAL_OVERRIDE','ERROR')),
                photo_path      TEXT
            )
            """
        )
        # Parking mode (open parking): the set of cars CURRENTLY INSIDE. A row is
        # created on entry and DELETED on exit, so storage only ever holds cars
        # that are parked right now — nothing accumulates for cars that left.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS parking_sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text  TEXT UNIQUE NOT NULL,
                entry_time  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                entry_photo TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS system_metrics (
                id                     INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                fps                    REAL,
                avg_latency_ms         REAL,
                gpu_memory_mb          REAL,
                cpu_usage_percent      REAL,
                rtsp_connected         INTEGER,
                total_detections_today INTEGER,
                uptime_percent         REAL
            )
            """
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Whitelist operations
    # ------------------------------------------------------------------ #
    def add_plate(self, plate_text: str, owner_name: str,
                  vehicle_type: str = "", notes: str | None = None) -> bool:
        """Register a plate. owner_name is required (NOT NULL). False on dup/error."""
        try:
            self.conn.execute(
                "INSERT INTO registered_plates "
                "(plate_text, owner_name, vehicle_type, notes) VALUES (?, ?, ?, ?)",
                (plate_text, owner_name or "unknown", vehicle_type, notes),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        except sqlite3.Error as exc:
            print(f"[PlateDatabase] add_plate error: {exc}")
            return False

    def is_registered(self, plate_text: str) -> bool:
        """True if the plate exists AND is 'active' (SRS SEC-002 exact match)."""
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

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        """Levenshtein edit distance (small strings; used by nearest_registered)."""
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

    def nearest_registered(self, plate_text: str,
                           max_distance: int = 1) -> tuple[str, int] | None:
        """Closest ACTIVE registered plate within `max_distance` edits, or None.

        SAFETY: this is a *candidate* lookup for the REVIEW_REQUIRED path only —
        it is deliberately separate from is_registered() (which stays an EXACT
        match and is the ONLY thing that may open the gate). A near match here
        must never auto-open; it only flags a suggestion for human review, so a
        1-character CRNN misread of a legitimate plate is surfaced instead of
        silently denied. Returns (matched_plate_text, distance).
        """
        if not plate_text or max_distance < 1:
            return None
        try:
            best, best_d = None, max_distance + 1
            for (reg,) in self.conn.execute(
                "SELECT plate_text FROM registered_plates WHERE status = 'active'"
            ):
                if reg == plate_text:
                    return (reg, 0)          # exact — is_registered handles opening
                d = self._edit_distance(plate_text, reg)
                if d < best_d:
                    best, best_d = reg, d
            return (best, best_d) if best is not None and best_d <= max_distance else None
        except sqlite3.Error as exc:
            print(f"[PlateDatabase] nearest_registered error: {exc}")
            return None

    # ------------------------------------------------------------------ #
    # Parking mode — "cars currently inside" (created on entry, deleted on exit)
    # ------------------------------------------------------------------ #
    def open_parking_session(self, plate_text: str,
                             entry_photo: str | None = None) -> bool:
        """Record a car as INSIDE (entry). No-op if already inside (re-read)."""
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO parking_sessions (plate_text, entry_photo) "
                "VALUES (?, ?)", (plate_text, entry_photo))
            self.conn.commit()
            return True
        except sqlite3.Error as exc:
            print(f"[PlateDatabase] open_parking_session error: {exc}")
            return False

    def is_inside(self, plate_text: str) -> bool:
        """True if this plate currently has an open parking session."""
        try:
            row = self.conn.execute(
                "SELECT 1 FROM parking_sessions WHERE plate_text = ? LIMIT 1",
                (plate_text,)).fetchone()
            return row is not None
        except sqlite3.Error:
            return False

    def nearest_inside(self, plate_text: str,
                       max_distance: int = 1) -> tuple[str, int] | None:
        """Closest currently-inside plate within `max_distance` edits, or None.
        Lets an exit match a car whose plate the CRNN misread by a character."""
        if not plate_text or max_distance < 1:
            return None
        try:
            best, best_d = None, max_distance + 1
            for (p,) in self.conn.execute("SELECT plate_text FROM parking_sessions"):
                if p == plate_text:
                    return (p, 0)
                d = self._edit_distance(plate_text, p)
                if d < best_d:
                    best, best_d = p, d
            return (best, best_d) if best is not None and best_d <= max_distance else None
        except sqlite3.Error:
            return None

    def close_parking_session(self, plate_text: str) -> bool:
        """Remove a car's session (exit). Returns True if a row was deleted."""
        try:
            cur = self.conn.execute(
                "DELETE FROM parking_sessions WHERE plate_text = ?", (plate_text,))
            self.conn.commit()
            return cur.rowcount > 0
        except sqlite3.Error as exc:
            print(f"[PlateDatabase] close_parking_session error: {exc}")
            return False

    def active_sessions(self) -> list[dict]:
        """All cars currently inside (most recent entry first)."""
        try:
            rows = self.conn.execute(
                "SELECT plate_text, entry_time, entry_photo FROM parking_sessions "
                "ORDER BY entry_time DESC").fetchall()
            return [{"plate_text": r[0], "entry_time": r[1], "entry_photo": r[2]}
                    for r in rows]
        except sqlite3.Error:
            return []

    def count_inside(self) -> int:
        try:
            return self.conn.execute(
                "SELECT COUNT(*) FROM parking_sessions").fetchone()[0]
        except sqlite3.Error:
            return 0

    def expire_stale_sessions(self, hours: float) -> int:
        """Delete sessions older than `hours` (cars whose exit read was missed),
        so orphaned rows don't linger. Returns how many were removed."""
        if hours <= 0:
            return 0
        try:
            cur = self.conn.execute(
                "DELETE FROM parking_sessions WHERE entry_time < "
                "datetime('now', ?)", (f"-{float(hours)} hours",))
            self.conn.commit()
            return cur.rowcount
        except sqlite3.Error:
            return 0

    def suspend_plate(self, plate_text: str) -> bool:
        """Set a plate's status to 'suspended' (SRS ADM-002)."""
        return self.set_status(plate_text, "suspended")

    def set_status(self, plate_text: str, status: str) -> bool:
        """Set a plate's status to 'active' / 'suspended' / 'expired' (ADM-002)."""
        if status not in ("active", "suspended", "expired"):
            return False
        try:
            cur = self.conn.execute(
                "UPDATE registered_plates SET status = ? WHERE plate_text = ?",
                (status, plate_text),
            )
            self.conn.commit()
            return cur.rowcount > 0
        except sqlite3.Error as exc:
            print(f"[PlateDatabase] set_status error: {exc}")
            return False

    def remove_plate(self, plate_text: str) -> bool:
        """Delete a plate from the whitelist entirely."""
        try:
            cur = self.conn.execute(
                "DELETE FROM registered_plates WHERE plate_text = ?", (plate_text,))
            self.conn.commit()
            return cur.rowcount > 0
        except sqlite3.Error as exc:
            print(f"[PlateDatabase] remove_plate error: {exc}")
            return False

    # ------------------------------------------------------------------ #
    # Audit log
    # ------------------------------------------------------------------ #
    def log_read(self, detected_plate: str, yolo_confidence: float,
                 crnn_confidence: float, action: str,
                 plate_text: str | None = None, location: str = "Main Gate",
                 photo_path: str | None = None) -> bool:
        """Append one detection event to the audit log (docs/database.md).

        detected_plate : what the CRNN actually read.
        plate_text     : the matched whitelist plate (None if no match).
        action         : one of VALID_ACTIONS.
        """
        if action not in VALID_ACTIONS:
            action = "ERROR"
        try:
            cur = self.conn.execute(
                "INSERT INTO plate_reads "
                "(plate_text, detected_plate, yolo_confidence, crnn_confidence, "
                " location, action, photo_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (plate_text, detected_plate, _clamp(yolo_confidence),
                 _clamp(crnn_confidence), location, action, photo_path),
            )
            self.conn.commit()
            return cur.lastrowid       # row id (so a live read can be upgraded)
        except sqlite3.Error as exc:
            print(f"[PlateDatabase] log_read error: {exc}")
            return None

    def update_read(self, read_id: int, detected_plate: str, crnn_confidence: float,
                    action: str, plate_text: str | None = None,
                    photo_path: str | None = None) -> bool:
        """Overwrite an existing audit row (used by live de-duplication to keep a
        single row per car and upgrade it to the best read seen during the visit)."""
        if action not in VALID_ACTIONS:
            action = "ERROR"
        try:
            self.conn.execute(
                "UPDATE plate_reads SET detected_plate = ?, crnn_confidence = ?, "
                "action = ?, plate_text = ?, photo_path = ? WHERE id = ?",
                (detected_plate, _clamp(crnn_confidence), action, plate_text,
                 photo_path, read_id))
            self.conn.commit()
            return True
        except sqlite3.Error as exc:
            print(f"[PlateDatabase] update_read error: {exc}")
            return False

    def log_metrics(self, fps: float, avg_latency_ms: float,
                    gpu_memory_mb: float = 0.0, cpu_usage_percent: float = 0.0,
                    rtsp_connected: bool = False,
                    total_detections_today: int = 0,
                    uptime_percent: float = 100.0) -> bool:
        """Insert one system_metrics sample (SRS HLT-002)."""
        try:
            self.conn.execute(
                "INSERT INTO system_metrics "
                "(fps, avg_latency_ms, gpu_memory_mb, cpu_usage_percent, "
                " rtsp_connected, total_detections_today, uptime_percent) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (fps, avg_latency_ms, gpu_memory_mb, cpu_usage_percent,
                 1 if rtsp_connected else 0, total_detections_today, uptime_percent),
            )
            self.conn.commit()
            return True
        except sqlite3.Error as exc:
            print(f"[PlateDatabase] log_metrics error: {exc}")
            return False

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def get_recent_reads(self, limit: int = 10) -> list[dict]:
        try:
            rows = self.conn.execute(
                "SELECT * FROM plate_reads ORDER BY id DESC LIMIT ?", (limit,)
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

    def search_reads(self, plate: str | None = None, action: str | None = None,
                     date: str | None = None, limit: int = 100) -> list[dict]:
        """Audit-log search by plate / action / date (YYYY-MM-DD) (SRS ADM-003)."""
        where, params = [], []
        if plate:
            where.append("(detected_plate LIKE ? OR plate_text LIKE ?)")
            params += [f"%{plate}%", f"%{plate}%"]
        if action:
            where.append("action = ?")
            params.append(action)
        if date:
            where.append("DATE(timestamp, 'localtime') = ?")
            params.append(date)
        sql = "SELECT * FROM plate_reads"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        try:
            rows = self.conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            print(f"[PlateDatabase] search_reads error: {exc}")
            return []

    def get_stats(self) -> dict:
        """Summary counters. 'today' uses the local calendar date."""
        stats = {"total_registered": 0, "total_reads": 0,
                 "allowed_today": 0, "denied_today": 0, "review_today": 0}
        try:
            cur = self.conn.cursor()
            stats["total_registered"] = cur.execute(
                "SELECT COUNT(*) FROM registered_plates").fetchone()[0]
            stats["total_reads"] = cur.execute(
                "SELECT COUNT(*) FROM plate_reads").fetchone()[0]
            for key, act in (("allowed_today", "ENTRY_ALLOWED"),
                             ("denied_today", "ENTRY_DENIED"),
                             ("review_today", "REVIEW_REQUIRED")):
                stats[key] = cur.execute(
                    "SELECT COUNT(*) FROM plate_reads WHERE action = ? "
                    "AND DATE(timestamp, 'localtime') = DATE('now', 'localtime')",
                    (act,),
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


def _clamp(v) -> float:
    """Keep confidence within [0,1] so the CHECK constraint never rejects a row."""
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0
