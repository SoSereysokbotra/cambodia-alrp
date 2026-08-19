"""
src/utils/gdrive_storage.py
============================
Google Drive integration layer for the Cambodian ALPR system.

Provides:
  - OAuth2 authentication (personal Google account, browser-based consent)
  - Structured folder creation mirroring the project layout on Drive
  - File upload / download / sync with MD5 dedup + resumable uploads
  - Background upload thread for non-blocking integration with the live pipeline
  - Sync-state tracking to avoid re-uploading unchanged files

Usage (standalone):
    from utils.gdrive_storage import GDriveStorage
    gd = GDriveStorage(config)
    gd.authenticate()
    gd.upload_file("photos/plate_001.jpg")
    gd.sync_folder("models/")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("gdrive")

# Scopes: full Drive access on the user's personal account
_SCOPES = ["https://www.googleapis.com/auth/drive"]

# Folder structure to create on Drive (mirrors project layout)
_DRIVE_FOLDERS = [
    "models",
    "models/detection",
    "models/recognition",
    "models/onnx",
    "models/pretrained",
    "data",
    "data/raw",
    "data/annotated",
    "data/crnn_crops",
    "data/crnn_crops_augmented",
    "data/synthetic",
    "data/number_detect",
    "data/number_annotation",
    "data/province_crops",
    "runs",
    "results",
    "outputs",
    "photos",
    "logs",
    "backups",
    "db_snapshots",
]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _md5(filepath: Path) -> str:
    """Compute MD5 hex digest of a local file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _human_size(nbytes: int | float) -> str:
    """Pretty-print a byte count."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


# ---------------------------------------------------------------------------
# Sync-state persistence
# ---------------------------------------------------------------------------

class SyncState:
    """Tracks which local files have been uploaded (path -> Drive file ID + md5).

    Stored as a JSON file at ``state_path``.  Thread-safe via a lock.
    """

    def __init__(self, state_path: Path) -> None:
        self._path = state_path
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, str]] = {}
        self._load()

    # -- persistence --

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # -- public API --

    def is_synced(self, rel_path: str, local_md5: str) -> bool:
        """Return True if ``rel_path`` was already uploaded with the same MD5."""
        with self._lock:
            entry = self._data.get(rel_path)
            return entry is not None and entry.get("md5") == local_md5

    def mark_synced(self, rel_path: str, drive_id: str, md5: str) -> None:
        with self._lock:
            self._data[rel_path] = {
                "drive_id": drive_id,
                "md5": md5,
                "synced_at": datetime.now().isoformat(timespec="seconds"),
            }
            self._save()

    def get_drive_id(self, rel_path: str) -> str | None:
        with self._lock:
            entry = self._data.get(rel_path)
            return entry["drive_id"] if entry else None

    def remove(self, rel_path: str) -> None:
        with self._lock:
            self._data.pop(rel_path, None)
            self._save()

    def all_entries(self) -> dict[str, dict[str, str]]:
        with self._lock:
            return dict(self._data)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GDriveStorage:
    """Google Drive integration for the Cambodian ALPR project.

    Parameters
    ----------
    config : dict
        The ``gdrive`` section of system_config.yaml.
    project_root : Path
        Absolute path to the project root (for resolving relative paths).
    """

    def __init__(self, config: dict, project_root: Path) -> None:
        self.config = config
        self.root = project_root
        self.enabled = config.get("enabled", False)

        # Paths
        cred_path = config.get("credentials_path", "configs/gdrive_credentials.json")
        self._cred_file = (
            Path(cred_path)
            if Path(cred_path).is_absolute()
            else self.root / cred_path
        )
        token_path = config.get("token_path", "configs/gdrive_token.json")
        self._token_file = (
            Path(token_path)
            if Path(token_path).is_absolute()
            else self.root / token_path
        )
        state_path = config.get("sync_state_path", "configs/gdrive_sync_state.json")
        self._state_file = (
            Path(state_path)
            if Path(state_path).is_absolute()
            else self.root / state_path
        )

        self._root_folder_name = config.get(
            "root_folder_name", "Cambodian ALPR Project"
        )
        self._service = None         # google API service object
        self._root_folder_id = None  # Drive ID of the project root folder
        self._folder_ids: dict[str, str] = {}   # rel_path -> Drive folder ID
        self._sync_state = SyncState(self._state_file)

        # Background upload queue
        self._upload_q: queue.Queue[tuple[Path, str]] = queue.Queue()
        self._bg_thread: threading.Thread | None = None
        self._bg_stop = threading.Event()

    # ------------------------------------------------------------------ #
    # Authentication
    # ------------------------------------------------------------------ #

    def authenticate(self) -> bool:
        """Run the OAuth2 flow (or load saved token).  Returns True on success."""
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
        except ImportError:
            logger.error(
                "Google API libraries not installed. Run:\n"
                "  pip install -r requirements-gdrive.txt"
            )
            return False

        creds = None

        # 1) Try loading a saved token
        if self._token_file.exists():
            try:
                creds = Credentials.from_authorized_user_file(
                    str(self._token_file), _SCOPES
                )
            except Exception:
                creds = None

        # 2) Refresh or re-auth
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None

        if not creds or not creds.valid:
            if not self._cred_file.exists():
                logger.error(
                    "OAuth credentials file not found: %s\n"
                    "Run: python main.py gdrive setup",
                    self._cred_file,
                )
                return False
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self._cred_file), _SCOPES
            )
            creds = flow.run_local_server(port=0)

            # Save the token for next time
            self._token_file.parent.mkdir(parents=True, exist_ok=True)
            self._token_file.write_text(creds.to_json(), encoding="utf-8")
            logger.info("Token saved to %s", self._token_file)

        self._service = build("drive", "v3", credentials=creds)
        logger.info("Google Drive authenticated successfully.")
        return True

    def is_authenticated(self) -> bool:
        return self._service is not None

    def _ensure_auth(self) -> None:
        if not self.is_authenticated():
            raise RuntimeError(
                "Not authenticated. Call authenticate() first or run:\n"
                "  python main.py gdrive setup"
            )

    # ------------------------------------------------------------------ #
    # Folder management
    # ------------------------------------------------------------------ #

    def _find_folder(self, name: str, parent_id: str | None = None) -> str | None:
        """Find a folder by name (optionally under a parent).  Returns ID or None."""
        q = (
            f"name = '{name}' and mimeType = 'application/vnd.google-apps.folder' "
            f"and trashed = false"
        )
        if parent_id:
            q += f" and '{parent_id}' in parents"
        resp = (
            self._service.files()
            .list(q=q, spaces="drive", fields="files(id, name)", pageSize=1)
            .execute()
        )
        files = resp.get("files", [])
        return files[0]["id"] if files else None

    def _create_folder(self, name: str, parent_id: str | None = None) -> str:
        """Create a folder and return its Drive ID."""
        meta: dict[str, Any] = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            meta["parents"] = [parent_id]
        folder = (
            self._service.files()
            .create(body=meta, fields="id")
            .execute()
        )
        return folder["id"]

    def _ensure_folder(self, name: str, parent_id: str | None = None) -> str:
        """Find-or-create a folder. Returns Drive ID."""
        fid = self._find_folder(name, parent_id)
        if fid:
            return fid
        return self._create_folder(name, parent_id)

    def ensure_folder_tree(self) -> None:
        """Create the full project folder tree on Drive (idempotent)."""
        self._ensure_auth()
        # Root folder
        self._root_folder_id = self._ensure_folder(self._root_folder_name)
        self._folder_ids[""] = self._root_folder_id
        logger.info(
            "Drive root folder: '%s' (id=%s)",
            self._root_folder_name,
            self._root_folder_id,
        )

        for rel in _DRIVE_FOLDERS:
            parts = rel.split("/")
            parent_id = self._root_folder_id
            for i, part in enumerate(parts):
                prefix = "/".join(parts[: i + 1])
                if prefix in self._folder_ids:
                    parent_id = self._folder_ids[prefix]
                else:
                    fid = self._ensure_folder(part, parent_id)
                    self._folder_ids[prefix] = fid
                    parent_id = fid
        logger.info("Folder tree ensured (%d folders).", len(self._folder_ids))

    def _resolve_drive_parent(self, rel_path: str) -> str:
        """Get the Drive folder ID for the parent directory of a relative path."""
        parent_rel = str(Path(rel_path).parent).replace("\\", "/")
        if parent_rel == ".":
            parent_rel = ""
        if parent_rel not in self._folder_ids:
            # create missing parent chain
            parts = parent_rel.split("/")
            parent_id = self._root_folder_id
            for i, part in enumerate(parts):
                prefix = "/".join(parts[: i + 1])
                if prefix in self._folder_ids:
                    parent_id = self._folder_ids[prefix]
                else:
                    fid = self._ensure_folder(part, parent_id)
                    self._folder_ids[prefix] = fid
                    parent_id = fid
        return self._folder_ids[parent_rel]

    # ------------------------------------------------------------------ #
    # Upload
    # ------------------------------------------------------------------ #

    def upload_file(
        self,
        local_path: Path | str,
        *,
        force: bool = False,
        show_progress: bool = False,
    ) -> str | None:
        """Upload a single file to Drive.  Returns the Drive file ID, or None on error.

        Skips the upload if the file is already synced (same MD5) unless ``force=True``.
        Uses resumable upload for files > 5 MB.
        """
        self._ensure_auth()
        local = Path(local_path)
        if not local.is_absolute():
            local = self.root / local
        if not local.is_file():
            logger.warning("File not found: %s", local)
            return None

        try:
            rel = local.relative_to(self.root).as_posix()
        except ValueError:
            rel = local.name

        md5 = _md5(local)
        if not force and self._sync_state.is_synced(rel, md5):
            logger.debug("Skip (unchanged): %s", rel)
            return self._sync_state.get_drive_id(rel)

        parent_id = self._resolve_drive_parent(rel)
        file_size = local.stat().st_size

        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(
            str(local),
            resumable=(file_size > 5 * 1024 * 1024),  # resumable for > 5 MB
        )

        # Check if file already exists on Drive (by name + parent) to update
        existing_id = self._sync_state.get_drive_id(rel)
        if existing_id:
            try:
                resp = (
                    self._service.files()
                    .update(fileId=existing_id, media_body=media, fields="id,md5Checksum")
                    .execute()
                )
                self._sync_state.mark_synced(rel, resp["id"], md5)
                logger.info("Updated: %s (%s)", rel, _human_size(file_size))
                return resp["id"]
            except Exception:
                # file may have been deleted on Drive; fall through to create
                pass

        meta = {"name": local.name, "parents": [parent_id]}
        try:
            resp = (
                self._service.files()
                .create(body=meta, media_body=media, fields="id,md5Checksum")
                .execute()
            )
        except Exception as exc:
            logger.error("Upload failed: %s — %s", rel, exc)
            return None

        self._sync_state.mark_synced(rel, resp["id"], md5)
        logger.info("Uploaded: %s (%s)", rel, _human_size(file_size))
        return resp["id"]

    def upload_folder(
        self,
        local_dir: Path | str,
        *,
        force: bool = False,
        show_progress: bool = True,
    ) -> dict[str, str | None]:
        """Upload all files in a directory (recursively).  Returns {rel_path: drive_id}."""
        self._ensure_auth()
        d = Path(local_dir)
        if not d.is_absolute():
            d = self.root / d
        if not d.is_dir():
            logger.warning("Not a directory: %s", d)
            return {}

        files = sorted(f for f in d.rglob("*") if f.is_file())
        results: dict[str, str | None] = {}

        if show_progress:
            try:
                from tqdm import tqdm
                files_iter = tqdm(files, desc=f"Uploading {d.name}", unit="file")
            except ImportError:
                files_iter = files
        else:
            files_iter = files

        for f in files_iter:
            fid = self.upload_file(f, force=force)
            try:
                rel = f.relative_to(self.root).as_posix()
            except ValueError:
                rel = f.name
            results[rel] = fid

        return results

    # ------------------------------------------------------------------ #
    # Download
    # ------------------------------------------------------------------ #

    def download_file(
        self, rel_path: str, *, overwrite: bool = False
    ) -> Path | None:
        """Download a file from Drive to the local project tree.

        ``rel_path`` is relative to the project root (e.g. 'models/detection/best.pt').
        """
        self._ensure_auth()
        local = self.root / rel_path
        if local.exists() and not overwrite:
            logger.info("Already exists locally: %s", rel_path)
            return local

        drive_id = self._sync_state.get_drive_id(rel_path)
        if not drive_id:
            logger.warning("Not tracked in sync state: %s", rel_path)
            return None

        from googleapiclient.http import MediaIoBaseDownload
        import io

        request = self._service.files().get_media(fileId=drive_id)
        local.parent.mkdir(parents=True, exist_ok=True)

        with open(local, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    logger.debug(
                        "Download %s: %d%%", rel_path, int(status.progress() * 100)
                    )

        logger.info("Downloaded: %s -> %s", rel_path, local)
        return local

    def download_folder(self, rel_dir: str, *, overwrite: bool = False) -> list[Path]:
        """Download all tracked files under ``rel_dir``."""
        results: list[Path] = []
        for rel, entry in self._sync_state.all_entries().items():
            if rel.startswith(rel_dir):
                p = self.download_file(rel, overwrite=overwrite)
                if p:
                    results.append(p)
        return results

    # ------------------------------------------------------------------ #
    # Sync
    # ------------------------------------------------------------------ #

    def sync_folder(
        self,
        rel_dir: str,
        *,
        direction: str = "upload_only",
        force: bool = False,
    ) -> dict[str, Any]:
        """Smart sync of a local folder with Drive.

        direction: "upload_only", "download_only", or "bidirectional".
        """
        self._ensure_auth()
        stats: dict[str, Any] = {"uploaded": 0, "downloaded": 0, "skipped": 0, "errors": 0}
        local_dir = self.root / rel_dir

        if direction in ("upload_only", "bidirectional"):
            if local_dir.is_dir():
                files = sorted(f for f in local_dir.rglob("*") if f.is_file())
                for f in files:
                    fid = self.upload_file(f, force=force)
                    if fid:
                        stats["uploaded"] += 1
                    else:
                        stats["errors"] += 1
            else:
                logger.info("Local dir does not exist for upload: %s", rel_dir)

        if direction in ("download_only", "bidirectional"):
            for rel, entry in self._sync_state.all_entries().items():
                if rel.startswith(rel_dir):
                    local = self.root / rel
                    if not local.exists():
                        p = self.download_file(rel)
                        if p:
                            stats["downloaded"] += 1
                        else:
                            stats["errors"] += 1

        return stats

    def sync_all(self, *, force: bool = False) -> dict[str, dict]:
        """Sync all configured folders according to their configured direction."""
        self._ensure_auth()
        self.ensure_folder_tree()

        sync_cfg = self.config.get("sync", {})
        all_stats: dict[str, dict] = {}

        for folder, direction in sync_cfg.items():
            logger.info("Syncing %s (%s)...", folder, direction)
            stats = self.sync_folder(folder, direction=direction, force=force)
            all_stats[folder] = stats
            logger.info("  -> %s", stats)

        return all_stats

    # ------------------------------------------------------------------ #
    # Background upload thread (for live pipeline integration)
    # ------------------------------------------------------------------ #

    def start_background_uploader(self) -> None:
        """Start a daemon thread that processes the upload queue."""
        if self._bg_thread and self._bg_thread.is_alive():
            return
        self._bg_stop.clear()
        self._bg_thread = threading.Thread(
            target=self._bg_upload_worker, daemon=True, name="gdrive-uploader"
        )
        self._bg_thread.start()
        logger.info("Background uploader started.")

    def stop_background_uploader(self, timeout: float = 10.0) -> None:
        """Signal the background thread to stop and wait for it."""
        self._bg_stop.set()
        if self._bg_thread and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=timeout)
        logger.info("Background uploader stopped.")

    def queue_upload(self, local_path: Path | str, rel_subdir: str = "") -> None:
        """Add a file to the background upload queue (non-blocking)."""
        self._upload_q.put((Path(local_path), rel_subdir))

    def _bg_upload_worker(self) -> None:
        """Worker loop: pull files from queue and upload them."""
        while not self._bg_stop.is_set():
            try:
                local_path, _ = self._upload_q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self.upload_file(local_path)
            except Exception as exc:
                logger.error("Background upload failed for %s: %s", local_path, exc)
            finally:
                self._upload_q.task_done()

    def pending_uploads(self) -> int:
        """Number of files waiting in the upload queue."""
        return self._upload_q.qsize()

    # ------------------------------------------------------------------ #
    # DB backup
    # ------------------------------------------------------------------ #

    def backup_database(self, db_path: Path | str) -> str | None:
        """Copy the DB file to a timestamped snapshot on Drive."""
        import shutil

        self._ensure_auth()
        db = Path(db_path) if Path(db_path).is_absolute() else self.root / db_path
        if not db.exists():
            logger.warning("DB not found: %s", db)
            return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_name = f"plates_backup_{ts}.db"
        snap_dir = self.root / "backups"
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap = snap_dir / snap_name
        shutil.copy2(str(db), str(snap))

        fid = self.upload_file(snap, force=True)
        logger.info("DB backed up to Drive: %s", snap_name)
        return fid

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    def cleanup_local(
        self, *, older_than_days: int = 30, dry_run: bool = True
    ) -> list[str]:
        """Delete local files that are confirmed uploaded to Drive.

        Only deletes files older than ``older_than_days``.
        Returns list of paths that were (or would be) deleted.
        """
        cutoff = time.time() - older_than_days * 86400
        to_delete: list[str] = []

        for rel, entry in self._sync_state.all_entries().items():
            local = self.root / rel
            if not local.exists():
                continue
            if local.stat().st_mtime > cutoff:
                continue
            to_delete.append(rel)

        if not dry_run:
            for rel in to_delete:
                try:
                    (self.root / rel).unlink()
                    logger.info("Deleted local: %s", rel)
                except Exception as exc:
                    logger.error("Failed to delete %s: %s", rel, exc)

        return to_delete

    # ------------------------------------------------------------------ #
    # Status / info
    # ------------------------------------------------------------------ #

    def get_status(self) -> dict[str, Any]:
        """Return a summary of sync status."""
        entries = self._sync_state.all_entries()
        total_synced = len(entries)
        total_local = 0
        total_bytes = 0

        for rel in entries:
            local = self.root / rel
            if local.exists():
                total_local += 1
                total_bytes += local.stat().st_size

        about = {}
        if self.is_authenticated():
            try:
                res = self._service.about().get(fields="storageQuota").execute()
                sq = res.get("storageQuota", {})
                about = {
                    "drive_used": _human_size(int(sq.get("usage", 0))),
                    "drive_limit": _human_size(int(sq.get("limit", 0))),
                }
            except Exception:
                pass

        return {
            "authenticated": self.is_authenticated(),
            "enabled": self.enabled,
            "synced_files": total_synced,
            "local_files": total_local,
            "local_size": _human_size(total_bytes),
            "pending_uploads": self.pending_uploads(),
            **about,
        }
