#!/usr/bin/env python3
"""
scripts/setup/setup_gdrive.py
==============================
Guided first-time Google Drive setup for the Cambodian ALPR project.

Walks the user through:
  1. Installing required Python packages
  2. Creating / locating OAuth credentials
  3. Running the browser-based authentication flow
  4. Creating the Drive folder tree
  5. Verifying with a test upload

Run:
    python scripts/setup/setup_gdrive.py
    (or)  python main.py gdrive setup
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

REQS_FILE = ROOT / "requirements-gdrive.txt"
CRED_FILE = ROOT / "configs" / "gdrive_credentials.json"
TOKEN_FILE = ROOT / "configs" / "gdrive_token.json"
CONFIG_FILE = ROOT / "configs" / "system_config.yaml"


def _step(n: int, title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Step {n}: {title}")
    print(f"{'='*60}")


def _check_packages() -> bool:
    """Check if Google API packages are importable."""
    try:
        import googleapiclient  # noqa: F401
        import google_auth_oauthlib  # noqa: F401
        return True
    except ImportError:
        return False


def step1_install() -> None:
    _step(1, "Install Google Drive dependencies")
    if _check_packages():
        print("  [OK] Google API packages already installed.")
        return

    print(f"  Installing from: {REQS_FILE}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", str(REQS_FILE)],
    )
    print("  [OK] Packages installed.")


def step2_credentials() -> None:
    _step(2, "OAuth Credentials")

    if CRED_FILE.exists():
        print(f"  [OK] Credentials file found: {CRED_FILE}")
        return

    print("""
  You need a Google Cloud OAuth2 Client ID (Desktop type).
  Follow these steps:

  1. Go to https://console.cloud.google.com/
  2. Create a new project (or use an existing one)
  3. Enable the "Google Drive API":
       APIs & Services → Library → search "Google Drive API" → Enable
  4. Create OAuth credentials:
       APIs & Services → Credentials → Create Credentials → OAuth Client ID
       Application type: "Desktop app"
       Name: "Cambodian ALPR"
  5. Download the JSON file
  6. Save it as:
       {cred_path}
""".format(cred_path=CRED_FILE))

    print("  After saving the file, press Enter to continue...")
    input("  > ")

    if not CRED_FILE.exists():
        print(f"  [ERROR] File not found: {CRED_FILE}")
        print("  Please download and save the credentials file, then try again.")
        sys.exit(1)
    print("  [OK] Credentials file found!")


def step3_authenticate() -> None:
    _step(3, "Authenticate with Google Drive")

    if TOKEN_FILE.exists():
        print(f"  [OK] Existing token found: {TOKEN_FILE}")
        reauth = input("  Re-authenticate? [y/N] ").strip().lower()
        if reauth != "y":
            return

    print("  Opening browser for Google sign-in...")
    print("  (Grant access to your Google Drive when prompted)\n")

    import yaml
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    gdrive_cfg = config.get("gdrive", {})
    # Override paths to ensure they point to the right files
    gdrive_cfg["credentials_path"] = str(CRED_FILE)
    gdrive_cfg["token_path"] = str(TOKEN_FILE)

    from utils.gdrive_storage import GDriveStorage
    gd = GDriveStorage(gdrive_cfg, ROOT)
    ok = gd.authenticate()

    if ok:
        print("\n  [OK] Authentication successful! Token saved.")
    else:
        print("\n  [ERROR] Authentication failed. Check the error above.")
        sys.exit(1)


def step4_folder_tree() -> None:
    _step(4, "Create Drive folder structure")

    import yaml
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    gdrive_cfg = config.get("gdrive", {})
    gdrive_cfg["credentials_path"] = str(CRED_FILE)
    gdrive_cfg["token_path"] = str(TOKEN_FILE)

    from utils.gdrive_storage import GDriveStorage
    gd = GDriveStorage(gdrive_cfg, ROOT)
    gd.authenticate()
    gd.ensure_folder_tree()
    print("  [OK] Folder tree created on Google Drive!")


def step5_test() -> None:
    _step(5, "Test upload / download")

    # Create a small test file
    test_file = ROOT / "configs" / "_gdrive_test.txt"
    test_file.write_text(
        f"Google Drive integration test — {__import__('datetime').datetime.now()}\n",
        encoding="utf-8",
    )

    import yaml
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    gdrive_cfg = config.get("gdrive", {})
    gdrive_cfg["credentials_path"] = str(CRED_FILE)
    gdrive_cfg["token_path"] = str(TOKEN_FILE)

    from utils.gdrive_storage import GDriveStorage
    gd = GDriveStorage(gdrive_cfg, ROOT)
    gd.authenticate()
    gd.ensure_folder_tree()

    fid = gd.upload_file(test_file, force=True)
    test_file.unlink(missing_ok=True)

    if fid:
        print(f"  [OK] Test file uploaded successfully! (Drive ID: {fid})")
    else:
        print("  [ERROR] Test upload failed.")
        sys.exit(1)


def main() -> None:
    print("""
╔══════════════════════════════════════════════════════════╗
║     Google Drive Setup — Cambodian ALPR Project         ║
║     5 TB cloud storage for models, data, & backups      ║
╚══════════════════════════════════════════════════════════╝
""")

    step1_install()
    step2_credentials()
    step3_authenticate()
    step4_folder_tree()
    step5_test()

    print(f"""
{'='*60}
  ✓ Google Drive integration is ready!
{'='*60}

  What you can do now:
    python main.py gdrive status     Show sync status & Drive usage
    python main.py gdrive sync       Sync all configured folders
    python main.py gdrive upload <p> Upload a specific file/folder
    python main.py gdrive download   Download from Drive to local
    python main.py gdrive cleanup    Delete local files backed up to Drive

  To enable auto-upload in the live pipeline, edit system_config.yaml:
    gdrive:
      enabled: true

  Your 5 TB Google Drive is now connected! 🎉
""")


if __name__ == "__main__":
    main()
