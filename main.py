#!/usr/bin/env python3
"""
main.py — single entry point for the Cambodian ALPR system
==========================================================
Run with NO arguments for an interactive menu:

    python main.py

Or run a task directly:

    python main.py dashboard     # live dashboard on the configured camera_source
    python main.py demo          # run on the real test images
    python main.py stream        # test the phone/IP-camera connection
    python main.py enroll        # whitelist a plate (type the text yourself)
    python main.py read <img>    # read an image -> exact plate text -> offer to enroll
    python main.py db            # view registered plates + recent reads
    python main.py inside        # parking mode: list cars currently inside
    python main.py admin         # web panel to manage the whitelist (browser)
    python main.py accept        # SRS acceptance test (16 checks)
    python main.py camera <url>  # set camera_source (phone URL, 0=webcam, or a folder)

Always run inside the venv (.\\.venv\\Scripts\\activate) so it uses the GPU build.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
PY = sys.executable                      # the venv python running this file
SYS_SCRIPTS = ROOT / "scripts" / "system"
CONFIG = ROOT / "configs" / "system_config.yaml"
sys.path.insert(0, str(ROOT / "src"))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _run(script: str, *args: str) -> None:
    """Run one of the scripts/system/*.py files with the venv python."""
    subprocess.run([PY, str(SYS_SCRIPTS / script), *args])


def _load_cfg() -> dict:
    import yaml
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _open_db():
    from utils.database import PlateDatabase
    cfg = _load_cfg()
    db_path = cfg.get("db_path", "plates.db")
    p = Path(db_path)
    return PlateDatabase(str(p if p.is_absolute() else ROOT / p))


def set_camera_source(url: str) -> None:
    """Rewrite the camera_source line in the config, preserving comments."""
    text = CONFIG.read_text(encoding="utf-8")
    new = re.sub(r'^camera_source:.*$',
                 f'camera_source: "{url}"   # set via main.py',
                 text, count=1, flags=re.MULTILINE)
    CONFIG.write_text(new, encoding="utf-8")
    print(f"[OK] camera_source = {url}")
    print("     Run 'python main.py stream' to test it, then 'dashboard'.")


def status() -> None:
    try:
        cfg = _load_cfg()
        db = _open_db()
        s = db.get_stats()
        print("-" * 60)
        print(f" camera_source : {cfg.get('camera_source')}")
        print(f" registered    : {s.get('total_registered', 0)} plates"
              f"   | total reads: {s.get('total_reads', 0)}")
        print("-" * 60)
        db.close()
    except Exception as exc:
        print(f" (status unavailable: {exc})")


def enroll() -> None:
    print("\n== Enroll a plate (whitelist) ==")
    print("Tip: paste the composed text shown on the dashboard, e.g. 'ភ្នំពេញ 3E-6694'")
    plate = input(" Plate text : ").strip()
    if not plate:
        print(" cancelled (no plate).")
        return
    owner = input(" Owner name : ").strip() or "Unknown"
    vtype = input(" Vehicle    : ").strip() or "car"
    db = _open_db()
    ok = db.add_plate(plate, owner, vtype, "enrolled via main.py")
    print(f" -> {'REGISTERED' if ok else 'already present / error'}: {plate}")
    db.close()


def read_image(path: str | None = None) -> None:
    """Run one image through the full pipeline, print the EXACT composed
    plate_text (what you'd enter in admin), and offer to enroll it."""
    import cv2
    if not path:
        path = input(" image path (e.g. C:/Users/TUF/Downloads/plate.jpg): ").strip().strip('"')
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        print(f" image not found: {p}")
        return
    frame = cv2.imread(str(p))
    if frame is None:
        print(" could not read that image file.")
        return
    print(" loading models (first run takes a few seconds)...")
    from core.alpr_system import ALPRSystem
    system = ALPRSystem(str(CONFIG))
    res = system.process_frame(frame)
    if not res["plates"]:
        print(" no plate detected in that image.")
        system.close()
        return
    pl = res["plates"][0]
    print("\n" + "=" * 54)
    print(f"  composed plate_text : {pl['plate_text']}")
    print(f"  number (CRNN)       : {pl['number']}   conf {pl['crnn_confidence']}")
    print(f"  province id          : {pl['province_id']}   conf {pl['province_confidence']}")
    print(f"  gate decision now    : {pl['action']}")
    print("=" * 54)
    print("  >> This exact 'composed plate_text' is what to put in the admin panel.")
    if pl["crnn_confidence"] < system.crnn_conf_threshold:
        print("  (note: confidence is below the 0.70 gate — a clearer photo reads better)")
    if input("\n  Authorize this plate now? [y/N] ").strip().lower() == "y":
        owner = input("   Owner name: ").strip() or "Unknown"
        ok = system.database.add_plate(pl["plate_text"], owner, "car", "enrolled via read")
        print(f"   -> {'REGISTERED' if ok else 'already present / failed'}: {pl['plate_text']}")
    system.close()


def launch_admin() -> None:
    """Start the web admin panel and open it in the default browser."""
    import webbrowser
    print(" starting web admin panel (Ctrl+C to stop)...")
    try:
        webbrowser.open("http://localhost:5000")
    except Exception:
        pass
    _run("admin_web.py")


def view_inside() -> None:
    """Parking mode: list the cars currently INSIDE (open sessions)."""
    db = _open_db()
    rows = db.active_sessions()
    print("\n== Cars currently INSIDE (parking mode) ==")
    if not rows:
        print("  (none — the lot is empty)")
    for r in rows:
        print(f"  {r.get('plate_text'):<26} in since {r.get('entry_time','')}")
    print(f"\n  {len(rows)} car(s) inside. A car is cleared automatically when it "
          "is read on the way out.")
    db.close()


def view_db() -> None:
    db = _open_db()
    plates = db.get_all_registered()
    auth = [p for p in plates if p.get("status") == "active"]
    blocked = [p for p in plates if p.get("status") != "active"]

    print("\n== AUTHORIZED plates (status 'active' -> gate opens on a confident read) ==")
    for r in auth:
        print(f"  [OK]   {r.get('plate_text'):<22} {r.get('owner_name','')} "
              f"({r.get('vehicle_type','')})")
    if blocked:
        print("\n== NOT authorized (suspended/expired -> gate stays closed) ==")
        for r in blocked:
            print(f"  [X]    {r.get('plate_text'):<22} {r.get('owner_name','')} "
                  f"[{r.get('status','')}]")
    print(f"\n  {len(auth)} authorized, {len(blocked)} blocked, {len(plates)} total.")
    print("  A live plate is ALLOWED only if its composed text EXACTLY matches an")
    print("  authorized row AND crnn confidence >= 0.70.")

    print("\n== Last 10 reads (audit log) ==")
    for r in db.get_recent_reads(limit=10):
        print(f"  {r.get('timestamp','')}  {str(r.get('detected_plate','')):<22} "
              f"{r.get('action','')}  crnn={r.get('crnn_confidence')}")
    db.close()


# --------------------------------------------------------------------------- #
# menu
# --------------------------------------------------------------------------- #
MENU = """
============================================================
   CAMBODIAN ALPR — main menu
============================================================
  1) Live dashboard        (phone / webcam / folder)
  2) Demo on test images    (offline, real plates)
  3) Test phone/camera stream
  4) Enroll a plate         (whitelist)
  5) View database          (registered + recent reads)
  6) SRS acceptance test    (16 checks)
  7) Set camera source      (phone URL, 0=webcam, folder)
  8) Admin panel (web)      (manage whitelist in a browser)
  9) Read an image          (get its exact plate text -> enroll)
  0) Quit
============================================================"""


def interactive() -> None:
    while True:
        status()
        print(MENU)
        choice = input(" choose > ").strip().lower()
        if choice in ("0", "q", "quit", "exit"):
            print("bye.")
            return
        elif choice == "1":
            _run("dashboard.py")
        elif choice == "2":
            _run("run_demo.py")
        elif choice == "3":
            _run("test_stream.py")
        elif choice == "4":
            enroll()
        elif choice == "5":
            view_db()
        elif choice == "6":
            _run("srs_acceptance_test.py")
        elif choice == "7":
            url = input(" new camera source (e.g. http://10.1.64.129:8080/video, 0, "
                        "or data/annotated/test/images/): ").strip()
            if url:
                set_camera_source(url)
        elif choice == "8":
            launch_admin()
        elif choice == "9":
            read_image()
        else:
            print(" ? unknown choice")
        input("\n[enter] to return to menu ...")


# --------------------------------------------------------------------------- #
# direct subcommands
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) == 1:
        interactive()
        return
    cmd = sys.argv[1].lower()
    rest = sys.argv[2:]
    if cmd in ("dashboard", "live"):
        _run("dashboard.py", *rest)
    elif cmd == "demo":
        _run("run_demo.py", *rest)
    elif cmd == "stream":
        _run("test_stream.py", *rest)
    elif cmd == "enroll":
        enroll()
    elif cmd == "db":
        view_db()
    elif cmd == "inside":
        view_inside()
    elif cmd == "admin":
        launch_admin()
    elif cmd == "read":
        read_image(rest[0] if rest else None)
    elif cmd in ("accept", "acceptance", "test"):
        _run("srs_acceptance_test.py", *rest)
    elif cmd == "camera":
        if rest:
            set_camera_source(rest[0])
        else:
            print("usage: python main.py camera <url|0|folder>")
    else:
        print(__doc__)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted.")
