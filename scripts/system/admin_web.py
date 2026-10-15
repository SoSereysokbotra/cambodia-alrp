#!/usr/bin/env python3
"""
scripts/system/admin_web.py
===========================
Web admin panel for the Cambodian ALPR whitelist (SRS Phase 8 — ADM-001/002/003).
A browser UI (works from the PC or a phone on the same Wi-Fi) to:
  * see AUTHORIZED vs blocked plates
  * add a plate (register)          -> ADM-001
  * suspend / reactivate / delete   -> ADM-002
  * search the audit log            -> ADM-003

No web framework needed: Python stdlib http.server + jinja2 (already installed).
Khmer text renders natively in the browser (unlike the OpenCV window).

Authentication (PLAN_V2 Phase 6.1)
----------------------------------
This panel can add, suspend and DELETE whitelist entries — i.e. it controls who the
gate opens for. It previously had no authentication at all, so anyone on the LAN
could take it over. It now requires a password login and **refuses to start until
one is set**:

    python scripts/system/admin_web.py --set-password     # prompts, stores a hash
    python scripts/system/admin_web.py                    # normal start

The password is stored as a PBKDF2-HMAC-SHA256 hash (200k iterations, per-install
random salt) in `configs/admin_auth.json`, which is gitignored — the plaintext is
never written to disk or to the repo. Sessions are server-side random tokens in a
cookie (HttpOnly, SameSite=Strict), so nothing sensitive lives in the browser.

`--no-auth` exists for a localhost-only demo and prints a loud warning; it refuses
to bind to a non-loopback address, so it cannot accidentally expose the panel.

Run:
    python scripts/system/admin_web.py            # http://<this-pc-ip>:5000
    python scripts/system/admin_web.py --port 8000
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from jinja2 import Template                       # noqa: E402
from utils.database import PlateDatabase          # noqa: E402

CONFIG = PROJECT_ROOT / "configs" / "system_config.yaml"
AUTH_FILE = PROJECT_ROOT / "configs" / "admin_auth.json"   # gitignored


# --------------------------------------------------------------------------- #
# Live view (--live): run the real pipeline in a background thread and serve the
# annotated frames as MJPEG, so the whole demo — gate AND whitelist — is in one
# browser window. Without --live the panel starts instantly and loads no models.
# --------------------------------------------------------------------------- #
class LiveWorker:
    """Owns the ALPRSystem. One thread reads frames, runs the pipeline and keeps
    the latest annotated JPEG; HTTP handlers only ever read that buffer."""

    def __init__(self) -> None:
        self.system = None
        self.frame_jpeg: bytes | None = None
        self.last = {}                       # last plate result, for the status strip
        self.error: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started_at = time.time()
        self._reg_plate, self._reg_val, self._reg_at = None, None, 0.0   # lookup cache

    # -- lifecycle --------------------------------------------------------- #
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.system is not None:
            try:
                self.system.camera.stop()
                self.system.close()
            except Exception:
                pass

    def _run(self) -> None:
        try:
            import cv2
            from core.alpr_system import ALPRSystem
            print("[live] loading models ...")
            self.system = ALPRSystem(str(CONFIG))
            self.system._live_dedup = True        # one audit row per car, as in run_video
            self.system.camera.start()
            print("[live] pipeline running")
        except Exception as exc:                  # never kill the admin panel
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[live] FAILED: {self.error}")
            return

        while not self._stop.is_set():
            frame = self.system.camera.get_frame(self.system.frame_timeout)
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                res = self.system.process_frame(frame)
            except Exception as exc:
                self.error = str(exc)
                time.sleep(0.5)
                continue
            out = res["frame"]
            if self.system.estop_active:
                cv2.putText(out, "*** EMERGENCY STOP ***",
                            (out.shape[1] // 2 - 180, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with self._lock:
                    self.frame_jpeg = buf.tobytes()
                    self.last = (res["plates"][0] if res["plates"] else {}) | {
                        "total_ms": round(res["total_ms"], 1)}

    # -- read side --------------------------------------------------------- #
    def _is_registered(self, plate: str) -> bool | None:
        """Read-only whitelist lookup, cached for a second.

        Must never open a writing connection: the pipeline holds the write lock
        almost continuously, so anything that commits here starves it.
        """
        if not plate or plate == "(unreadable)":
            return None
        now = time.time()
        if plate == self._reg_plate and now - self._reg_at < 1.0:
            return self._reg_val
        val = None
        try:
            import sqlite3
            path = Path(_cfg().get("db_path", "plates.db"))
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=3)
            try:
                val = con.execute(
                    "SELECT 1 FROM registered_plates "
                    "WHERE plate_text = ? AND status = 'active' LIMIT 1",
                    (plate,)).fetchone() is not None
            finally:
                con.close()
        except Exception:
            val = None
        self._reg_plate, self._reg_val, self._reg_at = plate, val, now
        return val

    def jpeg(self) -> bytes | None:
        with self._lock:
            return self.frame_jpeg

    def status(self) -> dict:
        with self._lock:
            p = dict(self.last)
        s = self.system
        if s is None:
            return {"ready": False, "error": self.error}
        st = s.get_session_stats()
        # Does the last read already have a whitelist row? This drives the
        # "Authorize" button on the live page, so the demo never has to leave it.
        #
        # Deliberately NOT PlateDatabase(): its __init__ runs CREATE TABLE IF NOT
        # EXISTS + commit(), i.e. a WRITE transaction. Called once per second from
        # the status poll it fought the pipeline's ~20 audit writes/sec and every
        # single write failed with "database is locked". Read-only connection,
        # one SELECT, and the answer is cached for a second.
        plate = p.get("plate_text") or ""
        registered = self._is_registered(plate)
        return {
            "ready": True, "error": self.error,
            "plate": p.get("plate_text", "—"),
            "registered": registered,
            "action": p.get("action", "—"),
            "crnn": p.get("crnn_confidence"),
            "latency_ms": p.get("total_ms"),
            "estop": bool(s.estop_active),
            "processed": st["processed"], "allowed": st["allowed"],
            "denied": st["denied"], "review": st["review"],
            "gate": st["gate_status"],
            "uptime_sec": round(time.time() - self.started_at),
        }


LIVE: LiveWorker | None = None       # set by main() when --live is passed

# --------------------------------------------------------------------------- #
# Authentication (PLAN_V2 Phase 6.1)
# --------------------------------------------------------------------------- #
PBKDF2_ITERATIONS = 200_000
SESSION_COOKIE = "alpr_admin_session"
SESSION_TIMEOUT_SEC = 8 * 3600         # re-login once a shift
LOGIN_MAX_FAILURES = 5                 # per client IP
LOGIN_LOCKOUT_SEC = 300                # then wait this long

# token -> expiry epoch. In-memory on purpose: a restart invalidates every session,
# and there is no session state worth persisting for a single-admin LAN panel.
_SESSIONS: dict[str, float] = {}
# client ip -> [failure count, first-failure epoch]
_LOGIN_FAILURES: dict[str, list] = {}


def hash_password(password: str, salt: bytes | None = None,
                  iterations: int = PBKDF2_ITERATIONS) -> dict:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return {"algorithm": "pbkdf2_sha256", "iterations": iterations,
            "salt": salt.hex(), "hash": dk.hex()}


def verify_password(password: str, record: dict) -> bool:
    """Constant-time check so a timing side-channel can't leak the hash."""
    try:
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                 bytes.fromhex(record["salt"]),
                                 int(record["iterations"]))
    except Exception:
        return False
    return hmac.compare_digest(dk.hex(), record.get("hash", ""))


def load_auth() -> dict | None:
    if not AUTH_FILE.exists():
        return None
    try:
        rec = json.loads(AUTH_FILE.read_text(encoding="utf-8"))
        return rec if rec.get("hash") and rec.get("salt") else None
    except Exception as exc:
        print(f"[admin] could not read {AUTH_FILE.name}: {exc}")
        return None


def set_password_cli() -> int:
    """Prompt for a new admin password and store only its hash."""
    import getpass
    pw = os.environ.get("ALPR_ADMIN_PASSWORD")
    if pw:
        print("[admin] using ALPR_ADMIN_PASSWORD from the environment")
    else:
        pw = getpass.getpass("New admin password: ")
        if pw != getpass.getpass("Repeat password: "):
            print("[X] passwords do not match")
            return 1
    if len(pw) < 8:
        print("[X] use at least 8 characters")
        return 1
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUTH_FILE.write_text(json.dumps(hash_password(pw), indent=2), encoding="utf-8")
    try:                      # best-effort: owner-only on POSIX, no-op on Windows
        os.chmod(AUTH_FILE, 0o600)
    except Exception:
        pass
    print(f"[ok] password set -> {AUTH_FILE.relative_to(PROJECT_ROOT)} "
          "(hash only; keep this file out of git)")
    return 0


def _new_session() -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    _SESSIONS[token] = now + SESSION_TIMEOUT_SEC
    for t, exp in list(_SESSIONS.items()):        # opportunistic cleanup
        if exp < now:
            _SESSIONS.pop(t, None)
    return token


def _session_valid(token: str | None) -> bool:
    if not token:
        return False
    exp = _SESSIONS.get(token)
    if exp is None:
        return False
    if exp < time.time():
        _SESSIONS.pop(token, None)
        return False
    return True


def _login_locked(ip: str) -> int:
    """Seconds remaining in a lockout for this IP (0 if not locked)."""
    rec = _LOGIN_FAILURES.get(ip)
    if not rec or rec[0] < LOGIN_MAX_FAILURES:
        return 0
    remaining = int(rec[1] + LOGIN_LOCKOUT_SEC - time.time())
    if remaining <= 0:
        _LOGIN_FAILURES.pop(ip, None)
        return 0
    return remaining


def _record_failure(ip: str) -> None:
    rec = _LOGIN_FAILURES.setdefault(ip, [0, time.time()])
    if rec[0] >= LOGIN_MAX_FAILURES and time.time() > rec[1] + LOGIN_LOCKOUT_SEC:
        rec[:] = [0, time.time()]
    rec[0] += 1
    if rec[0] == LOGIN_MAX_FAILURES:
        rec[1] = time.time()


LOGIN_PAGE = Template("""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign in — Cambodian ALPR</title>
<style>
  :root{
    --ink:#0b1f3a; --navy:#1f3b73; --accent:#2e7dd1; --accent-dark:#255f9e;
    --mute:#64748b; --line:#e2e8f0; --bg:#f5f7fa; --card:#ffffff;
    --ok:#16a34a; --ok-bg:#eafaf0; --bad:#dc2626; --bad-bg:#fdecec;
    --warn:#b45309; --warn-bg:#fdf6e7;
    --shadow:0 1px 2px rgba(16,24,40,.06), 0 1px 3px rgba(16,24,40,.08);
    --radius:10px;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font-family:system-ui,-apple-system,"Segoe UI","Noto Sans Khmer",sans-serif;
       font-size:15px;line-height:1.5}
  a{color:var(--accent);text-decoration:none}
  .shell{max-width:1180px;margin-inline:auto;padding:0 1.25rem 3rem}

  /* ---- top bar ---- */
  .topbar{background:var(--card);border-bottom:1px solid var(--line);
          box-shadow:var(--shadow);position:sticky;top:0;z-index:20}
  .topbar .inner{max-width:1180px;margin-inline:auto;padding:.7rem 1.25rem;
          display:flex;align-items:center;gap:1.25rem;flex-wrap:wrap}
  .brand{font-weight:800;font-size:1.02rem;color:var(--navy);letter-spacing:-.01em;
         display:flex;align-items:center;gap:.5rem}
  .brand .dot{width:9px;height:9px;border-radius:50%;background:var(--accent)}
  nav{display:flex;gap:.25rem;margin-left:auto}
  nav a{padding:.45rem .85rem;border-radius:8px;font-weight:600;font-size:.88rem;
        color:var(--mute)}
  nav a:hover{background:var(--bg);color:var(--ink)}
  nav a.on{background:var(--navy);color:#fff}
  nav a.quiet{color:var(--mute)}

  /* ---- page head ---- */
  .head{padding:1.6rem 0 1.1rem}
  .head h1{margin:0;font-size:1.55rem;font-weight:800;letter-spacing:-.02em}
  .head .sub{color:var(--mute);font-size:.92rem;margin-top:.3rem;max-width:70ch}

  /* ---- cards ---- */
  .card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
        box-shadow:var(--shadow);margin-bottom:1.1rem;overflow:hidden}
  .card > .ch{padding:.85rem 1.1rem;border-bottom:1px solid var(--line);
       display:flex;align-items:center;justify-content:space-between;gap:1rem;
       background:#fcfdfe}
  .card > .ch h2{margin:0;font-size:.98rem;font-weight:700}
  .card > .ch .n{color:var(--mute);font-weight:600;font-size:.85rem}
  .card > .cb{padding:1.1rem}
  .card > .cb.flush{padding:0;overflow-x:auto}
  .pager{padding:.3rem .65rem;border:1px solid var(--line);border-radius:7px;
         background:#fff;font-weight:700;font-size:.82rem;color:var(--ink)}
  .pager:hover{border-color:var(--accent);color:var(--accent)}

  /* ---- stats ---- */
  .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.7rem}
  .stat{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
        padding:.75rem .9rem;box-shadow:var(--shadow)}
  .stat b{display:block;font-size:1.5rem;font-weight:800;letter-spacing:-.02em;
          line-height:1.15}
  .stat span{color:var(--mute);font-size:.76rem;font-weight:600;
             text-transform:uppercase;letter-spacing:.05em}

  /* ---- tables ---- */
  table{width:100%;border-collapse:collapse}
  th{text-align:left;padding:.6rem 1.1rem;font-size:.74rem;font-weight:700;
     color:var(--mute);text-transform:uppercase;letter-spacing:.05em;
     background:#fcfdfe;border-bottom:1px solid var(--line);white-space:nowrap}
  td{padding:.65rem 1.1rem;border-bottom:1px solid var(--line);font-size:.92rem;
     vertical-align:middle}
  tbody tr:last-child td{border-bottom:0}
  tbody tr:hover{background:#fafcfe}
  .plate{font-weight:700;letter-spacing:-.01em}
  .mono{font-variant-numeric:tabular-nums;color:var(--mute);font-size:.86rem;
        white-space:nowrap}
  .empty{color:var(--mute);font-style:italic}

  /* ---- pills ---- */
  .pill{display:inline-block;padding:.12rem .55rem;border-radius:999px;
        font-size:.76rem;font-weight:700;white-space:nowrap}
  .p-ok{background:var(--ok-bg);color:var(--ok)}
  .p-bad{background:var(--bad-bg);color:var(--bad)}
  .p-warn{background:var(--warn-bg);color:var(--warn)}
  .p-mute{background:#eef2f7;color:var(--mute)}

  /* ---- forms ---- */
  label{display:block;font-size:.76rem;font-weight:700;color:var(--mute);
        text-transform:uppercase;letter-spacing:.04em;margin-bottom:.3rem}
  input,select{width:100%;padding:.52rem .7rem;border:1px solid var(--line);
       border-radius:8px;background:#fff;color:var(--ink);font-size:.93rem;
       font-family:inherit}
  input:focus,select:focus{outline:0;border-color:var(--accent);
       box-shadow:0 0 0 3px rgba(46,125,209,.14)}
  input::placeholder{color:#b4bfcd}
  .row{display:flex;gap:.8rem;flex-wrap:wrap;align-items:flex-end}
  .row > div{flex:1 1 170px;min-width:0}
  .row > div.sm{flex:0 0 auto}
  form.inline{display:inline}

  /* ---- buttons ---- */
  button{cursor:pointer;border:1px solid transparent;border-radius:8px;
         padding:.52rem .9rem;font-size:.86rem;font-weight:700;font-family:inherit;
         white-space:nowrap;transition:filter .12s}
  button:hover{filter:brightness(.94)}
  .b-primary{background:var(--accent);color:#fff}
  .b-ghost{background:#fff;color:var(--ink);border-color:var(--line)}
  .b-ok{background:var(--ok);color:#fff}
  .b-warn{background:#fff;color:var(--warn);border-color:#f0dfb8}
  .b-danger{background:#fff;color:var(--bad);border-color:#f3cfcf}
  .b-sm{padding:.34rem .62rem;font-size:.79rem}

  /* ---- messages ---- */
  .msg{padding:.7rem 1rem;border-radius:var(--radius);font-size:.9rem;
       font-weight:600;margin-bottom:1.1rem;border:1px solid}
  .msg.ok{background:var(--ok-bg);color:var(--ok);border-color:#bfe8cd}
  .msg.err{background:var(--bad-bg);color:var(--bad);border-color:#f3cfcf}
  .foot{color:var(--mute);font-size:.82rem;padding:.75rem 1.1rem;
        border-top:1px solid var(--line);background:#fcfdfe}
</style>
<style>
  body{display:grid;place-items:center;min-height:100vh;padding:1.25rem}
  .login{width:100%;max-width:370px}
  .login .cb{padding:1.6rem}
  .login h1{margin:0 0 .25rem;font-size:1.25rem;font-weight:800}
  .login .sub{color:var(--mute);font-size:.88rem;margin-bottom:1.3rem}
  .login button{width:100%;margin-top:.9rem;padding:.65rem}
  .note{text-align:center;color:var(--mute);font-size:.78rem;margin-top:1rem}
</style></head><body>
  <div class="login">
    <div class="card"><div class="cb">
      <h1>Cambodian ALPR</h1>
      <div class="sub">Whitelist &amp; audit control</div>
      {% if error %}<div class="msg err">{{ error }}</div>{% endif %}
      <form method="post" action="/login">
        <label for="pw">Admin password</label>
        <input id="pw" type="password" name="password" autofocus
               autocomplete="current-password" required>
        <button class="b-primary" type="submit">Sign in</button>
      </form>
    </div></div>
    <div class="note">This panel controls which plates open the gate.</div>
  </div>
</body></html>""")

LIVE_PAGE = Template("""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live gate — Cambodian ALPR</title>
<style>
  :root{
    --ink:#0b1f3a; --navy:#1f3b73; --accent:#2e7dd1; --accent-dark:#255f9e;
    --mute:#64748b; --line:#e2e8f0; --bg:#f5f7fa; --card:#ffffff;
    --ok:#16a34a; --ok-bg:#eafaf0; --bad:#dc2626; --bad-bg:#fdecec;
    --warn:#b45309; --warn-bg:#fdf6e7;
    --shadow:0 1px 2px rgba(16,24,40,.06), 0 1px 3px rgba(16,24,40,.08);
    --radius:10px;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font-family:system-ui,-apple-system,"Segoe UI","Noto Sans Khmer",sans-serif;
       font-size:15px;line-height:1.5}
  a{color:var(--accent);text-decoration:none}
  .shell{max-width:1180px;margin-inline:auto;padding:0 1.25rem 3rem}

  /* ---- top bar ---- */
  .topbar{background:var(--card);border-bottom:1px solid var(--line);
          box-shadow:var(--shadow);position:sticky;top:0;z-index:20}
  .topbar .inner{max-width:1180px;margin-inline:auto;padding:.7rem 1.25rem;
          display:flex;align-items:center;gap:1.25rem;flex-wrap:wrap}
  .brand{font-weight:800;font-size:1.02rem;color:var(--navy);letter-spacing:-.01em;
         display:flex;align-items:center;gap:.5rem}
  .brand .dot{width:9px;height:9px;border-radius:50%;background:var(--accent)}
  nav{display:flex;gap:.25rem;margin-left:auto}
  nav a{padding:.45rem .85rem;border-radius:8px;font-weight:600;font-size:.88rem;
        color:var(--mute)}
  nav a:hover{background:var(--bg);color:var(--ink)}
  nav a.on{background:var(--navy);color:#fff}
  nav a.quiet{color:var(--mute)}

  /* ---- page head ---- */
  .head{padding:1.6rem 0 1.1rem}
  .head h1{margin:0;font-size:1.55rem;font-weight:800;letter-spacing:-.02em}
  .head .sub{color:var(--mute);font-size:.92rem;margin-top:.3rem;max-width:70ch}

  /* ---- cards ---- */
  .card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
        box-shadow:var(--shadow);margin-bottom:1.1rem;overflow:hidden}
  .card > .ch{padding:.85rem 1.1rem;border-bottom:1px solid var(--line);
       display:flex;align-items:center;justify-content:space-between;gap:1rem;
       background:#fcfdfe}
  .card > .ch h2{margin:0;font-size:.98rem;font-weight:700}
  .card > .ch .n{color:var(--mute);font-weight:600;font-size:.85rem}
  .card > .cb{padding:1.1rem}
  .card > .cb.flush{padding:0}

  /* ---- stats ---- */
  .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.7rem}
  .stat{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
        padding:.75rem .9rem;box-shadow:var(--shadow)}
  .stat b{display:block;font-size:1.5rem;font-weight:800;letter-spacing:-.02em;
          line-height:1.15}
  .stat span{color:var(--mute);font-size:.76rem;font-weight:600;
             text-transform:uppercase;letter-spacing:.05em}

  /* ---- tables ---- */
  table{width:100%;border-collapse:collapse}
  th{text-align:left;padding:.6rem 1.1rem;font-size:.74rem;font-weight:700;
     color:var(--mute);text-transform:uppercase;letter-spacing:.05em;
     background:#fcfdfe;border-bottom:1px solid var(--line);white-space:nowrap}
  td{padding:.65rem 1.1rem;border-bottom:1px solid var(--line);font-size:.92rem;
     vertical-align:middle}
  tbody tr:last-child td{border-bottom:0}
  tbody tr:hover{background:#fafcfe}
  .plate{font-weight:700;letter-spacing:-.01em}
  .mono{font-variant-numeric:tabular-nums;color:var(--mute);font-size:.86rem;
        white-space:nowrap}
  .empty{color:var(--mute);font-style:italic}

  /* ---- pills ---- */
  .pill{display:inline-block;padding:.12rem .55rem;border-radius:999px;
        font-size:.76rem;font-weight:700;white-space:nowrap}
  .p-ok{background:var(--ok-bg);color:var(--ok)}
  .p-bad{background:var(--bad-bg);color:var(--bad)}
  .p-warn{background:var(--warn-bg);color:var(--warn)}
  .p-mute{background:#eef2f7;color:var(--mute)}

  /* ---- forms ---- */
  label{display:block;font-size:.76rem;font-weight:700;color:var(--mute);
        text-transform:uppercase;letter-spacing:.04em;margin-bottom:.3rem}
  input,select{width:100%;padding:.52rem .7rem;border:1px solid var(--line);
       border-radius:8px;background:#fff;color:var(--ink);font-size:.93rem;
       font-family:inherit}
  input:focus,select:focus{outline:0;border-color:var(--accent);
       box-shadow:0 0 0 3px rgba(46,125,209,.14)}
  input::placeholder{color:#b4bfcd}
  .row{display:flex;gap:.8rem;flex-wrap:wrap;align-items:flex-end}
  .row > div{flex:1 1 170px;min-width:0}
  .row > div.sm{flex:0 0 auto}
  form.inline{display:inline}

  /* ---- buttons ---- */
  button{cursor:pointer;border:1px solid transparent;border-radius:8px;
         padding:.52rem .9rem;font-size:.86rem;font-weight:700;font-family:inherit;
         white-space:nowrap;transition:filter .12s}
  button:hover{filter:brightness(.94)}
  .b-primary{background:var(--accent);color:#fff}
  .b-ghost{background:#fff;color:var(--ink);border-color:var(--line)}
  .b-ok{background:var(--ok);color:#fff}
  .b-warn{background:#fff;color:var(--warn);border-color:#f0dfb8}
  .b-danger{background:#fff;color:var(--bad);border-color:#f3cfcf}
  .b-sm{padding:.34rem .62rem;font-size:.79rem}

  /* ---- messages ---- */
  .msg{padding:.7rem 1rem;border-radius:var(--radius);font-size:.9rem;
       font-weight:600;margin-bottom:1.1rem;border:1px solid}
  .msg.ok{background:var(--ok-bg);color:var(--ok);border-color:#bfe8cd}
  .msg.err{background:var(--bad-bg);color:var(--bad);border-color:#f3cfcf}
  .foot{color:var(--mute);font-size:.82rem;padding:.75rem 1.1rem;
        border-top:1px solid var(--line);background:#fcfdfe}
</style>
<style>
  .wrap{display:grid;grid-template-columns:1fr 330px;gap:1.1rem;align-items:start}
  @media(max-width:900px){.wrap{grid-template-columns:1fr}}
  #cam{width:100%;display:block;background:#0b1f3a;border-radius:8px}
  #camload{aspect-ratio:1/1;display:grid;place-content:center;gap:.8rem;
           justify-items:center;background:#f8fafc;border-radius:8px;
           color:var(--mute);font-size:.9rem;text-align:center;padding:1rem}
  .spinner{width:26px;height:26px;border:3px solid var(--line);
           border-top-color:var(--accent);border-radius:50%;
           animation:spin .9s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  .last{font-size:1.45rem;font-weight:800;letter-spacing:-.02em;word-break:break-all;
        line-height:1.25;margin:.1rem 0 .5rem}
  .ctl{display:grid;gap:.55rem}
  .ctl button{width:100%;padding:.75rem;font-size:.92rem}
  .kv{display:flex;justify-content:space-between;font-size:.86rem;
      padding:.35rem 0;border-bottom:1px solid var(--line)}
  .kv:last-child{border-bottom:0}
  .kv span{color:var(--mute)}
  .kv b{font-variant-numeric:tabular-nums}
</style></head><body>
<div class="topbar"><div class="inner">
  <div class="brand"><span class="dot"></span> Cambodian ALPR</div>
  <nav>
    <a href="/live" class="on">Live gate</a>
    <a href="/" class="">Whitelist &amp; audit</a>
    <a href="/logout" class="quiet">Sign out</a>
  </nav>
</div></div>
<div class="shell">
  <div class="head">
    <h1>Live gate</h1>
    <div class="sub">{{ source }} — the pipeline decides, the gate stays closed unless a
      confident read matches an authorized plate.</div>
  </div>
  {% if error %}<div class="msg err">Pipeline error: {{ error }}</div>{% endif %}
  <div class="wrap">
    <div class="card"><div class="cb flush" style="position:relative">
      <div id="camwrap">
        <div id="camload">
          <div class="spinner"></div>
          <div>Loading the pipeline — four models, about ten seconds…</div>
        </div>
        <img id="cam" src="/stream" alt="live camera" style="display:none"
             onload="camUp()" onerror="camDown()">
      </div>
    </div></div>
    <div>
      <div class="card">
        <div class="ch"><h2>Last read</h2><span class="pill p-mute" id="gate">—</span></div>
        <div class="cb">
          <div class="last" id="plate">—</div>
          <span class="pill p-mute" id="action">waiting</span>
          <span class="pill p-ok" id="known" style="display:none">authorized</span>
          <form method="post" action="/live_authorize" id="authform"
                style="display:none;margin-top:.8rem"
                onsubmit="return confirm('Authorize this plate to open the gate?\n\n' + document.getElementById('plate').textContent)">
            <input type="hidden" name="plate_text" id="authplate">
            <button class="b-primary" style="width:100%" type="submit">
              Authorize this plate</button>
          </form>
          <div style="margin-top:.9rem">
            <div class="kv"><span>Reader confidence</span><b id="conf">—</b></div>
            <div class="kv"><span>Latency</span><b id="latency">—</b></div>
            <div class="kv"><span>Frames processed</span><b id="processed">—</b></div>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="ch"><h2>Session</h2></div>
        <div class="cb">
          <div class="stats" style="grid-template-columns:1fr 1fr 1fr">
            <div class="stat"><b id="allowed" style="color:var(--ok)">0</b><span>Allowed</span></div>
            <div class="stat"><b id="denied" style="color:var(--bad)">0</b><span>Denied</span></div>
            <div class="stat"><b id="review" style="color:var(--warn)">0</b><span>Review</span></div>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="ch"><h2>Operator controls</h2></div>
        <div class="cb"><div class="ctl">
          <form method="post" action="/live_open">
            <button class="b-ok" type="submit">Manual open</button></form>
          <form method="post" action="/live_estop">
            <button class="b-danger" id="estop" type="submit">Emergency stop</button></form>
        </div></div>
        <div class="foot">Both actions are written to the audit log.</div>
      </div>
    </div>
  </div>
</div>
<script>
async function tick(){
  try{
    const s = await (await fetch('/live_status.json',{cache:'no-store'})).json();
    if(!s.ready) return;
    const $ = i => document.getElementById(i);
    $('plate').textContent     = s.plate || '—';
    $('allowed').textContent   = s.allowed;
    $('denied').textContent    = s.denied;
    $('review').textContent    = s.review;
    $('processed').textContent = s.processed;
    $('latency').textContent   = s.latency_ms != null ? s.latency_ms + ' ms' : '—';
    $('conf').textContent      = s.crnn != null ? s.crnn : '—';
    const a = $('action');
    a.textContent = (s.action || '—').replace(/_/g,' ');
    a.className = 'pill ' + (s.action==='ENTRY_ALLOWED' ? 'p-ok'
                          : s.action==='ENTRY_DENIED'  ? 'p-bad'
                          : s.action==='—'             ? 'p-mute' : 'p-warn');
    const g = $('gate');
    g.textContent = s.estop ? 'E-STOP' : 'gate ' + s.gate;
    g.className = 'pill ' + (s.estop ? 'p-bad' : 'p-mute');
    $('estop').textContent = s.estop ? 'Clear emergency stop' : 'Emergency stop';
    // offer one-click enrolment only for a plate that is readable and unknown
    const plate = s.plate, known = s.registered === true;
    const show  = plate && plate !== '—' && plate !== '(unreadable)' && !known;
    $('authform').style.display = show ? 'block' : 'none';
    $('known').style.display    = known ? 'inline-block' : 'none';
    if (show) $('authplate').value = plate;
  }catch(e){}
}
function camUp(){
  document.getElementById('camload').style.display='none';
  document.getElementById('cam').style.display='block';
}
function camDown(){
  document.getElementById('cam').style.display='none';
  const l=document.getElementById('camload');
  l.style.display='grid';
  l.innerHTML='<div style="font-size:1.6rem">⚠</div>'
            + '<div>Video stream lost. The pipeline may have stopped — '
            + 'check the terminal, then reload.</div>';
}
setInterval(tick,1000); tick();
</script>
</body></html>""")


PAGE = Template("""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Whitelist &amp; audit — Cambodian ALPR</title>
<style>
  :root{
    --ink:#0b1f3a; --navy:#1f3b73; --accent:#2e7dd1; --accent-dark:#255f9e;
    --mute:#64748b; --line:#e2e8f0; --bg:#f5f7fa; --card:#ffffff;
    --ok:#16a34a; --ok-bg:#eafaf0; --bad:#dc2626; --bad-bg:#fdecec;
    --warn:#b45309; --warn-bg:#fdf6e7;
    --shadow:0 1px 2px rgba(16,24,40,.06), 0 1px 3px rgba(16,24,40,.08);
    --radius:10px;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font-family:system-ui,-apple-system,"Segoe UI","Noto Sans Khmer",sans-serif;
       font-size:15px;line-height:1.5}
  a{color:var(--accent);text-decoration:none}
  .shell{max-width:1180px;margin-inline:auto;padding:0 1.25rem 3rem}

  /* ---- top bar ---- */
  .topbar{background:var(--card);border-bottom:1px solid var(--line);
          box-shadow:var(--shadow);position:sticky;top:0;z-index:20}
  .topbar .inner{max-width:1180px;margin-inline:auto;padding:.7rem 1.25rem;
          display:flex;align-items:center;gap:1.25rem;flex-wrap:wrap}
  .brand{font-weight:800;font-size:1.02rem;color:var(--navy);letter-spacing:-.01em;
         display:flex;align-items:center;gap:.5rem}
  .brand .dot{width:9px;height:9px;border-radius:50%;background:var(--accent)}
  nav{display:flex;gap:.25rem;margin-left:auto}
  nav a{padding:.45rem .85rem;border-radius:8px;font-weight:600;font-size:.88rem;
        color:var(--mute)}
  nav a:hover{background:var(--bg);color:var(--ink)}
  nav a.on{background:var(--navy);color:#fff}
  nav a.quiet{color:var(--mute)}

  /* ---- page head ---- */
  .head{padding:1.6rem 0 1.1rem}
  .head h1{margin:0;font-size:1.55rem;font-weight:800;letter-spacing:-.02em}
  .head .sub{color:var(--mute);font-size:.92rem;margin-top:.3rem;max-width:70ch}

  /* ---- cards ---- */
  .card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
        box-shadow:var(--shadow);margin-bottom:1.1rem;overflow:hidden}
  .card > .ch{padding:.85rem 1.1rem;border-bottom:1px solid var(--line);
       display:flex;align-items:center;justify-content:space-between;gap:1rem;
       background:#fcfdfe}
  .card > .ch h2{margin:0;font-size:.98rem;font-weight:700}
  .card > .ch .n{color:var(--mute);font-weight:600;font-size:.85rem}
  .card > .cb{padding:1.1rem}
  .card > .cb.flush{padding:0}

  /* ---- stats ---- */
  .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.7rem}
  .stat{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
        padding:.75rem .9rem;box-shadow:var(--shadow)}
  .stat b{display:block;font-size:1.5rem;font-weight:800;letter-spacing:-.02em;
          line-height:1.15}
  .stat span{color:var(--mute);font-size:.76rem;font-weight:600;
             text-transform:uppercase;letter-spacing:.05em}

  /* ---- tables ---- */
  table{width:100%;border-collapse:collapse}
  th{text-align:left;padding:.6rem 1.1rem;font-size:.74rem;font-weight:700;
     color:var(--mute);text-transform:uppercase;letter-spacing:.05em;
     background:#fcfdfe;border-bottom:1px solid var(--line);white-space:nowrap}
  td{padding:.65rem 1.1rem;border-bottom:1px solid var(--line);font-size:.92rem;
     vertical-align:middle}
  tbody tr:last-child td{border-bottom:0}
  tbody tr:hover{background:#fafcfe}
  .plate{font-weight:700;letter-spacing:-.01em}
  .mono{font-variant-numeric:tabular-nums;color:var(--mute);font-size:.86rem;
        white-space:nowrap}
  .empty{color:var(--mute);font-style:italic}

  /* ---- pills ---- */
  .pill{display:inline-block;padding:.12rem .55rem;border-radius:999px;
        font-size:.76rem;font-weight:700;white-space:nowrap}
  .p-ok{background:var(--ok-bg);color:var(--ok)}
  .p-bad{background:var(--bad-bg);color:var(--bad)}
  .p-warn{background:var(--warn-bg);color:var(--warn)}
  .p-mute{background:#eef2f7;color:var(--mute)}

  /* ---- forms ---- */
  label{display:block;font-size:.76rem;font-weight:700;color:var(--mute);
        text-transform:uppercase;letter-spacing:.04em;margin-bottom:.3rem}
  input,select{width:100%;padding:.52rem .7rem;border:1px solid var(--line);
       border-radius:8px;background:#fff;color:var(--ink);font-size:.93rem;
       font-family:inherit}
  input:focus,select:focus{outline:0;border-color:var(--accent);
       box-shadow:0 0 0 3px rgba(46,125,209,.14)}
  input::placeholder{color:#b4bfcd}
  .row{display:flex;gap:.8rem;flex-wrap:wrap;align-items:flex-end}
  .row > div{flex:1 1 170px;min-width:0}
  .row > div.sm{flex:0 0 auto}
  form.inline{display:inline}

  /* ---- buttons ---- */
  button{cursor:pointer;border:1px solid transparent;border-radius:8px;
         padding:.52rem .9rem;font-size:.86rem;font-weight:700;font-family:inherit;
         white-space:nowrap;transition:filter .12s}
  button:hover{filter:brightness(.94)}
  .b-primary{background:var(--accent);color:#fff}
  .b-ghost{background:#fff;color:var(--ink);border-color:var(--line)}
  .b-ok{background:var(--ok);color:#fff}
  .b-warn{background:#fff;color:var(--warn);border-color:#f0dfb8}
  .b-danger{background:#fff;color:var(--bad);border-color:#f3cfcf}
  .b-sm{padding:.34rem .62rem;font-size:.79rem}

  /* ---- messages ---- */
  .msg{padding:.7rem 1rem;border-radius:var(--radius);font-size:.9rem;
       font-weight:600;margin-bottom:1.1rem;border:1px solid}
  .msg.ok{background:var(--ok-bg);color:var(--ok);border-color:#bfe8cd}
  .msg.err{background:var(--bad-bg);color:var(--bad);border-color:#f3cfcf}
  .foot{color:var(--mute);font-size:.82rem;padding:.75rem 1.1rem;
        border-top:1px solid var(--line);background:#fcfdfe}
</style>
</head><body>
<div class="topbar"><div class="inner">
  <div class="brand"><span class="dot"></span> Cambodian ALPR</div>
  <nav>
    {% if live %}<a href="/live" class="">Live gate</a>{% endif %}
    <a href="/" class="on">Whitelist &amp; audit</a>
    <a href="/logout" class="quiet">Sign out</a>
  </nav>
</div></div>
<div class="shell">
  <div class="head">
    <h1>Whitelist &amp; audit log</h1>
    <div class="sub">A plate opens the gate only if its composed text <b>exactly</b>
      matches an authorized row <b>and</b> the reader confidence is at least 0.70.</div>
  </div>

  {% if msg %}<div class="msg {{ msg_kind }}" id="flash">{{ msg }}</div>
  <script>
    // Drop msg/kind from the address bar so a refresh does not replay the banner.
    const u = new URL(location); u.searchParams.delete('msg'); u.searchParams.delete('kind');
    history.replaceState({}, '', u);
    setTimeout(() => { const f = document.getElementById('flash');
      if (f) { f.style.transition = 'opacity .4s'; f.style.opacity = 0;
               setTimeout(() => f.remove(), 400); } }, 4000);
  </script>{% endif %}

  <div class="stats" style="margin-bottom:1.1rem">
    <div class="stat"><b style="color:var(--ok)">{{ n_auth }}</b><span>Authorized</span></div>
    <div class="stat"><b style="color:var(--bad)">{{ n_blocked }}</b><span>Blocked</span></div>
    <div class="stat"><b>{{ n_total }}</b><span>Total plates</span></div>
    <div class="stat"><b>{{ n_reads_total }}</b><span>Audit reads</span></div>
  </div>

  <div class="card">
    <div class="ch"><h2>Register a plate</h2></div>
    <div class="cb">
      <form method="post" action="/add" class="row">
        <div><label>Plate text</label>
          <input name="plate_text" required placeholder="ភ្នំពេញ 3E-6694"></div>
        <div><label>Owner</label><input name="owner_name" placeholder="name"></div>
        <div><label>Vehicle</label><input name="vehicle_type" placeholder="car / moto"></div>
        <div class="sm"><button class="b-primary" type="submit">Add &amp; authorize</button></div>
      </form>
    </div>
    <div class="foot">Tip: copy the exact composed text from the live gate view.</div>
  </div>

  {% if candidates %}
  <div class="card">
    <div class="ch"><h2>Authorize from recent reads</h2>
      <span class="n">{{ candidates|length }} unregistered</span></div>
    <div class="cb flush"><table>
      <thead><tr><th>Plate as read</th><th>Result</th><th>When</th><th>Conf</th><th></th></tr></thead>
      <tbody>{% for c in candidates %}
        <tr><td class="plate">{{ c.plate_text }}</td>
          <td><span class="pill {{ 'p-ok' if c.cls=='a-allow' else 'p-bad' if c.cls=='a-deny' else 'p-warn' }}">{{ c.action }}</span></td>
          <td class="mono">{{ c.timestamp }}</td><td class="mono">{{ c.crnn_confidence }}</td>
          <td style="text-align:right"><form class="inline" method="post" action="/authorize">
            <input type="hidden" name="plate_text" value="{{ c.plate_text }}">
            <button class="b-ok b-sm">Authorize</button></form></td></tr>
      {% endfor %}</tbody></table></div>
  </div>
  {% endif %}

  {% if inside %}
  <div class="card">
    <div class="ch"><h2>Currently inside</h2><span class="n">{{ inside|length }} vehicle(s)</span></div>
    <div class="cb flush"><table>
      <thead><tr><th>Plate</th><th>Entered</th><th></th></tr></thead>
      <tbody>{% for c in inside %}
        <tr><td class="plate">{{ c.plate_text }}</td><td class="mono">{{ c.entry_time }}</td>
          <td style="text-align:right"><form class="inline" method="post" action="/checkout"
                onsubmit="return confirm('Clear {{ c.plate_text }} from inside?')">
            <input type="hidden" name="plate_text" value="{{ c.plate_text }}">
            <button class="b-warn b-sm">Clear</button></form></td></tr>
      {% endfor %}</tbody></table></div>
  </div>
  {% endif %}

  <div class="card">
    <div class="ch"><h2>Authorized plates</h2><span class="n">{{ n_auth }}</span></div>
    <div class="cb flush"><table>
      <thead><tr><th>Plate</th><th>Owner</th><th>Vehicle</th><th></th></tr></thead>
      <tbody>{% for p in authorized %}
        <tr><td class="plate">{{ p.plate_text }}</td><td>{{ p.owner_name }}</td>
          <td class="mono">{{ p.vehicle_type }}</td>
          <td style="text-align:right">
            <form class="inline" method="post" action="/suspend">
              <input type="hidden" name="plate_text" value="{{ p.plate_text }}">
              <button class="b-warn b-sm">Suspend</button></form>
            <form class="inline" method="post" action="/delete"
                  onsubmit="return confirm('Delete {{ p.plate_text }} permanently?')">
              <input type="hidden" name="plate_text" value="{{ p.plate_text }}">
              <button class="b-danger b-sm">Delete</button></form></td></tr>
      {% else %}<tr><td colspan="4" class="empty">No authorized plates yet.</td></tr>
      {% endfor %}</tbody></table></div>
  </div>

  {% if blocked %}
  <div class="card">
    <div class="ch"><h2>Blocked</h2><span class="n">suspended / expired</span></div>
    <div class="cb flush"><table>
      <thead><tr><th>Plate</th><th>Owner</th><th>Status</th><th></th></tr></thead>
      <tbody>{% for p in blocked %}
        <tr><td class="plate">{{ p.plate_text }}</td><td>{{ p.owner_name }}</td>
          <td><span class="pill p-bad">{{ p.status }}</span></td>
          <td style="text-align:right">
            <form class="inline" method="post" action="/activate">
              <input type="hidden" name="plate_text" value="{{ p.plate_text }}">
              <button class="b-ok b-sm">Reactivate</button></form>
            <form class="inline" method="post" action="/delete"
                  onsubmit="return confirm('Delete {{ p.plate_text }} permanently?')">
              <input type="hidden" name="plate_text" value="{{ p.plate_text }}">
              <button class="b-danger b-sm">Delete</button></form></td></tr>
      {% endfor %}</tbody></table></div>
  </div>
  {% endif %}

  <div class="card">
    <div class="ch"><h2>Audit log</h2>
      <form class="inline" method="post" action="/clear_reads"
            onsubmit="return confirm('Delete ALL {{ n_reads_total }} audit reads? This cannot be undone. (The whitelist is NOT affected.)')">
        <button class="b-danger b-sm">Delete all {{ n_reads_total }} reads</button></form>
    </div>
    <div class="cb">
      <form method="get" action="/" class="row">
        <div><label>Plate contains</label><input name="q" value="{{ q }}"></div>
        <div><label>Action</label>
          <select name="action"><option value="">any</option>
            {% for a in actions %}<option {{ 'selected' if a==action_f }}>{{ a }}</option>{% endfor %}
          </select></div>
        <div class="sm"><button class="b-ghost" type="submit">Search</button></div>
      </form>
    </div>
    <div class="cb flush"><table>
      <thead><tr><th>Time</th><th>Read</th><th>Action</th><th>Conf</th><th>Where</th><th></th></tr></thead>
      <tbody>{% for r in reads %}
        <tr><td class="mono">{{ r.timestamp }}</td><td class="plate">{{ r.detected_plate }}</td>
          <td><span class="pill {{ 'p-ok' if r.cls=='a-allow' else 'p-bad' if r.cls=='a-deny' else 'p-warn' }}">{{ r.action }}</span></td>
          <td class="mono">{{ r.crnn_confidence }}</td><td class="mono">{{ r.location }}</td>
          <td style="text-align:right"><form class="inline" method="post" action="/delete_read"
                onsubmit="return confirm('Delete this read ({{ r.detected_plate }})?')">
            <input type="hidden" name="read_id" value="{{ r.id }}">
            <button class="b-danger b-sm">Delete</button></form></td></tr>
      {% else %}<tr><td colspan="6" class="empty">No matching reads.</td></tr>{% endfor %}
      </tbody></table></div>
    <div class="foot" style="display:flex;justify-content:space-between;
         align-items:center;gap:1rem;flex-wrap:wrap">
      <span>{{ offset + 1 }}–{{ offset + reads|length }} of {{ n_reads_total }}{% if q or action_f %} (filtered){% endif %}</span>
      <span style="display:flex;gap:.4rem">
        {% if offset > 0 %}
          <a class="pager" href="/?q={{ q }}&action={{ action_f }}&offset={{ offset - per if offset - per > 0 else 0 }}">← Newer</a>
        {% endif %}
        {% if has_next %}
          <a class="pager" href="/?q={{ q }}&action={{ action_f }}&offset={{ offset + per }}">Older →</a>
        {% endif %}
      </span>
    </div>
  </div>
</div>
</body></html>""")

ACTIONS = ["ENTRY_ALLOWED", "ENTRY_DENIED", "REVIEW_REQUIRED", "MANUAL_OVERRIDE", "ERROR"]
_ACT_CLS = {"ENTRY_ALLOWED": "a-allow", "ENTRY_DENIED": "a-deny",
            "REVIEW_REQUIRED": "a-review"}


def _cfg() -> dict:
    import yaml
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _db() -> PlateDatabase:
    cfg = _cfg()
    p = Path(cfg.get("db_path", "plates.db"))
    return PlateDatabase(str(p if p.is_absolute() else PROJECT_ROOT / p))


def render(db: PlateDatabase, q: str = "", action_f: str = "",
           msg: str = "", msg_kind: str = "ok", offset: int = 0) -> bytes:
    plates = db.get_all_registered()
    authorized = [p for p in plates if p.get("status") == "active"]
    blocked = [p for p in plates if p.get("status") != "active"]
    # one extra row tells us whether a "next page" exists without a COUNT query
    PER = 25
    page = db.search_reads(plate=q or None, action=action_f or None,
                           limit=offset + PER + 1)[offset:]
    reads, has_next = page[:PER], len(page) > PER
    for r in reads:
        r["cls"] = _ACT_CLS.get(r.get("action"), "act")

    # One-click enrolment: recent reads that were NOT let in and are not already
    # registered. Authorising copies the EXACT text the camera read (no typing).
    registered_texts = {p.get("plate_text") for p in plates}
    candidates, seen = [], set()
    for r in db.get_recent_reads(limit=80):
        dp = (r.get("detected_plate") or "").strip()
        if (not dp or dp in registered_texts or dp in seen
                or dp == "MANUAL_OVERRIDE" or dp.endswith("(unreadable)")):
            continue
        if r.get("action") not in ("ENTRY_DENIED", "REVIEW_REQUIRED"):
            continue
        seen.add(dp)
        candidates.append({"plate_text": dp, "action": r.get("action"),
                           "timestamp": r.get("timestamp"),
                           "crnn_confidence": r.get("crnn_confidence"),
                           "cls": _ACT_CLS.get(r.get("action"), "act")})
        if len(candidates) >= 10:
            break

    n_reads_total = db.get_stats().get("total_reads", 0)
    html = PAGE.render(
        authorized=authorized, blocked=blocked,
        n_auth=len(authorized), n_blocked=len(blocked), n_total=len(plates),
        reads=reads, candidates=candidates, inside=db.active_sessions(),
        n_reads_total=n_reads_total,
        q=q, action_f=action_f, actions=ACTIONS, msg=msg, msg_kind=msg_kind,
        live=LIVE is not None, offset=offset, per=PER, has_next=has_next)
    return html.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    # A browser closing/reloading a tab aborts the socket mid-response; on Windows
    # that surfaces as WinError 10053. It is harmless (the server keeps running) but
    # BaseHTTPRequestHandler would print an alarming traceback. Swallow it everywhere.
    _CLIENT_GONE = (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)

    auth_required = True                  # set False only by --no-auth (localhost)

    def log_message(self, *a):            # quieter console
        pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except self._CLIENT_GONE:
            self.close_connection = True   # client vanished mid-request — ignore quietly

    def _send(self, body: bytes, status: int = 200, extra: list | None = None):
        try:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            # Defensive headers: the panel embeds no third-party content and must not
            # be framed (clickjacking a DELETE button would be trivial otherwise).
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or []):
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
        except self._CLIENT_GONE:
            self.close_connection = True   # browser closed the tab — nothing to send

    # -- live view --------------------------------------------------------- #
    def _see_other(self, where: str):
        """303 after a POST so a refresh never re-submits the action."""
        self.send_response(303)
        self.send_header("Location", where)
        self.end_headers()

    def _send_json(self, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_live(self):
        if LIVE is None:
            self._send(b"<h1>Live view is off</h1><p>Restart with "
                       b"<code>--live</code> to load the pipeline.</p>"
                       b"<p><a href='/'>Back to the whitelist</a></p>", status=404)
            return
        src = _cfg().get("camera_source", "?")
        body = LIVE_PAGE.render(source=f"source: {src}", error=LIVE.error).encode("utf-8")
        self._send(body)

    def _send_stream(self):
        """MJPEG: multipart/x-mixed-replace. Every browser renders this in an
        <img> with no JavaScript, which is why it beats a WebSocket here."""
        if LIVE is None:
            self._send(b"live view is off", status=404)
            return
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                jpg = LIVE.jpeg()
                if jpg is None:
                    time.sleep(0.1)
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n")
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(0.06)                  # ~16 fps to the browser
        except (BrokenPipeError, ConnectionResetError):
            pass                                  # the tab was closed; normal

    def _redirect(self, msg: str, kind: str = "ok"):
        from urllib.parse import quote
        self.send_response(303)
        self.send_header("Location", f"/?msg={quote(msg)}&kind={kind}")
        self.end_headers()

    # ---------------- auth helpers ---------------- #
    def _client_ip(self) -> str:
        return self.client_address[0] if self.client_address else "?"

    def _cookie_token(self) -> str | None:
        raw = self.headers.get("Cookie")
        if not raw:
            return None
        try:
            return SimpleCookie(raw).get(SESSION_COOKIE).value    # type: ignore[union-attr]
        except Exception:
            return None

    def _authed(self) -> bool:
        return (not self.auth_required) or _session_valid(self._cookie_token())

    def _send_login(self, error: str = "", status: int = 200):
        self._send(LOGIN_PAGE.render(error=error).encode("utf-8"), status)

    def _require_auth(self) -> bool:
        """True if the request may proceed; otherwise the response is already sent."""
        if self._authed():
            return True
        if self.command == "POST":
            # Never redirect a mutating request to a page — fail it outright.
            self._send(b"<h1>401 - sign in required</h1>", 401)
        else:
            self._send_login()
        return False

    def _do_login(self, form: dict):
        ip = self._client_ip()
        locked = _login_locked(ip)
        if locked:
            self._send_login(f"Too many attempts. Try again in {locked}s.", 429)
            return
        record = load_auth()
        if record and verify_password(form.get("password", [""])[0], record):
            _LOGIN_FAILURES.pop(ip, None)
            token = _new_session()
            cookie = (f"{SESSION_COOKIE}={token}; HttpOnly; SameSite=Strict; "
                      f"Path=/; Max-Age={SESSION_TIMEOUT_SEC}")
            self.send_response(303)
            self.send_header("Location", "/")
            self.send_header("Set-Cookie", cookie)
            self.end_headers()
            return
        _record_failure(ip)
        print(f"[admin] failed login from {ip}")
        self._send_login("Wrong password.", 401)

    def _do_logout(self):
        _SESSIONS.pop(self._cookie_token() or "", None)
        self.send_response(303)
        self.send_header("Location", "/")
        self.send_header("Set-Cookie",
                         f"{SESSION_COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/login":
            self._send_login() if not self._authed() else self._redirect("already signed in")
            return
        if path == "/logout":
            self._do_logout()
            return
        if not self._require_auth():
            return
        if path == "/live":
            self._send_live()
            return
        if path == "/stream":
            self._send_stream()
            return
        if path == "/live_status.json":
            self._send_json(LIVE.status() if LIVE else {"ready": False,
                                                        "error": "started without --live"})
            return
        qs = parse_qs(urlparse(self.path).query)
        db = _db()
        try:
            try:
                off = max(0, int(qs.get("offset", ["0"])[0]))
            except ValueError:
                off = 0
            body = render(db, q=qs.get("q", [""])[0], action_f=qs.get("action", [""])[0],
                          msg=qs.get("msg", [""])[0], msg_kind=qs.get("kind", ["ok"])[0],
                          offset=off)
        finally:
            db.close()
        self._send(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        form = parse_qs(self.rfile.read(n).decode("utf-8"))
        path = urlparse(self.path).path
        if path == "/login":
            self._do_login(form)
            return
        if path == "/live_authorize":
            if not self._require_auth():
                return
            plate = (form.get("plate_text", [""])[0] or "").strip()
            if not plate or plate == "(unreadable)":
                self._see_other("/live")
                return
            db = _db()
            try:
                db.add_plate(plate, form.get("owner_name", ["live"])[0] or "live",
                             "", "authorized from the live view")
            finally:
                db.close()
            self._see_other("/live")
            return
        if path in ("/live_open", "/live_estop"):
            if not self._require_auth():
                return
            if LIVE is None or LIVE.system is None:
                self._redirect("live view is not running", "err")
                return
            if path == "/live_open":
                LIVE.system.manual_override()     # MAN-001, logged as MANUAL_OVERRIDE
            else:
                LIVE.system.emergency_stop()      # MAN-002, toggles fail-safe
            self._see_other("/live")
            return
        if not self._require_auth():
            return

        # ---- audit-log record deletion (these carry no plate_text) ---- #
        if path == "/delete_read":
            db = _db()
            try:
                rid = form.get("read_id", [""])[0]
                if not rid.isdigit():
                    self._redirect("bad read id", "err"); return
                photo = db.delete_read(int(rid))          # returns its photo path or None
                if photo:
                    try:
                        (PROJECT_ROOT / photo).unlink(missing_ok=True)
                    except Exception:
                        pass
                self._redirect(f"deleted read #{rid}" if photo is not None
                               else f"read #{rid} not found",
                               "ok" if photo is not None else "err")
            finally:
                db.close()
            return
        if path == "/clear_reads":
            db = _db()
            try:
                photos = db.clear_reads()                 # deletes all rows, returns photo paths
                removed = 0
                for p in photos:
                    try:
                        (PROJECT_ROOT / p).unlink(missing_ok=True)
                        removed += 1
                    except Exception:
                        pass
                self._redirect(f"cleared all reads ({len(photos)} rows, "
                               f"{removed} evidence photos removed)")
            finally:
                db.close()
            return

        plate = form.get("plate_text", [""])[0].strip()
        db = _db()
        try:
            if not plate:
                self._redirect("no plate text given", "err"); return
            if path == "/add":
                ok = db.add_plate(plate, form.get("owner_name", [""])[0].strip() or "Unknown",
                                  form.get("vehicle_type", [""])[0].strip() or "car",
                                  "added via web admin")
                self._redirect(f"registered {plate}" if ok else f"{plate} already exists",
                               "ok" if ok else "err")
            elif path == "/authorize":
                # one-click enrol of the EXACT text the camera read
                ok = db.add_plate(plate, "Unknown", "car", "authorized from recent read")
                self._redirect(f"authorized {plate}" if ok else f"{plate} already authorized",
                               "ok" if ok else "err")
            elif path == "/suspend":
                ok = db.set_status(plate, "suspended")
                self._redirect(f"suspended {plate}" if ok else "not found",
                               "ok" if ok else "err")
            elif path == "/activate":
                ok = db.set_status(plate, "active")
                self._redirect(f"reactivated {plate}" if ok else "not found",
                               "ok" if ok else "err")
            elif path == "/delete":
                ok = db.remove_plate(plate)
                self._redirect(f"deleted {plate}" if ok else "not found",
                               "ok" if ok else "err")
            elif path == "/checkout":
                # manually clear a car from "inside" (missed exit scan)
                ok = db.close_parking_session(plate)
                self._redirect(f"cleared {plate} from inside" if ok else "not inside",
                               "ok" if ok else "err")
            else:
                self._redirect("unknown action", "err")
        finally:
            db.close()


def _lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--set-password", action="store_true",
                    help="set/replace the admin password, then exit")
    ap.add_argument("--no-auth", action="store_true",
                    help="DANGEROUS: serve with no login. Loopback binds only.")
    ap.add_argument("--open", action="store_true",
                    help="open the panel in a browser once the server is listening")
    ap.add_argument("--live", action="store_true",
                    help="also run the ALPR pipeline and serve it at /live, so the "
                         "gate demo and the whitelist are in one browser window. "
                         "Loads four models (~10 s, needs the GPU).")
    args = ap.parse_args()

    if args.set_password:
        sys.exit(set_password_cli())

    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    if args.no_auth:
        # An unauthenticated panel that can delete whitelist entries must never be
        # reachable from the network. Refuse rather than warn-and-continue.
        if not loopback:
            print("[X] --no-auth only works with --host 127.0.0.1 "
                  f"(you asked for {args.host!r}).")
            print("    This panel can add and DELETE whitelist plates; serving it")
            print("    unauthenticated on the LAN would hand the gate to anyone.")
            sys.exit(2)
        Handler.auth_required = False
        print("!" * 56)
        print(" WARNING: authentication DISABLED (--no-auth), localhost only.")
        print(" Anyone with access to this machine controls the whitelist.")
        print("!" * 56)
    elif load_auth() is None:
        print("=" * 56)
        print(" ALPR Whitelist Admin — NO PASSWORD SET")
        print("=" * 56)
        print(" This panel adds, suspends and DELETES whitelist plates — it decides")
        print(" who the gate opens for — so it will not start unauthenticated.")
        print("")
        print("   python scripts/system/admin_web.py --set-password")
        print("")
        print(" Or, for a localhost-only demo:")
        print("   python scripts/system/admin_web.py --no-auth --host 127.0.0.1")
        print("=" * 56)
        sys.exit(2)

    # Bind the socket BEFORE anything opens a browser. This is the fix for the
    # recurring "localhost refused to connect": the old flow opened the browser
    # first and the server bound a moment later, so the first page load always
    # failed. Now the browser is opened by a thread that fires only after the
    # server is already listening.
    # ThreadingHTTPServer is REQUIRED once /stream exists: an MJPEG response never
    # ends, so a single-threaded server would serve the video and nothing else.
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True

    global LIVE
    if args.live:
        LIVE = LiveWorker()
        LIVE.start()

    ip = _lan_ip()
    print("=" * 56)
    print(" ALPR Whitelist Admin — web panel")
    print("=" * 56)
    print(f"  On this PC   : http://localhost:{args.port}")
    if not loopback:
        print(f"  On your phone: http://{ip}:{args.port}   (same Wi-Fi)")
    print(f"  Auth         : {'DISABLED (--no-auth)' if args.no_auth else 'password required'}")
    print(f"  Live gate    : {'ON  -> /live' if args.live else 'off (start with --live)'}")
    print("  Ctrl+C to stop.")
    print("=" * 56)

    if args.open:
        import threading
        import webbrowser

        def _open_when_ready():
            time.sleep(1.0)          # the socket is already bound; give it a beat
            try:
                webbrowser.open(f"http://localhost:{args.port}")
            except Exception:
                pass
        threading.Thread(target=_open_when_ready, daemon=True).start()

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        if LIVE is not None:
            LIVE.stop()                 # release the camera and close the DB
        srv.server_close()


if __name__ == "__main__":
    main()
