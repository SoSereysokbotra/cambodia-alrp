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
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
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
<title>ALPR Admin — sign in</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, "Segoe UI", sans-serif; background: #0f1115;
         color: #e7e9ee; display: grid; place-items: center; min-height: 100vh;
         margin: 0; padding: 1rem; }
  .card { width: 100%; max-width: 340px; background: #161a22; padding: 1.5rem;
          border: 1px solid #232838; border-radius: 12px; }
  h1 { font-size: 1.15rem; margin: 0 0 .3rem; }
  .sub { color: #9aa3b2; font-size: .85rem; margin-bottom: 1.1rem; }
  input { width: 100%; padding: .6rem .7rem; border-radius: 8px;
          border: 1px solid #2a3040; background: #0f1115; color: #e7e9ee;
          font-size: 1rem; }
  button { width: 100%; margin-top: .8rem; padding: .6rem; border: 0;
           border-radius: 8px; background: #2b6cb0; color: #fff;
           font-weight: 700; font-size: 1rem; cursor: pointer; }
  .err { background: #3a1b1b; color: #ff6b6b; padding: .5rem .7rem;
         border-radius: 8px; font-size: .85rem; margin-bottom: .8rem; }
</style></head><body>
  <div class="card">
    <h1>ALPR Admin</h1>
    <div class="sub">Whitelist &amp; audit control</div>
    {% if error %}<div class="err">{{ error }}</div>{% endif %}
    <form method="post" action="/login">
      <input type="password" name="password" placeholder="Admin password"
             autofocus autocomplete="current-password" required>
      <button type="submit">Sign in</button>
    </form>
  </div>
</body></html>""")

PAGE = Template("""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ALPR Admin — Whitelist</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, "Segoe UI", "Noto Sans Khmer", sans-serif;
         margin: 0; padding: 1rem; max-width: 900px; margin-inline: auto;
         background: #0f1115; color: #e7e9ee; }
  h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 1.6rem; }
  .sub { color: #9aa3b2; font-size: .9rem; }
  table { width: 100%; border-collapse: collapse; margin-top: .5rem; }
  th, td { text-align: left; padding: .5rem .6rem; border-bottom: 1px solid #232838;
           font-size: .95rem; }
  th { color: #9aa3b2; font-weight: 600; }
  .plate { font-weight: 700; }
  .pill { padding: .1rem .5rem; border-radius: 999px; font-size: .78rem; font-weight: 700; }
  .ok { background: #12351f; color: #46d17f; }
  .bad { background: #3a1b1b; color: #ff6b6b; }
  form.inline { display: inline; }
  button { cursor: pointer; border: 0; border-radius: 6px; padding: .35rem .6rem;
           font-size: .82rem; font-weight: 600; color: #fff; }
  .b-add { background: #2f7d46; } .b-susp { background: #b5842a; }
  .b-act { background: #2f6d7d; } .b-del { background: #a33; }
  .card { background: #161a22; border: 1px solid #232838; border-radius: 10px;
          padding: 1rem; margin-top: 1rem; }
  input, select { padding: .5rem; border-radius: 6px; border: 1px solid #333a4d;
                  background: #0f1115; color: #e7e9ee; font-size: .95rem; }
  .row { display: flex; gap: .5rem; flex-wrap: wrap; align-items: end; }
  .row > div { display: flex; flex-direction: column; gap: .2rem; }
  label { font-size: .78rem; color: #9aa3b2; }
  .msg { padding: .6rem .8rem; border-radius: 8px; margin-top: .8rem; }
  .msg.ok { background: #12351f; color: #7be0a3; }
  .msg.err { background: #3a1b1b; color: #ff9b9b; }
  .act { color: #9aa3b2; } .a-allow { color: #46d17f; } .a-deny { color: #ff6b6b; }
  .a-review { color: #e0b23a; }
</style></head><body>
  <h1>🚗 ALPR Whitelist Admin</h1>
  <div class="sub">{{ n_auth }} authorized · {{ n_blocked }} blocked · {{ n_total }} total.
    A live plate opens the gate only if its composed text <b>exactly</b> matches an
    authorized row and confidence ≥ 0.70.</div>

  {% if msg %}<div class="msg {{ msg_kind }}">{{ msg }}</div>{% endif %}

  <div class="card">
    <h2 style="margin-top:0">➕ Register a plate</h2>
    <form method="post" action="/add" class="row">
      <div><label>Plate text (e.g. ភ្នំពេញ 3E-6694)</label>
        <input name="plate_text" required size="24" placeholder="province number"></div>
      <div><label>Owner</label><input name="owner_name" placeholder="name"></div>
      <div><label>Vehicle</label><input name="vehicle_type" placeholder="car / moto"></div>
      <div><button class="b-add" type="submit">Add / Authorize</button></div>
    </form>
  </div>

  <div class="card">
    <h2 style="margin-top:0">🆕 Authorize from recent reads</h2>
    <div class="sub">Cars the camera read but that aren't authorized yet. Click
      <b>Authorize</b> to add the <b>exact</b> text the camera read — no typing, no
      typos. Check it matches the plate before authorizing.</div>
    <table><tr><th>Plate (as read)</th><th>Result</th><th>When</th><th>Conf</th><th></th></tr>
    {% for c in candidates %}
      <tr><td class="plate">{{ c.plate_text }}</td>
        <td class="{{ c.cls }}">{{ c.action }}</td>
        <td class="sub">{{ c.timestamp }}</td>
        <td>{{ c.crnn_confidence }}</td>
        <td><form class="inline" method="post" action="/authorize">
          <input type="hidden" name="plate_text" value="{{ c.plate_text }}">
          <button class="b-add">Authorize</button></form></td></tr>
    {% else %}<tr><td colspan="5" class="sub">no un-authorized reads yet</td></tr>{% endfor %}
    </table>
  </div>

  <div class="card">
    <h2 style="margin-top:0">🅿️ Currently inside ({{ inside|length }})</h2>
    <div class="sub">Cars parked right now (parking mode). They clear automatically
      on exit — use <b>Clear</b> only if a car left without being scanned.</div>
    <table><tr><th>Plate</th><th>Entered</th><th></th></tr>
    {% for c in inside %}
      <tr><td class="plate">{{ c.plate_text }}</td>
        <td class="sub">{{ c.entry_time }}</td>
        <td><form class="inline" method="post" action="/checkout"
              onsubmit="return confirm('Clear {{ c.plate_text }} from inside?')">
          <input type="hidden" name="plate_text" value="{{ c.plate_text }}">
          <button class="b-susp">Clear</button></form></td></tr>
    {% else %}<tr><td colspan="3" class="sub">lot is empty</td></tr>{% endfor %}
    </table>
  </div>

  <h2>✅ Authorized ({{ n_auth }})</h2>
  <table><tr><th>Plate</th><th>Owner</th><th>Vehicle</th><th></th></tr>
  {% for p in authorized %}
    <tr><td class="plate">{{ p.plate_text }}</td><td>{{ p.owner_name }}</td>
      <td>{{ p.vehicle_type }}</td><td>
      <form class="inline" method="post" action="/suspend">
        <input type="hidden" name="plate_text" value="{{ p.plate_text }}">
        <button class="b-susp">Suspend</button></form>
      <form class="inline" method="post" action="/delete"
            onsubmit="return confirm('Delete {{ p.plate_text }}?')">
        <input type="hidden" name="plate_text" value="{{ p.plate_text }}">
        <button class="b-del">Delete</button></form></td></tr>
  {% else %}<tr><td colspan="4" class="sub">none</td></tr>{% endfor %}
  </table>

  {% if blocked %}
  <h2>⛔ Blocked (suspended / expired)</h2>
  <table><tr><th>Plate</th><th>Owner</th><th>Status</th><th></th></tr>
  {% for p in blocked %}
    <tr><td class="plate">{{ p.plate_text }}</td><td>{{ p.owner_name }}</td>
      <td><span class="pill bad">{{ p.status }}</span></td><td>
      <form class="inline" method="post" action="/activate">
        <input type="hidden" name="plate_text" value="{{ p.plate_text }}">
        <button class="b-act">Reactivate</button></form>
      <form class="inline" method="post" action="/delete"
            onsubmit="return confirm('Delete {{ p.plate_text }}?')">
        <input type="hidden" name="plate_text" value="{{ p.plate_text }}">
        <button class="b-del">Delete</button></form></td></tr>
  {% endfor %}</table>
  {% endif %}

  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center">
      <h2 style="margin:0">🔎 Audit log</h2>
      <form class="inline" method="post" action="/clear_reads"
            onsubmit="return confirm('Delete ALL {{ n_reads_total }} audit reads? This cannot be undone. (The whitelist is NOT affected.)')">
        <button class="b-del">🗑 Delete ALL reads ({{ n_reads_total }})</button></form>
    </div>
    <form method="get" action="/" class="row">
      <div><label>Plate contains</label><input name="q" value="{{ q }}"></div>
      <div><label>Action</label>
        <select name="action">
          <option value="">any</option>
          {% for a in actions %}<option {{ 'selected' if a==action_f }}>{{ a }}</option>{% endfor %}
        </select></div>
      <div><button class="b-act">Search</button></div>
    </form>
    <table><tr><th>Time</th><th>Read</th><th>Action</th><th>Conf</th><th>Where</th><th></th></tr>
    {% for r in reads %}
      <tr><td class="sub">{{ r.timestamp }}</td><td class="plate">{{ r.detected_plate }}</td>
        <td class="{{ r.cls }}">{{ r.action }}</td><td>{{ r.crnn_confidence }}</td>
        <td class="sub">{{ r.location }}</td>
        <td><form class="inline" method="post" action="/delete_read"
              onsubmit="return confirm('Delete this read ({{ r.detected_plate }})?')">
          <input type="hidden" name="read_id" value="{{ r.id }}">
          <button class="b-del">Delete</button></form></td></tr>
    {% else %}<tr><td colspan="6" class="sub">no matching reads</td></tr>{% endfor %}
    </table>
    <p class="sub">Showing the latest {{ reads|length }}. Use search to find older ones,
       or "Delete ALL" to clear the whole history.</p>
  </div>
</body></html>""")

ACTIONS = ["ENTRY_ALLOWED", "ENTRY_DENIED", "REVIEW_REQUIRED", "MANUAL_OVERRIDE", "ERROR"]
_ACT_CLS = {"ENTRY_ALLOWED": "a-allow", "ENTRY_DENIED": "a-deny",
            "REVIEW_REQUIRED": "a-review"}


def _db() -> PlateDatabase:
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    p = Path(cfg.get("db_path", "plates.db"))
    return PlateDatabase(str(p if p.is_absolute() else PROJECT_ROOT / p))


def render(db: PlateDatabase, q: str = "", action_f: str = "",
           msg: str = "", msg_kind: str = "ok") -> bytes:
    plates = db.get_all_registered()
    authorized = [p for p in plates if p.get("status") == "active"]
    blocked = [p for p in plates if p.get("status") != "active"]
    reads = db.search_reads(plate=q or None, action=action_f or None, limit=25)
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
        q=q, action_f=action_f, actions=ACTIONS, msg=msg, msg_kind=msg_kind)
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
        qs = parse_qs(urlparse(self.path).query)
        db = _db()
        try:
            body = render(db, q=qs.get("q", [""])[0], action_f=qs.get("action", [""])[0],
                          msg=qs.get("msg", [""])[0], msg_kind=qs.get("kind", ["ok"])[0])
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
    srv = HTTPServer((args.host, args.port), Handler)
    ip = _lan_ip()
    print("=" * 56)
    print(" ALPR Whitelist Admin — web panel")
    print("=" * 56)
    print(f"  On this PC   : http://localhost:{args.port}")
    if not loopback:
        print(f"  On your phone: http://{ip}:{args.port}   (same Wi-Fi)")
    print(f"  Auth         : {'DISABLED (--no-auth)' if args.no_auth else 'password required'}")
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
        srv.server_close()


if __name__ == "__main__":
    main()
