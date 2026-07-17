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

Run:
    python scripts/system/admin_web.py            # http://<this-pc-ip>:5000
    python scripts/system/admin_web.py --port 8000
"""
from __future__ import annotations

import argparse
import socket
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
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
    <h2 style="margin-top:0">🔎 Audit log</h2>
    <form method="get" action="/" class="row">
      <div><label>Plate contains</label><input name="q" value="{{ q }}"></div>
      <div><label>Action</label>
        <select name="action">
          <option value="">any</option>
          {% for a in actions %}<option {{ 'selected' if a==action_f }}>{{ a }}</option>{% endfor %}
        </select></div>
      <div><button class="b-act">Search</button></div>
    </form>
    <table><tr><th>Time</th><th>Read</th><th>Action</th><th>Conf</th><th>Where</th></tr>
    {% for r in reads %}
      <tr><td class="sub">{{ r.timestamp }}</td><td class="plate">{{ r.detected_plate }}</td>
        <td class="{{ r.cls }}">{{ r.action }}</td><td>{{ r.crnn_confidence }}</td>
        <td class="sub">{{ r.location }}</td></tr>
    {% else %}<tr><td colspan="5" class="sub">no matching reads</td></tr>{% endfor %}
    </table>
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

    html = PAGE.render(
        authorized=authorized, blocked=blocked,
        n_auth=len(authorized), n_blocked=len(blocked), n_total=len(plates),
        reads=reads, candidates=candidates, inside=db.active_sessions(),
        q=q, action_f=action_f, actions=ACTIONS, msg=msg, msg_kind=msg_kind)
    return html.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):            # quieter console
        pass

    def _send(self, body: bytes, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, msg: str, kind: str = "ok"):
        from urllib.parse import quote
        self.send_response(303)
        self.send_header("Location", f"/?msg={quote(msg)}&kind={kind}")
        self.end_headers()

    def do_GET(self):
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
        plate = form.get("plate_text", [""])[0].strip()
        path = urlparse(self.path).path
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
    args = ap.parse_args()

    srv = HTTPServer((args.host, args.port), Handler)
    ip = _lan_ip()
    print("=" * 56)
    print(" ALPR Whitelist Admin — web panel")
    print("=" * 56)
    print(f"  On this PC   : http://localhost:{args.port}")
    print(f"  On your phone: http://{ip}:{args.port}   (same Wi-Fi)")
    print("  Ctrl+C to stop.")
    print("=" * 56)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
        srv.server_close()


if __name__ == "__main__":
    main()
