#!/usr/bin/env python3
"""
scripts/test_confidence_gate.py
===============================
Unit test for the SRS REC-005 confidence gate — proves the three decision
branches and that a low-confidence read NEVER opens the gate.

Run:
    python scripts/test_confidence_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()), Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
from core.alpr_system import ALPRSystem  # noqa: E402

CONFIG = PROJECT_ROOT / "configs" / "system_config.yaml"
THRESH = 0.70


def decide(system, text, conf, is_reg) -> str:
    """Replicate process_frame's decision by patching reader + DB."""
    system.reader.read = lambda crop: (text, conf)
    system.database.is_registered = lambda t: is_reg
    frame = np.zeros((96, 320, 3), dtype=np.uint8)
    system.assume_crop = True  # skip YOLO, feed whole frame
    res = system.process_frame(frame)
    return res["plates"][0]["action"]


def main() -> None:
    system = ALPRSystem(str(CONFIG))
    gate_opened = {"n": 0}
    orig_open = system.gate.open_gate
    system.gate.open_gate = lambda *a, **k: (gate_opened.__setitem__("n", gate_opened["n"] + 1), orig_open(*a, **k))

    passed, failed = 0, 0

    def check(name, got, expect):
        nonlocal passed, failed
        ok = got == expect
        passed += ok
        failed += (not ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, expected {expect}")

    print("=" * 56)
    print(" SRS REC-005 CONFIDENCE GATE — UNIT TEST")
    print("=" * 56)

    # Branch 1: low confidence -> REVIEW_REQUIRED (even if registered)
    check("low conf + registered -> REVIEW",
          decide(system, "1AB-2345", 0.40, True), "REVIEW_REQUIRED")
    # Branch 2: high conf + registered -> ALLOWED
    check("high conf + registered -> ALLOWED",
          decide(system, "1AB-2345", 0.95, True), "ENTRY_ALLOWED")
    # Branch 3: high conf + not registered -> DENIED
    check("high conf + unknown -> DENIED",
          decide(system, "9ZZ-9999", 0.95, False), "ENTRY_DENIED")
    # Boundary: exactly below threshold -> REVIEW
    check("just below threshold -> REVIEW",
          decide(system, "1AB-2345", THRESH - 0.01, True), "REVIEW_REQUIRED")

    # Safety invariant: gate must NOT have opened on the two low-confidence cases
    print("-" * 56)
    print(f"  gate.open_gate() calls: {gate_opened['n']} "
          f"(expected 1 — only the ALLOWED case)")
    if gate_opened["n"] != 1:
        failed += 1
        print("  [FAIL] gate opened on a case it should not have!")
    else:
        passed += 1
        print("  [PASS] gate opened ONLY for high-confidence registered plate")

    print("=" * 56)
    print(f" RESULT: {passed} passed, {failed} failed")
    print("=" * 56)
    system.close()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] test failed to run: {exc}")
        sys.exit(1)
