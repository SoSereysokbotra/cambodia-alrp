# SRS Deviations — Cambodian ALPR System

This log records intentional, approved deviations from `docs/srs.md` v2.0.
Each deviation is a deliberate engineering decision, documented here so it is
traceable and not mistaken for an omission.

---

## DEV-001 — PORT-002: Operating System

**SRS Requirement (PORT-002):** Ubuntu 20.04/22.04 LTS.

**Actual Implementation:** **Windows 11**.

**Justification:**
- The existing development environment is fully functional on Windows 11.
- All dependencies are available and verified on Windows: Python 3.10 venv,
  PyTorch 2.5.1 + CUDA 12.1, YOLOv10 (Ultralytics), CRNN (PyTorch), OpenCV,
  SQLite, MQTT (paho / Mosquitto).
- There is no functional difference in system behaviour between the two OSes for
  this workload.

**Impact:** None functional. Only setup commands/paths differ (documented in
`docs/DEPLOYMENT.md`).

**Approval Date:** 2026-07-11
**Status:** Intentional, documented, non-functional deviation.

---

## DEV-002 — REC-002: Recognition Character Scope (interim)

**SRS Requirement (REC-002):** recognise Khmer consonants + vowels + Latin +
digits (50+ characters).

**Actual Implementation (current):** CRNN reads the **plate number** only
(Latin A–Z + digits + separators, 38 chars). The **Khmer province** is planned
to come from a separate province classifier (Plate_v4's 29 classes) and be
composed as `provinceKhmer + " " + number` — see Implementation Plan Phase 3
(Option A, COMMITTED).

**Justification:**
- Rendering correctly-shaped Khmer for synthetic training requires complex-script
  shaping (libraqm); the number is the identifying field and trains reliably.
- Option A reuses existing detection infrastructure and raises the probability of
  meeting CER ≤ 10% by the deadline.

**Impact:** Until Phase 3 lands, the recognised text is the number only. After
Phase 3, the full `provinceKhmer + number` string is produced (REC-002/REC-006).

**Approval Date:** 2026-07-11
**Status:** Interim deviation; closes when Phase 3 is complete.

---

## DEV-003 — GTC-002: MQTT Payload Format

**SRS Requirement (GTC-002):** topic `gate/control`, payload plain text
`"GATE_OPEN"` / `"GATE_CLOSE"`.

**Actual Implementation:** topic `alpr/{gate_id}/control`, **JSON** payload
(`command`, `plate`, `duration`, `timestamp`), which the ESP32 sketch parses.

**Justification:** the JSON form carries the plate + duration, enabling richer
gate behaviour and status logging on the ESP32.

**Planned resolution (Plan Phase 11):** add a config flag `mqtt.srs_compat` that
*also* publishes the SRS plain-text form on `gate/control`, satisfying both.

**Approval Date:** 2026-07-11
**Status:** Deviation with a planned compatibility bridge.

---

*Deviation log v1.0 — reviewed against `docs/srs.md` v2.0.*
