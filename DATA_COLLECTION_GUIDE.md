# 📸 Data Collection Guide — Cambodian ALPR (Week 1)

**Goal:** Collect **100 high-quality Cambodian license plate photos** in Phnom Penh
to bootstrap the YOLOv10 detection dataset (and later, CRNN text-reading crops).

This is the single most important week of the project. **Garbage in → garbage out.**
Spend the extra minute per photo to get it right — a clean, diverse 100-image set
beats a sloppy 300-image set.

---

## 1. Equipment Needed

| Item | Required? | Notes |
|------|-----------|-------|
| **Smartphone camera** | ✅ Required | 12 MP or higher. Clean the lens first. |
| **Tripod / phone clamp** | ⭐ Recommended | Keeps plates sharp, avoids motion blur. |
| **Power bank** | ⭐ Recommended | Photo sessions drain the battery fast. |
| **Notebook or the CSV log** | ✅ Required | Record metadata *as you shoot*, not later. |
| **Reflective vest / permission** | Optional | Useful when shooting inside private lots. |

**Camera settings:**
- Resolution: **highest available** (aim for **≥ 1920 × 1080**, ideally 12 MP full).
- Format: **JPG** (not HEIC — convert HEIC to JPG before uploading to Roboflow).
- Turn **grid lines ON** to help center the plate.
- **Disable** heavy "beauty"/AI filters and digital zoom (walk closer instead).
- Tap to focus **on the plate** before every shot.

---

## 2. Where to Collect Photos in Phnom Penh

Prioritise **variety of plate styles, provinces, and vehicle types**. Cambodian plates
include single-line and **multi-line Khmer text**, plus newer plates with **QR codes** —
capture all of these.

**Good, low-friction locations:**
- 🅿️ **Parking lots** — AEON Mall (Sen Sok / Mean Chey), Chip Mong, Exchange Square,
  hospital & university car parks. Rows of parked cars = many plates, no motion blur.
- 🛒 **Markets** — Central Market (Phsar Thmey), Russian Market (Toul Tom Poung),
  Orussey Market. Dense **motorbike** plates.
- 🏢 **Office / condo gates** — the actual deployment scenario. Ask building security
  for permission; explain it is a student project.
- 🏍️ **Street-side parking** along Norodom, Monivong, Sihanouk Blvd.
- ⛽ **Petrol stations & tuk-tuk stands** — mixed vehicle types.

**Aim for a spread:**
- Cars, SUVs, motorbikes/scooters, tuk-tuks, trucks.
- Different provinces (Phnom Penh, Kandal, Siem Reap, etc. — the Khmer province name
  differs per plate).
- Both **older plates** and **new QR-code plates**.

> ⚠️ **Privacy & etiquette:** License plates are not private, but photograph the
> **plate**, not people or building interiors. In private lots, **ask permission first**.
> If someone objects, stop and move on. Do not block traffic or gates.

---

## 3. Photo Requirements

Each photo must satisfy **all** of these:

| Requirement | Target |
|-------------|--------|
| **Resolution** | ≥ 1920 × 1080 (higher is better) |
| **Plate visibility** | Entire plate visible, all 4 corners in frame |
| **Sharpness** | Text readable to *your own eye* at 100% zoom |
| **Plate size in frame** | Plate occupies roughly **10–40%** of the image width |
| **Distance** | ~1.5–4 m from the plate (real gate camera distance) |
| **File format** | JPG |

### Angles to capture (this matters a lot)
Collect a **balanced mix** — the model must generalise, not memorise one viewpoint.

| Angle code | Description | Rough target count |
|------------|-------------|--------------------|
| `FRONT`    | Straight-on, camera perpendicular to plate | ~30 |
| `LEFT`     | ~15–30° from the left | ~20 |
| `RIGHT`    | ~15–30° from the right | ~20 |
| `HIGH`     | Camera above plate, looking down (gate-cam view) | ~15 |
| `LOW`      | Camera below plate, looking up | ~15 |

### Lighting conditions
Shoot across the day, not all at noon:

| Lighting code | When |
|---------------|------|
| `DAY`   | Bright daylight |
| `SHADE` | Overcast / under cover / shadow |
| `LOW`   | Dusk / dim indoor parking |
| `BACKLIT` | Sun or bright light behind the plate |

> 🎯 **Diversity target for the 100 images:** at least 3 angles and 2 lighting
> conditions well represented. Don't shoot 100 identical front-lit photos.

---

## 4. Naming Convention

Rename every photo to this exact pattern **before** uploading to Roboflow:

```
plate_LOCATION_DATE_TIME_ANGLE.jpg
```

| Field | Format | Example |
|-------|--------|---------|
| `LOCATION` | short lowercase tag, no spaces | `aeonsensok`, `russianmkt`, `norodom` |
| `DATE` | `YYYYMMDD` | `20260706` |
| `TIME` | `HHMM` (24-hour) | `1430` |
| `ANGLE` | one of `FRONT/LEFT/RIGHT/HIGH/LOW` | `FRONT` |

**Examples:**
```
plate_aeonsensok_20260706_1430_FRONT.jpg
plate_russianmkt_20260706_1512_LEFT.jpg
plate_norodom_20260706_1740_HIGH.jpg
```

Rules:
- Lowercase for location, **UPPERCASE** for the angle code (easy to scan).
- No spaces, no Khmer characters, no `#` in filenames.
- Keep names **unique** — if two photos would collide, add a `_2` before `.jpg`.

---

## 5. Metadata Log (CSV)

Maintain a running log at **`data/metadata/metadata_log.csv`**. Fill it in **while
shooting** — memory fades fast. Columns:

```csv
photo#,filename,angle,lighting,notes
1,plate_aeonsensok_20260706_1430_FRONT.jpg,FRONT,DAY,car - single-line plate
2,plate_aeonsensok_20260706_1431_LEFT.jpg,LEFT,DAY,car - slight glare top-left
3,plate_russianmkt_20260706_1512_FRONT.jpg,FRONT,SHADE,motorbike - multi-line Khmer
4,plate_norodom_20260706_1740_HIGH.jpg,HIGH,LOW,tuk-tuk - QR-code plate
5,plate_exchange_20260707_0915_BACKLIT.jpg,FRONT,BACKLIT,SUV - sun behind vehicle
```

**Copy-paste starter (save as `data/metadata/metadata_log.csv`):**
```csv
photo#,filename,angle,lighting,notes
```

**Notes column — record anything useful for later debugging:**
- Vehicle type (car / motorbike / tuk-tuk / truck)
- Plate type (single-line / multi-line Khmer / QR-code / new / old)
- Problems (glare, partial occlusion, dirty plate, blur)
- Province if identifiable

---

## 6. Folder Organisation

`setup_week1.py` already creates these folders. Sort your photos as you copy them off
the phone:

```
data/
├── raw/                         # ← original renamed photos live here
│   ├── by_angle/
│   │   ├── front/
│   │   ├── angled_left/
│   │   ├── angled_right/
│   │   ├── rear/                # (HIGH/LOW can go here or in front/)
│   ├── by_lighting/
│   │   ├── daylight/
│   │   ├── low_light/
│   │   ├── backlit/
├── metadata/
│   └── metadata_log.csv         # ← your CSV log
```

**Recommended workflow:**
1. Dump all renamed photos into `data/raw/` (this is the master copy — keep it intact).
2. **Copy** (don't move) into `by_angle/` and `by_lighting/` for your own review.
3. `data/raw/` is what you upload to Roboflow.

> 💡 Keep `data/raw/` as the untouched source of truth. Do all sorting/experiments
> on copies so you can never lose originals.

---

## 7. Per-Photo Quality Checklist

Before you accept a photo, confirm **every** box. Reject and reshoot if any fail.

- [ ] The **entire plate** is in frame (all four corners visible).
- [ ] Plate text is **sharp and readable** to your eye at 100% zoom.
- [ ] **No motion blur** (steady hands / tripod / subject not moving).
- [ ] Plate fills ~**10–40%** of the frame width (not tiny, not cropped).
- [ ] **No heavy glare / reflection** covering the characters.
- [ ] Plate is **not** severely occluded (bumper bars, dirt, stickers).
- [ ] Correct **exposure** — not blown-out white, not too dark to read.
- [ ] File is **JPG**, high resolution.
- [ ] Filename follows `plate_LOCATION_DATE_TIME_ANGLE.jpg`.
- [ ] Row added to `metadata_log.csv`.

### Collection-set checklist (after all 100)
- [ ] **100 photos** collected.
- [ ] At least **3 angles** well represented.
- [ ] At least **2 lighting** conditions represented.
- [ ] Mix of vehicle types (cars **and** motorbikes at minimum).
- [ ] Includes **multi-line Khmer** plates and at least a few **QR-code** plates.
- [ ] No exact duplicates (same car, same angle, same second).
- [ ] `metadata_log.csv` has exactly one row per photo (100 rows + header).

---

## 8. Common Mistakes to Avoid

| ❌ Mistake | ✅ Fix |
|-----------|-------|
| Using digital zoom | Physically walk closer |
| All photos front-on, same lot | Vary angle **and** location |
| Shooting only at noon | Include shade, dusk, backlit |
| HEIC files | Convert to JPG before Roboflow |
| Filling the CSV "later" | Log **while** shooting |
| Blurry "it's probably fine" shots | Reshoot — blur is unusable |
| Only cars | Motorbikes are the majority in Phnom Penh — include them |

---

**Next:** once all 100 photos are collected, renamed, sorted, and logged →
open **`roboflow_setup.py`** to upload, annotate, and export in YOLO format.
