"""
src/recognition/province_map.py
===============================
Phase 3 — map the province classifier's output to Khmer names and compose the
full plate text `provinceKhmer + " " + number`.

Class scheme (26 classes):
    0..24  -> the 25 Cambodian provinces (Khmer names below)
    25     -> "other" (non-province Plate_v4 categories: Cambodia/Police/RCAF/State)

The 25-province order is the project's canonical order (matches the classifier
head). `PLATE_V4_NAMES` + `plate_v4_to_class()` map the original 29-class
Plate_v4 labels onto this scheme when building the dataset.
"""

from __future__ import annotations

import re

N_PROVINCES = 25
OTHER_CLASS = 25          # folded non-province categories
N_CLASSES = 26            # 25 provinces + other

# class_id -> Khmer province name (0..24), plus 25 = other
PROVINCE_KHMER: dict[int, str] = {
    0:  "បន្ទាយមានជ័យ",
    1:  "បាត់ដំបង",
    2:  "កំពង់ចាម",
    3:  "កំពង់ឆ្នាំង",
    4:  "កំពង់ស្ពឺ",
    5:  "កំពង់ធំ",
    6:  "កំពត",
    7:  "កណ្តាល",
    8:  "កែប",
    9:  "កោះកុង",
    10: "ក្រចេះ",
    11: "មណ្ឌលគិរី",
    12: "ឧត្តរមានជ័យ",
    13: "ប៉ៃលិន",
    14: "ភ្នំពេញ",
    15: "ព្រះសីហនុ",
    16: "ព្រះវិហារ",
    17: "ព្រៃវែង",
    18: "ពោធិ៍សាត់",
    19: "រតនគិរី",
    20: "សៀមរាប",
    21: "ស្ទឹងត្រែង",
    22: "ស្វាយរៀង",
    23: "តាកែវ",
    24: "ត្បូងឃ្មុំ",
    25: "",              # "other" -> no province prefix
}

# Latin names for the 25 provinces, SAME order as the class ids above.
PROVINCE_LATIN = [
    "Banteay_Meanchey", "Battambang", "Kampong_Cham", "Kampong_Chhnang",
    "Kampong_Speu", "Kampong_Thom", "Kampot", "Kandal", "Kep", "Koh_Kong",
    "Kratie", "Mondul_Kiri", "Oudor_Meanchey", "Pailin", "Phnom_Penh",
    "Preah_Sihanouk", "Preah_Vihear", "Prey_Veng", "Pursat", "Ratanakiri",
    "Siem_Reap", "Stung_Treng", "Svay_Rieng", "Takeo", "Tboung_Khmum",
]

# The original Plate_v4 v3 `names` list (29 classes), used only when building
# the province dataset from that download.
PLATE_V4_NAMES = [
    "Banteay_Meanchey", "Battambang", "Cambodia", "Kampong_Cham",
    "Kampong_Chhnang", "Kampong_Speu", "Kampong_Thom", "Kampot", "Kandal",
    "Kep", "Koh_Kong", "Kratie", "Mondul_Kiri", "Oudor_Meanchey", "Pailin",
    "Phnom_Penh", "Police", "Preah_Sihanouk", "Preah_Vihear", "Prey_Veng",
    "Pursat", "RCAF", "Ratanakiri", "Siem_Reap", "State", "Stung_Treng",
    "Svay_Rieng", "Takeo", "Tboung_Khmum",
]

_LATIN_TO_CLASS = {name: i for i, name in enumerate(PROVINCE_LATIN)}


def plate_v4_to_class(plate_v4_id: int) -> int:
    """Map an original Plate_v4 class id (0..28) to this project's class id.
    Provinces -> 0..24; the 4 non-province categories -> OTHER_CLASS (25)."""
    if not (0 <= plate_v4_id < len(PLATE_V4_NAMES)):
        return OTHER_CLASS
    return _LATIN_TO_CLASS.get(PLATE_V4_NAMES[plate_v4_id], OTHER_CLASS)


def province_khmer(class_id: int) -> str:
    """Khmer name for a class id ('' for 'other' or unknown)."""
    return PROVINCE_KHMER.get(int(class_id), "")


def province_latin(class_id: int) -> str:
    if 0 <= class_id < N_PROVINCES:
        return PROVINCE_LATIN[class_id]
    return "Other"


def normalize_plate(text: str) -> str:
    """SRS REC-006 normalisation: collapse whitespace, tighten dashes, trim."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return text


def compose_plate(class_id: int, number: str) -> str:
    """Full plate = normalize(provinceKhmer + ' ' + number).

    compose_plate(14, '1AB-2345') -> 'ភ្នំពេញ 1AB-2345'
    compose_plate(25, '1AB-2345') -> '1AB-2345'   (other -> number only)
    """
    prov = province_khmer(class_id)
    return normalize_plate(f"{prov} {number}".strip())
