#!/usr/bin/env python3
"""
scripts/tools/test_nonplate_inputs.py
=====================================
Does the system read numbers that are NOT on a licence plate?

A fair question in a viva: "your CRNN reads digits — what if I show it a house
number, or a plate number written on paper?" This script answers it by running
the REAL deployed pipeline over non-plate images and printing what each stage
returned.

It includes a POSITIVE CONTROL — a genuine plate photo from the test set — in
the same run. Without it, "nothing detected" proves nothing: a broken pipeline
would give the same output. The control must come back ENTRY_DENIED (or
ENTRY_ALLOWED if that plate happens to be whitelisted), which shows the pipeline
was alive while the non-plate images produced nothing.

    python scripts/tools/test_nonplate_inputs.py
    python scripts/tools/test_nonplate_inputs.py --figure   # also save a montage

Outputs:
    results/nonplate/<case>.jpg          the exact inputs used
    results/nonplate/summary.txt         the table below
    results/nonplate/figure.png          montage for the slide appendix (--figure)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2                                           # noqa: E402
import numpy as np                                   # noqa: E402

OUT = PROJECT_ROOT / "results" / "nonplate"
CONFIG = PROJECT_ROOT / "configs" / "system_config.yaml"
TEST_IMAGES = PROJECT_ROOT / "data" / "annotated" / "test" / "images"


# --------------------------------------------------------------------------- #
# The inputs. Deterministic, so anyone re-running gets the same images.
# --------------------------------------------------------------------------- #
def blank_page():
    return np.full((640, 640, 3), 245, np.uint8)


def number_on_paper():
    """A REAL Cambodian plate number, printed on paper. The hardest case:
    the exact string the CRNN was trained to read, with no plate around it."""
    img = np.full((640, 640, 3), 250, np.uint8)
    cv2.putText(img, "2A-0243", (110, 350), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (20, 20, 20), 8)
    return img


def house_number():
    img = np.full((640, 640, 3), 185, np.uint8)
    cv2.putText(img, "128", (205, 365), cv2.FONT_HERSHEY_DUPLEX, 4.5, (40, 40, 40), 10)
    return img


def fake_plate_shape():
    """A white rectangle with digits — plate-SHAPED, but not a plate: no Khmer
    province line, no border, wrong proportions and colours."""
    img = np.full((640, 640, 3), 70, np.uint8)
    cv2.rectangle(img, (140, 260), (500, 400), (240, 240, 240), -1)
    cv2.putText(img, "1A-2345", (165, 355), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (25, 25, 25), 5)
    return img


def random_texture():
    rng = np.random.default_rng(0)
    return cv2.GaussianBlur(rng.integers(60, 200, (640, 640, 3), dtype=np.uint8), (21, 21), 0)


def real_plate():
    """POSITIVE CONTROL — a genuine scene from the held-out test set."""
    if not TEST_IMAGES.is_dir():
        return None
    for p in sorted(TEST_IMAGES.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            img = cv2.imread(str(p))
            if img is not None:
                return img
    return None


CASES = [
    ("1_blank_page", "blank white page", blank_page, False),
    ("2_number_on_paper", "plate NUMBER printed on paper", number_on_paper, False),
    ("3_house_number", "house number on a wall", house_number, False),
    ("4_plate_shaped_sign", "white sign, plate-shaped, with digits", fake_plate_shape, False),
    ("5_random_texture", "random photo texture", random_texture, False),
    ("6_REAL_PLATE", "POSITIVE CONTROL — real plate photo", real_plate, True),
]


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", action="store_true", help="also save a montage PNG")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    from core.alpr_system import ALPRSystem

    system = ALPRSystem(str(CONFIG))
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 78)
    say(" DOES THE SYSTEM READ NUMBERS THAT ARE NOT ON A PLATE?")
    say("=" * 78)
    say(f" detector confidence threshold : {system.detector.conf}")
    say(f" CRNN confidence gate          : {system.crnn_conf_threshold}")
    say(f" plate-format validation       : {system.format_validation}")
    say("-" * 78)
    say(f" {'input':<40}{'detected':<10}{'read':<14}{'decision'}")
    say("-" * 78)

    results = []
    for key, label, fn, is_control in CASES:
        img = fn()
        if img is None:
            say(f" {label:<40}{'SKIPPED — no test images on disk'}")
            continue
        cv2.imwrite(str(OUT / f"{key}.jpg"), img)
        res = system.process_frame(img)
        if not res["plates"]:
            say(f" {label:<40}{'NO':<10}{'—':<14}{'(pipeline never ran)'}")
            results.append((label, img, False, None, None, is_control))
        else:
            p = res["plates"][0]
            say(f" {label:<40}{'YES':<10}{p['number']:<14}{p['action']}")
            results.append((label, img, True, p["number"], p["action"], is_control))

    say("-" * 78)

    # ---- verdict ---------------------------------------------------------- #
    non_plate = [r for r in results if not r[5]]
    control = next((r for r in results if r[5]), None)
    n_det = sum(1 for r in non_plate if r[2])
    say(f" non-plate images that produced a detection : {n_det} of {len(non_plate)}")
    if control is not None:
        say(f" positive control (real plate) detected     : {'YES' if control[2] else 'NO'}"
            f"   -> {control[4]}")
    say("")
    if n_det == 0 and control is not None and control[2]:
        say(" VERDICT: the pipeline was working (the control was read) and NONE of the")
        say("          non-plate images produced a detection. The CRNN is never reached")
        say("          because the detector hands it nothing — it is not a general OCR.")
    elif control is not None and not control[2]:
        say(" VERDICT: INCONCLUSIVE — the control was not detected either, so this run")
        say("          does not show anything. Check the weights and the config.")
    else:
        say(" VERDICT: at least one non-plate image produced a detection — inspect it")
        say("          above and check whether the confidence gate still refused it.")
    say("=" * 78)

    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved -> {(OUT / 'summary.txt').relative_to(PROJECT_ROOT)}")

    if args.figure:
        _figure(results)
    system.close()


def _figure(results) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 4.0))
    for ax, (label, img, det, num, action, is_ctrl) in zip(np.atleast_1d(axes), results):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        if det:
            title = f"detected\n{num}\n{action}"
            colour = "#2e9e6b" if is_ctrl else "#d64545"
        else:
            title = "no detection\nCRNN never runs"
            colour = "#2e9e6b"
        ax.set_title(f"{label}\n", fontsize=8, color="#333")
        ax.text(0.5, -0.08, title, transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color=colour, weight="bold")
    fig.suptitle("Non-plate inputs produce no detection; the real plate does "
                 "(positive control, far right)", fontsize=11)
    fig.tight_layout()
    path = OUT / "figure.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()