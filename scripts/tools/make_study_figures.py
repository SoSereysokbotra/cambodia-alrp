#!/usr/bin/env python3
"""
scripts/tools/make_study_figures.py
===================================
Build every table and figure for the province-classification comparison study
from the artefacts each training run leaves in results/province_study/<run>/
(history.csv, run.json, test_predictions.csv). Nothing is re-trained here.

Outputs (results/province_study/):
    summary.csv                       one row per run: params, time, val/test acc, macro-F1
    summary.md                        the same table in Markdown, ready for the README
    figures/fig1_test_accuracy.png    test accuracy + macro-F1 per approach
    figures/fig2_learning_curves.png  train vs val loss and accuracy, all runs overlaid
    figures/fig3_confusion_<best>.png normalised confusion matrix of the best run
    figures/fig4_per_class_<best>.png per-class test accuracy vs train-set size
    figures/fig5_failures_<best>.png  the most confident wrong predictions, as images
    figures/fig6_tuning_heatmap.png   lr x weight-decay -> val accuracy (if tuning.csv exists)

Run:
    python scripts/tools/make_study_figures.py
    python scripts/tools/make_study_figures.py --best C_resnet18_finetune
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src" / "recognition"))

import numpy as np                                   # noqa: E402
import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

from province_map import province_latin              # noqa: E402

STUDY = PROJECT_ROOT / "results" / "province_study"
FIGS = STUDY / "figures"
MANIFEST = PROJECT_ROOT / "data" / "province_crops" / "manifest.json"

# one colour per run letter, reused across every figure
PALETTE = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728"}


def load_runs() -> list[dict]:
    """The main-study runs (A, B, C, D...) - every run folder with a run.json
    EXCEPT the tune_* grid, which belongs to fig6 only."""
    runs = []
    for rj in sorted(STUDY.glob("*/run.json")):
        if rj.parent.name.startswith("tune_"):
            continue
        r = json.loads(rj.read_text(encoding="utf-8"))
        r["_dir"] = rj.parent
        hist = rj.parent / "history.csv"
        r["_hist"] = list(csv.DictReader(open(hist, encoding="utf-8"))) if hist.exists() else []
        runs.append(r)
    if not runs:
        sys.exit(f"[X] no run.json under {STUDY} - train first")
    return runs


def colour(run: dict) -> str:
    return PALETTE.get(run["run_name"][0].upper(), "#7f7f7f")


def label(run: dict) -> str:
    """'A  resnet18 / scratch'"""
    return f"{run['run_name'][0].upper()}  {run['arch']} / {run['mode']}"


# --------------------------------------------------------------------------- #
def add_clean_accuracy(runs: list[dict]) -> int:
    """Accuracy on the test crops whose source photo is NOT in train/val
    (list written by check_province_leakage.py). Returns the clean count."""
    lst = STUDY / "leaked_test_crops.txt"
    leaked = set(lst.read_text(encoding="utf-8").split()) if lst.exists() else set()
    n_clean = 0
    for r in runs:
        preds = load_predictions(r)
        clean = [p for p in preds if p["image"] not in leaked]
        n_clean = len(clean)
        r["test_acc_clean"] = (sum(p["true_class"] == p["pred_class"] for p in clean) / len(clean)
                               if clean else float("nan"))
    return n_clean


def write_summary(runs: list[dict]) -> None:
    n_clean = add_clean_accuracy(runs)
    cols = ["run_name", "arch", "mode", "pretrained", "frozen_backbone", "lr",
            "weight_decay", "trainable_params", "total_params", "epochs",
            "best_epoch", "best_val_acc", "test_acc", "test_acc_clean", "test_macro_f1",
            "train_wall_sec", "hardware", "seed"]
    with open(STUDY / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(runs)

    lines = ["| run | architecture | strategy | trainable params | best epoch | "
             f"val acc | **test acc** (n=567) | test acc, clean subset (n={n_clean}) | "
             "test macro-F1 | train time | hardware |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        lines.append(
            f"| {r['run_name']} | {r['arch']} | {r['mode']} | {r['trainable_params']:,} | "
            f"{r['best_epoch']} | {100*r['best_val_acc']:.1f} % | **{100*r['test_acc']:.1f} %** | "
            f"{100*r['test_acc_clean']:.1f} % | "
            f"{100*r['test_macro_f1']:.1f} % | {r['train_wall_sec']/60:.1f} min | {r['hardware']} |")
    (STUDY / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
def fig1_bar(runs: list[dict]) -> None:
    x = np.arange(len(runs))
    acc = [100 * r["test_acc"] for r in runs]
    f1 = [100 * r["test_macro_f1"] for r in runs]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - 0.2, acc, 0.4, label="test accuracy", color=[colour(r) for r in runs])
    b2 = ax.bar(x + 0.2, f1, 0.4, label="test macro-F1",
                color=[colour(r) for r in runs], alpha=0.45, hatch="//")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                    f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([label(r).replace("  ", "\n") for r in runs], fontsize=9)
    ax.set_ylabel("% on the 567 held-out test crops")
    ax.set_ylim(0, 105)
    ax.set_title("Province classification - test performance per approach")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_test_accuracy.png", dpi=150)
    plt.close(fig)


def fig2_curves(runs: list[dict]) -> None:
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in runs:
        h = r["_hist"]
        if not h:
            continue
        ep = [int(row["epoch"]) for row in h]
        c = colour(r)
        ax_l.plot(ep, [float(row["train_loss"]) for row in h], color=c, lw=1.8,
                  label=f"{label(r)} - train")
        ax_l.plot(ep, [float(row["val_loss"]) for row in h], color=c, lw=1.8, ls="--",
                  label=f"{label(r)} - val")
        ax_a.plot(ep, [100 * float(row["train_acc"]) for row in h], color=c, lw=1.8)
        ax_a.plot(ep, [100 * float(row["val_acc"]) for row in h], color=c, lw=1.8, ls="--")
    ax_l.set_title("Cross-entropy loss (solid = train, dashed = val)")
    ax_l.set_xlabel("epoch"); ax_l.set_ylabel("loss"); ax_l.grid(alpha=0.3)
    ax_a.set_title("Accuracy (solid = train, dashed = val)")
    ax_a.set_xlabel("epoch"); ax_a.set_ylabel("%"); ax_a.set_ylim(0, 102); ax_a.grid(alpha=0.3)
    ax_l.legend(fontsize=7, ncol=2)
    fig.suptitle("Learning curves - same split, same augmentation, 40 epochs, seed 42")
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_learning_curves.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def load_predictions(run: dict) -> list[dict]:
    p = run["_dir"] / "test_predictions.csv"
    return list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []


def fig3_confusion(run: dict, preds: list[dict]) -> None:
    classes = sorted({int(r["true_class"]) for r in preds} | {int(r["pred_class"]) for r in preds})
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)))
    for r in preds:
        cm[idx[int(r["true_class"])], idx[int(r["pred_class"])]] += 1
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    names = [province_latin(c).replace("_", " ") for c in classes]
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(names, rotation=90, fontsize=8); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] and i != j:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7, color="red")
            elif i == j:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03, label="fraction of true class")
    ax.set_title(f"Confusion matrix (row-normalised) - {label(run)}  "
                 f"test acc {100*run['test_acc']:.1f} %\nred numbers = off-diagonal counts")
    fig.tight_layout()
    fig.savefig(FIGS / f"fig3_confusion_{run['run_name']}.png", dpi=150)
    plt.close(fig)


def fig4_per_class(run: dict, preds: list[dict]) -> None:
    train_n = {}
    if MANIFEST.exists():
        train_n = {int(k): v for k, v in
                   json.loads(MANIFEST.read_text(encoding="utf-8"))["per_class"]["train"].items()}
    tot, ok = {}, {}
    for r in preds:
        t = int(r["true_class"])
        tot[t] = tot.get(t, 0) + 1
        ok[t] = ok.get(t, 0) + (r["true_class"] == r["pred_class"])
    classes = sorted(tot, key=lambda c: train_n.get(c, 0))
    acc = [100 * ok[c] / tot[c] for c in classes]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    bars = ax.bar(range(len(classes)), acc, color=colour(run))
    for b, c in zip(bars, classes):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"n={train_n.get(c, '?')}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([f"{province_latin(c).replace('_', ' ')} ({tot[c]})" for c in classes],
                       rotation=90, fontsize=8)
    ax.set_ylabel("test accuracy (%)"); ax.set_ylim(0, 118)
    ax.set_title(f"Per-class test accuracy, sorted by TRAIN-set size (n above bar; test count in "
                 f"brackets) - {label(run)}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / f"fig4_per_class_{run['run_name']}.png", dpi=150)
    plt.close(fig)


def fig5_failures(run: dict, preds: list[dict], n: int = 16) -> None:
    import cv2
    wrong = [r for r in preds if r["true_class"] != r["pred_class"]]
    wrong.sort(key=lambda r: -float(r["confidence"]))      # most confident mistakes first
    wrong = wrong[:n]
    if not wrong:
        return
    cols = 4
    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
    for ax, r in zip(axes.flat, wrong):
        img = cv2.imread(str(PROJECT_ROOT / r["image"]))
        if img is not None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"true: {province_latin(int(r['true_class']))}\n"
                     f"pred: {province_latin(int(r['pred_class']))}  ({float(r['confidence']):.2f})",
                     fontsize=8, color="darkred")
        ax.axis("off")
    for ax in list(axes.flat)[len(wrong):]:
        ax.axis("off")
    fig.suptitle(f"Most confident WRONG predictions - {label(run)}  "
                 f"({len([r for r in preds if r['true_class'] != r['pred_class']])} errors / {len(preds)})")
    fig.tight_layout()
    fig.savefig(FIGS / f"fig5_failures_{run['run_name']}.png", dpi=130)
    plt.close(fig)


def fig6_tuning() -> None:
    p = STUDY / "tuning.csv"
    if not p.exists():
        return
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    lrs = sorted({float(r["lr"]) for r in rows})
    wds = sorted({float(r["weight_decay"]) for r in rows})
    grid = np.full((len(wds), len(lrs)), np.nan)
    for r in rows:
        grid[wds.index(float(r["weight_decay"])), lrs.index(float(r["lr"]))] = 100 * float(r["best_val_acc"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    im = ax.imshow(grid, cmap="viridis")
    ax.set_xticks(range(len(lrs))); ax.set_xticklabels([f"{v:g}" for v in lrs])
    ax.set_yticks(range(len(wds))); ax.set_yticklabels([f"{v:g}" for v in wds])
    ax.set_xlabel("learning rate"); ax.set_ylabel("weight decay")
    for i in range(len(wds)):
        for j in range(len(lrs)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", color="w", fontsize=10)
    fig.colorbar(im, ax=ax, label="best val accuracy (%)")
    ax.set_title("Approach C: lr x weight decay (selected on validation, never on test)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_tuning_heatmap.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--best", default=None,
                    help="run_name to use for the confusion / per-class / failure figures "
                         "(default: highest VALIDATION accuracy - never chosen on test)")
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    write_summary(runs)
    fig1_bar(runs)
    fig2_curves(runs)

    best = (next(r for r in runs if r["run_name"] == args.best) if args.best
            else max(runs, key=lambda r: r["best_val_acc"]))
    preds = load_predictions(best)
    if preds:
        fig3_confusion(best, preds)
        fig4_per_class(best, preds)
        fig5_failures(best, preds)
    fig6_tuning()

    print(f"\nfigures -> {FIGS.relative_to(PROJECT_ROOT)}  (best run for detail figures: {best['run_name']})")
    for p in sorted(FIGS.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
