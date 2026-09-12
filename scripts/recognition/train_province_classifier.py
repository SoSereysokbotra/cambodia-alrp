#!/usr/bin/env python3
"""
scripts/recognition/train_province_classifier.py
=================================================
Train a ResNet18 province classifier (26 classes) on data/province_crops/.

Prerequisite:
    python scripts/recognition/build_province_dataset.py

Outputs:
    models/recognition/province_classifier_best.pth
    models/recognition/province_classifier_config.json

Run:
    python scripts/recognition/train_province_classifier.py
    python scripts/recognition/train_province_classifier.py --epochs 40 --pretrained
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").is_dir()),
                    Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT_ROOT / "src" / "recognition"))

import numpy as np
import torch
import torch.nn as nn

from province_classifier import build_model      # noqa: E402
from province_map import N_CLASSES                # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "province_crops"
OUT_DIR = PROJECT_ROOT / "models" / "recognition"
RESULTS_DIR = PROJECT_ROOT / "results" / "province_study"   # one folder per run
IMG_SIZE = 128
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed(seed: int) -> None:
    """Fix every random source (Python, NumPy, PyTorch CPU + CUDA) so a run is
    reproducible. cuDNN autotuning is disabled because it picks kernels
    non-deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(batch: int, workers: int, framing_aug: bool = True,
                 rotate: bool = False, rotate180: bool = False,
                 seed: int = 42):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    if framing_aug:
        # FRAMING-INVARIANCE augmentation (MODEL_IMPROVEMENT_PLAN Step 3).
        # The classifier was fragile to WHERE the detector box lands: jittering the
        # box flipped 70% of plates to 2-3 provinces (the live flicker). These
        # transforms simulate that box variation at train time — random zoom/crop
        # position (RandomResizedCrop) + random shift/scale/rotate (RandomAffine),
        # with black fill so it also sees the box catching background — so the model
        # learns the province regardless of exact framing.
        # ROTATION (--rotate): degrees=180 makes the AFFINE rotate the crop to ANY
        # angle, so the classifier reads the province upside-down / sideways too.
        # --rotate180 (preferred for upside-down plates): flip EXACTLY 180 deg
        # half the time. --rotate spreads the model over every angle in
        # [-180,180], including 90 deg where a wide Khmer line is mostly cut off
        # by the square crop — capacity spent on orientations no plate ever has.
        # A plate on a car is upright or upside-down, so train for those two.
        rot_deg = 180 if rotate else 6
        steps = [transforms.RandomResizedCrop(IMG_SIZE, scale=(0.60, 1.0), ratio=(0.6, 1.7))]
        if rotate180:
            steps.append(transforms.RandomApply(
                [transforms.RandomRotation((180, 180))], p=0.5))
        steps += [
            transforms.RandomAffine(degrees=rot_deg, translate=(0.12, 0.12),
                                    scale=(0.9, 1.1), fill=0),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
        train_tf = transforms.Compose(steps)
    else:
        # original (tight-crop) augmentation — kept so the old behaviour is reproducible
        train_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(6),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_ds = datasets.ImageFolder(str(DATA_DIR / "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(str(DATA_DIR / "val"), transform=eval_tf)
    test_ds = datasets.ImageFolder(str(DATA_DIR / "test"), transform=eval_tf)
    nw = workers if sys.platform != "win32" else 0
    # seeded generator -> the shuffle order is the same on every run
    gen = torch.Generator().manual_seed(seed)
    return (DataLoader(train_ds, batch, shuffle=True, num_workers=nw, generator=gen),
            DataLoader(val_ds, batch, shuffle=False, num_workers=nw),
            DataLoader(test_ds, batch, shuffle=False, num_workers=nw),
            train_ds)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None) -> tuple[float, float]:
    """Return (accuracy, mean loss) over a loader. Loss is 0.0 if no criterion."""
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            loss_sum += criterion(logits, y).item() * y.numel()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1), loss_sum / max(total, 1)


@torch.no_grad()
def evaluate_detailed(model, loader, device, criterion, idx_to_class):
    """Test-set evaluation with everything the report needs.

    Returns (accuracy, mean loss, macro-F1, rows) where rows are
    (image_path, true_class_id, pred_class_id, confidence) per test image —
    class ids are the TRUE province ids (via idx_to_class), not ImageFolder's.
    Macro-F1 averages F1 over classes with equal weight, so the 5-image
    provinces count as much as Phnom Penh — the right metric for this
    imbalance. Computed from the confusion matrix with NumPy (no sklearn).
    """
    model.eval()
    n = len(idx_to_class)
    cm = np.zeros((n, n), dtype=np.int64)          # rows = true, cols = pred
    rows, loss_sum, total = [], 0.0, 0
    paths = [p for p, _ in loader.dataset.samples]
    i = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * y.numel()
        conf, pred = torch.softmax(logits, 1).max(1)
        for t, p, c in zip(y.tolist(), pred.tolist(), conf.tolist()):
            cm[t, p] += 1
            rel = Path(paths[i]).relative_to(PROJECT_ROOT).as_posix()
            rows.append((rel, idx_to_class[t], idx_to_class[p], round(c, 4)))
            i += 1
        total += y.numel()
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return tp.sum() / max(total, 1), loss_sum / max(total, 1), float(f1.mean()), rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="L2 regularisation strength for Adam (0 = off). The "
                         "regularisation choice swept in the hyperparameter study.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--arch", choices=["resnet18", "small_cnn"], default="resnet18",
                    help="model architecture: resnet18 (11 M params) or a 4-block "
                         "small_cnn baseline (~0.4 M params, from scratch only)")
    ap.add_argument("--pretrained", action="store_true",
                    help="Use ImageNet-pretrained ResNet18 (downloads weights).")
    ap.add_argument("--freeze", action="store_true",
                    help="FEATURE EXTRACTION: freeze the backbone, train only the "
                         "classifier head. Use WITH --pretrained. Without this the "
                         "run is a full FINE-TUNE (every layer learns).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path,
                    default=OUT_DIR / "province_classifier_best.pth",
                    help="weights output path. Pass a NEW name (e.g. "
                         "province_classifier_framing.pth) to keep the deployed model "
                         "intact while validating a candidate.")
    ap.add_argument("--framing-aug", dest="framing_aug", action="store_true",
                    default=True, help="framing-invariance augmentation (default ON)")
    ap.add_argument("--no-framing-aug", dest="framing_aug", action="store_false",
                    help="reproduce the old tight-crop augmentation")
    ap.add_argument("--rotate", action="store_true",
                    help="add FULL rotation (any angle in [-180,180]). Broad but "
                         "wasteful — see --rotate180.")
    ap.add_argument("--rotate180", action="store_true",
                    help="flip exactly 180 deg with p=0.5 — the targeted fix for "
                         "upside-down plates (mirrors finetune_crnn.py --rotate180)")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for Python / NumPy / PyTorch (reproducibility)")
    ap.add_argument("--run-name", default=None,
                    help="folder name under results/province_study/ for this run's "
                         "history/metrics (default: <arch>_<mode>_s<seed>)")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from results/province_study/"
                         "<run-name>/last.pth (pass the same flags as the original run)")
    args = ap.parse_args()
    out_pth = args.out
    out_cfg = out_pth.with_name(out_pth.stem + "_config.json")
    if args.run_name is None:
        _mode = ("frozen" if (args.freeze and args.pretrained)
                 else "finetune" if args.pretrained else "scratch")
        args.run_name = f"{args.arch}_{_mode}_s{args.seed}"

    if not (DATA_DIR / "train").is_dir():
        print(f"[X] {DATA_DIR/'train'} not found. Run build_province_dataset.py first.")
        sys.exit(1)

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print(f" TRAIN PROVINCE CLASSIFIER ({args.arch}, 26 classes)")
    print(f"   device {device} | epochs {args.epochs} | batch {args.batch}")
    print("=" * 60)

    mode = ("feature-extraction" if (args.freeze and args.pretrained)
            else "fine-tuning" if args.pretrained else "from-scratch")
    print(f"   output {out_pth.name} | mode={mode} | framing_aug={args.framing_aug} | "
          f"rotate={args.rotate} | rotate180={args.rotate180} | seed={args.seed}")
    train_dl, val_dl, test_dl, train_ds = make_loaders(args.batch, args.workers,
                                                       args.framing_aug, args.rotate,
                                                       args.rotate180, args.seed)

    # ImageFolder sorts class folders LEXICOGRAPHICALLY ('0','1','10',..,'2',..),
    # so its label index is NOT our numeric province id. Record the true mapping:
    #   model output index i  ->  idx_to_class[i]  (the real province class id)
    idx_to_class = [int(c) for c in train_ds.classes]
    n_head = len(idx_to_class)
    assert train_ds.classes == test_dl.dataset.classes == val_dl.dataset.classes, \
        "train/val/test class folders differ — re-run build_province_dataset.py"
    print(f"train={len(train_dl.dataset)} val={len(val_dl.dataset)} "
          f"test={len(test_dl.dataset)} | classes={n_head} | idx_to_class={idx_to_class}")

    # Head is sized to the number of populated folders; predictions are
    # translated back to true province ids via idx_to_class (saved in config).
    if args.pretrained and args.arch != "resnet18":
        print(f"[!] --pretrained has no ImageNet weights for arch={args.arch}; ignoring it.")
        args.pretrained = False
    try:
        model = build_model(args.arch, n_head, pretrained=args.pretrained)
    except Exception as exc:
        print(f"[!] pretrained unavailable ({exc}); training from scratch.")
        args.pretrained = False
        model = build_model(args.arch, n_head, pretrained=False)

    # TRANSFER LEARNING MODE (--freeze) — "feature extraction".
    # Lock every ImageNet layer and train ONLY the new classification head, so
    # the pretrained convolutions act as a fixed feature extractor. Contrast with
    # the default, "fine-tuning", where the whole network keeps learning.
    # Feature extraction is the safer choice on a small dataset (far fewer
    # trainable parameters, so less to overfit); fine-tuning usually wins when
    # there is enough data to adapt the features to the new domain.
    frozen = trainable = 0
    if args.freeze:
        if not args.pretrained:
            print("[!] --freeze without --pretrained freezes RANDOM weights — "
                  "that is not feature extraction. Add --pretrained.")
        for name, prm in model.named_parameters():
            if not name.startswith("fc."):
                prm.requires_grad = False
                frozen += prm.numel()
            else:
                trainable += prm.numel()
        print(f"[freeze] FEATURE EXTRACTION: {frozen:,} frozen, "
              f"{trainable:,} trainable ({100*trainable/(frozen+trainable):.2f}% of the model)")
    else:
        trainable = sum(p.numel() for p in model.parameters())
        print(f"[freeze] FINE-TUNING: all {trainable:,} parameters trainable")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Only optimise what is actually trainable (an optimiser handed frozen
    # params still carries their state and can silently update them via
    # weight decay in some optimisers).
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    # Per-epoch history -> results/province_study/<run>/history.csv, so learning
    # curves (train vs val loss/accuracy) can be plotted after the run.
    run_dir = RESULTS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    hist_path = run_dir / "history.csv"
    print(f"   history -> {hist_path.relative_to(PROJECT_ROOT)}")

    # Full-state checkpoint (model + optimizer + scheduler + progress) written
    # after EVERY epoch so a Colab disconnect costs at most one epoch:
    #   python train_province_classifier.py <same flags> --resume
    last_ckpt = run_dir / "last.pth"
    best_acc, best_epoch, start_epoch = 0.0, 0, 1
    if args.resume:
        if not last_ckpt.exists():
            print(f"[X] --resume but no checkpoint at {last_ckpt}")
            sys.exit(1)
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        best_acc, best_epoch = ck["best_acc"], ck["best_epoch"]
        start_epoch = ck["epoch"] + 1
        print(f"[resume] continuing from epoch {start_epoch} "
              f"(best val {best_acc*100:.2f}% at epoch {best_epoch})")
    else:
        with open(hist_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc",
                                    "val_loss", "val_acc", "lr", "epoch_sec"])

    t_train = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        t_epoch = time.perf_counter()
        model.train()
        running = n = 0.0
        correct = seen = 0
        for x, y in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item(); n += 1
            correct += (logits.argmax(1) == y).sum().item(); seen += y.numel()
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        train_loss, train_acc = running / max(n, 1), correct / max(seen, 1)
        val_acc, val_loss = evaluate(model, val_dl, device, criterion)
        epoch_sec = time.perf_counter() - t_epoch
        print(f"Epoch {epoch}/{args.epochs} | loss {train_loss:.4f} acc {train_acc*100:.2f}% "
              f"| val loss {val_loss:.4f} acc {val_acc*100:.2f}% | {epoch_sec:.0f}s")
        with open(hist_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.5f}", f"{train_acc:.5f}",
                                    f"{val_loss:.5f}", f"{val_acc:.5f}",
                                    f"{cur_lr:.2e}", f"{epoch_sec:.1f}"])
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            out_pth.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_pth)
            out_cfg.write_text(json.dumps({
                "n_classes": n_head, "idx_to_class": idx_to_class,
                "img_size": IMG_SIZE, "mean": MEAN, "std": STD, "arch": args.arch,
            }, indent=2), encoding="utf-8")
            print(f"    [saved] best val_acc {val_acc*100:.2f}%")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "best_acc": best_acc, "best_epoch": best_epoch}, last_ckpt)
    train_sec = time.perf_counter() - t_train

    # Test the BEST-validation checkpoint, not whatever the last epoch left
    # behind (model selection on val, one final measurement on test).
    model.load_state_dict(torch.load(out_pth, map_location=device, weights_only=True))
    test_acc, test_loss, test_f1, preds = evaluate_detailed(model, test_dl, device,
                                                            criterion, idx_to_class)
    with open(run_dir / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "true_class", "pred_class", "confidence"])
        w.writerows(preds)

    # Everything the rubric asks to report per approach, in one JSON.
    run_meta = {
        "run_name": args.run_name, "arch": args.arch, "mode": mode,
        "pretrained": bool(args.pretrained), "frozen_backbone": bool(args.freeze),
        "seed": args.seed, "epochs": args.epochs, "batch": args.batch,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "framing_aug": bool(args.framing_aug), "rotate180": bool(args.rotate180),
        "trainable_params": trainable, "total_params": trainable + frozen,
        "train_images": len(train_dl.dataset), "val_images": len(val_dl.dataset),
        "test_images": len(test_dl.dataset), "n_classes": n_head,
        "hardware": (torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu"),
        "train_wall_sec": round(train_sec, 1),      # this process only if resumed
        "resumed": bool(args.resume),
        "best_epoch": best_epoch, "best_val_acc": round(best_acc, 5),
        "test_acc": round(test_acc, 5), "test_loss": round(test_loss, 5),
        "test_macro_f1": round(test_f1, 5),
        "weights": str(out_pth), "config": str(out_cfg),
    }
    (run_dir / "run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print("-" * 60)
    print(f"Best val acc: {best_acc*100:.2f}% (epoch {best_epoch}) | "
          f"Test acc: {test_acc*100:.2f}% | Test macro-F1: {test_f1*100:.2f}% | "
          f"train time {train_sec/60:.1f} min on {run_meta['hardware']}")
    print(f"Weights -> {out_pth}")
    print(f"Config  -> {out_cfg}")
    print(f"Run dir -> {run_dir.relative_to(PROJECT_ROOT)}  (history.csv, run.json, test_predictions.csv)")
    print("\nNext: validate framing-stability before/after:")
    print(f"  python scripts/tools/test_province_stability.py --weights {out_pth} --config {out_cfg}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] training failed: {exc}")
        raise
