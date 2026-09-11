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

import torch
import torch.nn as nn

from province_classifier import build_resnet18   # noqa: E402
from province_map import N_CLASSES                # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "province_crops"
OUT_DIR = PROJECT_ROOT / "models" / "recognition"
IMG_SIZE = 128
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def make_loaders(batch: int, workers: int, framing_aug: bool = True,
                 rotate: bool = False, rotate180: bool = False):
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
    return (DataLoader(train_ds, batch, shuffle=True, num_workers=nw),
            DataLoader(val_ds, batch, shuffle=False, num_workers=nw),
            DataLoader(test_ds, batch, shuffle=False, num_workers=nw),
            train_ds)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
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
    args = ap.parse_args()
    out_pth = args.out
    out_cfg = out_pth.with_name(out_pth.stem + "_config.json")

    if not (DATA_DIR / "train").is_dir():
        print(f"[X] {DATA_DIR/'train'} not found. Run build_province_dataset.py first.")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print(" TRAIN PROVINCE CLASSIFIER (ResNet18, 26 classes)")
    print(f"   device {device} | epochs {args.epochs} | batch {args.batch}")
    print("=" * 60)

    mode = ("feature-extraction" if (args.freeze and args.pretrained)
            else "fine-tuning" if args.pretrained else "from-scratch")
    print(f"   output {out_pth.name} | mode={mode} | framing_aug={args.framing_aug} | "
          f"rotate={args.rotate} | rotate180={args.rotate180}")
    train_dl, val_dl, test_dl, train_ds = make_loaders(args.batch, args.workers,
                                                       args.framing_aug, args.rotate,
                                                       args.rotate180)

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
    if args.pretrained:
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, n_head)
        except Exception as exc:
            print(f"[!] pretrained unavailable ({exc}); training from scratch.")
            model = build_resnet18(n_head)
    else:
        model = build_resnet18(n_head)

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
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = n = 0.0
        for x, y in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item(); n += 1
        scheduler.step()
        val_acc = evaluate(model, val_dl, device)
        print(f"Epoch {epoch}/{args.epochs} | loss {running/max(n,1):.4f} "
              f"| val_acc {val_acc*100:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            out_pth.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_pth)
            out_cfg.write_text(json.dumps({
                "n_classes": n_head, "idx_to_class": idx_to_class,
                "img_size": IMG_SIZE, "mean": MEAN, "std": STD, "arch": "resnet18",
            }, indent=2), encoding="utf-8")
            print(f"    [saved] best val_acc {val_acc*100:.2f}%")

    test_acc = evaluate(model, test_dl, device)
    print("-" * 60)
    print(f"Best val acc: {best_acc*100:.2f}% | Test acc: {test_acc*100:.2f}%")
    print(f"Weights -> {out_pth}")
    print(f"Config  -> {out_cfg}")
    print("\nNext: validate framing-stability before/after:")
    print(f"  python scripts/tools/test_province_stability.py --weights {out_pth} --config {out_cfg}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[X] training failed: {exc}")
        raise
