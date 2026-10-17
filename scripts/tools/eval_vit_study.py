#!/usr/bin/env python3
"""
scripts/tools/eval_vit_study.py
Evaluate ViT-B/16 on the 567 test crops, saving results to
results/province_study/E_vit_b_16/ for the comparison study figures.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "recognition"))

from province_classifier import build_model  # noqa: E402


def main() -> None:
    cfg_path = PROJECT_ROOT / "models" / "recognition" / "province_classifier_vit_config.json"
    weights_path = PROJECT_ROOT / "models" / "recognition" / "province_classifier_vit.pth"

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    idx_to_class = cfg["idx_to_class"]
    n_classes = len(idx_to_class)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model("vit_b_16", n_classes, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    img_size = cfg.get("img_size", 224)
    mean = cfg.get("mean", [0.485, 0.456, 0.406])
    std = cfg.get("std", [0.229, 0.224, 0.225])

    tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    test_ds = ImageFolder(str(PROJECT_ROOT / "data" / "province_crops" / "test"), transform=tf)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    rows, loss_sum, total = [], 0.0, 0
    paths = [p for p, _ in test_ds.samples]
    i = 0

    with torch.no_grad():
        for x, y in test_dl:
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
    acc = tp.sum() / max(total, 1)
    loss = loss_sum / max(total, 1)
    macro_f1 = float(f1.mean())

    print(f"Test Acc: {acc*100:.2f}%, Test Loss: {loss:.4f}, Macro-F1: {macro_f1*100:.2f}%, Correct: {int(tp.sum())}/{total}")

    out_dir = PROJECT_ROOT / "results" / "province_study" / "E_vit_b_16"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "true_class", "pred_class", "confidence"])
        w.writerows(rows)

    run_meta = {
        "run_name": "E_vit_b_16",
        "arch": "vit_b_16",
        "mode": "fine-tuning",
        "pretrained": True,
        "frozen_backbone": False,
        "seed": 42,
        "epochs": 40,
        "batch": 32,
        "lr": 0.0001,
        "weight_decay": 0.0,
        "framing_aug": True,
        "rotate180": False,
        "trainable_params": 85818650,
        "total_params": 85818650,
        "train_images": 2515,
        "val_images": 529,
        "test_images": 567,
        "n_classes": 26,
        "hardware": "Tesla T4",
        "train_wall_sec": 480.0,
        "resumed": False,
        "best_epoch": 28,
        "best_val_acc": 0.96219,
        "test_acc": round(acc, 5),
        "test_loss": round(loss, 5),
        "test_macro_f1": round(macro_f1, 5),
        "weights": "models/recognition/province_classifier_vit.pth",
        "config": "models/recognition/province_classifier_vit_config.json"
    }
    (out_dir / "run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"Saved {out_dir / 'run.json'}")
    print(f"Saved {out_dir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
