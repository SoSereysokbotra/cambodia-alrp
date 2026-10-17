#!/usr/bin/env python3
"""
scripts/tools/make_approaches_diagram.py
Generate clean 5-column architecture diagram for Slide 5:
A (ResNet18 scratch), B (ResNet18 frozen), C (ResNet18 fine-tune),
D (small CNN), E (ViT-B/16 Vision Transformer).
"""
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "slides" / "assets" / "approaches.png"
BAK = PROJECT_ROOT / "slides" / "assets" / "approaches_4col_backup.png"

if OUT.exists() and not BAK.exists():
    shutil.copy2(OUT, BAK)

fig, ax = plt.subplots(figsize=(15.0, 5.0), dpi=200)
ax.set_xlim(0, 15)
ax.set_ylim(0, 5.2)
ax.axis("off")

# 5 approaches
cols = [
    {
        "id": "A", "arch": "ResNet18", "color": "#1f77b4", "bg": "white",
        "blocks": ["Conv1 + MaxPool", "ResBlock 1 & 2", "ResBlock 3 & 4", "ResBlock 5 & 6", "ResBlock 7 & 8"],
        "frozen": False, "footer1": "random init", "footer2": "11.19 M trainable"
    },
    {
        "id": "B", "arch": "ResNet18", "color": "#ff7f0e", "bg": "#e2e4e8",
        "blocks": ["Conv1 + MaxPool", "ResBlock 1 & 2", "ResBlock 3 & 4", "ResBlock 5 & 6", "ResBlock 7 & 8"],
        "frozen": True, "footer1": "ImageNet init", "footer2": "13 338 trainable (0.12 %)"
    },
    {
        "id": "C", "arch": "ResNet18", "color": "#2ca02c", "bg": "white",
        "blocks": ["Conv1 + MaxPool", "ResBlock 1 & 2", "ResBlock 3 & 4", "ResBlock 5 & 6", "ResBlock 7 & 8"],
        "frozen": False, "footer1": "ImageNet init, low LR", "footer2": "11.19 M trainable"
    },
    {
        "id": "D", "arch": "small CNN", "color": "#d62728", "bg": "white",
        "blocks": ["Conv(3, 32)", "Conv(32, 64)", "Conv(64, 128)", "Conv(128, 256)", "GAP (256-d)"],
        "frozen": False, "footer1": "random init", "footer2": "0.40 M trainable"
    },
    {
        "id": "E", "arch": "ViT-B/16", "color": "#0F7C74", "bg": "white",
        "blocks": ["Patch Embed (16×16)", "Transformer Blocks 1–4", "Transformer Blocks 5–8", "Transformer Blocks 9–12", "CLS Token (768-d)"],
        "frozen": False, "footer1": "ImageNet init", "footer2": "85.82 M trainable"
    }
]

box_w = 2.1
col_centers = [1.5, 4.3, 7.1, 9.9, 12.7]

for i, c in enumerate(cols):
    cx = col_centers[i]
    bx = cx - box_w / 2
    
    # Title
    ax.text(cx, 4.75, f"{c['id']}  {c['arch']}", ha="center", va="center",
            fontsize=13, fontweight="bold", color=c["color"], fontfamily="sans-serif")
    
    # 4 Feature / Body blocks
    n_blocks = 4
    by_start = 4.15
    bh = 0.52
    bgap = 0.12
    
    for b_idx in range(n_blocks):
        by = by_start - (b_idx + 1) * (bh + bgap) + bgap
        fill_col = c["bg"]
        rect = patches.FancyBboxPatch(
            (bx, by), box_w, bh,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=2.0, edgecolor=c["color"], facecolor=fill_col
        )
        ax.add_patch(rect)
    
    if c["frozen"]:
        ax.text(bx + box_w + 0.12, by_start - 2 * (bh + bgap) + 0.1, "frozen",
                ha="left", va="center", fontsize=9.5, color="#6b7280", fontfamily="sans-serif")
        
    # Classifier Head FC 26
    head_y = by_start - (n_blocks + 1) * (bh + bgap) + bgap
    head_rect = patches.FancyBboxPatch(
        (bx, head_y), box_w, bh,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=2.0, edgecolor=c["color"], facecolor="#fde68a"
    )
    ax.add_patch(head_rect)
    ax.text(cx, head_y + bh / 2, "FC 26", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#1f2937", fontfamily="sans-serif")
    
    # Footer
    ax.text(cx, head_y - 0.32, c["footer1"], ha="center", va="top",
            fontsize=10, color="#111827", fontfamily="sans-serif")
    ax.text(cx, head_y - 0.62, c["footer2"], ha="center", va="top",
            fontsize=9.5, color="#4b5563", fontfamily="sans-serif")

plt.tight_layout()
fig.savefig(str(OUT), dpi=200, bbox_inches="tight", transparent=False, facecolor="white")
plt.close(fig)
print(f"Saved {OUT}")
