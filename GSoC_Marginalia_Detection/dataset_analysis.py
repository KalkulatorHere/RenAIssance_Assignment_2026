import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
from collections import Counter

DATASET_ROOT = Path(__file__).resolve().parent.parent / "YOLOv7_85-15"
OUTPUT_DIR   = Path(__file__).resolve().parent / "results" / "dataset_analysis"

SPLITS = {
    "train": DATASET_ROOT / "train",
    "valid": DATASET_ROOT / "valid",
    "test":  DATASET_ROOT / "test",
}

CLASS_NAMES = {0: "marginalia"}


def parse_yolo_label(label_path: Path):
    boxes = []
    if not label_path.exists() or label_path.stat().st_size == 0:
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:])
                boxes.append((cls_id, x_c, y_c, w, h))
    return boxes


def yolo_to_xyxy(box, img_w, img_h):
    _, x_c, y_c, w, h = box
    x1 = (x_c - w / 2) * img_w
    y1 = (y_c - h / 2) * img_h
    x2 = (x_c + w / 2) * img_w
    y2 = (y_c + h / 2) * img_h
    return x1, y1, x2, y2


def compute_split_stats():
    print("=" * 60)
    print("DATASET SPLIT STATISTICS")
    print("=" * 60)

    total_images = 0
    total_boxes  = 0
    all_widths   = []
    all_heights  = []
    box_counts   = []

    rows = []
    for name, split_dir in SPLITS.items():
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        img_files = sorted(img_dir.iterdir()) if img_dir.exists() else []
        n_imgs = len(img_files)
        total_images += n_imgs

        n_boxes  = 0
        n_pos    = 0
        n_neg    = 0
        for img_path in img_files:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            boxes = parse_yolo_label(lbl_path)
            n_boxes += len(boxes)
            box_counts.append(len(boxes))
            if len(boxes) > 0:
                n_pos += 1
            else:
                n_neg += 1
            for box in boxes:
                all_widths.append(box[3])
                all_heights.append(box[4])

        total_boxes += n_boxes
        rows.append((name, n_imgs, n_pos, n_neg, n_boxes,
                      n_boxes / n_imgs if n_imgs else 0))

    print(f"{'Split':<8} {'Images':>8} {'Pos':>6} {'Neg':>6} {'Boxes':>8} {'Avg/img':>8}")
    print("-" * 50)
    for r in rows:
        print(f"{r[0]:<8} {r[1]:>8} {r[2]:>6} {r[3]:>6} {r[4]:>8} {r[5]:>8.2f}")
    print("-" * 50)
    print(f"{'TOTAL':<8} {total_images:>8} {'':>6} {'':>6} {total_boxes:>8} "
          f"{total_boxes / total_images:>8.2f}")
    print()

    return all_widths, all_heights, box_counts


def plot_bbox_analysis(widths, heights, box_counts, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.scatter(widths, heights, alpha=0.3, s=10, c="#6C5CE7")
    ax.set_xlabel("Box Width (normalised)")
    ax.set_ylabel("Box Height (normalised)")
    ax.set_title("Bounding-Box Size Distribution")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1]
    ax.hist(box_counts, bins=range(0, max(box_counts) + 2),
            color="#00B894", edgecolor="white")
    ax.set_xlabel("Boxes per Image")
    ax.set_ylabel("Count")
    ax.set_title("Annotations per Image")

    areas = [w * h for w, h in zip(widths, heights)]
    ax = axes[2]
    ax.hist(areas, bins=40, color="#FD79A8", edgecolor="white")
    ax.set_xlabel("Box Area (normalised)")
    ax.set_ylabel("Count")
    ax.set_title("Bounding-Box Area Distribution")

    plt.tight_layout()
    save_path = out_dir / "bbox_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved bbox analysis → {save_path}")


def visualize_samples(n=6, out_dir: Path = OUTPUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    positives = []
    for name, split_dir in SPLITS.items():
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            boxes = parse_yolo_label(lbl_path)
            if boxes:
                positives.append((img_path, boxes, name))

    samples = random.sample(positives, min(n, len(positives)))

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]

    for idx, (img_path, boxes, split) in enumerate(samples):
        ax = axes[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        ax.imshow(np.array(img))

        for box in boxes:
            x1, y1, x2, y2 = yolo_to_xyxy(box, w, h)
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=colors[idx % len(colors)],
                facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 4, "marginalia",
                    fontsize=8, color="white",
                    bbox=dict(facecolor=colors[idx % len(colors)], alpha=0.7, pad=1))

        ax.set_title(f"[{split}] {img_path.name}\n{len(boxes)} box(es)",
                     fontsize=9)
        ax.axis("off")

    plt.suptitle("Sample Annotations — YOLOv7_85-15 Marginalia Dataset",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = out_dir / "sample_annotations.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved sample annotations → {save_path}")


def augmentation_breakdown():
    print("AUGMENTATION ANALYSIS (train split)")
    print("=" * 60)
    img_dir = SPLITS["train"] / "images"
    if not img_dir.exists():
        print("  Train image directory not found.")
        return

    aug_types = Counter()
    for f in img_dir.iterdir():
        stem = f.stem
        if "-saturation" in stem:
            aug_types["saturation"] += 1
        elif "-noise" in stem:
            aug_types["noise"] += 1
        elif "-illumination" in stem:
            aug_types["illumination"] += 1
        elif "-contrast" in stem:
            aug_types["contrast"] += 1
        else:
            aug_types["original"] += 1

    for aug, count in sorted(aug_types.items(), key=lambda x: -x[1]):
        print(f"  {aug:<16} {count:>6}")
    print(f"  {'TOTAL':<16} {sum(aug_types.values()):>6}")
    print()

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    widths, heights, box_counts = compute_split_stats()
    augmentation_breakdown()
    plot_bbox_analysis(widths, heights, box_counts, OUTPUT_DIR)
    visualize_samples(n=6, out_dir=OUTPUT_DIR)

    print("\n✅ Dataset analysis complete! Results saved to:", OUTPUT_DIR)
