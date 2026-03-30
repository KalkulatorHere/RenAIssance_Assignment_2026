"""
Visualise Results — Marginalia Detection
=========================================
Generates side-by-side comparisons of ground truth annotations
vs YOLOv8 predictions on test images, plus training metric plots.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
DATASET_ROOT  = BASE_DIR.parent / "YOLOv7_85-15"
RESULTS_DIR   = BASE_DIR / "results" / "predictions"
TRAIN_RUN_DIR = BASE_DIR / "runs" / "detect" / "marginalia_1epoch"

CLASS_NAMES = {0: "marginalia"}
COLORS_GT   = "#2ECC71"   # green for ground truth
COLORS_PRED = "#E74C3C"   # red for prediction


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


def draw_boxes(ax, boxes_xyxy, color, label_prefix, scores=None):
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        txt = label_prefix
        if scores is not None and i < len(scores):
            txt += f" {scores[i]:.2f}"
        ax.text(x1, max(y1 - 5, 0), txt, fontsize=7, color="white",
                bbox=dict(facecolor=color, alpha=0.8, pad=1))


# ── 1. Prediction comparisons ─────────────────────────────────
def generate_prediction_comparisons(model_path: Path, n_images: int = 8):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    test_img_dir = DATASET_ROOT / "test" / "images"
    test_lbl_dir = DATASET_ROOT / "test" / "labels"

    img_files = sorted(test_img_dir.iterdir())[:n_images]

    for img_path in img_files:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img_np = np.array(img)

        # Ground truth
        lbl_path = test_lbl_dir / (img_path.stem + ".txt")
        gt_boxes = parse_yolo_label(lbl_path)
        gt_xyxy  = [yolo_to_xyxy(b, w, h) for b in gt_boxes]

        # Predictions
        results = model.predict(str(img_path), imgsz=640, conf=0.25, verbose=False)
        pred_boxes  = []
        pred_scores = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                pred_boxes.append((x1, y1, x2, y2))
                pred_scores.append(float(box.conf[0].cpu()))

        # Plot side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        ax1.imshow(img_np)
        draw_boxes(ax1, gt_xyxy, COLORS_GT, "GT")
        ax1.set_title(f"Ground Truth ({len(gt_xyxy)} boxes)", fontsize=12)
        ax1.axis("off")

        ax2.imshow(img_np)
        draw_boxes(ax2, pred_boxes, COLORS_PRED, "Pred", pred_scores)
        ax2.set_title(f"YOLOv8 Predictions ({len(pred_boxes)} boxes)", fontsize=12)
        ax2.axis("off")

        plt.suptitle(img_path.name, fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = RESULTS_DIR / f"comparison_{img_path.stem}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path.name}")


# ── 2. Grid of all test predictions ───────────────────────────
def generate_prediction_grid(model_path: Path):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    test_img_dir = DATASET_ROOT / "test" / "images"
    img_files = sorted(test_img_dir.iterdir())

    n = len(img_files)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, img_path in enumerate(img_files):
        ax = axes[idx]
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        w, h = img.size
        ax.imshow(img_np)

        results = model.predict(str(img_path), imgsz=640, conf=0.25, verbose=False)
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu())
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=COLORS_PRED, facecolor="none"
                )
                ax.add_patch(rect)
                ax.text(x1, max(y1 - 4, 0), f"{conf:.2f}", fontsize=6,
                        color="white",
                        bbox=dict(facecolor=COLORS_PRED, alpha=0.7, pad=0.5))

        ax.set_title(img_path.name, fontsize=7)
        ax.axis("off")

    # Hide remaining axes
    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Test Set Predictions — YOLOv8 (1 Epoch)", fontsize=16,
                 fontweight="bold")
    plt.tight_layout()
    save_path = RESULTS_DIR / "test_predictions_grid.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Locate best weights
    best = TRAIN_RUN_DIR / "weights" / "best.pt"
    if not best.exists():
        best = TRAIN_RUN_DIR / "weights" / "last.pt"
    if not best.exists():
        print("❌ No trained weights found! Run train.py first.")
        sys.exit(1)

    print(f"Using weights: {best}\n")

    print("1/2  Generating side-by-side comparisons...")
    generate_prediction_comparisons(best, n_images=8)

    print("\n2/2  Generating test prediction grid...")
    generate_prediction_grid(best)

    print(f"\n✅ All results saved to: {RESULTS_DIR}")
