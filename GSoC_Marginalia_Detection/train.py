"""
YOLOv8 Training Script — Marginalia Detection (1 Epoch Demo)
=============================================================
Fine-tunes YOLOv8s on the YOLOv7_85-15 marginalia dataset for 1 epoch.
Uses Ultralytics API with YOLO-format labels (fully compatible).

Hardware: NVIDIA RTX 3050 Laptop GPU (4 GB VRAM)
"""

from ultralytics import YOLO
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────
DATA_YAML  = Path(__file__).resolve().parent / "data.yaml"
PROJECT    = Path(__file__).resolve().parent / "runs" / "detect"
NAME       = "marginalia_1epoch"

MODEL      = "yolov8s.pt"        # small model — fits in 4 GB VRAM
EPOCHS     = 1                   # demo run
IMG_SIZE   = 640                 # safe for 4 GB VRAM
BATCH_SIZE = 8                   # conservative for RTX 3050
DEVICE     = 0                   # GPU 0
WORKERS    = 4                   # data-loading threads


def main():
    print("=" * 60)
    print("  MARGINALIA DETECTION — YOLOv8 FINE-TUNING (1 Epoch)")
    print("=" * 60)
    print(f"  Model     : {MODEL}")
    print(f"  Dataset   : {DATA_YAML}")
    print(f"  Epochs    : {EPOCHS}")
    print(f"  Image Size: {IMG_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print()

    # Load pre-trained YOLOv8s
    model = YOLO(MODEL)

    # Train for 1 epoch
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=str(PROJECT),
        name=NAME,
        exist_ok=True,
        pretrained=True,
        verbose=True,
        plots=True,          # save training plots
        save=True,           # save checkpoints
        val=True,            # run validation after each epoch
    )

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Results saved to: {PROJECT / NAME}")
    print()

    # ── Run validation on the test set ────────────────────────
    print("Running evaluation on TEST set...")
    best_weights = PROJECT / NAME / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = PROJECT / NAME / "weights" / "last.pt"

    model_eval = YOLO(str(best_weights))
    metrics = model_eval.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(PROJECT),
        name=NAME + "_test_eval",
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print("  TEST EVALUATION RESULTS")
    print("=" * 60)
    print(f"  mAP@50      : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95   : {metrics.box.map:.4f}")
    print(f"  Precision    : {metrics.box.mp:.4f}")
    print(f"  Recall       : {metrics.box.mr:.4f}")
    print("=" * 60)

    return results, metrics


if __name__ == "__main__":
    main()
