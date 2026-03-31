# GSoC Marginalia Detection — YOLOv8 Fine-tuning Demo



Marginalia detection in early modern books using YOLO object detection.  
This is a reproduction and adaptation of [MRE-Detecting-Marginalia](../MRE-Detecting-Marginalia/) for a GSoC test submission.

## Dataset: YOLOv7_85-15

| Property | Value |
|---|---|
| Source | [Archaeology of Reading](https://archaeologyofreading.org/), [UCLA Clark Library (Calisphere)](https://calisphere.org/collections/26771/), [NLS Chapbooks](https://data.nls.uk/data/digitised-collections/chapbooks-printed-in-scotland/) |
| Annotation Tool | [RocketAnnotator](https://www.kaggle.com/datasets/chantalb/marginalia-training-data) |
| Format | YOLO (`class x_center y_center width height`, normalised 0–1) |
| Classes | 1 — `marginalia` |
| Train / Valid / Test | 1,585 / 39 / 17 images |
| Augmentations | saturation, noise, illumination, contrast |

### Bounding Box Format

Each `.txt` label file has one line per annotation:
```
<class_id> <x_center> <y_center> <width> <height>
```
- All coordinates are **normalised** to `[0, 1]` relative to image dimensions
- `x_center, y_center` = centre of the bounding box
- `width, height` = box dimensions
- Class `0` = marginalia (handwritten annotations in book margins)

## Original Approach (MRE)

The original notebook fine-tuned **YOLOv7-e6** for 150 epochs at 1280px using `train_aux.py` from the [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) repo, with data augmentation via [data_augmentation_yolov7](https://github.com/MinoruHenrique/data_augmentation_yolov7).

## Our Reproduction

We use **YOLOv8s** (Ultralytics) for simplicity and compatibility:
- Same YOLO label format — zero data conversion needed
- 1-epoch demo training on RTX 3050 Laptop GPU
- Image size 640px, batch size 8

## Quick Start

```bash
# 1. Install dependencies
pip install ultralytics matplotlib pillow numpy

# 2. Run dataset analysis
python dataset_analysis.py

# 3. Train 1 epoch
python train.py

# 4. Visualise results
python visualize_results.py
```

## Project Structure

```
GSoC_Marginalia_Detection/
├── data.yaml                 # Dataset config
├── dataset_analysis.py       # Dataset stats & visualisations
├── train.py                  # YOLOv8 training (1 epoch)
├── visualize_results.py      # Prediction comparisons
├── README.md
├── results/
│   ├── dataset_analysis/     # Stats plots & sample annotations
│   └── predictions/          # GT vs prediction comparisons
└── runs/
    └── detect/
        └── marginalia_1epoch/  # Training outputs & weights
```

## License

Educational use — GSoC test submission.
