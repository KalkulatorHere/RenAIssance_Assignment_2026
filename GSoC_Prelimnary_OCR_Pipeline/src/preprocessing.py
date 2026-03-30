import cv2
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

from config import SCORE_THRESHOLD, AREA_THRESHOLD_PERCENT, MIN_ASPECT_RATIO, SPLIT_RATIO

class TextlineExtractor:
    def __init__(self, model_path: str):
        self.cfg = self.setup_cfg(model_path)
        self.predictor = DefaultPredictor(self.cfg)
        
    def setup_cfg(self, model_path):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
        cfg.MODEL.WEIGHTS = model_path
        cfg.DATASETS.TEST = ("page_test",)
        cfg.DATALOADER.NUM_WORKERS = 2
        MetadataCatalog.get("page_test").thing_classes = ["textline", "baseline"]
        return cfg
    
    def calculate_dynamic_padding(self, boxes, image_shape):
        if len(boxes) < 2: return {"top": 10, "bottom": 10, "left": 8, "right": 8}
        
        centers = np.array([[(x1+x2)/2, (y1+y2)/2] for (x1,y1,x2,y2) in boxes])
        vertical_distances, horizontal_distances = [], []
        
        sorted_indices = np.argsort(centers[:, 1])
        sorted_boxes = boxes[sorted_indices]
        
        for i in range(len(sorted_boxes) - 1):
            curr, nxt = sorted_boxes[i], sorted_boxes[i + 1]
            if abs((curr[0]+curr[2])/2 - (nxt[0]+nxt[2])/2) < image_shape[1] * 0.3:
                gap = nxt[1] - curr[3]
                if gap > 0: vertical_distances.append(gap)
                
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                box1, box2 = boxes[i], boxes[j]
                if abs((box1[1]+box1[3])/2 - (box2[1]+box2[3])/2) < (box1[3]-box1[1]) * 0.8:
                    gap = min(box2[0]-box1[2] if box1[2]<box2[0] else float('inf'),
                              box1[0]-box2[2] if box2[2]<box1[0] else float('inf'))
                    if gap > 0 and gap != float('inf'): horizontal_distances.append(gap)
                    
        avg_v = np.median(vertical_distances) if vertical_distances else 20
        avg_h = np.median(horizontal_distances) if horizontal_distances else 15
        
        vp = max(5, min(25, avg_v / 2))
        hp = max(3, min(20, avg_h / 3))
        
        avg_height = np.mean([b[3]-b[1] for b in boxes])
        vp = max(vp, avg_height * max(0.1, min(0.3, avg_height / 100)))
        
        return {"top": int(vp * 0.95), "bottom": int(vp * 1.2), "left": int(hp), "right": int(hp)}

    def filter_margin_boxes_by_area(self, boxes, scores):
        if len(boxes) == 0: return np.array([]), np.array([]), np.array([]), np.array([])
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        thresh = np.mean(areas) * (AREA_THRESHOLD_PERCENT / 100.0)
        
        main_idx = areas >= thresh
        margin_idx = ~main_idx
        return boxes[main_idx], scores[main_idx], boxes[margin_idx], scores[margin_idx]
    
    def detect_columns_and_sort_reading_order(self, boxes, scores):
        if len(boxes) == 0: return boxes, scores, []
        centers = np.array([[(x1+x2)/2, (y1+y2)/2] for (x1,y1,x2,y2) in boxes])
        y_sort_indices = np.argsort(centers[:, 1])
        
        reading_order = [
            {'original_index': int(idx), 'column': 0, 'position_in_column': int(pos), 'reading_order_index': int(pos)}
            for pos, idx in enumerate(y_sort_indices)
        ]
        return boxes[y_sort_indices], scores[y_sort_indices], reading_order
        
    def extract_textlines(self, image):
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")
        mask = instances.pred_classes == 0
        return instances.pred_boxes[mask].tensor.numpy(), instances.scores[mask].numpy(), outputs
        
    def crop_textlines_with_dynamic_padding(self, image, boxes, use_margin_filtering=True):
        if len(boxes) == 0: return [], [], {}
        
        boxes_for_pad = self.filter_margin_boxes_by_area(boxes, np.ones(len(boxes)))[0] if use_margin_filtering else boxes
        if len(boxes_for_pad) == 0: boxes_for_pad = boxes
            
        padding = self.calculate_dynamic_padding(boxes_for_pad, image.shape)
        h, w = image.shape[:2]
        
        cropped, padded = [], []
        for (x1, y1, x2, y2) in boxes.astype(int):
            x1p, y1p = max(0, x1 - padding["left"]), max(0, y1 - padding["top"])
            x2p, y2p = min(w, x2 + padding["right"]), min(h, y2 + padding["bottom"])
            cropped.append(image[y1p:y2p, x1p:x2p])
            padded.append([x1p, y1p, x2p, y2p])
            
        return cropped, padded, padding

    def should_split(self, image: np.ndarray) -> bool:
        h, w = image.shape[:2]
        return (w / h) >= MIN_ASPECT_RATIO

    def split_image(self, image: np.ndarray):
        h, w = image.shape[:2]
        split_x = int(w * SPLIT_RATIO)
        return image[:, :split_x], image[:, split_x:], split_x


def detect_text_regions(extractor: TextlineExtractor, image: np.ndarray):
    """Convenience wrapper for the entire extraction -> filtering -> cropping logic."""
    boxes, scores, _ = extractor.extract_textlines(image)
    if len(boxes) == 0: return np.array([]), np.array([]), [], [], {}
    
    f_boxes, f_scores, _, _ = extractor.filter_margin_boxes_by_area(boxes, scores)
    if len(f_boxes) == 0: return np.array([]), np.array([]), [], [], {}
        
    o_boxes, o_scores, ro_info = extractor.detect_columns_and_sort_reading_order(f_boxes, f_scores)
    crops, p_boxes, split_pad = extractor.crop_textlines_with_dynamic_padding(image, o_boxes, False)
    
    return o_boxes, o_scores, ro_info, crops, split_pad
