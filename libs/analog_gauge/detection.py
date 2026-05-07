from ultralytics import YOLO
import numpy as np
import os
from typing import List, Dict, Any
import torch

class GaugeDetector:
    def __init__(self, config: dict):
        self.config = config
        self.model_path = self.config.get('model_path', '')
        self.conf = self.config.get('conf', 0.5)
        self.iou = self.config.get('iou', 0.5)
        self.verbose = self.config.get('verbose',False)
        requested_device = self.config.get('device', 0)
        if isinstance(requested_device, (int, list)) and not torch.cuda.is_available():
            print("Warning: GPU requested but CUDA is not available. Falling back to CPU.")
            self.device = 'cpu'
        else:
            self.device = requested_device
        self.model = None
        self._load_model()

    def _load_model(self):

        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")

            self.model = YOLO(self.model_path,
                              task= 'detect')
            print(f"Successfully loaded model from {self.model_path} on {self.device}")
            
        except Exception as e:
            print(f"Error during model loading: {e}")

    def get_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        
        if self.model is None: return []
        
        results = self.model.predict(source=image,
                                     device = self.device, 
                                     conf=self.conf,
                                     iou=self.iou,
                                     verbose=self.verbose)
        detections = []
        
        for result in results:
            for box in result.boxes:
                
                xyxy = box.xyxy[0].cpu().numpy().astype(int) # [x1, y1, x2, y2]
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                
                detections.append({
                    "bbox": xyxy.tolist(),
                    "class": label,
                    "conf": float(box.conf[0])
                })
        return detections

    def crop_image(self, image: np.ndarray, bbox: list) -> np.ndarray:
        x1, y1, x2, y2 = bbox
       
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return image[y1:y2, x1:x2]