import cv2
import numpy as np
from typing import List, Dict, Any

class BaseVisualizer:
    def __init__(self):
        self.colors = {} 
        self.palette = [
            (0, 0, 255),    
            (0, 255, 0),    
            (255, 0, 0),    
            (0, 255, 255),  
            (255, 0, 255),  
            (255, 255, 0),  
            (128, 0, 128),  
            (0, 165, 255),  
            (128, 128, 0),  
            (0, 128, 128),  
            (128, 0, 0),   
        ]
        self.palette_index = 0

    def _get_color(self, class_name: str) -> tuple:
        if class_name not in self.colors:
            color = self.palette[self.palette_index % len(self.palette)]
            self.palette_index += 1
            self.colors[class_name] = color
        return self.colors[class_name]

class DetectionVisualizer(BaseVisualizer):
    def draw(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        vis_image = image.copy()
        for det in detections:
            bbox = det['bbox']
            class_name = det['class']
            conf = det['conf']
            color = self._get_color(class_name)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis_image, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(vis_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return vis_image

class SegmentationVisualizer(BaseVisualizer):
    def draw(self, image: np.ndarray, segmentations: List[Dict[str, Any]]) -> np.ndarray:
        vis_image = image.copy()
        overlay = vis_image.copy()
        for seg in segmentations:
            mask_points = np.array(seg['mask'], dtype=np.int32)
            class_name = seg['class']
            conf = seg['conf']
            color = self._get_color(class_name)
            cv2.fillPoly(overlay, [mask_points], color)
            cv2.polylines(vis_image, [mask_points], True, color, 2)
            bbox = seg['bbox']
            x1, y1 = bbox[0], bbox[1]
            label = f"{class_name} {conf:.2f}"
            cv2.putText(vis_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        alpha = 0.4  
        cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0, vis_image)
        return vis_image

class EllipseVisualizer(BaseVisualizer):
    def draw(self, image: np.ndarray, ellipse_results: List[Dict[str, Any]]) -> np.ndarray:
        vis_image = image.copy()
        
        for item in ellipse_results:

            if 'opencv_params' not in item:
                continue
                
            params = item['opencv_params']
            class_name = item.get('class', 'unknown')
            angle = item.get('angle_deg', 0)
            color = self._get_color(class_name)
            
            cv2.ellipse(vis_image, params, color, 2)
            
            (xc, yc) = params[0]
            cv2.circle(vis_image, (int(xc), int(yc)), 4, (0, 0, 255), -1)
            
            # label = f"{class_name} {angle:.1f}deg"
            # cv2.putText(vis_image, label, (int(xc) + 10, int(yc)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return vis_image