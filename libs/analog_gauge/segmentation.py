import torch
from ultralytics import YOLO
import numpy as np
import os
from typing import List, Dict, Any

class GaugeSegmentor:
    def __init__(self, config: dict):
        self.config = config
        self.model_path = self.config.get('model_path', '')
        self.conf = self.config.get('conf', 0.5)
        self.iou = self.config.get('iou', 0.5)
        self.verbose = self.config.get('verbose', False)
        self.use_triton = self.config.get('use_triton', False)
        self.triton_url = self.config.get('triton_url', 'localhost:8000')

        requested_device = self.config.get('device', 0)
        if isinstance(requested_device, (int, list)) and not torch.cuda.is_available():
            print("Warning: GPU requested but CUDA is not available. Falling back to CPU for Segmentation.")
            self.device = 'cpu'
        else:
            self.device = requested_device

        self.model = None
        self.triton_client = None
        self._load_model()

    def _load_model(self):
        if self.use_triton:
            try:
                from .triton_clients import TritonSegmentationClient
                self.triton_client = TritonSegmentationClient(
                    triton_url=self.triton_url,
                    model_name='segmentation'
                )
                print(f"Successfully connected to Triton segmentation model at {self.triton_url}")
            except Exception as e:
                print(f"Error connecting to Triton segmentation: {e}")
                print("Falling back to local model...")
                self.use_triton = False

        if not self.use_triton:
            try:
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"Segmentation Model file not found at: {self.model_path}")

                self.model = YOLO(self.model_path, task='segment')
                print(f"Successfully loaded segmentation model from {self.model_path} on {self.device}")

            except Exception as e:
                print(f"Error during segmentation model loading: {e}")

    def get_segmentation(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        bbox, class และ mask (polygon points)
        """
        if self.use_triton and self.triton_client:
            return self.triton_client.get_segmentation(image)
        elif self.model is None:
            return []

        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=self.verbose,
            retina_masks=True,
            save=False
        )

        segmentations = []
        for result in results:
            if result.masks is None:
                continue

            try:
                xyxys = result.boxes.xyxy.cpu().numpy().astype(int)
                clss = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()

                masks_xy_list = list(result.masks.xy)
            except Exception as e:
                continue

            n = min(len(xyxys), len(masks_xy_list))
            for i in range(n):
                label = self.model.names[clss[i]]
                mask_points = masks_xy_list[i].astype(int)

                segmentations.append({
                    "bbox": xyxys[i].tolist(),
                    "mask": mask_points.tolist(),
                    "class": label,
                    "conf": float(confs[i]),
                })

        return segmentations

    def crop_object(self, image: np.ndarray, bbox: list) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return image[y1:y2, x1:x2]