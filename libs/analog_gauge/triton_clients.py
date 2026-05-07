import numpy as np
import cv2
import tritonclient.http as httpclient
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TritonSegmentationClient:
    def __init__(self, triton_url: str = "localhost:8000", model_name: str = "segmentation", model_version: str = "1"):
        self.triton_url = triton_url
        self.model_name = model_name
        self.model_version = model_version
        self.client = httpclient.InferenceServerClient(url=triton_url)

        # Check if model is ready
        if not self.client.is_model_ready(model_name, model_version):
            raise RuntimeError(f"Model {model_name}:{model_version} is not ready")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for YOLO segmentation model"""
        # Resize to 640x640, normalize to 0-1
        resized = cv2.resize(image, (640, 640))
        # Convert BGR to RGB and normalize
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize to 0-1 and change to CHW format
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        return chw

    def postprocess_results(self, outputs: List, original_shape: tuple) -> List[Dict[str, Any]]:
        """Convert Triton outputs back to segmentation format"""
        # This is a simplified version - actual postprocessing would be more complex
        # In real implementation, you'd need to decode YOLO outputs properly
        segmentations = []

        # Placeholder implementation - replace with actual YOLO postprocessing
        logger.warning("Triton segmentation postprocessing not fully implemented")
        return segmentations

    def get_segmentation(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run segmentation inference via Triton"""
        try:
            # Preprocess
            input_data = self.preprocess_image(image)
            input_tensor = httpclient.InferInput("images", input_data.shape, "FP32")
            input_tensor.set_data_from_numpy(input_data)

            # Run inference
            outputs = self.client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=[input_tensor]
            )

            # Postprocess
            result = self.postprocess_results(outputs, image.shape)
            return result

        except Exception as e:
            logger.error(f"Triton segmentation inference failed: {e}")
            return []


class TritonOCRClient:
    def __init__(self, triton_url: str = "localhost:8000", model_name: str = "ocr", model_version: str = "1",
                 vocab: str = " %./0123456789ABCFGHIKLMNOPRSW\\abcfghiklmnpqrsx\u00b0\u00b2\u0e08"):
        self.triton_url = triton_url
        self.model_name = model_name
        self.model_version = model_version
        self.vocab = vocab
        self.client = httpclient.InferenceServerClient(url=triton_url)

        if not self.client.is_model_ready(model_name, model_version):
            raise RuntimeError(f"Model {model_name}:{model_version} is not ready")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for Doctr OCR model"""
        # Resize to 32x128, convert to RGB, normalize
        resized = cv2.resize(image, (128, 32))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        return chw

    def postprocess_results(self, outputs: List) -> tuple:
        """Convert Triton outputs to text and confidence"""
        # Get output tensor
        output_data = outputs[0].as_numpy()

        # Simple argmax decoding (simplified - actual Doctr decoding is more complex)
        pred_indices = np.argmax(output_data, axis=1)[0]

        # Convert indices to characters
        text = ""
        for idx in pred_indices:
            if idx < len(self.vocab):
                text += self.vocab[idx]

        # Calculate confidence (simplified)
        confidence = float(np.max(output_data, axis=1).mean())

        return text.strip(), confidence

    def predict(self, image: np.ndarray) -> tuple:
        """Run OCR inference via Triton"""
        try:
            # Preprocess
            input_data = self.preprocess_image(image)
            input_tensor = httpclient.InferInput("input", input_data.shape, "FP32")
            input_tensor.set_data_from_numpy(input_data)

            # Run inference
            outputs = self.client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=[input_tensor]
            )

            # Postprocess
            text, confidence = self.postprocess_results(outputs)
            return text, confidence

        except Exception as e:
            logger.error(f"Triton OCR inference failed: {e}")
            return "", 0.0


class TritonSuperResolutionClient:
    def __init__(self, triton_url: str = "localhost:8000", model_name: str = "superresolution", model_version: str = "1"):
        self.triton_url = triton_url
        self.model_name = model_name
        self.model_version = model_version
        self.client = httpclient.InferenceServerClient(url=triton_url)

        if not self.client.is_model_ready(model_name, model_version):
            logger.warning(f"Model {model_name}:{model_version} is not ready - SR will be skipped")

    def get_superresolution(self, image: np.ndarray) -> np.ndarray:
        """Run super-resolution via Triton (placeholder - actual implementation needed)"""
        logger.warning("Triton super-resolution not implemented yet - returning original image")
        return image