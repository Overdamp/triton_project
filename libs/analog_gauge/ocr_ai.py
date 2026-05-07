import os
os.environ["LRU_CACHE_CAPACITY"] = "1"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import cv2
import json
import numpy as np
import torch
from torchvision.transforms.v2 import Resize, Compose, ToImage, ToDtype
from doctr.models import recognition

class DoctrOCR:
    def __init__(self, model_dir: str, device: str = "cpu", use_triton: bool = False, triton_url: str = "localhost:8000"):
        self.device = device
        self.model_dir = model_dir
        self.use_triton = use_triton
        self.triton_url = triton_url

        if use_triton:
            self._init_triton_client()
        else:
            self._init_local_model()

    def _init_triton_client(self):
        try:
            from .triton_clients import TritonOCRClient
            config_path = os.path.join(self.model_dir, "config.json")
            vocab = " %./0123456789ABCFGHIKLMNOPRSW\\abcfghiklmnpqrsx\u00b0\u00b2\u0e08"
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                    vocab = cfg.get("vocab", vocab)

            self.triton_client = TritonOCRClient(
                triton_url=self.triton_url,
                model_name='ocr',
                vocab=vocab
            )
            print(f"Successfully connected to Triton OCR model at {self.triton_url}")
        except Exception as e:
            print(f"Error connecting to Triton OCR: {e}")
            print("Falling back to local model...")
            self.use_triton = False
            self._init_local_model()

    def _init_local_model(self):
        config_path = os.path.join(self.model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")

        with open(config_path, "r") as f:
            self.cfg = json.load(f)

        self.vocab = self.cfg.get("vocab")
        self.input_size = tuple(self.cfg.get("INPUT_SIZE", [32, 128]))
        self.model_arch = self.cfg.get("MODEL_ARCH", "parseq")

        self.model = recognition.__dict__[self.model_arch](
            pretrained=False,
            vocab=self.vocab,
            input_shape=(3, self.input_size[0], self.input_size[1])
        )

        ckpt_path = os.path.join(self.model_dir, "best_model.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

        state_dict = torch.load(ckpt_path, map_location=self.device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(clean_state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transforms = Compose([
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Resize(self.input_size, antialias=True),
        ])

    def predict(self, img_bgr: np.ndarray):
        if self.use_triton and hasattr(self, 'triton_client'):
            return self.triton_client.predict(img_bgr)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = self.transforms(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor, target=None, return_preds=True)

        if "preds" in output:
            pred_text, conf = output['preds'][0]
        else:
            pred_text, conf = output[0]

        return pred_text, conf