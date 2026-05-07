import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging


class Visualizer:

    def __init__(self, config: dict):
        self.logger = logging.getLogger("AIPipeline")
        self.font_path = config.get('system', {}).get('font_path', 'fonts/tahoma.ttf')
        self.fonts = {}

    def _get_font(self, size):
        if size not in self.fonts:
            try:
                self.fonts[size] = ImageFont.truetype(self.font_path, size)
            except IOError:
                self.fonts[size] = ImageFont.load_default()
        return self.fonts[size]

    def draw_label_bar(self, img_bgr, text, color=(0, 255, 0), bar_height=None):
     
        h, w = img_bgr.shape[:2]
        if bar_height is None:
            bar_height = max(24, int(h * 0.08))

        font_size = max(14, int(bar_height * 0.65))
        font = self._get_font(font_size)
        cv2.rectangle(img_bgr, (0, 0), (w, bar_height), (0, 0, 0), -1)
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = max(4, (w - tw) // 2)
        ty = (bar_height - th) // 2

        b, g, r = color
        draw.text((tx, ty), text, font=font, fill=(r, g, b))

        result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        np.copyto(img_bgr, result)

        return img_bgr

    def draw_text_fast(self, img_bgr, text, position,
                       font_scale=None, color=(0, 255, 0), thickness=None):

        h, w = img_bgr.shape[:2]

        if font_scale is None:
            font_scale = max(0.35, w / 1920.0)
        if thickness is None:
            thickness = max(1, int(font_scale * 2.5))

        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = position

        x = max(2, min(x, w - tw - 4))
        y = max(th + 4, min(y, h - baseline - 2))

        cv2.rectangle(
            img_bgr,
            (x - 2, y - th - 4),
            (x + tw + 4, y + baseline + 2),
            (0, 0, 0), -1
        )
        cv2.putText(img_bgr, text, (x, y), font, font_scale, color, thickness)
        return img_bgr

   
