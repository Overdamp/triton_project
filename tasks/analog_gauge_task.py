import logging
import traceback
import numpy as np
import cv2
from typing import Optional, Dict, Any, List

from libs.analog_gauge import (
    GaugeSegmentor,
    EllipseFitter,
    GaugeCalculator,
    DoctrOCR,
    GaugeScaleSuperResolution
)
from libs.analog_gauge.gauge_debug import GaugeDebugger
import concurrent.futures 
import time


class AnalogGaugeTask:


    def __init__(self, config: dict):
        self.logger = logging.getLogger("AIPipeline")
        self.config = config.get("analog_gauge", {})
        self.segmentor = None
        self.fitter = EllipseFitter()
        self.calculator = GaugeCalculator()
        self.ocr_model = None

        debug_cfg = self.config.get("debug", {})
        self.debug_enabled = debug_cfg.get("enabled", False)
        self.debug_output_dir = debug_cfg.get("output_dir", "debug_output")
        self.debugger = GaugeDebugger(
            output_dir=self.debug_output_dir,
            enabled=self.debug_enabled,
        )

        self._initialize_models()

    def _initialize_models(self):
        seg_cfg = self.config.get("segmentation", {})
        seg_cfg['use_triton'] = self.config.get("use_triton", False)
        seg_cfg['triton_url'] = self.config.get("triton_url", "localhost:8000")
        if seg_cfg:
            try:
                self.logger.info("[AnalogGaugeTask] Loading Segmentation model...")
                self.segmentor = GaugeSegmentor(seg_cfg)
                self.logger.info("[AnalogGaugeTask] Segmentation model loaded.")
            except Exception as e:
                self.logger.error(f"[AnalogGaugeTask] Failed to load Segmentor: {e}")
                self.logger.debug(traceback.format_exc())

        ocr_model_dir = self.config.get("ocr_model_dir", "")
        ocr_device = self.config.get("device", "cpu")
        ocr_use_triton = self.config.get("use_triton", False)
        ocr_triton_url = self.config.get("triton_url", "localhost:8000")
        if ocr_model_dir:
            try:
                self.ocr_model = DoctrOCR(model_dir=ocr_model_dir, device=ocr_device,
                                        use_triton=ocr_use_triton, triton_url=ocr_triton_url)
                self.logger.info("[AnalogGaugeTask] DoctrOCR loaded.")
            except Exception as e:
                self.logger.error(f"[AnalogGaugeTask] Failed to load DoctrOCR: {e}")

        sr_cfg= self.config.get("superresolution",{})
        if sr_cfg:
            try:
                self.logger.info("[AnalogGaugeTask] Loading SuperResolution model...")
                self.superresolution= GaugeScaleSuperResolution(sr_cfg)
            except Exception as e:
                self.logger.error(f"[AnalogGaugeTask] Failed to load SuperResolution: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, cropped_img: np.ndarray, filename: str = "") -> Optional[Dict[str, Any]]:

        debug_data = {
            "original_img": cropped_img.copy() if cropped_img is not None else None,
            "segmentations": [],
            "ellipse_results": [],
            "center": None,
            "ocr_results": [],
            "needle_scores": [],
            "calibration_data": {},
            "result": None,
            "unit": "",
        }

        if self.segmentor is None or cropped_img is None or cropped_img.size == 0:

            self._save_debug(filename, debug_data)
            return None

        try:
            # ===== Step 1: Segmentation =====
            segmentations = self.segmentor.get_segmentation(cropped_img)
            debug_data["segmentations"] = segmentations
            if not segmentations:
                self._save_debug(filename, debug_data)
                return None

            # ===== Step 2: Ellipse Fitting =====
            ellipse_result = self._get_component_ellipse(segmentations)
            debug_data["ellipse_results"] = ellipse_result

            # ===== Step 3: OCR =====
            ocr_results = self._run_ocr(cropped_img, segmentations)
            debug_data["ocr_results"] = ocr_results

            for ocr in ocr_results:
                self.logger.debug(
                    f"[OCR] cls={ocr['class']} text='{ocr['text']}' "
                    f"conf={ocr['confidence']:.2f} center={ocr.get('mask_center')}"
                )

            # ===== Step 4: Gauge Reading =====
            result = self._get_gauge_reading(
                cropped_img, segmentations, ellipse_result, ocr_results, debug_data
            )
            debug_data["result"] = result

            # ===== Step 5: Debug Report =====
            self._save_debug(filename, debug_data)

            return result

        except Exception as e:
            self.logger.error(f"[AnalogGaugeTask] Error: {e}")
            self.logger.debug(traceback.format_exc())
            self._save_debug(filename, debug_data)
            return None

    # ------------------------------------------------------------------
    # Internal: Ellipse Fitting
    # ------------------------------------------------------------------

    def _get_component_ellipse(self, segmentation):
        ellipse_results = []
        best_max, best_min, other = None, None, []
        for seg in segmentation:
            c = seg["class"]
            if c == "max-value":
                if best_max is None or seg["conf"] > best_max["conf"]: best_max = seg
            elif c == "min-value":
                if best_min is None or seg["conf"] > best_min["conf"]: best_min = seg
            elif c not in ["unit", "needle"]:
                other.append(seg)
        for seg in other + [s for s in [best_max, best_min] if s]:
            fd = self.fitter.fit(seg["mask"])
            if fd:
                r = seg.copy(); r.update(fd); ellipse_results.append(r)
        return ellipse_results

    # ------------------------------------------------------------------
    # Internal: OCR
    # ------------------------------------------------------------------

    def _preprocess_for_ocr(self, crop_bgr):
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(enhanced, -1, kernel)

    # def _run_ocr(self, frame, segmentations):
    #     target_classes = ["max-value", "min-value", "unit", "scale_number"]
    #     filtered = [s for s in segmentations if s["class"] in target_classes]

    #     best_max, best_min, others = None, None, []
    #     for seg in filtered:
    #         if seg["class"] == "max-value":
    #             if best_max is None or seg["conf"] > best_max["conf"]: best_max = seg
    #         elif seg["class"] == "min-value":
    #             if best_min is None or seg["conf"] > best_min["conf"]: best_min = seg
    #         else:
    #             others.append(seg)

    #     final = others + [s for s in [best_max, best_min] if s]
    #     ocr_results = []
    #     img_h, img_w = frame.shape[:2]
    #     i=0

    #     for seg in final:
    #         mask_points = np.array(seg["mask"])
    #         mask_center = None
    #         if len(mask_points) > 0:
    #             M = np.mean(mask_points, axis=0)
    #             mask_center = (int(M[0]), int(M[1]))

    #         # ใช้ YOLO bbox
    #         bbox = seg.get("bbox")
    #         if bbox:
    #             bx1, by1, bx2, by2 = bbox
    #         else:
    #             x, y, w, h = cv2.boundingRect(mask_points)
    #             bx1, by1, bx2, by2 = x, y, x+w, y+h

    #         w_box, h_box = bx2 - bx1, by2 - by1
    #         pad_x = max(10, int(w_box * 0.30))
    #         pad_y = max(10, int(h_box * 0.30))
    #         x1 = max(0, bx1 - pad_x)
    #         y1 = max(0, by1 - pad_y)
    #         x2 = min(img_w, bx2 + pad_x)
    #         y2 = min(img_h, by2 + pad_y)

    #         crop_img = frame[y1:y2, x1:x2].copy()
           
            
    #         crop_img = self._preprocess_for_ocr(crop_img)
            
    #         import time
    #         # resize_crop = cv2.resize(crop_img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    #         if hasattr(self, 'superresolution') and self.superresolution is not None:
    #             try:
    #                 start_time=time.time()
    #                 resize_crop = self.superresolution.get_superresolution(crop_img)
    #                 cv2.imwrite(f"/home/engineer00/winworkspace/PTTEP/AnalogGaugeReading/gauge-win-v4/debug_ocr/crop_{i}.png",resize_crop)
    #                 print("process_time",time.time()-start_time,"s")
    #             except Exception as e:
    #                 self.logger.warning(f"[OCR] SuperResolution failed for class {seg['class']}: {e}")
    #         i+=1

    #         text, conf = "", 0.0
    #         if self.ocr_model:
    #             try:
    #                 text, conf = self.ocr_model.predict(resize_crop)
    #             except Exception as e:
    #                 self.logger.debug(f"[OCR] predict failed: {e}")

    #         ocr_results.append({
    #             "class": seg["class"], "text": text,
    #             "confidence": float(conf), "mask_center": mask_center,
    #         })
    #     return ocr_results
    
    ###### case 2
    # def _run_ocr(self, frame, segmentations):
    #     target_classes = ["max-value", "min-value", "unit", "scale_number"]
    #     filtered = [s for s in segmentations if s["class"] in target_classes]

    #     best_max, best_min, others = None, None, []
    #     for seg in filtered:
    #         if seg["class"] == "max-value":
    #             if best_max is None or seg["conf"] > best_max["conf"]: best_max = seg
    #         elif seg["class"] == "min-value":
    #             if best_min is None or seg["conf"] > best_min["conf"]: best_min = seg
    #         else:
    #             others.append(seg)

    #     final = others + [s for s in [best_max, best_min] if s]
    #     ocr_results = []
    #     img_h, img_w = frame.shape[:2]

    #     # ฟังก์ชันย่อยสำหรับประมวลผลแต่ละ Crop ภาพ
    #     def process_single_crop(seg):
    #         mask_points = np.array(seg["mask"])
    #         mask_center = None
    #         if len(mask_points) > 0:
    #             M = np.mean(mask_points, axis=0)
    #             mask_center = (int(M[0]), int(M[1]))

    #         bbox = seg.get("bbox")
    #         if bbox:
    #             bx1, by1, bx2, by2 = bbox
    #         else:
    #             x, y, w, h = cv2.boundingRect(mask_points)
    #             bx1, by1, bx2, by2 = x, y, x+w, y+h

    #         w_box, h_box = bx2 - bx1, by2 - by1
    #         pad_x = max(10, int(w_box * 0.30))
    #         pad_y = max(10, int(h_box * 0.30))
    #         x1 = max(0, bx1 - pad_x)
    #         y1 = max(0, by1 - pad_y)
    #         x2 = min(img_w, bx2 + pad_x)
    #         y2 = min(img_h, by2 + pad_y)

    #         crop_img = frame[y1:y2, x1:x2].copy()
    #         crop_img = self._preprocess_for_ocr(crop_img)
            
        
    #         processed_crop = crop_img
            
    #         if hasattr(self, 'superresolution') and self.superresolution is not None:
    #             try:
    #                 # start_time = time.time()
             
    #                 processed_crop = self.superresolution.get_superresolution(crop_img)
    #                 # print("process_time",time.time()-start_time,"s")
                    
    #                 # cv2.imwrite(f"/home/engineer00/winworkspace/PTTEP/AnalogGaugeReading/gauge-win-v4/debug_ocr/crop_x.png",processed_crop)
    #                 # self.logger.debug(f"[OCR] SR process_time: {time.time() - start_time:.3f}s")
    #             except Exception as e:
    #                 self.logger.warning(f"[OCR] SuperResolution failed for class {seg['class']}: {e}")
                   
    #                 processed_crop = cv2.resize(crop_img, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    #         else:
               
    #             processed_crop = cv2.resize(crop_img, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    #         # --- OCR ---
    #         text, conf = "", 0.0
    #         if self.ocr_model:
    #             try:
    #                 text, conf = self.ocr_model.predict(processed_crop)
    #             except Exception as e:
    #                 self.logger.debug(f"[OCR] predict failed: {e}")

    #         return {
    #             "class": seg["class"], 
    #             "text": text,
    #             "confidence": float(conf), 
    #             "mask_center": mask_center,
    #         }

        
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            
    #         futures = [executor.submit(process_single_crop, seg) for seg in final]
            
    #         for future in concurrent.futures.as_completed(futures):
    #             try:
    #                 result = future.result()
    #                 ocr_results.append(result)
    #             except Exception as exc:
    #                 self.logger.error(f"[OCR] Parallel processing generated an exception: {exc}")

    #     return ocr_results

    ####  Batch OCR
    def _run_ocr(self, frame, segmentations):
        
        target_classes = ["max-value", "min-value", "unit", "scale_number"]
        filtered = [s for s in segmentations if s["class"] in target_classes]

        best_max, best_min, others = None, None, []
        for seg in filtered:
            if seg["class"] == "max-value":
                if best_max is None or seg["conf"] > best_max["conf"]: best_max = seg
            elif seg["class"] == "min-value":
                if best_min is None or seg["conf"] > best_min["conf"]: best_min = seg
            else:
                others.append(seg)

        final = others + [s for s in [best_max, best_min] if s]
        if not final:
            return []

        img_h, img_w = frame.shape[:2]
        
        crop_images_list = []
        seg_metadata_list = []

        for seg in final:
            mask_points = np.array(seg["mask"])
            mask_center = None
            if len(mask_points) > 0:
                M = np.mean(mask_points, axis=0)
                mask_center = (int(M[0]), int(M[1]))

            bbox = seg.get("bbox")
            if bbox:
                bx1, by1, bx2, by2 = bbox
            else:
                x, y, w, h = cv2.boundingRect(mask_points)
                bx1, by1, bx2, by2 = x, y, x+w, y+h

            w_box, h_box = bx2 - bx1, by2 - by1
            pad_x = max(10, int(w_box * 0.30))
            pad_y = max(10, int(h_box * 0.30))
            x1 = max(0, bx1 - pad_x)
            y1 = max(0, by1 - pad_y)
            x2 = min(img_w, bx2 + pad_x)
            y2 = min(img_h, by2 + pad_y)

            crop_img = frame[y1:y2, x1:x2].copy()
            crop_img = self._preprocess_for_ocr(crop_img)
            crop_img=cv2.resize(crop_img, (0, 0), fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
            
            crop_images_list.append(crop_img)
            
            seg_metadata_list.append({
                "class": seg["class"],
                "mask_center": mask_center
            })


        sr_images = []
        if hasattr(self, 'superresolution') and self.superresolution is not None:
            try:
                start_time = time.time()
                sr_images = self.superresolution.get_superresolution_batch(crop_images_list)
                # print("process_time",time.time()-start_time,"s")      
                # for idx, sr_img in enumerate(sr_images):
                #     save_path = f"/home/engineer00/winworkspace/PTTEP/AnalogGaugeReading/gauge-win-v4/debug_ocr/crop_batch_{idx}.png"
                #     cv2.imwrite(save_path, sr_img)
                # print(f"[OCR] Batch SuperResolution processed {len(crop_images_list)} images in {time.time()-start_time:.3f}s")
            except Exception as e:
                self.logger.warning(f"[OCR] Batch SuperResolution failed: {e}. Fallback to cv2.resize.")
                sr_images = [cv2.resize(img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC) for img in crop_images_list]
        else:

            sr_images = [cv2.resize(img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC) for img in crop_images_list]


        ocr_results = []
        

        for i, sr_img in enumerate(sr_images):
            text, conf = "", 0.0
            if self.ocr_model:
                try:
                    # ส่งภาพเข้า DoctrOCR
                    text, conf = self.ocr_model.predict(sr_img)
                except Exception as e:
                    self.logger.debug(f"[OCR] predict failed for image {i}: {e}")

            ocr_results.append({
                "class": seg_metadata_list[i]["class"], 
                "text": text,
                "confidence": float(conf), 
                "mask_center": seg_metadata_list[i]["mask_center"],
            })

        return ocr_results

    # ------------------------------------------------------------------
    # Internal: Needle Selection (anti-shadow)
    # ------------------------------------------------------------------

    def _select_best_needle(self, needle_segs, center, debug_data):

        if len(needle_segs) == 1:
            debug_data["needle_scores"] = [{
                "idx": 0, "min_dist": 0, "max_dist": 0,
                "score": 0, "selected": True,
            }]
            return needle_segs[0]

        best_seg, best_score = None, -1
        scores = []

        for i, seg in enumerate(needle_segs):
            pts = np.array(seg["mask"])
            if len(pts) == 0: continue
            dists = np.sqrt((pts[:, 0]-center[0])**2 + (pts[:, 1]-center[1])**2)
            min_d, max_d = float(np.min(dists)), float(np.max(dists))
            score = max_d / (min_d + 1.0) * (0.5 + seg["conf"])

            scores.append({
                "idx": i, "min_dist": min_d, "max_dist": max_d,
                "score": score, "selected": False,
            })

            if score > best_score:
                best_score = score
                best_seg = seg
                best_idx = len(scores) - 1

        if scores:
            scores[best_idx]["selected"] = True
        debug_data["needle_scores"] = scores

        self.logger.debug(
            f"[Needle] {len(needle_segs)} candidates, selected #{scores[best_idx]['idx'] if scores else '?'} "
            f"(score={best_score:.1f})"
        )
        return best_seg

    # ------------------------------------------------------------------
    # Internal: Unit Detection
    # ------------------------------------------------------------------

    def _detect_unit(self, ocr_results):
        for ocr in ocr_results:
            if ocr["class"] == "unit" and ocr["confidence"] > 0.5:
                clean = "".join(c for c in ocr["text"] if c.isalpha() or c in "°/%")
                if clean: return clean
        for ocr in ocr_results:
            if ocr["class"] in ["max-value", "min-value", "scale_number"]:
                alpha = "".join(c for c in ocr["text"] if c.isalpha())
                if len(alpha) >= 2: return alpha
        return ""

    # ------------------------------------------------------------------
    # Internal: Gauge Reading
    # ------------------------------------------------------------------

    def _get_gauge_reading(self, crop_img, segmentations, ellipse_result, ocr_results, debug_data):
        if not ellipse_result: return None

        # Center
        center, best_ellipse = None, None
        for cls_list in [["centre", "center"], ["gauge"]]:
            cands = [r for r in ellipse_result if r["class"] in cls_list]
            if cands:
                best = max(cands, key=lambda x: x.get("conf", 0))
                center, best_ellipse = best["center"], best
                break
        if center is None and ellipse_result:
            center, best_ellipse = ellipse_result[0]["center"], ellipse_result[0]
        if center is None: return None

        debug_data["center"] = center

        # Needle (anti-shadow)
        needle_mask = []
        needle_cands = [s for s in segmentations if s["class"] == "needle"]
        if needle_cands:
            best_needle = self._select_best_needle(needle_cands, center, debug_data)
            if best_needle:
                needle_mask = best_needle["mask"]

    
        unit = self._detect_unit(ocr_results)
        debug_data["unit"] = unit

    
        result = self.calculator.process_gauge(center, needle_mask, ocr_results, best_ellipse)

        if result and "_calibration_debug" in result:
            debug_data["calibration_data"] = result.pop("_calibration_debug")

        if result and result.get("r2_score", 0) < 0.5:
            self.logger.warning(
                f"[AnalogGaugeTask] Low R²={result['r2_score']:.3f} "
                f"pts={result.get('fit_points', 0)}"
            )

       
        debug_img = self._draw_debug(crop_img, center, result, unit)

        if result:
            result["unit"] = unit
            result["debug_img"] = debug_img
            return result

        return {"debug_img": debug_img} if debug_img is not None else None

    # ------------------------------------------------------------------
    # Debug Report
    # ------------------------------------------------------------------

    def _save_debug(self, filename: str, debug_data: Dict):
        
        if not self.debug_enabled or not filename:
            return
        try:
            path = self.debugger.generate_report(filename, debug_data)
            if path:
                self.logger.info(f"[Debug] Report saved: {path}")
        except Exception as e:
            self.logger.warning(f"[Debug] Failed to save report: {e}")

    # ------------------------------------------------------------------
    # Legacy Debug Image
    # ------------------------------------------------------------------

    def _draw_debug(self, crop_img, center, result, unit):
        W, H = 400, 400
        vis = cv2.resize(crop_img.copy(), (W, H))
        oh, ow = crop_img.shape[:2]
        sx, sy = W / ow, H / oh
        cx, cy = int(center[0]*sx), int(center[1]*sy)
        f = cv2.FONT_HERSHEY_SIMPLEX

        cv2.circle(vis, (cx, cy), 6, (0,0,255), -1)
        cv2.circle(vis, (cx, cy), 3, (255,255,255), -1)

        if result and "value" in result:
            nt = (int(result["needle_tip"][0]*sx), int(result["needle_tip"][1]*sy))
            cv2.line(vis, (cx, cy), nt, (0,0,255), 2)
            u = f" {unit}" if unit else ""
            txt = f"Read: {result['value']:.1f}{u}"
            (tw,th),_ = cv2.getTextSize(txt, f, 0.7, 2)
            p = (10, th+10)
            cv2.rectangle(vis, (p[0]-2,p[1]-th-5),(p[0]+tw+2,p[1]+3),(255,255,255),-1)
            r2 = result.get('r2_score', 0)
            clr = (0,150,0) if r2>=0.8 else (0,165,255) if r2>=0.5 else (0,0,255)
            cv2.putText(vis, txt, p, f, 0.7, clr, 2)
            info = f"R2:{r2:.2f} | pts:{result.get('fit_points','?')} | {'CW' if result['slope']>0 else 'CCW'}"
            (dw,dh),_ = cv2.getTextSize(info, f, 0.45, 1)
            dp = (10, p[1]+dh+10)
            cv2.rectangle(vis, (dp[0]-2,dp[1]-dh-5),(dp[0]+dw+2,dp[1]+3),(255,255,255),-1)
            cv2.putText(vis, info, dp, f, 0.45, (0,0,0), 1)
        else:
            cv2.putText(vis, "Calc Failed", (10,30), f, 0.7, (0,0,255), 2)
        return vis