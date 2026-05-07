import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import math


class GaugeDebugger:

    PW, PH = 400, 400
    
    CLASS_COLORS = {
        "needle":      (0, 0, 255),      
        "gauge":       (255, 200, 0),     
        "centre":      (0, 255, 255),     
        "center":      (0, 255, 255),     
        "max-value":   (0, 165, 255),     
        "min-value":   (0, 255, 0),       
        "scale_number":(255, 0, 255),     
        "unit":        (255, 255, 0),     
    }
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, output_dir: str = "debug_output", enabled: bool = True):
        self.output_dir = output_dir
        self.enabled = enabled
        if enabled:
            os.makedirs(output_dir, exist_ok=True)

    def _get_color(self, cls_name: str) -> Tuple[int, int, int]:
        return self.CLASS_COLORS.get(cls_name, (200, 200, 200))

    def _make_panel(self, title: str, bg_color=(40, 40, 40)) -> np.ndarray:
        
        panel = np.full((self.PH, self.PW, 3), bg_color, dtype=np.uint8)
        cv2.putText(panel, title, (8, 22), self.FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(panel, (0, 28), (self.PW, 28), (100, 100, 100), 1)
        return panel

    def _resize_to_panel(self, img: np.ndarray) -> np.ndarray:
        
        h, w = img.shape[:2]
        avail_h = self.PH - 32
        avail_w = self.PW - 4
        scale = min(avail_w / w, avail_h / h)
        nw, nh = int(w * scale), int(h * scale)
        return cv2.resize(img, (nw, nh)), scale

    def _paste_on_panel(self, panel: np.ndarray, img: np.ndarray, y_offset: int = 32):
        
        resized, scale = self._resize_to_panel(img)
        rh, rw = resized.shape[:2]
        x_off = (self.PW - rw) // 2
        panel[y_offset:y_offset+rh, x_off:x_off+rw] = resized
        return scale, x_off, y_offset

    # ==================================================================
    # Panel 1: Original Image
    # ==================================================================
    def _panel_original(self, img: np.ndarray) -> np.ndarray:
        panel = self._make_panel("1. Original Input")
        self._paste_on_panel(panel, img)
        h, w = img.shape[:2]
        cv2.putText(panel, f"{w}x{h}", (8, self.PH - 10), self.FONT, 0.4, (150, 150, 150), 1)
        return panel

    # ==================================================================
    # Panel 2: Segmentation Overlay
    # ==================================================================
    def _panel_segmentation(self, img: np.ndarray, segmentations: List[Dict]) -> np.ndarray:
        panel = self._make_panel("2. Segmentation")
        vis = img.copy()
        overlay = vis.copy()

        for seg in segmentations:
            pts = np.array(seg["mask"], dtype=np.int32)
            cls = seg["class"]
            conf = seg["conf"]
            color = self._get_color(cls)

            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(vis, [pts], True, color, 2)

            M = np.mean(pts, axis=0).astype(int)
            label = f"{cls} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, self.FONT, 0.35, 1)
            cv2.rectangle(vis, (M[0]-1, M[1]-th-2), (M[0]+tw+1, M[1]+2), (0, 0, 0), -1)
            cv2.putText(vis, label, (M[0], M[1]), self.FONT, 0.35, color, 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
        self._paste_on_panel(panel, vis)

        y = self.PH - 12
        x = 8
        classes_found = list(set(s["class"] for s in segmentations))
        for cls in classes_found:
            color = self._get_color(cls)
            cv2.circle(panel, (x, y), 4, color, -1)
            cv2.putText(panel, cls, (x + 8, y + 4), self.FONT, 0.3, (200, 200, 200), 1)
            x += 10 + len(cls) * 7
            if x > self.PW - 40:
                break

        return panel

    # ==================================================================
    # Panel 3: Ellipse Fitting
    # ==================================================================
    def _panel_ellipse(self, img: np.ndarray, ellipse_results: List[Dict],
                       center: Optional[Tuple] = None) -> np.ndarray:
        panel = self._make_panel("3. Ellipse Fitting")
        vis = img.copy()

        for item in ellipse_results:
            if 'opencv_params' not in item:
                continue
            params = item['opencv_params']
            cls = item.get('class', '?')
            color = self._get_color(cls)

            try:
                cv2.ellipse(vis, params, color, 2)
            except Exception:
                pass

            xc, yc = params[0]
            cv2.circle(vis, (int(xc), int(yc)), 4, color, -1)
            label = f"{cls}"
            cv2.putText(vis, label, (int(xc)+6, int(yc)-6), self.FONT, 0.35, color, 1)

        # Real center 
        if center:
            cx, cy = int(center[0]), int(center[1])
            cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(vis, f"CENTER ({cx},{cy})", (cx+10, cy+5), self.FONT, 0.4, (0, 0, 255), 1)

        self._paste_on_panel(panel, vis)

        info = f"Found {len(ellipse_results)} ellipses"
        cv2.putText(panel, info, (8, self.PH-10), self.FONT, 0.35, (150,150,150), 1)
        return panel

    # ==================================================================
    # Panel 4: OCR Results (crop grid + text)
    # ==================================================================
    def _panel_ocr(self, img: np.ndarray, segmentations: List[Dict],
                   ocr_results: List[Dict]) -> np.ndarray:
    
        panel = self._make_panel("4. OCR Results (All Detections)")

        target_classes = ["max-value", "min-value", "unit", "scale_number"]
        ocr_segs = [s for s in segmentations if s["class"] in target_classes]

        if not ocr_results:
            cv2.putText(panel, "No OCR results", (20, 100), self.FONT, 0.5, (100, 100, 255), 1)
            return panel

        n = len(ocr_results)
        cols = min(4, n)
        rows = math.ceil(n / cols)

        cell_w = (self.PW - 8) // cols
        cell_h = min(80, (self.PH - 40) // max(rows, 1))

        for idx, ocr in enumerate(ocr_results):
            r, c = divmod(idx, cols)
            x0 = 4 + c * cell_w
            y0 = 34 + r * cell_h

            cls = ocr["class"]
            text = ocr["text"]
            conf = ocr["confidence"]
            color = self._get_color(cls)

            matching_seg = None
            ocr_center = ocr.get("mask_center")
            
            if ocr_center:
                
                min_dist = float('inf')
                for seg in ocr_segs:
                    if seg["class"] != cls:
                        continue
                    seg_mask = np.array(seg["mask"])
                    if len(seg_mask) == 0:
                        continue
                    seg_center = np.mean(seg_mask, axis=0)
                    dist = np.sqrt((ocr_center[0] - seg_center[0])**2 + 
                                 (ocr_center[1] - seg_center[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        matching_seg = seg
            
            
            if not matching_seg:
                for seg in ocr_segs:
                    if seg["class"] == cls:
                        matching_seg = seg
                        break
                    
            if matching_seg and img is not None:
                bbox = matching_seg.get("bbox")
                if bbox:
                    bx1, by1, bx2, by2 = [max(0, v) for v in bbox]
                    by2 = min(by2, img.shape[0])
                    bx2 = min(bx2, img.shape[1])
                    crop = img[by1:by2, bx1:bx2]
                    if crop.size > 0:
                        crop_h = cell_h - 20
                        crop_w = cell_w - 4
                        if crop_h > 0 and crop_w > 0:
                            scale = min(crop_w / max(crop.shape[1], 1), crop_h / max(crop.shape[0], 1))
                            nw = max(1, int(crop.shape[1] * scale))
                            nh = max(1, int(crop.shape[0] * scale))
                            mini = cv2.resize(crop, (nw, nh))
                            ey = min(y0 + nh, panel.shape[0])
                            ex = min(x0 + nw, panel.shape[1])
                            panel[y0:ey, x0:ex] = mini[:ey-y0, :ex-x0]

            info_y = y0 + cell_h - 6
            if info_y < self.PH:
                cv2.putText(panel, f"{cls[:8]}", (x0, info_y - 10), self.FONT, 0.28, color, 1)
                status_color = (0, 255, 0) if conf > 0.7 else (0, 200, 255) if conf > 0.4 else (0, 0, 255)
                cv2.putText(panel, f"'{text}' {conf:.2f}", (x0, info_y), self.FONT, 0.28, status_color, 1)

            cv2.rectangle(panel, (x0-1, y0-1), (x0+cell_w-2, y0+cell_h-2), color, 1)

        cv2.putText(panel, f"Total: {n} detections (may include duplicates - see Panel 6 for cleaned)", 
                   (8, self.PH-10), self.FONT, 0.3, (150,150,150), 1)

        return panel

    # ==================================================================
    # Panel 5: Needle Selection
    # ==================================================================
    def _panel_needle(self, img: np.ndarray, segmentations: List[Dict],
                      center: Optional[Tuple], needle_scores: List[Dict] = None) -> np.ndarray:
        panel = self._make_panel("5. Needle Selection")
        vis = img.copy()

        needle_segs = [s for s in segmentations if s["class"] == "needle"]

        if center:
            cx, cy = int(center[0]), int(center[1])
            cv2.drawMarker(vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 15, 1)

        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)]
        for i, seg in enumerate(needle_segs):
            pts = np.array(seg["mask"], dtype=np.int32)
            color = colors[i % len(colors)]
            cv2.polylines(vis, [pts], True, color, 2)

            if center and len(pts) > 0:
                
                dists = np.sqrt((pts[:, 0]-center[0])**2 + (pts[:, 1]-center[1])**2)
                tip = pts[np.argmax(dists)]
                cv2.circle(vis, tuple(tip), 5, color, -1)
                cv2.line(vis, (cx, cy), tuple(tip), color, 1)

                min_d = np.min(dists)
                max_d = np.max(dists)
                score = max_d / (min_d + 1.0) * (0.5 + seg["conf"])

                label = f"#{i} conf={seg['conf']:.2f} score={score:.1f}"
                cv2.putText(vis, label, (pts[0][0], pts[0][1]-5),
                           self.FONT, 0.3, color, 1)

        scale, x_off, y_off = self._paste_on_panel(panel, vis)

        
        n = len(needle_segs)
        info = f"{n} needle(s) detected"
        if n > 1:
            info += " (shadow?)"
        cv2.putText(panel, info, (8, self.PH-10), self.FONT, 0.35, (150,150,150), 1)

        if needle_scores:
            y = self.PH - 28
            for ns in needle_scores:
                txt = f"#{ns['idx']}: min_d={ns['min_dist']:.0f} max_d={ns['max_dist']:.0f} score={ns['score']:.1f} {'<< SELECTED' if ns.get('selected') else ''}"
                color = (0, 255, 0) if ns.get('selected') else (150, 150, 150)
                cv2.putText(panel, txt, (8, y), self.FONT, 0.28, color, 1)
                y -= 14

        return panel

    # ==================================================================
    # Panel 6: Calibration Plot (Angle vs Value)
    # ==================================================================
    def _panel_calibration(self, cal_data: Dict) -> np.ndarray:
    
        panel = self._make_panel("6. Calibration (Deduplicated)")

        if not cal_data or not cal_data.get("all_points"):
            cv2.putText(panel, "No calibration data", (20, 100), self.FONT, 0.5, (100,100,255), 1)
            return panel

        all_pts = cal_data["all_points"]       # list of (val, ang, class, text, radius) - DEDUPLICATED
        clean_data = cal_data.get("clean_data")  # list of (val, ang)
        slope = cal_data.get("slope", 0)
        intercept = cal_data.get("intercept", 0)
        needle_ang = cal_data.get("needle_angle_final")
        needle_val = cal_data.get("needle_value")
        r2 = cal_data.get("r2", 0)

        plot_x, plot_y = 50, 40
        plot_w, plot_h = self.PW - 70, self.PH - 80

        all_angs = [p[1] for p in all_pts]
        all_vals = [p[0] for p in all_pts]
        if needle_ang is not None:
            all_angs.append(needle_ang)
        if needle_val is not None:
            all_vals.append(needle_val)

        ang_min, ang_max = min(all_angs), max(all_angs)
        val_min, val_max = min(all_vals), max(all_vals)
        ang_range = max(ang_max - ang_min, 1)
        val_range = max(val_max - val_min, 1)

        ang_min -= ang_range * 0.1; ang_max += ang_range * 0.1
        val_min -= val_range * 0.1; val_max += val_range * 0.1
        ang_range = ang_max - ang_min
        val_range = val_max - val_min

        def to_px(ang, val):
            px = plot_x + int((ang - ang_min) / ang_range * plot_w)
            py = plot_y + plot_h - int((val - val_min) / val_range * plot_h)
            return (px, py)

        cv2.rectangle(panel, (plot_x, plot_y), (plot_x+plot_w, plot_y+plot_h), (80,80,80), 1)
        cv2.putText(panel, "Angle", (plot_x + plot_w//2 - 15, self.PH-5), self.FONT, 0.3, (150,150,150), 1)
        cv2.putText(panel, "Val", (2, plot_y + plot_h//2), self.FONT, 0.3, (150,150,150), 1)

        for i in range(5):
            a = ang_min + i * ang_range / 4
            v = val_min + i * val_range / 4
            px_a = plot_x + int(i * plot_w / 4)
            py_v = plot_y + plot_h - int(i * plot_h / 4)
            cv2.putText(panel, f"{a:.0f}", (px_a-8, plot_y+plot_h+12), self.FONT, 0.25, (120,120,120), 1)
            cv2.putText(panel, f"{v:.0f}", (2, py_v+3), self.FONT, 0.25, (120,120,120), 1)

        if slope != 0 or intercept != 0:
            a1, a2 = ang_min, ang_max
            v1, v2 = slope * a1 + intercept, slope * a2 + intercept
            p1, p2 = to_px(a1, v1), to_px(a2, v2)
            cv2.line(panel, p1, p2, (100, 200, 100), 1, cv2.LINE_AA)

        for val, ang, cls, text, radius in all_pts:
            color = self._get_color(cls)
            px, py = to_px(ang, val)
            cv2.circle(panel, (px, py), 3, (100, 100, 100), -1)
            cv2.putText(panel, text[:6], (px+4, py-4), self.FONT, 0.25, (100,100,100), 1)

        if clean_data:
            for val, ang in clean_data:
                px, py = to_px(ang, val)
                cv2.circle(panel, (px, py), 5, (0, 255, 0), -1)
                cv2.putText(panel, f"{val:.0f}", (px+6, py-2), self.FONT, 0.3, (0, 255, 0), 1)

        if needle_ang is not None and needle_val is not None:
            px, py = to_px(needle_ang, needle_val)
            cv2.drawMarker(panel, (px, py), (0, 0, 255), cv2.MARKER_DIAMOND, 12, 2)
            cv2.putText(panel, f"NEEDLE={needle_val:.1f}", (px+8, py+4), self.FONT, 0.35, (0, 0, 255), 1)

        # R2 info
        cv2.putText(panel, f"R2={r2:.3f} slope={slope:.4f}", (plot_x, plot_y-5), self.FONT, 0.3, (200,200,200), 1)

        return panel

    # ==================================================================
    # Panel 7: Radius Analysis (dual-scale detection)
    # ==================================================================
    def _panel_radius(self, cal_data: Dict) -> np.ndarray:
        panel = self._make_panel("7. Radius Analysis (Dual-Scale)")

        if not cal_data or not cal_data.get("all_points"):
            cv2.putText(panel, "No data", (20, 100), self.FONT, 0.5, (100,100,255), 1)
            return panel

        all_pts = cal_data["all_points"]   # (val, ang, class, text, radius)
        groups = cal_data.get("radius_groups", [])

        if len(all_pts[0]) >= 5:
            radii = [p[4] for p in all_pts]
            vals = [p[0] for p in all_pts]
            texts = [p[3] for p in all_pts]
        else:
            cv2.putText(panel, "No radius data", (20, 100), self.FONT, 0.5, (100,100,255), 1)
            return panel

        plot_x, plot_y = 60, 40
        plot_w, plot_h = self.PW - 80, self.PH - 60

        r_min, r_max = min(radii), max(radii)
        r_range = max(r_max - r_min, 1)

        n = len(radii)
        bar_h = min(25, max(8, plot_h // max(n, 1)))
        
        group_colors = [(0, 200, 255), (255, 100, 100), (100, 255, 100)]

        for i, (r, v, t) in enumerate(sorted(zip(radii, vals, texts), key=lambda x: x[0])):
            y = plot_y + i * bar_h
            if y + bar_h > plot_y + plot_h:
                break
            
            bar_len = int((r - r_min) / r_range * plot_w) if r_range > 0 else plot_w // 2
            bar_len = max(5, bar_len)

            color = (180, 180, 180)
            for gi, group in enumerate(groups):
                if any(abs(p.get('r', 0) - r) < 1 for p in group):
                    color = group_colors[gi % len(group_colors)]
                    break

            cv2.rectangle(panel, (plot_x, y+2), (plot_x + bar_len, y + bar_h - 2), color, -1)
            cv2.putText(panel, f"{t[:6]}={v:.0f}", (4, y + bar_h - 4), self.FONT, 0.28, (200,200,200), 1)
            cv2.putText(panel, f"r={r:.0f}", (plot_x + bar_len + 4, y + bar_h - 4), self.FONT, 0.28, (200,200,200), 1)

        if radii:
            med = np.median(radii)
            med_x = plot_x + int((med - r_min) / r_range * plot_w) if r_range > 0 else plot_x + plot_w // 2
            cv2.line(panel, (med_x, plot_y), (med_x, plot_y + plot_h), (0, 255, 255), 1)
            cv2.putText(panel, f"median={med:.0f}", (med_x+3, plot_y+plot_h+12), self.FONT, 0.3, (0,255,255), 1)

        n_groups = len(groups) if groups else 1
        cv2.putText(panel, f"Groups: {n_groups} | {'DUAL SCALE' if n_groups > 1 else 'Single Scale'}",
                   (8, self.PH-10), self.FONT, 0.35, (0,255,255) if n_groups>1 else (150,150,150), 1)

        return panel

    # ==================================================================
    # Panel 8: Final Result
    # ==================================================================
    def _panel_final(self, img: np.ndarray, result: Optional[Dict],
                     center: Optional[Tuple], unit: str) -> np.ndarray:
        panel = self._make_panel("8. Final Result")
        vis = img.copy()

        if center:
            cx, cy = int(center[0]), int(center[1])
            cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)

        if result and "value" in result:
            tip = result.get("needle_tip")
            if tip and center:
                cv2.line(vis, (cx, cy), (int(tip[0]), int(tip[1])), (0, 0, 255), 2)
                cv2.circle(vis, (int(tip[0]), int(tip[1])), 5, (0, 0, 255), -1)

        self._paste_on_panel(panel, vis)

        if result and "value" in result:
            r2 = result.get("r2_score", 0)
            val = result["value"]
            pts = result.get("fit_points", "?")
            vmin = result.get("min_found", 0)
            vmax = result.get("max_found", 0)
            slope = result.get("slope", 0)

            if r2 >= 0.8 and int(str(pts)) >= 3:
                status = "GOOD"
                status_color = (0, 200, 0)
            elif r2 >= 0.5:
                status = "FAIR"
                status_color = (0, 200, 255)
            else:
                status = "POOR"
                status_color = (0, 0, 255)

            u = f" {unit}" if unit else ""
            lines = [
                (f"VALUE: {val:.2f}{u}", 0.6, status_color, 2),
                (f"Status: {status} | R2: {r2:.3f}", 0.4, status_color, 1),
                (f"Cal Points: {pts} | Range: [{vmin:.0f}, {vmax:.0f}]", 0.35, (200,200,200), 1),
                (f"Direction: {'CW' if slope > 0 else 'CCW'} | Slope: {slope:.4f}", 0.35, (200,200,200), 1),
            ]
            y = self.PH - 70
            for text, scale, color, thick in lines:
                cv2.putText(panel, text, (8, y), self.FONT, scale, color, thick, cv2.LINE_AA)
                y += 16
        else:
            cv2.putText(panel, "CALCULATION FAILED", (20, self.PH - 40),
                       self.FONT, 0.7, (0, 0, 255), 2)

        return panel

    # ==================================================================
    # Generate Full Report
    # ==================================================================
    def generate_report(self, filename: str, debug_data: Dict[str, Any]) -> Optional[str]:

        if not self.enabled:
            return None

        img = debug_data.get("original_img")
        if img is None:
            return None

        segmentations = debug_data.get("segmentations", [])
        ellipse_results = debug_data.get("ellipse_results", [])
        center = debug_data.get("center")
        ocr_results = debug_data.get("ocr_results", [])
        needle_scores = debug_data.get("needle_scores", [])
        cal_data = debug_data.get("calibration_data", {})
        result = debug_data.get("result")
        unit = debug_data.get("unit", "")

        p1 = self._panel_original(img)
        p2 = self._panel_segmentation(img, segmentations)
        p3 = self._panel_ellipse(img, ellipse_results, center)
        p4 = self._panel_ocr(img, segmentations, ocr_results)
        p5 = self._panel_needle(img, segmentations, center, needle_scores)
        p6 = self._panel_calibration(cal_data)
        p7 = self._panel_radius(cal_data)
        p8 = self._panel_final(img, result, center, unit)

        row1 = np.hstack([p1, p2, p3, p4])
        row2 = np.hstack([p5, p6, p7, p8])
        report = np.vstack([row1, row2])

        title_h = 36
        title_bar = np.full((title_h, report.shape[1], 3), (30, 30, 30), dtype=np.uint8)
        status = "OK" if result and result.get("r2_score", 0) >= 0.8 else "WARN" if result else "FAIL"
        val_str = f"{result['value']:.2f}" if result and 'value' in result else "N/A"
        title_text = f"DEBUG REPORT: {filename} | Value={val_str} {unit} | {status}"
        cv2.putText(title_bar, title_text, (10, 25), self.FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        report = np.vstack([title_bar, report])

        out_path = os.path.join(self.output_dir, f"debug_{os.path.splitext(filename)[0]}.jpg")
        cv2.imwrite(out_path, report, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return out_path
