import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger("AIPipeline")


class GaugeCalculator:
    def __init__(self):
        pass

    def get_farthest_point(self, center, points):
        pts = np.array(points)
        if len(pts) == 0: return center
        dists = np.sqrt((pts[:, 0] - center[0])**2 + (pts[:, 1] - center[1])**2)
        return tuple(pts[np.argmax(dists)])

    def correct_perspective_angle(self, center, point, axes, angle_tilt):
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        rad = np.radians(-angle_tilt)
        dx_rot = dx * np.cos(rad) - dy * np.sin(rad)
        dy_rot = dx * np.sin(rad) + dy * np.cos(rad)
        a, b = max(axes[0], 1e-6), max(axes[1], 1e-6)
        angle_deg = np.degrees(np.arctan2(dy_rot / b, dx_rot / a))
        return angle_deg % 360

    def calculate_radius(self, center, point):
        return np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)

    def unwrap_angles(self, angles):
        u = np.copy(angles)
        for i in range(1, len(u)):
            d = u[i] - u[i-1]
            if d < -180: u[i:] += 360
            elif d > 180: u[i:] -= 360
        return u

    def run_ransac(self, x, y, max_iters=100, threshold_ratio=0.15):
        n = len(x)
        if n < 2: return None, None, np.array([])
        best_c, best_m, best_mask = 0, None, np.zeros(n, dtype=bool)
        thr = max(np.ptp(y), 1.0) * threshold_ratio
        for _ in range(max_iters):
            ii = np.random.choice(n, 2, replace=False)
            dx = x[ii[1]] - x[ii[0]]
            if abs(dx) < 1e-6: continue
            s = (y[ii[1]] - y[ii[0]]) / dx
            b = y[ii[0]] - s * x[ii[0]]
            mask = np.abs(y - (s * x + b)) < thr
            c = np.sum(mask)
            if c > best_c: best_c, best_m, best_mask = c, (s, b), mask
        if best_m is None:
            s, b = np.polyfit(x, y, 1)
            return s, b, np.ones(n, dtype=bool)
        return best_m[0], best_m[1], best_mask

    def _split_by_radius(self, points):
        if len(points) <= 2: return [points]
        sorted_pts = sorted(points, key=lambda p: p['r'])
        radii = [p['r'] for p in sorted_pts]
        r_range = radii[-1] - radii[0]
        if r_range < 1e-6: return [points]
        max_gap, split_idx = 0, -1
        for i in range(len(radii) - 1):
            gap = radii[i+1] - radii[i]
            if gap > max_gap: max_gap = gap; split_idx = i
        if max_gap > r_range * 0.25 and split_idx >= 0:
            g1, g2 = sorted_pts[:split_idx+1], sorted_pts[split_idx+1:]
            groups = [g for g in [g1, g2] if len(set(p['val'] for p in g)) >= 2]
            if groups: return groups
        return [points]

    def _unwrap_needle_angle(self, needle_angle, clean_angs, clean_vals, slope, intercept):
        vmin, vmax = np.min(clean_vals), np.max(clean_vals)
        ang_mid = (np.min(clean_angs) + np.max(clean_angs)) / 2
        candidates = [needle_angle + k * 360 for k in range(-3, 4)]
        def score(c):
            p = slope * c + intercept
            d_val = max(0, vmin - p, p - vmax)
            return (d_val, abs(c - ang_mid))
        return min(candidates, key=score)

    def _fit_group(self, points, needle_angle):
        points.sort(key=lambda x: x['val'])
        vals = np.array([p['val'] for p in points])
        angs = self.unwrap_angles(np.array([p['ang'] for p in points]))
        if len(vals) >= 4:
            _, _, mask = self.run_ransac(angs, vals)
            ca, cv = angs[mask], vals[mask]
            if len(cv) < 2: ca, cv = angs, vals
        else:
            ca, cv = angs, vals
        slope, intercept = np.polyfit(ca, cv, 1)
        y_pred = slope * ca + intercept
        ss_res = np.sum((cv - y_pred)**2)
        ss_tot = np.sum((cv - np.mean(cv))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
        na = self._unwrap_needle_angle(needle_angle, ca, cv, slope, intercept)
        predicted = slope * na + intercept
        vmin, vmax = float(np.min(cv)), float(np.max(cv))
        margin = (vmax - vmin) * 0.10
        predicted = float(np.clip(predicted, vmin - margin, vmax + margin))
        return {
            "value": predicted, "slope": slope, "intercept": intercept,
            "r2": r2, "n_pts": len(cv),
            "clean_angs": ca, "clean_vals": cv,
            "needle_angle_final": na, "vmin": vmin, "vmax": vmax,
        }

    def process_gauge(self, center, needle_mask, ocr_data, ellipse_data=None):
        """
        Returns:
            dict with keys:
                value, min_found, max_found, fit_points, total_points_detected,
                r2_score, needle_tip, slope,
                debug_cleaned_data,
                _calibration_debug 
        """
        if center is None or len(needle_mask) == 0:
            return None

        axes, tilt = (1.0, 1.0), 0.0
        if ellipse_data and 'axes' in ellipse_data:
            axes = ellipse_data['axes']
            tilt = ellipse_data.get('angle_deg', 0.0)

        needle_tip = self.get_farthest_point(center, needle_mask)
        needle_angle = self.correct_perspective_angle(center, needle_tip, axes, tilt)

        # --- Collect calibration points ---
        raw_points = []
        for item in ocr_data:
            if item.get('confidence', 0) < 0.3: continue
            if item.get('class', '') == 'unit': continue
            mc = item.get('mask_center')
            if not mc: continue
            text = ''.join(c for c in item.get('text', '') if c.isdigit() or c == '.').strip('.')
            if not text or text.count('.') > 1: continue
            try: val = float(text)
            except ValueError: continue
            ang = self.correct_perspective_angle(center, mc, axes, tilt)
            r = self.calculate_radius(center, mc)
            raw_points.append({
                'val': val, 'ang': ang, 'r': r,
                'text': item.get('text', ''), 'conf': item.get('confidence', 0),
                'class': item.get('class', ''),
            })

        if len(raw_points) < 2: return None

        # Enforce max > min
        max_pts = [p for p in raw_points if p['class'] == 'max-value']
        min_pts = [p for p in raw_points if p['class'] == 'min-value']
        if max_pts and min_pts and max_pts[0]['val'] < min_pts[0]['val']:
            max_pts[0]['val'], min_pts[0]['val'] = min_pts[0]['val'], max_pts[0]['val']

        
        groups = defaultdict(list)
        for p in raw_points: groups[p['val']].append(p)
        deduped = [max(g, key=lambda x: x['conf']) for g in groups.values()]
        if len(set(p['val'] for p in deduped)) < 2: return None
        
        logger.info(f"[GaugeCalc] OCR: {len(raw_points)} detections → {len(deduped)} unique values")

        
        all_points_debug = [(p['val'], p['ang'], p['class'], p['text'], p['r']) for p in deduped]

        scale_groups = self._split_by_radius(deduped)
        if len(scale_groups) > 1:
            logger.info(f"[GaugeCalc] Dual-scale: {[len(g) for g in scale_groups]} groups")

        
        best_result = None
        for group in scale_groups:
            if len(set(p['val'] for p in group)) < 2: continue
            fit = self._fit_group(group, needle_angle)
            if best_result is None or (fit['n_pts'], fit['r2']) > (best_result['n_pts'], best_result['r2']):
                best_result = fit

        if best_result is None:
            best_result = self._fit_group(deduped, needle_angle)

        
        cal_debug = {
            "all_points": all_points_debug,
            "clean_data": list(zip(best_result["clean_vals"].tolist(), best_result["clean_angs"].tolist())),
            "slope": best_result["slope"],
            "intercept": best_result["intercept"],
            "r2": best_result["r2"],
            "needle_angle_raw": needle_angle,
            "needle_angle_final": best_result["needle_angle_final"],
            "needle_value": best_result["value"],
            "radius_groups": scale_groups,
        }

        return {
            "value": best_result["value"],
            "min_found": best_result["vmin"],
            "max_found": best_result["vmax"],
            "fit_points": best_result["n_pts"],
            "total_points_detected": len(raw_points),
            "r2_score": best_result["r2"],
            "needle_tip": needle_tip,
            "slope": best_result["slope"],
            "debug_cleaned_data": list(zip(best_result["clean_vals"].tolist(), best_result["clean_angs"].tolist())),
            "_calibration_debug": cal_debug,
        }