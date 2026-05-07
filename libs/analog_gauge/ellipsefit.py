import numpy as np
import cv2
from skimage.measure import EllipseModel, ransac

class EllipseFitter:
    def fit(self, mask_points: list, apply_convex_hull: bool = True) -> dict:
        if EllipseModel is None:
            return None

        pts = np.array(mask_points, dtype=np.float64)
        
        
        if len(pts) < 5:
            return None
            
        try:

            if apply_convex_hull:
                pts_cv = np.array(pts, dtype=np.float32)
                hull = cv2.convexHull(pts_cv)
                pts = hull.reshape(-1, 2)

                if len(pts) < 5:
                    return None


            model, inliers = ransac(pts, EllipseModel, min_samples=5, 
                                    residual_threshold=3.0, max_trials=100)
            
            if model is None:
                return None
            

            if hasattr(model, 'center'):
                xc, yc = model.center
                a, b = model.axis_lengths
                theta = model.theta
            else:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    xc, yc, a, b, theta = model.params

            
            width = 2 * a 
            height = 2 * b 
            angle_deg = np.degrees(theta)
            
            if np.isnan(xc) or np.isnan(width) or np.isinf(width):
                return None

            return {
                "opencv_params": ((xc, yc), (width, height), angle_deg),
                "center": (xc, yc),
                "axes": (a, b),
                "angle_deg": angle_deg,
                "angle_rad": theta,
                "inliers_used": inliers
            }

        except Exception as e:
            # print(f"Skimage Fit Error: {e}") 
            return None