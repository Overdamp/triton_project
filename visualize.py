import os
import cv2
import numpy as np
import glob
from cores.config_loader import load_config
from libs.analog_gauge.segmentation import GaugeSegmentor

def main():
    config = load_config("configs/config.yaml")
    
    input_folder = "/home/engineer00/winworkspace/PTTEP/Objectdetection/crop_gauge_result/analog-gauge"        
    output_folder = "/home/engineer00/winworkspace/PTTEP/AnalogGaugeReading/gauge-win-v3/segment_results"   
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    print("Loading Segmentation Model...")
    seg_config = config.get("analog_gauge", {}).get("segmentation", {})
    segmentor = GaugeSegmentor(seg_config)

    valid_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
        image_paths.extend(glob.glob(os.path.join(input_folder, ext.upper())))

    if not image_paths:
        print(f" Not Found Folder: {input_folder}")
        return

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        print(f"Processing: {filename}")

        segmentations = segmentor.get_segmentation(img)
        
        if not segmentations:
            print(f" Not Found Object in image {filename}")
            continue

        for seg in segmentations:
            cls_name = seg['class']
            mask_points = np.array(seg['mask'], dtype=np.int32)
            
            cv2.drawContours(img, [mask_points], -1, (0, 255, 0), 2)
            
            x, y, w, h = cv2.boundingRect(mask_points)
            cv2.putText(img, cls_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        all_mask_path = os.path.join(output_folder, f"{name}_ALL_MASKS{ext}")
        cv2.imwrite(all_mask_path, img)
        print(f" save overview image: {all_mask_path}")

if __name__ == "__main__":
    main()