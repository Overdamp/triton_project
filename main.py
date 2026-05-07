import os
import cv2
import glob
import time
from cores.config_loader import load_config
from cores.logger import setup_logger
from tasks.analog_gauge_task import AnalogGaugeTask


def main():
    config = load_config("configs/config.yaml")
    logger = setup_logger(config)
    logger.info("=== Starting Analog Gauge Batch Inference ===")



    debug_cfg = config.get("analog_gauge", {}).get("debug", {})
    if debug_cfg.get("enabled"):
        logger.info(f"[DEBUG MODE ON] Reports will be saved to: {debug_cfg.get('output_dir', 'debug_output')}")
    else:
        logger.warning("Debug mode is OFF - no reports will be generated")

    logger.info("Initializing Analog Gauge Models...")
    try:
        analog_task = AnalogGaugeTask(config)
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return

    input_folder = "/home/engineer00/winworkspace/PTTEP/Objectdetection/crop_gauge_result/analog-gauge"
    output_folder = "/home/engineer00/winworkspace/PTTEP/AnalogGaugeReading/gauge-win-v3/results"

    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    valid_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
        image_paths.extend(glob.glob(os.path.join(input_folder, ext.upper())))

    if not image_paths:
        logger.warning(f"No images found in '{input_folder}' folder.")
        return

    logger.info(f"Found {len(image_paths)} images to process.")

    success_count = 0
    start_time = time.time()

    for img_path in image_paths:
        filename = os.path.basename(img_path)

        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"Cannot read image: {filename}")
            continue

        logger.info(f"Processing: {filename} ...")

        result = analog_task.execute(img, filename=filename)

        if result and "value" in result:
            value = result.get("value", 0.0)
            unit = result.get("unit", "")
            r2_score = result.get("r2_score", 0.0)

            logger.info(f"-> Result [{filename}]: {value:.2f} {unit} (R2: {r2_score:.2f})")
            success_count += 1

            debug_img = result.get("debug_img")
            if debug_img is not None:
                out_path = os.path.join(output_folder, f"result_{filename}")
                cv2.imwrite(out_path, debug_img)
        else:
            logger.warning(f"-> Failed to calculate gauge value for: {filename}")

            if result and result.get("debug_img") is not None:
                out_path = os.path.join(output_folder, f"failed_{filename}")
                cv2.imwrite(out_path, result["debug_img"])

    elapsed = time.time() - start_time
    logger.info("=== Inference Completed ===")
    logger.info(f"Successfully processed {success_count}/{len(image_paths)} images in {elapsed:.2f} seconds.")
    logger.info(f"Results saved to '{output_folder}' directory.")

    if debug_cfg.get("enabled"):
        logger.info(f"✓ Debug reports for ALL images saved to: {debug_cfg.get('output_dir', 'debug_output')} directory.")
        logger.info(f"  Total debug files: ~{len(image_paths)} images")


if __name__ == "__main__":
    main()