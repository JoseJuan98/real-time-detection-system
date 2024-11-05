# -*- utf-8 -*-
"""Yolov11 experiment."""
import cv2
import numpy
import yaml
from skimage import feature
from ultralytics import YOLO
from ultralytics.engine.results import Results

from common.config import Config
from common.log import get_logger, msg_task


def is_decoy(object_roi: numpy.ndarray) -> bool:
    """Determine if object is a decoy based on Local Binary Pattern (LBP) histogram for texture analysis.

    Args:
        object_roi (numpy.ndarray): Region of interest containing the object.

    Returns:
        is_decoy: True if object is a decoy, False otherwise.
    """
    lbp = feature.local_binary_pattern(object_roi, P=24, R=3, method="uniform")
    lbp_hist, _ = numpy.histogram(lbp.ravel(), bins=numpy.arange(0, 27), range=(0, 26))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= lbp_hist.sum() + 1e-6

    # Thresholding based on empirical observation
    if numpy.max(lbp_hist) > 0.3:
        return True
    return False


def get_3d_position(depth_image: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray, K: numpy.ndarray) -> numpy.ndarray:
    """Get 3D position of object.

    Args:
        depth_image (numpy.ndarray): Depth image.
        x (numpy.ndarray): x-coordinate of object.
        y (numpy.ndarray): y-coordinate of object.
        K (numpy.ndarray): Camera intrinsic matrix.

    Returns:
        position: 3D position of object
    """
    depth = depth_image[y, x]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    return numpy.array([X, Y, Z])


def process_frame(
    color_image: numpy.ndarray, depth_image: numpy.ndarray, model: YOLO, K: numpy.ndarray
) -> list[numpy.ndarray]:
    """Process a frame.

    Args:
        color_image (numpy.ndarray): Color image.
        depth_image (numpy.ndarray): Depth image.
        model (YOLO): YOLOv11 model.
        K (numpy.ndarray): Camera intrinsic matrix.
    """
    # FIXME: only get the prediction over a threshold confidence
    detections: list[Results] = model(color_image)

    positions = []
    for box in detections[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        object_roi = color_image[y1:y2, x1:x2].reshape(-1, 3)

        if is_decoy(object_roi=object_roi):
            print("Decoy detected!")
        else:
            position = get_3d_position(depth_image=depth_image, K=K, x=(x1 + x2) // 2, y=(y1 + y2) // 2)
            print(f"Real object detected at {position}")
            positions.append(position)

    return positions


def main():
    """Main function."""
    color_img_dir = Config.data_dir / "raw" / "test" / "camera_color_image_raw"
    depth_img_dir = Config.data_dir / "raw" / "test" / "camera_depth_image_raw"

    model_pt = Config.model_dir.parents[1] / "models" / "yolo_v11_extinguiser" / "weights" / "best.pt"
    model = YOLO(model=model_pt, task="detect", verbose=True)

    # Image with several objects
    # detection = model(
    #     Config.data_dir / "FireExtinguiser" / "valid" / "images" / "30_jpg.rf.7a274213763dbf3a7a0c74e44bb34f24.jpg"
    # )

    # Image with decoys
    # decoy_img = (
    #     Config.data_dir / "raw" / "test" / "camera_color_image_raw" / "camera_color_image_1727164485882972273.png"
    # )
    # importing camera parameters from yaml
    with open(
        color_img_dir.parent / "camera_color_camera_info" / "camera_color_info_1727164479160357265.txt", "r"
    ) as file:
        camera_params = yaml.safe_load(file)

    K = numpy.array(camera_params["K"]).reshape(3, 3)

    color_imgs = sorted(color_img_dir.glob("*.png"))
    depth_imgs = sorted(depth_img_dir.glob("*.png"))

    color_image = cv2.imread(color_imgs[0], cv2.IMREAD_COLOR)
    depth_image = cv2.imread(depth_imgs[0], cv2.IMREAD_UNCHANGED)

    logger = get_logger(log_filename=Config.log_dir / "yolov11_experiment.log")

    msg_task(msg="3D Poisition estimation - Yolov11 Experiment", logger=logger)

    logger.info(f"Load {len(color_imgs)} images of shape {color_image.shape} and {depth_image.shape} loaded.")

    positions = process_frame(color_image=color_image, depth_image=depth_image, model=model, K=K)

    if not positions:
        logger.info("No objects detected.")

    else:

        logger.info(f"Positions:")
        for idx, position in enumerate(positions):
            logger.info(f"Position for extinguiser {idx}: {position}")


if __name__ == "__main__":
    main()
