# -*- utf-8 -*-
"""Yolov11 experiment."""
import cv2
import numpy
from skimage import feature
from ultralytics import YOLO
from ultralytics.engine.results import Results

from common.config import Config
from common.log import get_logger


def is_decoy(object_roi: numpy.ndarray) -> bool:
    """Determine if object is a decoy.

    Args:
        object_roi (numpy.ndarray): Region of interest containing the object.

    Returns:
        is_decoy: True if object is a decoy, False otherwise.
    """
    # Example: Use texture analysis to determine if object is a decoy
    lbp = feature.local_binary_pattern(object_roi, P=24, R=3, method="uniform")
    lbp_hist, _ = numpy.histogram(lbp.ravel(), bins=numpy.arange(0, 27), range=(0, 26))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= lbp_hist.sum() + 1e-6
    # Thresholding based on empirical observation
    if numpy.max(lbp_hist) > 0.3:
        return True
    return False


def get_3d_position(depth_image: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
    """Get 3D position of object.

    Args:
        depth_image (numpy.ndarray): Depth image.
        x (numpy.ndarray): x-coordinate of object.
        y (numpy.ndarray): y-coordinate of object.

    Returns:
        position: 3D position of object
    """
    depth = depth_image[y, x]
    fx, fy, cx, cy = 525, 525, 319.5, 239.5  # Example intrinsic parameters
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    return numpy.array([X, Y, Z])


def process_frame(color_image: numpy.ndarray, depth_image: numpy.ndarray, model: YOLO) -> None:
    """Process a frame.

    Args:
        color_image (numpy.ndarray): Color image.
        depth_image (numpy.ndarray): Depth image.
        model (YOLO): YOLOv11 model.
    """
    detections: list[Results] = model(color_image)

    for det in detections[0].boxes.xyxy:
        x1, y1, x2, y2 = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        object_roi = color_image[y1:y2, x1:x2]

        if not is_decoy(object_roi):
            position = get_3d_position(depth_image=depth_image, x=(x1 + x2) // 2, y=(y1 + y2) // 2)
            print(f"Real object detected at {position}")


def main():
    """Main function."""
    color_img_dir = Config.data_dir / "raw" / "test" / "camera_color_image_raw"
    depth_img_dir = Config.data_dir / "raw" / "test" / "camera_depth_image_raw"

    model_pt = Config.model_dir.parents[1] / "models" / "yolo_v11_extinguiser" / "weights" / "best.pt"
    model = YOLO(model=model_pt, task="detect", verbose=True)

    # detection = model(
    #     Config.data_dir / "FireExtinguiser" / "valid" / "images" / "30_jpg.rf.7a274213763dbf3a7a0c74e44bb34f24.jpg"
    # )

    color_imgs = sorted(color_img_dir.glob("*.png"))
    depth_imgs = sorted(depth_img_dir.glob("*.png"))

    color_image = cv2.imread(
        str(
            Config.data_dir / "FireExtinguiser" / "valid" / "images" / "30_jpg.rf.7a274213763dbf3a7a0c74e44bb34f24.jpg"
        ),
        cv2.IMREAD_COLOR,
    )
    depth_image = cv2.imread(str(depth_imgs[0]), cv2.IMREAD_UNCHANGED)

    logger = get_logger(log_filename=Config.log_dir / "yolov11_experiment.log")
    logger.info(f"Images of shape {color_image.shape} and {depth_image.shape} loaded.")

    process_frame(color_image=color_image, depth_image=depth_image, model=model)


if __name__ == "__main__":
    main()
