# -*- utf-8 -*-
"""Yolov11 experiment."""
import cv2
import numpy
from skimage import feature
from ultralytics import YOLO

from common.config import Config
from common.log import get_logger


def detect_objects(image: numpy.ndarray, model: YOLO) -> numpy.ndarray:
    """Detect objects in an image using YOLOv11.

    Args:
        image (numpy.ndarray): Image.
        model (YOLO): YOLOv11 model.

    Returns:
        detections: Detected objects.
    """
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    layer_names = model.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]
    detections = model.forward(output_layers)
    return detections


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
    detections = detect_objects(image=color_image, model=model)
    for det in detections:
        x, y, w, h = det["bbox"]
        object_roi = color_image[y : y + h, x : x + w]
        if not is_decoy(object_roi):
            position = get_3d_position(depth_image, x + w // 2, y + h // 2)
            print(f"Real object detected at {position}")


def main():
    """Main function."""
    color_img_dir = Config.data_dir / "raw" / "test" " camera_color_image_raw"
    depth_img_dir = Config.data_dir / "raw" / "test" " camera_depth_image_raw"

    model = YOLO(model=Config.model_dir / "yolov11.pt", task="detection", verbose=True)

    color_imgs = sorted(color_img_dir.glob("*.png"))
    depth_imgs = sorted(depth_img_dir.glob("*.png"))

    color_image = cv2.imread(str(color_imgs[0]), cv2.IMREAD_COLOR)
    depth_image = cv2.imread(str(depth_imgs[0]), cv2.IMREAD_UNCHANGED)

    logger = get_logger(log_filename=Config.log_dir / "yolov11_experiment.log")
    logger.info(f"Images of shape {color_image.shape} and {depth_image.shape} loaded.")

    process_frame(color_image=color_image, depth_image=depth_image, model=model)


if __name__ == "__main__":
    main()
