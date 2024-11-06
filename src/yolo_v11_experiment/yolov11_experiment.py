# -*- utf-8 -*-
"""Yolov11 experiment."""
import pathlib

import cv2
import numpy
import yaml
import pyvista
from numpy.ma.core import shape
from skimage import feature
from matplotlib import pyplot
from ultralytics import YOLO
from ultralytics.engine.results import Results

from pypcd4 import PointCloud

from common.config import Config
from common.log import get_logger, msg_task


def is_decoy(object_roi: numpy.ndarray) -> bool:
    """Determine if object is a decoy based on Local Binary Pattern (LBP) histogram for texture analysis.

    Args:
        object_roi (numpy.ndarray): Region of interest containing the object.

    Returns:
        is_decoy: True if object is a decoy, False otherwise.
    """
    # if it's a color image, convert to grayscale
    if object_roi.ndim == 3:
        object_roi = cv2.cvtColor(object_roi, cv2.COLOR_BGR2GRAY)

    lbp = feature.local_binary_pattern(object_roi, P=24, R=3, method="uniform")
    lbp_hist, _ = numpy.histogram(lbp.ravel(), bins=numpy.arange(0, 27), range=(0, 26))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= lbp_hist.sum() + 1e-6

    # Thresholding based on empirical observation
    if lbp_hist.max().item() > 0.3:
        # plot histogram
        pyplot.figure(figsize=(16, 9))
        pyplot.bar(numpy.arange(0, 26), lbp_hist)
        pyplot.title("LBP Histogram")
        # FIXME:
        pyplot.savefig("lbp_hist.png")
        pyplot.close()
        return False
    return True


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
    X = abs((x - cx) * depth / fx)
    Y = abs((y - cy) * depth / fy)
    Z = depth
    return numpy.array([X, Y, Z]).astype(numpy.float32)


def plot_and_save(img: numpy.ndarray, file_path: pathlib.Path | str, cmap: str = None, plot: bool = False) -> None:
    """Save plot to file.

    Args:
        file_path (pathlib.Path | str): File path to save plot.
        img (numpy.ndarray): Image to plot.
        cmap (str): Colormap to use.
        plot (bool): Plot the image. Default is False.
    """
    pyplot.figure(figsize=(16, 9))
    pyplot.imshow(X=img, cmap=cmap)
    pyplot.savefig(file_path, bbox_inches="tight", pad_inches=0.1)

    if plot:
        pyplot.show()

    pyplot.close()


def process_frame(
    color_image: numpy.ndarray, depth_image: numpy.ndarray, model: YOLO, K: numpy.ndarray, plot_dir: pathlib.Path
) -> (list[numpy.ndarray], int):
    """Process a frame.

    Args:
        color_image (numpy.ndarray): Color image.
        depth_image (numpy.ndarray): Depth image.
        model (YOLO): YOLOv11 model.
        K (numpy.ndarray): Camera intrinsic matrix.
        plot_dir (pathlib.Path): Directory to save plots.

    Returns:
        list: List of 3D positions of objects.
        int: total number of extinguisers detected without filtering.
    """
    detections: list[Results] = model(color_image)

    # Create plot directory if it doesn't exist
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_and_save(
        img=color_image,
        file_path=plot_dir / "color_image.png",
    )
    plot_and_save(
        img=depth_image,
        file_path=plot_dir / "depth_image.png",
        cmap="viridis",
    )

    pyplot.figure(figsize=(16, 9))
    detections[0].plot(filename=str(plot_dir / "detections.png"), save=True)

    positions = []
    confidence_treshold = 0.65
    for box in detections[0].boxes:

        # Skip boxes with low confidence
        if box.conf[0].item() < confidence_treshold:
            continue

        x1, y1, x2, y2 = box.xyxy[0]

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        object_roi = color_image[y1:y2, x1:x2]

        if not is_decoy(object_roi=object_roi):
            # get 3D centric position
            position = get_3d_position(depth_image=depth_image, K=K, x=(x1 + x2) // 2, y=(y1 + y2) // 2)
            positions.append(position)

    # plot img with positions
    # plot the 3D position of the first object
    pyplot.figure(figsize=(16, 9))
    pyplot.imshow(X=cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    idx = 0
    for box in detections[0].boxes:

        # Skip boxes with low confidence
        if box.conf[0].item() < confidence_treshold:
            continue

        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Draw the bounding box
        rect = pyplot.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2)
        pyplot.gca().add_patch(rect)
        pos_3d_text = (
            f"({positions[idx][0].item():.2f}, {positions[idx][1].item():.2f}, {positions[idx][2].item():.2f})"
        )
        pyplot.text(
            x1, y1, pos_3d_text, fontsize=8, color="blue", verticalalignment="bottom", horizontalalignment="left"
        )
        idx += 1
    pyplot.savefig(plot_dir / "detection_with_3D_positions.png", bbox_inches="tight", pad_inches=0.1)
    pyplot.show()

    return positions, len(detections[0])


def visualize_3d_positions(positions: numpy.ndarray, point_cloud: numpy.ndarray, plot_path: pathlib.Path) -> None:
    """Visualize 3D positions.

    Args:
        positions (list[numpy.ndarray]): List of 3D positions.
        point_cloud (PointCloud): Point cloud.
    """
    # normalize point cloud based in the range of the point cloud itself
    normalized_points = (positions - point_cloud.min()) / (point_cloud.max() - point_cloud.min()) / 100
    normalized_points[:, 2] = normalized_points[:, 2] / 10

    point_cloud = numpy.concatenate((point_cloud, normalized_points), axis=0)

    # Create RGB array for coloring the point cloud
    rgb = numpy.zeros(shape=(point_cloud.shape[0], 3), dtype=numpy.uint8)
    rgb[-positions.shape[0]] = numpy.ones(shape=(positions.shape[0], 3), dtype=numpy.uint8) * 255

    pyvista.plot(
        point_cloud,
        point_size=5,
        show_edges=True,
        scalars=rgb,
        cpos="xy",
    )
    # pl = pyvista.Plotter(off_screen=True)
    # pl.add_mesh(
    #     point_cloud,
    #     style="points_gaussian",
    #     color="#fff7c2",
    #     scalars=rgb,
    #     opacity=0.25,
    #     point_size=5,
    #     show_scalar_bar=False,
    #     show_edges=True,
    # )
    # pl.background_color = "k"
    # pl.show(auto_close=False)
    # path = pl.generate_orbital_path(n_points=36, shift=point_cloud.shape[0], factor=3.0)
    # pl.open_gif(plot_path)
    # pl.orbit_on_path(path, write_frames=True)
    # pl.close()


def main():
    """Main function."""
    data_dir = Config.data_dir / "raw" / "test"
    color_img_dir = data_dir / "camera_color_image_raw"
    depth_img_dir = data_dir / "camera_depth_image_raw"
    pcd_dir = data_dir / "camera_depth_points"
    plot_dir = Config.plot_dir / "yolov11_experiment"

    model_pt = Config.model_dir.parents[1] / "models" / "yolo_v11_extinguiser" / "weights" / "best.pt"
    model = YOLO(model=model_pt, task="detect", verbose=True)

    # importing camera parameters from yaml
    with open(
        color_img_dir.parent / "camera_color_camera_info" / "camera_color_info_1727164479160357265.txt", "r"
    ) as file:
        camera_params = yaml.safe_load(file)

    K = numpy.array(camera_params["K"]).reshape(3, 3)

    color_imgs = sorted(color_img_dir.glob("*.png"))
    depth_imgs = sorted(depth_img_dir.glob("*.png"))
    point_clouds = sorted(pcd_dir.glob("*.pcd"))

    # Image with several objects
    # detection = model(
    #     Config.data_dir / "FireExtinguiser" / "valid" / "images" / "30_jpg.rf.7a274213763dbf3a7a0c74e44bb34f24.jpg"
    # )

    # Image with decoys
    # color_imgs[0] = (
    #     Config.data_dir / "raw" / "test" / "camera_color_image_raw" / "camera_color_image_1727164485882972273.png"
    # )

    color_imgs = [color_imgs[0]]
    depth_imgs = [depth_imgs[0]]
    point_clouds = [point_clouds[0]]

    logger = get_logger(log_filename="yolov11_experiment.log")

    msg_task(msg="3D Poisition estimation - Yolov11 Experiment", logger=logger)

    logger.info(
        f"Loaded {len(color_imgs)} images for experiment of shape (640x400) and depth images of shape (640x400)."
    )

    for idx, (color_img, depth_img, point_cloud) in enumerate(zip(color_imgs, depth_imgs, point_clouds)):
        color_image = cv2.imread(color_img, cv2.IMREAD_COLOR)
        depth_image = cv2.imread(depth_img, cv2.IMREAD_UNCHANGED)
        point_cloud = PointCloud.from_path(point_cloud).numpy(fields=["x", "y", "z"])

        positions, n_extinguisers = process_frame(
            color_image=color_image, depth_image=depth_image, model=model, K=K, plot_dir=plot_dir / f"img{idx}"
        )

        if not positions:
            logger.info("No objects detected.")

        else:

            logger.info(f"Out of {n_extinguisers} extinguisers detected, {len(positions)} are detected to be real.")

            for idx, position in enumerate(positions):
                logger.info(f"Image {color_img.name} -> real position for extinguiser {idx}: {position}")

            visualize_3d_positions(
                positions=numpy.array(positions),
                point_cloud=point_cloud,
                plot_path=plot_dir / f"img{idx}" / "3d_positions.gif",
            )

        logger.info("Experiment completed.")


if __name__ == "__main__":
    main()
