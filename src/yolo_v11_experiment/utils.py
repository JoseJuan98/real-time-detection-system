# -*- coding: utf-8 -*-
"""Utility functions."""

import pathlib
import os

import numpy
from matplotlib import pyplot


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


def get_depth_image(color_image_name, depth_images_dir):
    """Get Name of the depth image in close timeframe.

    Args:
        color_image_name: Color image name.
        depth_images_dir: path of the depth images directory

    Returns:
        closest_depth_image: Name of the depth image.
    """
    # Extract timestamp from the color image filename
    color_timestamp = int(color_image_name.rsplit("_", 1)[1].split(".")[0])

    # List all depth images
    depth_images = [f for f in os.listdir(depth_images_dir) if os.path.isfile(os.path.join(depth_images_dir, f))]

    # Initialize variables to track the closest image
    closest_depth_image = None
    min_time_diff = float("inf")

    # Find the depth image with the closest timestamp
    for depth_image in depth_images:
        # Extract timestamp from the depth image filename
        depth_timestamp = int(depth_image.rsplit("_", 1)[1].split(".")[0])

        # Calculate time difference
        time_diff = abs(color_timestamp - depth_timestamp)

        # Update the closest image if a closer one is found
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_depth_image = depth_image

    return closest_depth_image
