# -*- utf-8 -*-
"""Yolov11 training."""
import torch
from ultralytics import YOLO

from common.config import Config
from common.log import get_logger


def train_yolov11():
    """Train YOLOv11 model."""
    logger = get_logger(log_filename=Config.log_dir / "train_yolov11.log")

    data_yaml = Config.data_dir / "FireExtinguiser" / "data.yaml"
    model_pt = Config.model_dir / "yolo11m.pt"

    # Purge CUDA cache
    torch.cuda.empty_cache()

    model = YOLO(model=model_pt, task="detect", verbose=True)

    # Train the model
    train_results = model.train(
        data=data_yaml,  # path to dataset YAML
        model=Config.model_dir / "pretrained_yolo11m.pt",  # path to model.pt
        epochs=10,  # number of training epochs
        imgsz=640,  # training image size
        device="cuda:0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        # workers=12,  # number of workers
        batch=1,  # batch size
        exist_ok=True,
        #save_json=True,
    )

    # Export the model to ONNX format
    path = model.export()  # return path to exported model

    logger.info(f"\n\nModel exported to {path}\n")

    # Evaluate model performance on the validation set
    metrics = model.val()

    logger.info(f"Training results: {metrics}\n\n")
    logger.info(f"Training results: {train_results}")


if __name__ == "__main__":
    train_yolov11()
