"""Initial YOLOv8n settings and prepared dataset checks."""

from pathlib import Path
import random

import torch
from ultralytics import YOLO
import yaml


PROJECT_DIR = Path(__file__).resolve().parent
DATA_YAML = PROJECT_DIR / "data.yaml"
MODEL_NAME = "yolov8n.pt"
SEED = 42
IMAGE_SIZE = 640
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def check_dataset() -> None:
    if not DATA_YAML.is_file():
        raise SystemExit("data.yaml was not found. Run prepare_dataset.py first.")
    with DATA_YAML.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not config.get("names"):
        raise SystemExit("data.yaml must contain train/val/test paths and names.")

    data_root = Path(config.get("path", PROJECT_DIR))
    if not data_root.is_absolute():
        data_root = PROJECT_DIR / data_root
    print("Classes: {}".format(config["names"]))
    for split in ("train", "val", "test"):
        if split not in config:
            raise SystemExit("data.yaml is missing the path {}.".format(split))
        images_dir = Path(config[split])
        if not images_dir.is_absolute():
            images_dir = data_root / images_dir
        labels_dir = images_dir.parent.parent / "labels" / split
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise SystemExit("The {} folders were not found. Run prepare_dataset.py.".format(split))
        images = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        if not images:
            raise SystemExit("The {} image folder is empty.".format(split))
        if any(not (labels_dir / (p.stem + ".txt")).is_file() for p in images):
            raise SystemExit("An image in {} has no matching .txt file.".format(split))
        print("{}: {} images, labels found".format(split, len(images)))


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    print("Model: {}".format(MODEL_NAME))
    print("Seed: {}; image size: {}; device: {}".format(SEED, IMAGE_SIZE, DEVICE))
    print("Dataset configuration: {}".format(DATA_YAML))
    check_dataset()
    print("Initial settings are ready.")


if __name__ == "__main__":
    main()
