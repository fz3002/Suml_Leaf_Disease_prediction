import random
from pathlib import Path
import kagglehub
import os
import shutil
import yaml

CONFIG_PATH: Path = Path(__file__).parent.parent.parent


def download_dataset() -> None:
    config = load_config_file()
    dest_path = config["path"]["raw"]

    path = kagglehub.dataset_download("warcoder/mango-leaf-disease-dataset")
    print("[INFO] download_dataset() - Downloading Dataset ", path)

    if os.path.exists(dest_path):
        print("[INFO] download_dataset() - Destination path already exists")
        print("[INFO] download_dataset() - Deleting destination path")
        shutil.rmtree(dest_path)
    else:
        os.makedirs(dest_path, exist_ok=True)
        print("[INFO] download_dataset() - Creating destination path")

    shutil.copytree(path, dest_path)

    print(print("[INFO] download_dataset() - Saved dataset to: ", dest_path))


def split_dataset(config_part: str) -> None:
    config = load_config_file()
    processed_path: str = config["path"][config_part]["root"]
    raw_path: str = config["path"]["1_raw"]
    split_info: dict = config["preprocess"]["split"]

    source_path: str = os.path.join(raw_path, "MangoLeafBD Dataset")

    if split_info.get("enable"):
        train_dir: str = os.path.join(processed_path, "train")
        val_dir: str = os.path.join(processed_path, "val")
        test_dir: str = os.path.join(processed_path, "test")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        train_ratio: float = split_info["train_ratio"]
        val_ratio: float = split_info["val_ratio"]
        shuffle: bool = split_info["shuffle"]
        seed: int = split_info["seed"]

        random.seed(seed)

        for class_name in os.listdir(source_path):
            source_class_dir: str = os.path.join(source_path, class_name)

            if not os.path.isdir(source_class_dir):
                continue

            class_train_dir = os.path.join(train_dir, class_name)
            class_val_dir = os.path.join(val_dir, class_name)
            class_test_dir = os.path.join(test_dir, class_name)

            os.makedirs(class_train_dir, exist_ok=True)
            os.makedirs(class_val_dir, exist_ok=True)
            os.makedirs(class_test_dir, exist_ok=True)

            images = [
                f for f in os.listdir(source_class_dir)
                if os.path.isfile(os.path.join(source_class_dir, f))
            ]

            if shuffle:
                random.shuffle(images)

            total = len(images)
            train_end = int(total * train_ratio)
            val_end   = train_end + int(total * val_ratio)

            train_files = images[:train_end]
            val_files   = images[train_end:val_end]
            test_files  = images[val_end:]

            for f in train_files:
                shutil.copy(
                    os.path.join(source_class_dir, f),
                    os.path.join(class_train_dir, f)
                )

            for f in val_files:
                shutil.copy(
                    os.path.join(source_class_dir, f),
                    os.path.join(class_val_dir, f)
                )

            for f in test_files:
                shutil.copy(
                    os.path.join(source_class_dir, f),
                    os.path.join(class_test_dir, f)
                )


def load_config_file() -> dict:
    config_path = CONFIG_PATH / 'config.yaml'
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config
