import random
from pathlib import Path
import kagglehub
import os
import shutil
import yaml

CONFIG_PATH: Path = Path(__file__).parent.parent.parent


def download_dataset(dest_path=None):
    path = kagglehub.dataset_download("warcoder/mango-leaf-disease-dataset")
    print("Dataset pobrany do:", path)

    if dest_path is None:
        dest_path = os.path.join(os.getcwd(), 'data')
    print("Docelowa ścieżka:", dest_path)

    if os.path.exists(dest_path):
        print("Folder docelowy istnieje – usuwam...")
        shutil.rmtree(dest_path)

    shutil.copytree(path, dest_path)

    print("Dataset zapisany w:", dest_path)
    return dest_path


def prepare_raw_dataset() -> None:
    config = load_config_file()

    processed_path = config["path"]["3_processed"]["root"]
    raw_path = config["path"]["1_raw"]
    split_info = config["preprocess"]["split"]

    source_path = os.path.join(raw_path, "MangoLeafBD Dataset")

    if split_info.get("enable"):
        train_dir = os.path.join(processed_path, "train")
        val_dir   = os.path.join(processed_path, "val")
        test_dir  = os.path.join(processed_path, "test")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir,   exist_ok=True)
        os.makedirs(test_dir,  exist_ok=True)

        train_ratio = split_info["train_ratio"]
        val_ratio   = split_info["val_ratio"]
        shuffle     = split_info["shuffle"]
        seed        = split_info["seed"]

        random.seed(seed)

        for class_name in os.listdir(source_path):
            source_class_dir = os.path.join(source_path, class_name)

            if not os.path.isdir(source_class_dir):
                continue

            class_train_dir = os.path.join(train_dir, class_name)
            class_val_dir   = os.path.join(val_dir, class_name)
            class_test_dir  = os.path.join(test_dir, class_name)

            os.makedirs(class_train_dir, exist_ok=True)
            os.makedirs(class_val_dir,   exist_ok=True)
            os.makedirs(class_test_dir,  exist_ok=True)

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

            print(f"[INFO] Klasa '{class_name}': "
                  f"{len(train_files)} train, {len(val_files)} val, {len(test_files)} test")


def load_config_file() -> dict:
    config_path = CONFIG_PATH / 'config.yaml'
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


