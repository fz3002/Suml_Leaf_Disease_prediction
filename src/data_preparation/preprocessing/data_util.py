import random
from pathlib import Path
import kagglehub
import os
import shutil
import yaml
from tqdm import tqdm


CONFIG_PATH: Path = Path(__file__).parent.parent.parent.parent


def download_dataset() -> None:
    """
    Download the Mango Leaf Disease dataset from Kaggle and store it locally.

    The function uses `kagglehub` to download the dataset
    "warcoder/mango-leaf-disease-dataset" and copies it into the raw data
    directory specified in `config.yaml`.

    If the destination directory already exists, it is removed before copying
    the newly downloaded dataset to ensure a clean state.

    :return: None
    """
    # Load configuration file
    config: dict = load_config_file()

    # Resolve project and destination paths
    project_path: str = config["path"]["project_path"]
    dest_path: str = os.path.join(project_path, config["path"]["raw"])

    # Download dataset using KaggleHub
    path: str = kagglehub.dataset_download(
        "warcoder/mango-leaf-disease-dataset"
    )
    print("[INFO] download_dataset() - Downloading Dataset ", path)

    # Handle destination directory
    if os.path.exists(dest_path):
        print("[INFO] download_dataset() - Destination path already exists")
        print("[INFO] download_dataset() - Deleting destination path")
        shutil.rmtree(dest_path)
    else:
        os.makedirs(dest_path, exist_ok=True)
        print("[INFO] download_dataset() - Creating destination path")

    # Copy downloaded dataset to the destination directory
    shutil.copytree(path, dest_path, dirs_exist_ok=True)

    print(print("[INFO] download_dataset() - Saved dataset to: ", dest_path))


def split_dataset(config_part: str) -> None:
    """
    Split a raw image dataset into training, validation, and test subsets.

    The function reads split configuration parameters from `config.yaml`,
    creates the required directory structure, and copies images from the
    raw dataset into train/val/test folders according to the specified ratios.

    The split is performed independently for each class directory found
    in the raw dataset path.

    :param config_part: Split configuration parameter.
    :return: None
    """

    if config_part not in ('split', 'processed'):
        raise ValueError("[WARNING] split_dataset() - config_part must be 'split' or 'processed'")

    # Load configuration file
    config: dict = load_config_file()

    # Resolve main project paths
    project_path: str = config["path"]["project_path"]
    dest_path: str = os.path.join(project_path, config["path"][config_part]["root"])
    raw_path: str = os.path.join(project_path, config["path"]["raw"])

    # Dataset split configuration
    split_info: dict = config["preprocess"]["split"]

    # Path to the raw dataset directory
    source_path: str = os.path.join(raw_path, "MangoLeafBD Dataset")

    # Create destination root directory if it does not exist
    if not os.path.exists(dest_path):
        os.makedirs(dest_path, exist_ok=True)
        print("[INFO] split_dataset() - Creating destination path")

    if split_info.get("enable"):
        # Target directories for each dataset split
        train_dir: str = os.path.join(dest_path, "train")
        val_dir: str = os.path.join(dest_path, "val")
        test_dir: str = os.path.join(dest_path, "test")

        # Create split directories
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Read split parameters
        train_ratio: float = split_info["train_ratio"]
        val_ratio: float = split_info["val_ratio"]
        shuffle: bool = split_info["shuffle"]
        seed: int = split_info["seed"]

        # Initialize random seed for reproducibility
        random.seed(seed)

        # Iterate over each class directory in the raw dataset
        for class_name in tqdm(os.listdir(source_path), desc="Split dataset "):
            source_class_dir: str = os.path.join(source_path, class_name)

            # Skip non-directory entries
            if not os.path.isdir(source_class_dir):
                continue

            # Create class subdirectories for each split
            class_train_dir: str = os.path.join(train_dir, class_name)
            class_val_dir: str = os.path.join(val_dir, class_name)
            class_test_dir: str = os.path.join(test_dir, class_name)

            os.makedirs(class_train_dir, exist_ok=True)
            os.makedirs(class_val_dir, exist_ok=True)
            os.makedirs(class_test_dir, exist_ok=True)

            # Collect all image files for the current class
            images: list[str] = [f for f in os.listdir(source_class_dir) if os.path.isfile(os.path.join(source_class_dir, f))]

            # Shuffle images if enabled
            if shuffle:
                random.shuffle(images)

            total: int = len(images)
            train_end: int = int(total * train_ratio)
            val_end: int = train_end + int(total * val_ratio)

            # Split image lists
            train_files: list[str] = images[:train_end]
            val_files: list[str] = images[train_end:val_end]
            test_files: list[str] = images[val_end:]

            # Copy training images
            for f in train_files:
                shutil.copy(os.path.join(source_class_dir, f), os.path.join(class_train_dir, f))

            # Copy validation images
            for f in val_files:
                shutil.copy(os.path.join(source_class_dir, f), os.path.join(class_val_dir, f))

            # Copy test images
            for f in test_files:
                shutil.copy(os.path.join(source_class_dir, f), os.path.join(class_test_dir, f))


def load_config_file() -> dict:
    """
    Load the YAML configuration file and return its contents as a dictionary.

    This function reads the `config.yaml` file located at `CONFIG_PATH`
    and parses it using `yaml.safe_load`.

    :return: Parsed contents of the YAML configuration file.
    """

    # Path to the configuration file
    config_path = CONFIG_PATH / 'config.yaml'

    with open(config_path, 'r', encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    return config


def load_labels_map() -> dict[str, int]:
    """
    Create a mapping from class names to integer labels.

    The function:
    1. First tries to read class names from dataset directory (dynamic loading)
    2. If dataset not found, falls back to static JSON file (for inference without raw data)
    3. Assigns consecutive integer labels to each class in alphabetical order.

    :return: Dictionary mapping class names to integer indices.
    """
    
    config: dict = load_config_file()
    label_map: dict[str, int] = {}
    project_path: str = config["path"]["project_path"]
    raw_path: str = config["path"]["raw"]

    # Path to the dataset directory containing class subfolders
    source_path = os.path.join(project_path, raw_path, "MangoLeafBD Dataset")

    # Try to load from dataset directory first (dynamic loading)
    if os.path.exists(source_path):
        try:
            classes: list[str] = os.listdir(source_path)
            classes = sorted(classes)
            
            # Assign integer labels to classes
            index: int = 0
            for class_name in classes:
                label_map[class_name] = index
                index += 1
            
            return label_map
        except Exception:
            pass  # Fall through to static JSON fallback
    
    # Fallback: try to load from static JSON file
    static_map_path = Path(__file__).parent.parent.parent.parent / "frontend" / "data" / "labels_map.json"
    if static_map_path.exists():
        try:
            import json
            with open(static_map_path, "r", encoding="utf-8") as f:
                label_map = json.load(f)
            return label_map
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load labels map from dataset ({source_path}) or static JSON ({static_map_path}). Error: {e}"
            )
    
    raise FileNotFoundError(
        f"Dataset path not found ({source_path}) and static labels_map.json not found ({static_map_path})"
    )
