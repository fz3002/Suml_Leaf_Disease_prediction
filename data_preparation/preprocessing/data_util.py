import random
from pathlib import Path
import kagglehub
import os
import shutil
import yaml
from tqdm import tqdm

CONFIG_PATH: Path = Path(__file__).parent.parent.parent


def download_dataset() -> None:
    """
    Funckja pobiera dataset Mango Leaf Disease z Kaggle i umieszcza go w folderze
    data/1_raw
    :return: None
    """
    config: dict = load_config_file()
    project_path: str = config["path"]["project_path"]
    dest_path: str = os.path.join(project_path, config["path"]["raw"])

    path: str = kagglehub.dataset_download("warcoder/mango-leaf-disease-dataset")
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
    """

    :param config_part: Albo split albo processed
    :return: None
    """

    if config_part not in ('split', 'processed'):
        raise ValueError("[WARNING] split_dataset() - config_part must be 'split' or 'processed'")

    # Pobieramy potrzebne ścieżki: projektu i folderów
    config: dict = load_config_file()   # Pobieramy plik config.yaml w postaci słownika
    project_path: str = config["path"]["project_path"]  # Ścieżka do projektu
    processed_path: str = os.path.join(project_path, config["path"][config_part]["root"])   # Wybrana ścieżka root
    raw_path: str = os.path.join(project_path, config["path"]["raw"])   # Ścieżka do surowego datasetu
    split_info: dict = config["preprocess"]["split"]    # Informacja o tym jak splitować dataset
    source_path: str = os.path.join(raw_path, "MangoLeafBD Dataset")    # Ściezka do surowego datasetu

    if split_info.get("enable"):
        # Docelowe ścieżki z podziałem na proces
        train_dir: str = os.path.join(processed_path, "train")
        val_dir: str = os.path.join(processed_path, "val")
        test_dir: str = os.path.join(processed_path, "test")

        # Tworzymy docelowe ścieżki
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Bierzemy dokłądne wartości co do splitu
        train_ratio: float = split_info["train_ratio"]
        val_ratio: float = split_info["val_ratio"]
        shuffle: bool = split_info["shuffle"]
        seed: int = split_info["seed"]

        # Inicjalizujemy ziarno
        random.seed(seed)

        # Dla każdego folderu (klasa) w folderze z surowymi danymi
        for class_name in tqdm(os.listdir(source_path), desc="Split dataset "):
            source_class_dir: str = os.path.join(source_path, class_name)

            # Jeżeli to nie folder to pomijamy
            if not os.path.isdir(source_class_dir):
                continue

            # Tworzymy ścieżki foldery klasy w każdym splicie
            class_train_dir: str = os.path.join(train_dir, class_name)
            class_val_dir: str = os.path.join(val_dir, class_name)
            class_test_dir: str = os.path.join(test_dir, class_name)

            # Tworzymy te foldery
            os.makedirs(class_train_dir, exist_ok=True)
            os.makedirs(class_val_dir, exist_ok=True)
            os.makedirs(class_test_dir, exist_ok=True)

            # Zbieramy ścieżki wszystkich zdjęć z danej klasy z surowego folderu
            images: list[str] = [
                f for f in os.listdir(source_class_dir)
                if os.path.isfile(os.path.join(source_class_dir, f))
            ]

            # Mieszamy zdjęcia
            if shuffle:
                random.shuffle(images)

            total: int = len(images) # Ilość wszystkich zdjęć na klasę
            train_end: int = int(total * train_ratio) # Ilość zdjęć treningowych
            val_end: int = train_end + int(total * val_ratio) # Ilość zdjęć walidacyjnych

            train_files: list[str] = images[:train_end] # Ścieżki ze zdjęciemi traningowymi
            val_files: list[str] = images[train_end:val_end] # Ścieżki ze zdjęciami walidacyjnymi
            test_files: list[str] = images[val_end:] # Ścieżki ze zdjęciami testowymi

            # Kopiujemy do folderu treningowego
            for f in train_files:
                shutil.copy(
                    os.path.join(source_class_dir, f),
                    os.path.join(class_train_dir, f)
                )

            # kopiujemy do folderu walidacyjnego
            for f in val_files:
                shutil.copy(
                    os.path.join(source_class_dir, f),
                    os.path.join(class_val_dir, f)
                )

            # Kopiujemy do folderu testowego
            for f in test_files:
                shutil.copy(
                    os.path.join(source_class_dir, f),
                    os.path.join(class_test_dir, f)
                )


def load_config_file() -> dict:
    """
    Funkcja wczytuje plik .yaml i zwraca cały plik jako słownik
    :return: Zawartość pliku .yaml jako słownik
    """
    config_path = CONFIG_PATH / 'config.yaml' # Ścieżka do pliku config.yaml

    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def load_labels_map() -> dict[str, int]:
    config = load_config_file()
    label_map: dict[str, int] = {}
    project_path: str = config["path"]["project_path"]
    raw_path: str = config["path"]["raw"]

    source_path = os.path.join(project_path, raw_path, "MangoLeafBD Dataset")

    classes: list[str] = os.listdir(source_path)
    classes = sorted(classes)
    index: int = 0
    for class_name in classes:
        label_map[class_name] = 0
        index += 1
    return label_map
