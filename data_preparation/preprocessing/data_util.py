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
    pass


def load_config_path() -> dict:
    config_path = CONFIG_PATH / 'config.yaml'
    config = yaml.safe_load(str(config_path))
    return config


