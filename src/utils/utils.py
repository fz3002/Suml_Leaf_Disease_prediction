from pathlib import Path
import yaml


CONFIG_PATH: Path = Path(__file__).parent.parent.parent / "config"

def load_model_config():
    with open(CONFIG_PATH / 'model_config.yaml', 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    return model_config


def load_dataset_config():
    with open(CONFIG_PATH / 'dataset_config.yaml', 'r') as f:
        dataset_config = yaml.load(f, Loader=yaml.FullLoader)
    return dataset_config


def load_process_config():
    with open(CONFIG_PATH / 'process_config.yaml', 'r') as f:
        process_config = yaml.load(f, Loader=yaml.FullLoader)
    return process_config
