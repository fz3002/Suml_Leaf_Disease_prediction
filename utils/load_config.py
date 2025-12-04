from pathlib import Path

import yaml
from dataclasses import dataclass


CONFIG_PATH = Path(__file__).parents[1] / 'config.yaml'


@dataclass
class DataPaths:
    project_path: str
    raw: str
    split: dict[str, str]
    processed: dict[str, str]
    model_weights: str

@dataclass
class PreprocessConfig:
    denoise: dict | None
    clahe: dict | None
    gamma_correction: dict | None
    white_balance: dict | None
    validate_images: dict | None
    remove_duplicates: dict | None
    split: dict

@dataclass
class TransformConfig:
    transform_dict: dict

@dataclass
class ModelConfig:
    name: str
    version: str
    num_classes: int
    init_weights: bool
    dropout: float

@dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    scheduler: dict
    num_workers: int
    device: str
    seed: int
    run_name: str

@dataclass
class ProcessedConfig:
    batch_size: int
    num_workers: int

@dataclass
class MLFlowConfig:
    ui_enabled: bool
    tracking_url: str
    artifact_root: str
    experiment_name: str
    host: str
    port: int
    log_params: bool
    log_metrics: bool
    log_artifacts: bool
    artifacts: dict


@dataclass
class ConfigContent:
    data_paths: DataPaths
    preprocess: PreprocessConfig
    train_transforms: TransformConfig
    val_transforms: TransformConfig
    test_transforms: TransformConfig
    model: ModelConfig
    train: TrainConfig
    val: ProcessedConfig
    test: ProcessedConfig
    mlflow: MLFlowConfig


def load_config_file() -> ConfigContent:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        content = yaml.load(f, Loader=yaml.FullLoader)

    config_content = ConfigContent(
        data_paths=DataPaths(**content.get('path')),
        preprocess=PreprocessConfig(**content.get('preprocess')),
        train_transforms = TransformConfig(content.get('transforms').get('train')),
        val_transforms = TransformConfig(content.get('transforms').get('val')),
        test_transforms = TransformConfig(content.get('transforms').get('test')),
        model=ModelConfig(**content.get('model')),
        train=TrainConfig(**content.get('train')),
        val=ProcessedConfig(**content.get('val')),
        test=ProcessedConfig(**content.get('test')),
        mlflow=MLFlowConfig(**content.get('mlflow')),
    )

    return config_content