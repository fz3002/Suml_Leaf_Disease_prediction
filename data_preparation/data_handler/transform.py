import torchvision.transforms as t
from data_preparation.preprocessing.data_util import load_config_file


class Transforms:
    def __init__(self):
        self.config: dict = load_config_file()

    def get_train_transform(self) -> t.Compose:
        train_config: dict = self.config["transforms"]["train"]
        transform: list = []

        if train_config.get("to_tensor", False):
            transform.append(t.ToTensor())

        if train_config.get('resize') is not None:
            transform.append(t.Resize(tuple(train_config['resize'])))

        if train_config.get("random_rotation") is not None:
            transform.append(t.RandomRotation(degrees=train_config['random_rotation']))

        if train_config.get("random_horizontal_flip", False):
            transform.append(t.RandomHorizontalFlip())

        color_jitter: dict = train_config.get("color_jitter")
        if color_jitter is not None:
            transform.append(t.ColorJitter(
                brightness=color_jitter['brightness'],
                contrast=color_jitter['contrast'],
                saturation=color_jitter['saturation'],
            ))
        if train_config.get("normalize") is not None:
            normalize = train_config.get("normalize")
            transform.append(t.Normalize(mean=normalize['mean'], std=normalize['std']))

        return t.Compose(transform)

    def get_val_transform(self) -> t.Compose:
        val_config: dict = self.config["transforms"]["val"]
        transform: list = []

        if val_config.get("to_tensor", False):
            transform.append(t.ToTensor())

        if val_config.get("resize") is not None:
            transform.append(t.Resize(tuple(val_config['resize'])))

        if val_config.get("center_crop") is not None:
            transform.append(t.CenterCrop(val_config['center_crop']))

        if val_config.get("normalize") is not None:
            normalize = val_config.get("normalize")
            transform.append(t.Normalize(mean=normalize['mean'], std=normalize['std']))

        return t.Compose(transform)

    def get_test_transform(self) -> t.Compose:
        test_config: dict = self.config["transforms"]["test"]
        transform: list = []
        if test_config.get("to_tensor", False):
            transform.append(t.ToTensor())

        if test_config.get("resize") is not None:
            transform.append(t.Resize(tuple(test_config['resize'])))

        if test_config.get("normalize") is not None:
            normalize = test_config.get("normalize")
            transform.append(t.Normalize(mean=normalize['mean'], std=normalize['std']))

        return t.Compose(transform)



