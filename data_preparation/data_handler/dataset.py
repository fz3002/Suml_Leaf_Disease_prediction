import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from data_preparation.preprocessing.data_util import load_config_file, load_labels_map


class LeafDataset(Dataset):
    def __init__(self,
                 transform: Compose,
                 split: str = 'train'
                 ):

        if split not in ('train', 'val', 'test'):
            raise ValueError('[WARNING] LeafDataset __init__ - split should be one of train, val or test')
        self.transform: Compose = transform
        self.split: str = split
        self.paths: list[tuple[str, str]] = self._load_paths()
        self.labels_map: dict = load_labels_map()

    def _load_paths(self) -> list[tuple[str, str]]:
        config: dict = load_config_file()
        project_path: str = config['path']['project_path']
        split_path: str = config['path']['processed'][self.split]
        source_path: str = os.path.join(project_path, split_path)
        paths: list[tuple[str, str]] = []

        for class_name in os.listdir(source_path):
            class_path: str = os.path.join(source_path, class_name)
            for image_path in os.listdir(class_path):
                paths.append((os.path.join(class_path, image_path), class_name))

        random.shuffle(paths)
        return paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple:
        path, label = self.paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.labels_map[label]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

