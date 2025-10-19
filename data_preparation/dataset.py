import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, transform=None):
        self.root: str = data_path
        self.data: list[str] = []
        self.labels: list[str] = []
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.data[index])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.data)

    def visualize(self, index) -> None:
        img = Image.open(self.data[index])
        if self.transform is not None:
            img = self.transform(img)
        plt.imshow(img)
        plt.show()

    def show_four_images(self, random: bool= True) -> None:
        pass

    def show_data_stats(self) -> None:
        pass

    def get_data_classes(self) -> list[str]:
        pass



