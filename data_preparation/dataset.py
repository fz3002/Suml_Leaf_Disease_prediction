import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import random


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, transform=None):
        self.root: str = os.path.join(data_path, 'MangoLeafBD Dataset')
        self.data: list[str] = []
        self.labels: list[str] = []
        self.classes: dict[str, int] = {}
        self.train_data: list[tuple[str, str]] = []
        self.test_data: list[tuple[str, str]] = []
        self.transform = transform
        self.__upload_images()

    def __upload_images(self):
        for dir_name in os.listdir(self.root):
            self.classes[dir_name] = 0
            dir_path: str = os.path.join(self.root, dir_name)
            for filename in os.listdir(dir_path):
                self.data.append(os.path.join(dir_path, filename))
                self.labels.append(dir_name)
                self.classes[dir_name] = self.classes[dir_name] + 1

    def train_test_split(self, train_size: float = 0.8, random_state: int = 42):
        np.random.seed(random_state)
        indexes_randomised = np.random.choice(len(self.data), size=int(len(self.data)), replace=False)
        for index in range(len(indexes_randomised)):
            if index > train_size * len(indexes_randomised):
                self.test_data.append((self.data[indexes_randomised[index]], self.labels[indexes_randomised[index]]))
            else:
                self.train_data.append((self.data[indexes_randomised[index]], self.labels[indexes_randomised[index]]))

    def get_training_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def __getitem__(self, index):
        img = Image.open(self.data[index])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.data)

    def visualize(self, index) -> None:
        img = self.__getitem__(index)
        plt.imshow(img)
        plt.show()

    def show_four_images(self, random_indexes: bool= True) -> None:
        if random_indexes:
            indexes = [random.randint(0, len(self.data) - 1) for _ in range(4)]
        else:
            indexes = [i for i in range(4)]

        img1 = self.__getitem__(indexes[0])
        img2 = self.__getitem__(indexes[1])
        img3 = self.__getitem__(indexes[2])
        img4 = self.__getitem__(indexes[3])

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(img1)
        axs[0, 0].set_title('img1')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(img2)
        axs[0, 1].set_title('img2')
        axs[0, 1].axis('off')

        axs[1, 0].imshow(img3)
        axs[1, 0].set_title('img3')
        axs[1, 0].axis('off')

        axs[1, 1].imshow(img4)
        axs[1, 1].set_title('img4')
        axs[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

    def show_data_stats(self) -> None:
        print('Dataset statistics:')
        print('  number of images: {}'.format(len(self.data)))
        print('  number of classes: {}'.format(len(self.classes)))
        print('  number of images per class: ', self.classes)

    def get_data_classes(self) -> list[str]:
        return list(self.classes.keys())



