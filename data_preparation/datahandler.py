from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class ImageDataHandler(Dataset):
    def __init__(self, data: list[tuple[str, str]], transform: transforms.Compose = None,
                 batch_size: int = 32, shuffle: bool = True):
        self.data = data
        self.transform = transform
        self.loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx][0]
        label = self.data[idx][1]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __iter__(self):
        return iter(self.loader)



