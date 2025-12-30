from torch.utils.data import DataLoader
from data_preparation.data_handler.dataset import LeafDataset
from data_preparation.data_handler.transform import Transforms
from data_preparation.preprocessing.data_util import load_config_file


class DataHandler:
    def __init__(self):
        self.config = load_config_file()
        self.transform = Transforms()
        self.train_dataset = LeafDataset(transform=self.transform.get_train_transform(), split='train')
        self.val_dataset = LeafDataset(transform=self.transform.get_val_transform(), split='val')
        self.test_dataset = LeafDataset(transform=self.transform.get_test_transform(), split='test')
        self.dataloaders: dict[str, DataLoader] = self._set_data_loaders()

    def _set_data_loaders(self) -> dict[str, DataLoader]:
        data_loaders: dict[str, DataLoader] = {}

        for split in ['train', 'val', 'test']:
            batch_size = self.config[split]['batch_size']
            num_workers = self.config[split]['num_workers']
            if split == 'train':
                data_loaders[split] = DataLoader(
                    self.train_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            elif split == 'val':
                data_loaders[split] = DataLoader(
                    self.val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            elif split == 'test':
                data_loaders[split] = DataLoader(
                    self.test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    pin_memory=True,
                )

        return data_loaders

    def get_dataloader(self, split: str) -> DataLoader:
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'[WARNING] DataHandler get_dataloaders() - split {split} is not supported')
        return self.dataloaders[split]