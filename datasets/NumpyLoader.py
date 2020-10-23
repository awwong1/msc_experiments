import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split


class NumpyLoader(Dataset):
    def __init__(self, numpy_file: str):
        super().__init__()
        self.numpy_file = numpy_file
        self.matrix = np.load(numpy_file, mmap_mode="r")

    def __len__(self):
        return len(self.matrix)

    def __getitem__(self, idx):
        return np.array(self.matrix[idx]).astype(np.float32)


class NumpyLoaderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        numpy_file: str,
        batch_size: int = 32,
        train_workers: int = 8,
        val_workers: int = 4,
        ratio_train: float = 0.8,
    ):
        super().__init__()
        self.numpy_file = numpy_file
        self.batch_size = batch_size
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.ratio_train = ratio_train

    def setup(self, stage=None):
        dataset = NumpyLoader(self.numpy_file)
        train_len = int(len(dataset) * self.ratio_train)
        val_len = len(dataset) - train_len
        train, val = random_split(dataset, [train_len, val_len])

        self.train_ds = train
        self.val_ds = val

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.train_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.val_workers,
        )
