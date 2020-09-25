from typing import List, Union
from operator import itemgetter

import numpy as np
import pytorch_lightning as pl
import torch
import zarr
from torch.utils.data import DataLoader, Dataset, random_split

# https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189
torch.multiprocessing.set_sharing_strategy('file_system')


class SequenceZarr(Dataset):
    """PyTorch Dataset class for returning sequences of beat windows."""

    def __init__(
        self,
        zarr_group_path: str = "data/ecgs.zarr",
        window_size: int = 400,
        sequence_length: int = 10,
        record_idxs: Union[None, List[int]] = None,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.root = zarr.open_group(zarr_group_path, mode="r")
        if record_idxs is None:
            self.record_idxs = self.list(
                range(len(self.root[f"beats/window_size_{window_size}_normalized"]))
            )
        else:
            # Check that all of the provided record idxs are in range
            # assert all(record_id in list(
            #     range(len(self.root[f"beats/window_size_{window_size}_normalized"]))
            # ) for record_id in record_idxs)
            self.record_idxs = sorted(record_idxs)

    def __len__(self):
        return len(self.record_idxs)

    def __getitem__(self, idx):
        record_idx = self.record_idxs[idx]
        # Get the normalized beat windows and the outlier index
        beat_windows = self.root[f"beats/window_size_{self.window_size}_normalized"][
            record_idx
        ].reshape(
            self.root[f"beats/window_size_{self.window_size}_shape"][record_idx],
            order="C",
        )
        outlier = self.root[f"beats/window_size_{self.window_size}_outlier"][record_idx]

        # Return a sequence of `sequence_length` beat windows, containin the outlier in the middle
        start_idx = 0
        end_idx = len(beat_windows) - 1

        # if sequence length is greater than length of beat_windows, return entire beat_windows
        if self.sequence_length < len(beat_windows):
            # return a sequence that contains the outlier beat (try to put it in the middle of the array)
            half_len = self.sequence_length // 2
            if outlier - half_len in range(len(beat_windows)):
                start_idx = outlier - half_len
                end_offset = 0
            else:
                # cannot place outlier in the 'middle', offset end
                end_offset = abs(outlier - half_len)

            if outlier + half_len + end_offset in range(len(beat_windows)):
                end_idx = outlier + half_len + end_offset
            else:
                # cannot outlier in the middle, offset start
                end_idx = len(beat_windows) - 1
                start_idx = end_idx - self.sequence_length

        # make writeable
        beat_seq = np.array(beat_windows[start_idx:end_idx])
        del beat_windows

        # change all nan to 0
        np.nan_to_num(beat_seq, copy=False)
        return beat_seq


def collate_fn(batch):
    # custom collate function to support variable length inputs
    data = [torch.Tensor(beat_window) for beat_window in batch]
    lengths = [len(beat_window) for beat_window in batch]
    # sort such that lengths are from longest to shortest
    data, lengths = zip(*sorted(zip(data, lengths), key=itemgetter(1), reverse=True))
    return [data, lengths]


class SequenceZarrDataModule(pl.LightningDataModule):
    def __init__(
        self,
        zarr_group_path: str = "data/ecgs.zarr",
        window_size: int = 400,
        sequence_length: int = 10,
        record_splits: Union[None, dict] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        batch_size: int = 128,
        train_workers: int = 16,
        val_workers: int = 8,
        test_workers: int = 8,
    ):
        super().__init__()
        self.zarr_group_path = zarr_group_path
        self.window_size = window_size
        self.batch_size = batch_size

        self.train_workers = train_workers
        self.val_workers = val_workers
        self.test_workers = test_workers

        # If the record_splits are not provided, randomly split records
        # otherwise use provided splits.
        if record_splits is None:
            test_ratio = 1 - train_ratio - val_ratio
            assert test_ratio >= 0, "Ratios must sum to 1"
            train_ratio = train_ratio
            val_ratio = val_ratio
            test_ratio = test_ratio

            # determine how to split the zarr records
            root = zarr.open_group(self.zarr_group_path, mode="r")
            all_record_idxs = list(
                range(len(root[f"beats/window_size_{self.window_size}_shape"]))
            )

            total_num_records = len(all_record_idxs)
            num_train_records = int(total_num_records * train_ratio)
            num_val_records = int(total_num_records * val_ratio)
            num_test_records = total_num_records - num_train_records - num_val_records

            train_records, val_records, test_records = random_split(
                all_record_idxs, [num_train_records, num_val_records, num_test_records]
            )
            self.train_records = train_records
            self.val_records = val_records
            self.test_records = test_records
        else:
            self.train_records = record_splits["train_records"]
            self.val_records = record_splits["val_records"]
            self.test_records = record_splits["test_records"]

        self.ds_train = SequenceZarr(
            zarr_group_path=self.zarr_group_path,
            window_size=self.window_size,
            record_idxs=self.train_records,
        )
        self.ds_val = SequenceZarr(
            zarr_group_path=self.zarr_group_path,
            window_size=self.window_size,
            record_idxs=self.val_records,
        )
        self.ds_test = SequenceZarr(
            zarr_group_path=self.zarr_group_path,
            window_size=self.window_size,
            record_idxs=self.test_records,
        )

        print(
            f"Train on {len(self.train_records)} records ({len(self.ds_train)} samples)"
        )
        print(f"Val on {len(self.val_records)} records ({len(self.ds_val)} samples)")
        print(f"Test on {len(self.test_records)} records ({len(self.ds_test)} samples)")

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.train_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            num_workers=self.val_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            num_workers=self.test_workers,
            collate_fn=collate_fn,
        )

    def data_config(self):
        return {
            "train_records": self.train_records,
            "val_records": self.val_records,
            "test_records": self.test_records,
        }
