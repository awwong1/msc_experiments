from typing import List, Union

import numpy as np
import pytorch_lightning as pl
import zarr
from torch.utils.data import DataLoader, Dataset, random_split

from utils import RangeKeyDict


class BeatsZarr(Dataset):
    """
    PyTorch Dataset class for Normalized beat windows
    beats/window_size_{window_size}_normalized

    Ensures no invalid split among records!
    """

    def __init__(
        self,
        zarr_group_path: str = "data/ecgs.zarr",
        window_size: int = 400,
        record_idxs: Union[None, List[int]] = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.root = zarr.open_group(zarr_group_path, mode="r")

        # If the record_idxs is None, we want to use all of the available records
        all_record_idxs = list(
            range(len(self.root[f"beats/window_size_{self.window_size}_shape"]))
        )
        if record_idxs is None:
            self.record_idxs = all_record_idxs
        else:
            # Check that all of the provided record idxs are in range
            assert all(record_id in all_record_idxs for record_id in record_idxs)
            self.record_idxs = record_idxs

        # Because each record may have multiple beats, create a map of beat_index to records
        beat_range_to_record_idx = {}
        beat_idx_accum = 0
        for record_idx, zarr_record_idx in enumerate(self.record_idxs):
            num_beats, _win_size, _num_leads = self.root[
                f"beats/window_size_{self.window_size}_shape"
            ][zarr_record_idx]
            beat_range_to_record_idx[
                (beat_idx_accum, beat_idx_accum + num_beats)
            ] = record_idx
            beat_idx_accum += num_beats
        self.num_beats = beat_idx_accum
        self.beat_range_to_record_idx = RangeKeyDict(beat_range_to_record_idx)
        # self.beat_range_to_record_idx = beat_range_to_record_idx
        self.record_idx_to_beat_range = dict(
            [(v, k) for (k, v) in beat_range_to_record_idx.items()]
        )

    def __len__(self):
        return self.num_beats

    def __getitem__(self, idx):
        record_idx = self.beat_range_to_record_idx[idx]
        zarr_idx = self.record_idxs[record_idx]
        l_offset, _ = self.record_idx_to_beat_range[record_idx]
        beat_offset = idx - l_offset

        # Return the beat window instance
        beat_window = self.root[f"beats/window_size_{self.window_size}"][
            zarr_idx
        ].reshape(self.root[f"beats/window_size_{self.window_size}_shape"][zarr_idx])[
            beat_offset
        ]

        # make writeable
        return np.array(beat_window)


class BeatsZarrDataModule(pl.LightningDataModule):
    def __init__(
        self,
        zarr_group_path: str = "data/ecgs.zarr",
        window_size: int = 400,
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

        # Determine train/validation/test splits of the available records
        test_ratio = 1 - train_ratio - val_ratio
        assert test_ratio >= 0, "Ratios must sum to 1"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def setup(self, *args):
        # determine how to split the zarr records
        root = zarr.open_group(self.zarr_group_path, mode="r")
        all_record_idxs = list(
            range(len(root[f"beats/window_size_{self.window_size}_shape"]))
        )

        total_num_records = len(all_record_idxs)
        num_train_records = int(total_num_records * self.train_ratio)
        num_val_records = int(total_num_records * self.val_ratio)
        num_test_records = total_num_records - num_train_records - num_val_records

        train_records, val_records, test_records = random_split(
            all_record_idxs, [num_train_records, num_val_records, num_test_records]
        )
        self.train_records = train_records
        self.val_records = val_records
        self.test_records = test_records

        self.ds_train = BeatsZarr(
            zarr_group_path=self.zarr_group_path,
            window_size=self.window_size,
            record_idxs=self.train_records,
        )
        self.ds_val = BeatsZarr(
            zarr_group_path=self.zarr_group_path,
            window_size=self.window_size,
            record_idxs=self.val_records,
        )
        self.ds_test = BeatsZarr(
            zarr_group_path=self.zarr_group_path,
            window_size=self.window_size,
            record_idxs=self.test_records,
        )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.train_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            num_workers=self.val_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            num_workers=self.test_workers,
        )
