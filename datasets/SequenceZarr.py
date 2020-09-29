import json
from typing import List, Union
from operator import itemgetter

import numpy as np
import pytorch_lightning as pl
import torch
import zarr
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset, random_split

# https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189
torch.multiprocessing.set_sharing_strategy("file_system")

SCORED_CODES = [
    270492004,  # ['IAVB', '1st degree av block']
    164889003,  # ['AF', 'atrial fibrillation']
    164890007,  # ['AFL', 'atrial flutter']
    426627000,  # ['Brady', 'bradycardia']
    713427006,  # ['CRBBB', 'complete right bundle branch block']
    713426002,  # ['IRBBB', 'incomplete right bundle branch block']
    445118002,  # ['LAnFB', 'left anterior fascicular block']
    39732003,  # ['LAD', 'left axis deviation']
    164909002,  # ['LBBB', 'left bundle branch block']
    251146004,  # ['LQRSV', 'low qrs voltages']
    698252002,  # ['NSIVCB', 'nonspecific intraventricular conduction disorder']
    10370003,  # ['PR', 'pacing rhythm']
    284470004,  # ['PAC', 'premature atrial contraction']
    427172004,  # ['PVC', 'premature ventricular contractions']
    164947007,  # ['LPR', 'Prolonged PR interval']
    111975006,  # ['LQT', 'prolonged qt interval']
    164917005,  # ['QAb', 'qwave abnormal']
    47665007,  # ['RAD', 'right axis deviation']
    59118001,  # ['RBBB', 'right bundle branch block']
    427393009,  # ['SA', 'sinus arrhythmia']
    426177001,  # ['SB', 'sinus bradycardia']
    426783006,  # ['SNR', 'sinus rhythm']
    427084000,  # ['STach', 'sinus tachycardia']
    63593006,  # ['SVPB', 'supraventricular premature beats']
    164934002,  # ['TAb', 't wave abnormal']
    59931005,  # ['TInv', 't wave inversion']
    17338001,  # ['VPB', 'ventricular premature beats']
]


class SequenceZarr(Dataset):
    """PyTorch Dataset class for returning sequences of beat windows."""

    def __init__(
        self,
        zarr_group_path: str = "data/ecgs.zarr",
        snomed_map_pth: str = "data/snomed_ct_dx_map.json",
        window_size: int = 400,
        sequence_length: int = 10,
        mlb_classes: List[int] = SCORED_CODES,
        record_idxs: Union[None, List[int]] = None,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.root = zarr.open_group(zarr_group_path, mode="r")
        if record_idxs is None:
            self.record_idxs = list(
                range(len(self.root[f"beats/window_size_{window_size}_normalized"]))
            )
        else:
            # Check that all of the provided record idxs are in range
            # assert all(record_id in list(
            #     range(len(self.root[f"beats/window_size_{window_size}_normalized"]))
            # ) for record_id in record_idxs)
            self.record_idxs = sorted(record_idxs)

        self.mlb = MultiLabelBinarizer(classes=mlb_classes)
        with open(snomed_map_pth) as f:
            self.snomed_map = json.load(f)
        self.label_weights = None
        self.set_label_weights()

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

        # get the dx and return sparse y_indicator
        raw_dx = self.root["raw/dx"][record_idx]
        sparse_dx = self.mlb.fit_transform([raw_dx, ])[0]
        str_dx_code = tuple(str(dx) for dx in sorted(raw_dx))
        str_dx_abbrv = tuple(self.snomed_map.get(str(dx), [str(dx), None])[0] for dx in sorted(raw_dx))

        return beat_seq, sparse_dx, str_dx_abbrv, str_dx_code

    def set_label_weights(self):
        if self.label_weights is None:
            all_sample_labels = self.root["raw/dx"][:][self.record_idxs]
            all_sparse_labels = self.mlb.fit_transform(all_sample_labels)
            num_total, classes = all_sparse_labels.shape
            positive = all_sparse_labels.sum(axis=0)
            negative = num_total - positive

            # number of negative samples over positive samples
            label_weights = negative / positive

            # assert all weights are defined.
            assert all(np.isfinite(label_weights)), "Split of data missing required label(s)."

            # rescale such that min is 1
            # self.label_weights = label_weights / min(label_weights)

            # rescale such that max is 1
            self.label_weights = label_weights / (max(label_weights) / 1)
        return self.label_weights

    @staticmethod
    def collate_fn(batch):
        # custom collate function to support variable length inputs
        data_len_dxs = [(torch.Tensor(s[0]), len(s[0]), torch.Tensor(s[1]), s[2], s[3]) for s in batch]

        # sort such that lengths are from longest to shortest
        data, lengths, sparse_dxs, str_abbrv_dxs, str_code_dxs = zip(*sorted(data_len_dxs, key=itemgetter(1), reverse=True))
        sparse_dxs = torch.stack(sparse_dxs)
        return [data, lengths, sparse_dxs, str_abbrv_dxs, str_code_dxs]


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
        mlb_classes: List[int] = SCORED_CODES,
    ):
        super().__init__()
        self.zarr_group_path = zarr_group_path
        self.window_size = window_size
        self.batch_size = batch_size

        self.train_workers = train_workers
        self.val_workers = val_workers
        self.test_workers = test_workers

        self.mlb_classes = mlb_classes

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
            mlb_classes=self.mlb_classes,
        )
        self.ds_val = SequenceZarr(
            zarr_group_path=self.zarr_group_path,
            window_size=self.window_size,
            record_idxs=self.val_records,
            mlb_classes=self.mlb_classes,
        )
        self.ds_test = SequenceZarr(
            zarr_group_path=self.zarr_group_path,
            window_size=self.window_size,
            record_idxs=self.test_records,
            mlb_classes=self.mlb_classes,
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
            collate_fn=SequenceZarr.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            num_workers=self.val_workers,
            collate_fn=SequenceZarr.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            num_workers=self.test_workers,
            collate_fn=SequenceZarr.collate_fn,
        )

    def data_config(self):
        return {
            "train_records": self.train_records,
            "train_weights": self.ds_train.label_weights,
            "val_records": self.val_records,
            "val_weights": self.ds_val.label_weights,
            "test_records": self.test_records,
            "test_weights": self.ds_test.label_weights,
            "classes": self.mlb_classes,
            "snomed_map": self.ds_train.snomed_map
        }
