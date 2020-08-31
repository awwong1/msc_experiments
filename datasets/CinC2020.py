import os
import json
from typing import Union
from functools import partial
import joblib
from torch.utils.data import Dataset
from wfdb import rdrecord
import scipy.signal as ss
import numpy as np

from datasets.utils import parse_comments, walk_files, RangeKeyDict


class CinC2020(Dataset):
    """
    PyTorch Dataset class for PhysioNet/CinC 2020 Challenge:
    12-Lead Electrocardiogram Classification
    """

    def __init__(
        self,
        root: str = "data",
        set_seq_len: Union[int, None] = None,
        fs: int = 500,
    ):
        """Initialize te PhysioNet/CinC2020 Challenge Dataset
        root: base path containing extracted *.hea/*.mat files (default: "data")
        set_seq_len: Set length of returned tensors, for batching (default: full signal length)
        fs: Sampling frequency to return tensors as (default: 500)
        """
        self.ecg_records = tuple(
            walk_files(root, suffix=".hea", prefix=True, remove_suffix=True)
        )
        self.set_seq_len = set_seq_len
        self.fs = fs

        root = os.path.expanduser(root)
        len_data_fp = os.path.join(root, f"fs_{fs}_lens.json")

        self.len_data = CinC2020.generate_record_length_cache(
            len_data_fp, fs, self.ecg_records
        )

        # Generate fragment to idx mapping
        self.generate_index_record_map()

    def generate_index_record_map(self):
        name_map_idx = {}
        idx = 0
        for ecg_record in self.ecg_records:
            seq_len = self.len_data.get(ecg_record)

            idx_start = idx
            idx_end = idx

            if self.set_seq_len is None:
                idx_end = idx_start + 1
            else:
                quot = seq_len // self.set_seq_len
                mod = seq_len % self.set_seq_len

                if quot > 0:
                    idx_end += quot
                if mod > 0:
                    idx_end += 1

            name_map_idx[ecg_record] = (int(idx_start), int(idx_end))
            idx = idx_end
        self.name_map_idx = name_map_idx
        self.idx_map_name = RangeKeyDict(
            dict((v, k) for (k, v) in name_map_idx.items())
        )

    @staticmethod
    def generate_record_length_cache(len_data_fp: str, fs: int, ecg_records: list):
        find_record_length = partial(CinC2020.find_record_length, fs=fs)

        if os.path.isfile(len_data_fp):
            with open(len_data_fp) as f:
                len_data = json.load(f)
        else:
            len_data = dict(
                joblib.Parallel(n_jobs=-1, verbose=0)(
                    joblib.delayed(find_record_length)(ecg_record)
                    for ecg_record in ecg_records
                )
            )
            with open(len_data_fp, "w") as f:
                json.dump(len_data, f)

        return len_data

    @staticmethod
    def find_record_length(record_fp: str, fs: int = 500):
        record = rdrecord(record_fp)
        seq_len, num_channels = record.p_signal.shape
        sampling_rate = record.fs
        if sampling_rate != fs:
            # what would the sequence length be at the defined frequency?
            duration = seq_len / sampling_rate
            return record_fp, int(fs * duration)
        else:
            return record_fp, int(seq_len)

    def __len__(self):
        # return the sum of the index mapping range lengths
        return sum([len(range(*v)) for v in self.name_map_idx.values()])

    def __getitem__(self, idx: int):
        """
        p_signal.shape is (seq_len, num_channels)
        Return: p_signal, sampling_rate, age, sex, dx
        """
        record_path = self.idx_map_name[idx]
        idx_start, _idx_end = self.name_map_idx[record_path]

        record = rdrecord(record_path)
        age, sex, dx = parse_comments(record.comments)

        p_signal = record.p_signal
        sampling_rate = record.fs
        fs_len = self.len_data[record_path]

        if sampling_rate != self.fs:
            # resample signal to match new sampling rate
            p_signal = ss.resample(p_signal, fs_len, axis=0)

        # calculate the offset of base signal according to idx
        if self.set_seq_len is not None:
            offset_idx = (idx - idx_start) * self.set_seq_len

            # check for offset greater than set_seq_len (last index)
            if offset_idx + self.set_seq_len > fs_len:
                offset_idx = max(fs_len - self.set_seq_len, 0)

            # check for partial sequence less than set_seq_len, pad
            p_signal = p_signal[offset_idx : offset_idx + self.set_seq_len]
            if len(p_signal) < self.set_seq_len:
                pad_all = self.set_seq_len - len(p_signal)
                pad_left = pad_all // 2
                pad_right = pad_left + pad_all % 2

                p_signal = np.pad(
                    p_signal, [(pad_left, pad_right), (0, 0)],  "constant", constant_values=0.0
                )

        return p_signal, self.fs, age, sex, dx
