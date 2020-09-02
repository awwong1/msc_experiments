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
        clip_min_max: Union[tuple, None] = None,  # (-3, 3),
        clean_signal: bool = True,
        standardize: bool = False,
        scale_0_1: bool = False
    ):
        """Initialize the PhysioNet/CinC2020 Challenge Dataset
        root: base path containing extracted *.hea/*.mat files (default: "data")
        set_seq_len: Set length of returned tensors, for batching (default: full signal length)
        fs: Sampling frequency to return tensors as (default: 500)
        clip_min_max: Signal value clip min_max, if None no clip (default: None)
        clean_signal: If True, clean ecg signal using nk2 ecg_clean approach (default: True)
        standardize: If True, each signal subtract mean, divide by std (default: False)
        scale_0_1: If True, each signal is bound between the range [0, 1]
        """
        self.ecg_records = tuple(
            walk_files(root, suffix=".hea", prefix=True, remove_suffix=True)
        )
        self.set_seq_len = set_seq_len
        self.fs = fs
        self.clip_min_max = clip_min_max
        self.clean_signal = clean_signal
        self.standardize = standardize
        self.scale_0_1 = scale_0_1

        root = os.path.expanduser(root)
        len_data_fp = os.path.join(root, f"fs_{fs}_lens.json")

        self.len_data = CinC2020._generate_record_length_cache(
            len_data_fp, fs, self.ecg_records
        )

        # Generate fragment to idx mapping
        self.generate_index_record_map()

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
                    p_signal,
                    [(pad_left, pad_right), (0, 0)],
                    "constant",
                    constant_values=0.0,
                )
        # clean the ecg signal
        if self.clean_signal:
            p_signal = CinC2020._clean_ecg_nk2(p_signal, sampling_rate=self.fs)

        if self.clip_min_max:
            # set the signal maximum and minimum bounds
            p_signal = np.clip(p_signal, self.clip_min_max[0], self.clip_min_max[1], out=p_signal)

        if self.standardize:
            # z-score normalization
            p_signal = (p_signal - np.mean(p_signal, axis=0)) / np.std(p_signal, axis=0)

        if self.scale_0_1:
            _, num_leads = p_signal.shape
            p_signal = np.stack(
                list(map(CinC2020._rescale_signal, [p_signal[:, li] for li in range(num_leads)])),
                axis=1
            )

        # set signal datatype to float32 (default was float64)
        p_signal = p_signal.astype(np.float32)

        # TODO: custom collate for multiple dx
        dx = dx[0]

        return p_signal, self.fs, age, sex, dx

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
    def _rescale_signal(sig, clamp_range=(0, 1)):
        return np.interp(sig, (sig.min(), sig.max()), clamp_range)

    @staticmethod
    def _clean_ecg_nk2(ecg_signal, sampling_rate=500):
        """
        Parallelized version of neurokit2 ecg_clean(method="neurokit")
        ecg_signal shape should be (signal_length, number of leads)
        """
        # Remove slow drift with highpass Butterworth.
        sos = ss.butter(
            5,
            (0.5,),
            btype="highpass",
            output="sos",
            fs=sampling_rate,
        )
        clean = ss.sosfiltfilt(sos, ecg_signal, axis=0).T

        # DC offset removal with 50hz powerline filter (convolve average kernel)
        if sampling_rate >= 100:
            b = np.ones(int(sampling_rate / 50))
        else:
            b = np.ones(2)
        a = [
            len(b),
        ]
        clean = np.copy(ss.filtfilt(b, a, clean, method="pad", axis=1).T)

        return clean

    @staticmethod
    def _generate_record_length_cache(len_data_fp: str, fs: int, ecg_records: list):
        _find_record_length = partial(CinC2020._find_record_length, fs=fs)

        if os.path.isfile(len_data_fp):
            with open(len_data_fp) as f:
                len_data = json.load(f)
        else:
            len_data = dict(
                joblib.Parallel(n_jobs=-1, verbose=0)(
                    joblib.delayed(_find_record_length)(ecg_record)
                    for ecg_record in ecg_records
                )
            )
            with open(len_data_fp, "w") as f:
                json.dump(len_data, f)

        return len_data

    @staticmethod
    def _find_record_length(record_fp: str, fs: int = 500):
        record = rdrecord(record_fp)
        seq_len, num_channels = record.p_signal.shape
        sampling_rate = record.fs
        if sampling_rate != fs:
            # what would the sequence length be at the defined frequency?
            duration = seq_len / sampling_rate
            return record_fp, int(fs * duration)
        else:
            return record_fp, int(seq_len)
