import json
import os
import warnings

import joblib
import neurokit2 as nk
import numpy as np
import scipy.signal as ss
from sklearn.neighbors import KernelDensity
from torch.utils.data import Dataset
from wfdb import rdrecord

from datasets.utils import RangeKeyDict, clean_ecg_nk2, parse_comments, walk_files


class CinC2020Beats(Dataset):
    """
    PyTorch Dataset class for PhysioNet/CinC 2020 Challenge:
    12-Lead Electrocardiogram Classification

    - Normalize the output by beats, so returned samples contain a PQRST heart beat window
    """

    def __init__(
        self,
        root: str = "data",
        pqrst_window_size: int = 400,
    ):
        """Initialize the PhysioNet/CinC2020 Challenge Dataset
        Beat dataloader, only returns isolated R-peak windows (heuristic average)
        """
        self.ecg_records = tuple(
            walk_files(root, suffix=".hea", prefix=True, remove_suffix=True)
        )
        self.pqrst_window_size = pqrst_window_size
        self.root = os.path.expanduser(root)

        len_data_fp = os.path.join(root, f".cache_beat_{pqrst_window_size}_lens.json")
        self.len_data = CinC2020Beats._generate_record_length_cache(
            len_data_fp, pqrst_window_size, self.ecg_records, root=self.root
        )
        self.generate_index_record_map()

    def __len__(self):
        return sum([len(range(*v)) for v in self.name_map_idx.values()])

    def __getitem__(self, idx):
        record_path = self.idx_map_name[idx]
        idx_start, _idx_end = self.name_map_idx[record_path]
        valid_windows, age, sex, dx = CinC2020Beats._generate_beat_file(
            record_path, self.pqrst_window_size, root=self.root
        )

        offset_idx = idx - idx_start
        window = valid_windows[offset_idx]

        # set signal datatype to float32 (default was float64)
        window = window.astype(np.float32)

        # TODO: custom collate for multiple dx
        dx = dx[0]

        return window, age, sex, dx

    def generate_index_record_map(self):
        name_map_idx = {}
        idx = 0
        for ecg_record in self.ecg_records:
            seq_len = self.len_data.get(ecg_record)
            idx_start = idx
            idx_end = idx_start + seq_len
            name_map_idx[ecg_record] = (int(idx_start), int(idx_end))
            idx = idx_end
        self.name_map_idx = name_map_idx
        self.idx_map_name = RangeKeyDict(
            dict((v, k) for (k, v) in name_map_idx.items())
        )

    @staticmethod
    def _generate_record_length_cache(
        len_data_fp: str, pqrst_window_size: int, ecg_records: list, root: str = ""
    ):
        def get_beat_file_shape(record_path):
            try:
                valid_windows, _age, _sex, _dx = CinC2020Beats._generate_beat_file(
                    record_path, pqrst_window_size=pqrst_window_size, root=root
                )
            except Exception:
                raise Exception(record_path)
            return record_path, valid_windows.shape[0]

        if os.path.isfile(len_data_fp):
            with open(len_data_fp) as f:
                len_data = json.load(f)
        else:
            len_data = dict(
                joblib.Parallel(n_jobs=-1, verbose=1)(
                    joblib.delayed(get_beat_file_shape)(ecg_record)
                    for ecg_record in ecg_records
                )
            )
            with open(len_data_fp, "w") as f:
                json.dump(len_data, f)
        return len_data

    @staticmethod
    def _generate_beat_file(record_path: str, pqrst_window_size: int, root: str = ""):
        # allow this to be cached to file
        cache_file = os.path.join(
            root, f".cache_beat_{pqrst_window_size}", f"{record_path}.npz"
        )
        if os.path.isfile(cache_file):
            with open(cache_file, "rb") as f:
                valid_windows = np.load(f)
                age = np.load(f)
                sex = np.load(f)
                dx = np.load(f)
        else:
            record = rdrecord(record_path)
            age, sex, dx = parse_comments(record.comments)

            p_signal = record.p_signal
            sampling_rate = record.fs

            # clean the signal
            p_signal = clean_ecg_nk2(p_signal, sampling_rate)

            # find all of the R-peaks in the 12-lead record
            def nk_ecg_peaks(cleaned_signal):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            action="ignore", message="Mean of empty slice"
                        )
                        return nk.ecg_peaks(
                            cleaned_signal,
                            sampling_rate=sampling_rate,
                            method="neurokit",
                            correct_artifacts=True,
                        )
                except Exception:
                    return None, {"ECG_R_Peaks": []}

            _r_peaks_df, r_peak_dicts = zip(*map(nk_ecg_peaks, p_signal.T))

            # join all of these locations into a single vector
            r_peaks = np.concatenate([rd["ECG_R_Peaks"] for rd in r_peak_dicts])
            r_peaks = r_peaks[:, np.newaxis]
            sig_range = np.linspace(0, r_peaks.max(), len(p_signal))[:, np.newaxis]

            # Find the peaks with bandwidth proportional to rough mean RR
            rough_meanrr = np.mean(
                [
                    np.diff(rd["ECG_R_Peaks"]).mean()
                    for rd in r_peak_dicts
                    if len(rd["ECG_R_Peaks"]) >= 2
                ]
            )
            kde = KernelDensity(bandwidth=rough_meanrr / 10).fit(r_peaks)
            log_dens = kde.score_samples(sig_range)
            dens = np.exp(log_dens)
            peaks, _ = ss.find_peaks(dens)

            # keep only the peaks that are greater than threshold density
            threshold_peak_density = dens[peaks].mean() - dens[peaks].std()
            valid_peaks = peaks[dens[peaks] > threshold_peak_density]

            # resample the signal such that the mean distance between
            # valid R-peaks is equal to `pqrst_window_size`
            scaling_indicies = sig_range[valid_peaks].squeeze()
            scaling_indicies = np.insert(scaling_indicies, 0, 0)
            scaling_indicies = np.append(scaling_indicies, len(p_signal))
            peak_diff_dist = np.diff(scaling_indicies)
            mean_peak_diff_dist = peak_diff_dist.mean()
            resamp_to_len = (len(p_signal) / mean_peak_diff_dist) * pqrst_window_size
            resamp_to_len = int(np.ceil(resamp_to_len))
            p_signal = ss.resample(p_signal, resamp_to_len)

            # resample the peaks so we don't have to calculate again
            scaling_factor = pqrst_window_size / mean_peak_diff_dist
            upscaled_peak_diff_dist = peak_diff_dist * scaling_factor
            scaled_indicies = np.r_[
                scaling_indicies[0], upscaled_peak_diff_dist
            ].cumsum()
            scaled_indicies = scaled_indicies[1:-1]

            # slice up the windows and return the new matrices
            valid_windows = []
            sc_loffset = pqrst_window_size * 0.35
            for sc in scaled_indicies:
                left_offset = int(np.floor(sc - sc_loffset))
                scaled_window = p_signal[left_offset : left_offset + pqrst_window_size]
                if len(scaled_window) != pqrst_window_size:
                    # ignore windows that don't fit into window size
                    continue
                valid_windows.append(scaled_window)

            valid_windows = np.stack(valid_windows)

            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "wb") as f:
                np.save(f, valid_windows)
                np.save(f, age)
                np.save(f, sex)
                np.save(f, dx)

        return valid_windows, age, sex, dx
