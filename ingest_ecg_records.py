#!/usr/bin/env python3
import warnings

import joblib
import neurokit2 as nk
import numcodecs
import numpy as np
import pandas as pd
import scipy.signal as ss
import wfdb
import zarr
from sklearn.neighbors import KernelDensity, LocalOutlierFactor
from sklearn.preprocessing import normalize

from utils import clean_ecg_nk2, parse_comments, walk_files, RangeKeyDict

numcodecs.blosc.use_threads = False
record_files = tuple(walk_files("data", suffix=".hea", prefix=True, remove_suffix=True))
store = zarr.DirectoryStore("data/ecgs.zarr")
root = zarr.group(store=store)
raw = root.require_group("raw")
cleaned = root.require_group("cleaned")
beats = root.require_group("beats")
meta = root.require_group("meta")
cinc_mfe = root.require_group("cinc_mfe")  # mfe: manual feature extraction

print(f"Number of records to process: {len(record_files)}")

# ONLY SET ONE OF THESE AT A TIME
parse_raw = False  # FIRST, convert wfdb files into normal unit numpy arrays
parse_clean = False  # SECOND, run nk2 signal cleaning
generate_beats = False  # THIRD, run nk2 beat annotation and parse out raw windows
find_outlier_beats = False  # FOURTH, run l2 normalization and sklearn outlier detector
flatten_beat_chunks = False  # FIFTH, improve load speed by storing beats/record
replicate_old_mfe = True  # SIXTH, replicate prior work from CinC2020 paper

window_size = 400  # RR Interval distance to resample for

if replicate_old_mfe:
    # https://github.com/awwong1/physionet-challenge-2020/blob/dd36e42e7803e7eb20e4bbaccbf4b29b57cf981d/neurokit2_parallel.py#L888
    engineered_features = cinc_mfe.empty(
        "all_cinc_2020_features",
        shape=(len(record_files), 19334),  # idk y not 18950,
        synchronizer=zarr.ProcessSynchronizer(".zarr_engineered_features"),
    )
    # submodule symlink and manual mapping
    from utils.neurokit2_parallel import lead_to_feature_dataframe, ECG_LEAD_NAMES

    def _engineer_features(idx):
        raw_signal = root["raw/p_signal"][idx].reshape(
            root["raw/p_signal_shape"][idx], order="C"
        )
        cleaned_signal = root["cleaned/p_signal"][idx].reshape(
            root["raw/p_signal_shape"][idx], order="C"
        )
        age, sex, fs = root["raw/meta"][idx]

        sig_len, num_leads = raw_signal.shape
        record_features = joblib.Parallel(n_jobs=num_leads, verbose=0)(
            joblib.delayed(lead_to_feature_dataframe)(
                raw_signal[:, i], cleaned_signal[:, i], ECG_LEAD_NAMES[i], fs, None
            )
            for i in range(num_leads)
        )
        df = pd.concat(
            [pd.DataFrame({"age": (age,), "sex": (sex,)})] + record_features, axis=1
        )
        engineered_features[idx] = df.to_numpy()[0]

    joblib.Parallel(n_jobs=16, verbose=1, backend="multiprocessing")(
        joblib.delayed(_engineer_features)(idx)
        for idx in range(len(record_files))
    )


elif flatten_beat_chunks:
    window_meta = meta.empty(
        f"record_idx_to_window_{window_size}_range",
        shape=1,
        dtype=object,
        object_codec=numcodecs.JSON(),
        chunks=1,
    )
    all_record_idxs = list(range(len(root[f"beats/window_size_{window_size}_shape"])))
    beat_range_to_record_idx = {}
    beat_idx_accum = 0
    print("Creating beat_record maps...")
    for record_idx, zarr_record_idx in enumerate(all_record_idxs):
        num_beats, win_size, num_leads = map(
            int, root[f"beats/window_size_{window_size}_shape"][zarr_record_idx]
        )
        beat_range_to_record_idx[
            (beat_idx_accum, beat_idx_accum + num_beats)
        ] = record_idx
        beat_idx_accum += num_beats
    num_beats = beat_idx_accum
    beat_range_to_record_idx = RangeKeyDict(beat_range_to_record_idx)
    # beat_range_to_record_idx = beat_range_to_record_idx
    record_idx_to_beat_range = dict(
        [(v, k) for (k, v) in beat_range_to_record_idx.items()]
    )
    window_meta[0] = record_idx_to_beat_range

    print(f"Moving {num_beats} beats to flattened array store")
    normalized_windows_flattened = beats.empty(
        f"window_size_{window_size}_normalized_flattened",
        shape=(num_beats, window_size, num_leads),
        dtype=np.float32,
        chunks=(1, None),
        synchronizer=zarr.ProcessSynchronizer(
            ".zarr_beat_windows_normalized_flattened"
        ),
    )

    def _store_beat_window(idx):
        record_idx = beat_range_to_record_idx[idx]
        zarr_idx = all_record_idxs[record_idx]
        l_offset, _ = record_idx_to_beat_range[record_idx]
        beat_offset = idx - l_offset

        beat_window = root[f"beats/window_size_{window_size}_normalized"][
            zarr_idx
        ].reshape(root[f"beats/window_size_{window_size}_shape"][zarr_idx])[beat_offset]
        normalized_windows_flattened[idx] = beat_window

    joblib.Parallel(n_jobs=-1, verbose=1, backend="multiprocessing")(
        joblib.delayed(_store_beat_window)(idx) for idx in range(num_beats)
    )

elif find_outlier_beats:
    normalized_windows = beats.empty(
        f"window_size_{window_size}_normalized",
        shape=len(record_files),
        dtype=object,
        object_codec=numcodecs.VLenArray(np.float32),
        chunks=1,
        synchronizer=zarr.ProcessSynchronizer(".zarr_beat_windows_normalized"),
    )
    window_outlier = beats.empty(
        f"window_size_{window_size}_outlier",
        shape=len(record_files),
        dtype=np.intc,
        chunks=1,
        synchronizer=zarr.ProcessSynchronizer(".zarr_beat_outlier"),
    )

    def _find_normalized_outlier(idx):
        try:
            windows = root[f"beats/window_size_{window_size}"][idx].reshape(
                root[f"beats/window_size_{window_size}_shape"][idx], order="C"
            )
            # l2 normalize the windows
            norm_windows = np.nan_to_num(windows)
            norm_windows = np.transpose(
                np.stack(
                    list(map(normalize, np.transpose(norm_windows, axes=(0, 2, 1))))
                ),
                axes=(0, 2, 1),
            )
            normalized_windows[idx] = norm_windows.flatten()

            # find the local outlier within the normalized windows
            outlier_idx = 0  # only one beat extracted case
            if len(norm_windows) > 1:
                clf = LocalOutlierFactor(
                    n_neighbors=max(
                        1, min(len(norm_windows) - 1, len(norm_windows) // 2)
                    )
                )
                clf.fit_predict(norm_windows.reshape(len(norm_windows), -1))
                outlier_idx = clf.negative_outlier_factor_.argmin()
            window_outlier[idx] = outlier_idx
        except Exception:
            raise Exception(idx)

    joblib.Parallel(n_jobs=-1, verbose=1, backend="multiprocessing")(
        joblib.delayed(_find_normalized_outlier)(idx)
        for idx in range(len(record_files))
    )

elif generate_beats:
    r_peak_idxs = beats.empty(
        "r_peak_idxs",
        shape=len(record_files),
        dtype=object,
        object_codec=numcodecs.JSON(),
        chunks=1,
        synchronizer=zarr.ProcessSynchronizer(".zarr_r_peak_idxs"),
    )

    valid_r_peak_idxs = beats.empty(
        "valid_r_peak_idxs",
        shape=len(record_files),
        dtype=object,
        object_codec=numcodecs.VLenArray(np.intc),
        chunks=1,
        synchronizer=zarr.ProcessSynchronizer(".zarr_valid_r_peak_idxs"),
    )

    beat_windows = beats.empty(
        f"window_size_{window_size}",
        shape=len(record_files),
        dtype=object,
        object_codec=numcodecs.VLenArray(np.float32),
        chunks=1,
        synchronizer=zarr.ProcessSynchronizer(".zarr_beat_windows"),
    )

    beat_window_shapes = beats.empty(
        f"window_size_{window_size}_shape",
        shape=(len(record_files), 3),  # num_windows, window_size, num_leads
        dtype=np.intc,
        chunks=(1, 3),
        synchronizer=zarr.ProcessSynchronizer(".zarr_beat_window_shapes"),
    )

    def _generate_beats(idx):
        p_signal = root["cleaned/p_signal"][idx].reshape(
            root["raw/p_signal_shape"][idx], order="C"
        )
        _, _, fs = root["raw/meta"][idx]

        def nk_ecg_peaks(cleaned_signal, sampling_rate=fs):
            try:
                _, ecg_r_peaks_dict = nk.ecg_peaks(
                    cleaned_signal,
                    sampling_rate=sampling_rate,
                    method="neurokit",
                    correct_artifacts=True,
                )
                r_peaks = ecg_r_peaks_dict["ECG_R_Peaks"].tolist()
                return r_peaks
            except Exception:
                return []

        all_r_peaks = list(map(nk_ecg_peaks, p_signal.T))

        # Join all of the R-peaks into a single vector
        all_r_peaks_flat = np.concatenate(all_r_peaks)[:, np.newaxis]
        sig_range = np.linspace(0, all_r_peaks_flat.max(), len(p_signal))[:, np.newaxis]

        # Find the peaks with bandwidth proportional to rough mean RR
        mean_beats_detected = np.mean([len(r_peaks) for r_peaks in all_r_peaks])
        rough_meanrr = np.mean(
            [
                np.diff(r_peaks).mean()
                for r_peaks in all_r_peaks
                if len(r_peaks) >= 2 and len(r_peaks) >= mean_beats_detected
            ]
        )
        kde = KernelDensity(bandwidth=rough_meanrr / 4).fit(all_r_peaks_flat)
        log_dens = kde.score_samples(sig_range)
        dens = np.exp(log_dens)
        peaks, _ = ss.find_peaks(dens)

        # keep only the peaks that are greater than thereshold density
        threshold_peak_density = dens[peaks].mean() - (2 * dens[peaks].std())
        valid_peaks = peaks[dens[peaks] > threshold_peak_density]

        r_peak_idxs[idx] = all_r_peaks
        valid_r_peak_idxs[idx] = valid_peaks

        # resample the signal such that the mean distance between
        # valid R-peaks is equal to `window_size`
        scaling_indicies = sig_range[valid_peaks].squeeze()
        scaling_indicies = np.insert(scaling_indicies, 0, 0)
        scaling_indicies = np.append(scaling_indicies, len(p_signal))
        peak_diff_dist = np.diff(scaling_indicies)
        mean_peak_diff_dist = peak_diff_dist.mean()
        resamp_to_len = (len(p_signal) / mean_peak_diff_dist) * window_size
        resamp_to_len = int(np.ceil(resamp_to_len))
        p_signal = ss.resample(p_signal, resamp_to_len)

        # resample the peaks so we don't have to calculate again
        scaling_factor = window_size / mean_peak_diff_dist
        upscaled_peak_diff_dist = peak_diff_dist * scaling_factor
        scaled_indicies = np.r_[scaling_indicies[0], upscaled_peak_diff_dist].cumsum()
        scaled_indicies = scaled_indicies[1:-1]

        # slice up the windows and return the new matrices
        valid_windows = []
        sc_loffset = window_size * 0.33
        for sc in scaled_indicies:
            left_offset = int(np.floor(sc - sc_loffset))
            scaled_window = p_signal[left_offset : left_offset + window_size]
            if len(scaled_window) != window_size:
                # ignore windows that don't fit into window size
                continue
            valid_windows.append(scaled_window)

        valid_windows = np.stack(valid_windows)
        beat_windows[idx] = valid_windows.flatten(order="C")
        beat_window_shapes[idx] = valid_windows.shape

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Mean of empty slice")

        joblib.Parallel(n_jobs=-1, verbose=1, backend="multiprocessing")(
            joblib.delayed(_generate_beats)(idx) for idx in range(len(record_files))
        )

elif parse_clean:
    clean_p_signal = cleaned.empty(
        f"p_signal",
        shape=len(record_files),
        dtype=object,
        object_codec=numcodecs.VLenArray(np.float32),
        chunks=100,
        synchronizer=zarr.ProcessSynchronizer(".zarr_sync_psignal"),
    )

    def _clean_p_signal(idx):
        signal = root["raw/p_signal"][idx].reshape(
            root["raw/p_signal_shape"][idx], order="C"
        )
        _age, _sex, fs = root["raw/meta"][idx]
        # dx = root["raw/dx"][idx]
        cleaned_signal = clean_ecg_nk2(signal, sampling_rate=fs)
        clean_p_signal[idx] = cleaned_signal.flatten(order="C")

    joblib.Parallel(n_jobs=-1, verbose=1, backend="multiprocessing")(
        joblib.delayed(_clean_p_signal)(idx) for idx in range(len(record_files))
    )
elif parse_raw:
    raw_p_signal = raw.empty(
        f"p_signal",
        shape=len(record_files),
        dtype=object,
        object_codec=numcodecs.VLenArray(np.float32),
        chunks=100,
        synchronizer=zarr.ProcessSynchronizer(".zarr_sync_psignal"),
    )
    raw_shape = raw.empty(
        f"p_signal_shape",
        shape=(len(record_files), 2),
        dtype=np.intc,
        chunks=100,
        synchronizer=zarr.ProcessSynchronizer(".zarr_sync_psignal_shape"),
    )
    raw_meta = raw.empty(
        f"meta",
        shape=(len(record_files), 3),  # age, sex, sampling_rate
        dtype=np.intc,
        chunks=100,
        synchronizer=zarr.ProcessSynchronizer(".zarr_sync_meta"),
    )
    raw_dx = raw.empty(
        f"dx",
        shape=len(record_files),
        dtype=object,
        object_codec=numcodecs.VLenArray(np.intc),
        chunks=100,
        synchronizer=zarr.ProcessSynchronizer(".zarr_sync_dx"),
    )

    def _set_record(idx, record_file):
        # idx, record_file = idx_record_file
        r = wfdb.rdrecord(record_file)
        age, sex, dx = parse_comments(r.comments)
        raw_p_signal[idx] = r.p_signal.flatten(order="C")
        raw_shape[idx] = np.asarray(r.p_signal.shape)
        raw_meta[idx] = np.asarray([age, sex, r.fs])
        raw_dx[idx] = np.asarray(dx)

    joblib.Parallel(n_jobs=-1, verbose=1, backend="multiprocessing")(
        joblib.delayed(_set_record)(idx, record)
        for idx, record in enumerate(record_files)
    )
