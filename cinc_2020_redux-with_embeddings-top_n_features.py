#!/usr/bin/env python3
# this is a script version of ./cinc_2020_redux-embeddings.ipynb

import os
import json
import pickle
from glob import glob

import torch
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import zarr

from utils import ElapsedTimer
from utils.evaluation_helper import evaluate_score_batch
from utils.evaluate_12ECG_score import load_table

experiment_dir = "cinc-2020-redux-embedding"

print(experiment_dir)
print("Loading features...")

with ElapsedTimer() as t:
    features = pd.read_csv("full_output/features.csv", index_col="header_file")
    features.sort_values(by=["header_file"], inplace=True)
print(f"Took {t.duration:.2f}s")

dxs = {}
print("Loading labels...")
with open("full_output/dxs.txt") as f:
    for line in f.readlines():
        k, v = json.loads(line)
        dxs[k] = v

with open("data/snomed_ct_dx_map.json", "r") as f:
    SNOMED_CODE_MAP = json.load(f)

root = zarr.open_group("data/ecgs.zarr", mode="r")
print(root.info)

weights_file = "evaluation-2020/weights.csv"
rows, cols, all_weights = load_table(weights_file)
assert rows == cols
scored_codes = rows

torch_checkpoints = glob("log_beat_autoencoder/*/checkpoints/*.ckpt")
data_configs = {}
for torch_checkpoint in torch_checkpoints:
    state_dict = torch.load(torch_checkpoint)
    raw_data_config = state_dict["hyper_parameters"]["data_config"]

    data_config = {}
    for k, v in raw_data_config.items():
        data_config[k] = v.indices
    version_str = torch_checkpoint.split("/")[1]
    data_config["checkpoint"] = torch_checkpoint
    data_configs[version_str] = data_config


def _determine_sample_weights(
    data_set, scored_codes, label_weights, weight_threshold=0.5
):
    """Using the scoring labels weights to increase the dataset size of positive labels"""
    data_labels = []
    sample_weights = []
    for dt in data_set:
        sample_weight = None
        for dx in dt:
            if str(dx) in scored_codes:
                _sample_weight = label_weights[scored_codes.index(str(dx))]
                if _sample_weight < weight_threshold:
                    continue
                if sample_weight is None or _sample_weight > sample_weight:
                    sample_weight = _sample_weight

        if sample_weight is None:
            # not a scored label, treat as a negative example (weight of 1)
            sample_weight = 1.0
            data_labels.append(False)
        else:
            data_labels.append(True)
        sample_weights.append(sample_weight)
    return data_labels, sample_weights


def _train_label_classifier(
    sc,
    idx_sc,
    all_weights,
    train_features,
    train_labels,
    eval_features,
    eval_labels,
    scored_codes,
    early_stopping_rounds,
    num_gpus,
):
    label_weights = all_weights[idx_sc]
    train_labels, train_weights = _determine_sample_weights(
        train_labels, scored_codes, label_weights
    )
    eval_labels, eval_weights = _determine_sample_weights(
        eval_labels, scored_codes, label_weights
    )

    # try negative over positive https://machinelearningmastery.com/xgboost-for-imbalanced-classification/
    pos_count = len([e for e in train_labels if e])
    pos_count = max(pos_count, 1)
    scale_pos_weight = (len(train_labels) - pos_count) / pos_count

    model = XGBClassifier(
        booster="dart",  # gbtree, dart or gblinear
        verbosity=0,
        tree_method="gpu_hist",
        sampling_method="gradient_based",
        scale_pos_weight=scale_pos_weight,
    )

    eval_set = [(train_features, train_labels), (eval_features, eval_labels)]
    sample_weight_eval_set = [train_weights, eval_weights]

    model = model.fit(
        train_features,
        train_labels,
        sample_weight=train_weights,
        eval_set=eval_set,
        sample_weight_eval_set=sample_weight_eval_set,
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )

    return sc, model


def train_experiment(
    data_config,
    old_experiment,
    top_n=1000,
    all_weights=all_weights,
    scored_codes=scored_codes,
    features=features,
    root=root,
    early_stopping_rounds=20,
):
    train_idxs = data_config["train_records"]
    val_idxs = data_config["val_records"]
    test_idxs = data_config["test_records"]

    checkpoint = data_config["checkpoint"]
    version_str = checkpoint.split("/")[1]

    embeddings = root[f"seq_embeddings/{version_str}"]
    raw_features = features.to_numpy()
    embd_features = np.concatenate((raw_features, embeddings), axis=1)

    train_features, train_labels = np.take(embd_features, train_idxs, axis=0), np.take(
        root["raw/dx"], train_idxs
    )
    eval_features, eval_labels = np.take(embd_features, val_idxs, axis=0), np.take(
        root["raw/dx"], val_idxs
    )
    test_features, test_labels = np.take(embd_features, test_idxs, axis=0), np.take(
        root["raw/dx"], test_idxs
    )

    classes = []
    labels = []
    scores = []

    models = {}

    for idx_sc, sc in enumerate(scored_codes):
        with ElapsedTimer() as t:
            print(f"Loading old experiment features for {SNOMED_CODE_MAP[sc][1]}")
            sorted_feat_idxs = old_experiment[sc].feature_importances_.argsort()
            top_n_feat_idxs = sorted_feat_idxs[::-1][:top_n]

            _train_features = train_features[:, top_n_feat_idxs]
            _eval_features = eval_features[:, top_n_feat_idxs]
            _test_features = test_features[:, top_n_feat_idxs]

            print(f"Training {SNOMED_CODE_MAP[sc][1]} classifier...", end="")
            sc, model = _train_label_classifier(
                sc,
                idx_sc,
                all_weights,
                _train_features,
                train_labels,
                _eval_features,
                eval_labels,
                scored_codes,
                early_stopping_rounds,
                1,
            )
            classes.append(str(sc))
            labels.append(model.predict(_test_features).tolist())
            scores.append(model.predict_proba(_test_features)[:, 1].tolist())
            models[sc] = model
        print(f"Took {t.duration:.2f} seconds")

    (
        classes,
        auroc,
        auprc,
        auroc_classes,
        auprc_classes,
        accuracy,
        f_measure,
        f_measure_classes,
        f_beta_measure,
        g_beta_measure,
        challenge_metric,
    ) = evaluate_score_batch(
        predicted_classes=classes,
        predicted_labels=np.array(labels).T,
        predicted_probabilities=np.array(scores).T,
        raw_ground_truth_labels=test_labels,
    )

    log = {
        "test_auroc": auroc,
        "test_auprc": auprc,
        "test_accuracy": accuracy,
        "test_f_measure": f_measure,
        "test_f_beta_measure": f_beta_measure,
        "test_g_beta_measure": g_beta_measure,
        "test_challenge_metric": challenge_metric,
    }
    class_output_string = "Classes,{}\nAUROC,{}\nAUPRC,{}\nF-measure,{}".format(
        ",".join(classes),
        ",".join("{:.3f}".format(x) for x in auroc_classes),
        ",".join("{:.3f}".format(x) for x in auprc_classes),
        ",".join("{:.3f}".format(x) for x in f_measure_classes),
    )

    print(log)
    return log, class_output_string, models


def main():
    top_n = 100

    os.makedirs(os.path.join(experiment_dir, f"top_{top_n}"), exist_ok=True)

    for dc_idx, data_config in data_configs.items():
        print(f"Experiment {dc_idx}")
        with ElapsedTimer() as t:
            with open(os.path.join(experiment_dir, f"{dc_idx}_models.pkl"), "rb") as f:
                old_experiment = pickle.load(f)

            log, class_output_string, models = train_experiment(data_config, old_experiment, top_n=top_n)

            with open(
                os.path.join(experiment_dir, f"top_{top_n}", f"{dc_idx}_test_results.json"), "w"
            ) as f:
                json.dump(log, f)
            with open(
                os.path.join(experiment_dir, f"top_{top_n}", f"{dc_idx}_test_class_labelwise.csv"), "w"
            ) as f:
                f.write(class_output_string)
            with open(os.path.join(experiment_dir, f"top_{top_n}", f"{dc_idx}_models.pkl"), "wb") as f:
                pickle.dump(models, f)

        print(os.path.join(experiment_dir, f"top_{top_n}", f"{dc_idx}_models.pkl"))
        print(f"Experiment {dc_idx} Took {t.duration:.2f}s")


if __name__ == "__main__":
    main()
