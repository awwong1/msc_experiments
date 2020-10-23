#!/usr/bin/env python3
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_logger import LearningRateLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import roc_curve
from torch.nn.utils.rnn import pack_sequence

from datasets import SequenceZarrDataModule
from linear_beat_autoencoder import BeatAutoEncoder
from utils.evaluation_helper import evaluate_score_batch


class SequenceAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        bae_checkpoint_path: str,
        base_lr: float = 1e-2,
        max_lr: float = 1e-3,
        num_layers: int = 2,
        lstm_dropout: float = 0.1,
        momentum: float = 0.9,
        sequence_length: int = 10,
        hidden_size: int = 256,
        dropout: float = 0.1,
        data_config: dict = {},
    ):
        super().__init__()

        self.save_hyperparameters()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.num_layers = num_layers
        state_dict = torch.load(bae_checkpoint_path)
        self.embedding_dim = state_dict["hyper_parameters"]["embedding_dim"]
        self.lstm_dropout = lstm_dropout
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.momentum = momentum
        self.data_config = data_config
        output_size = len(data_config["classes"])
        self.label_thresholds = None

        # not supported due to collate/split with len
        # self.example_input_array = torch.rand(sequence_length, 1, self.embedding_dim)

        # beat autoencoder should not be trainable
        self.beat_autoencoder = BeatAutoEncoder.load_from_checkpoint(
            bae_checkpoint_path
        )
        for param in self.beat_autoencoder.parameters():
            param.requires_grad = False

        self.lstm_enc = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            num_layers=self.num_layers,
            dropout=self.lstm_dropout,
        )
        self.lstm_dec = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            num_layers=self.num_layers,
            dropout=self.lstm_dropout,
        )

        self.seq_classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_size),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, output_size),
        )

    @staticmethod
    def emb_loss_function(x_hat, x):
        # x_hat and x are lists of tensors, each element is a sample
        # the entire list being the batch
        sq_err = torch.sum(
            torch.stack(
                [
                    F.mse_loss(r_x, s_x, reduction="sum")  # Sum over sample
                    for r_x, s_x in zip(x_hat, x)
                ]
            )
        )
        msq_err = sq_err / len(x_hat)  # Average over batches
        return msq_err

    @staticmethod
    def class_loss_function(y_hat, y, weights):
        # return F.multilabel_soft_margin_loss(
        #     y_hat, y, weight=torch.tensor(weights, device=y.device), reduction="sum"
        # )
        return F.binary_cross_entropy_with_logits(
            y_hat, y, pos_weight=torch.tensor(weights, device=y.device), reduction="sum"
        )

    def forward(self, beat_windows, lstm_input_lengths):
        # beat_windows, lstm_input_lengths = x

        # Convert into beat autoencoder embedding
        raw_seq_enc_input, _recon = zip(*map(self.beat_autoencoder, beat_windows))
        # lstm_input_lengths = [len(inp) for inp in raw_seq_enc_input]

        max_seq_length = max(lstm_input_lengths)
        # batch_size = len(raw_seq_enc_input)

        # pack variable length sequences for batched LSTM encoder training
        seq_enc_input = pack_sequence(raw_seq_enc_input)
        _seq_enc_outputs, (h_n, c_n) = self.lstm_enc(seq_enc_input)
        # seq_unpacked, lens_unpacked = pad_packed_sequence(lstm_enc_out)
        # length_idxs = [i - 1 for i in lens_unpacked]
        # batch_idxs = [i for i in range(batch_size)]
        # Bottleneck representation of sequences
        # seq_bottleneck = seq_unpacked[length_idxs, batch_idxs, :]
        # print(all((h_n[-1] == seq_bottleneck).flatten()))
        seq_bottleneck = h_n[-1]  # equivalent to all commented out code

        # rebuild the input using lstm decoder
        rebuild_hidden = torch.zeros_like(h_n), torch.zeros_like(c_n)
        rebuild_next = torch.unsqueeze(seq_bottleneck, 0)
        recon_seq = []
        for idx in range(max_seq_length):
            rebuild_next, rebuild_hidden = self.lstm_dec(rebuild_next, rebuild_hidden)
            recon_seq.append(rebuild_next)

        # full reconstructed sequence, slice to batch beat window lengths
        recon_seq = torch.cat(recon_seq, dim=0)
        seq_dec_output = []
        for idx, length in enumerate(lstm_input_lengths):
            seq_dec_output.append(recon_seq[0:length, idx, :])
        #         print(len(seq_dec_output))
        #         print([bwo.shape for bwo in seq_dec_output])

        pred_classes = self.seq_classifier(seq_bottleneck)

        return pred_classes, seq_bottleneck, raw_seq_enc_input, seq_dec_output

    def training_step(self, batch, batch_idx):
        beat_windows, seq_lens, dxs, str_abbrv_dxs, str_code_dxs = batch
        pred_classes, _, x_source, x_hat = self(beat_windows, seq_lens)
        emb_loss = self.emb_loss_function(x_hat, x_source)
        class_loss = self.class_loss_function(
            pred_classes, dxs, self.data_config["train_weights"]
        )
        loss = emb_loss + class_loss

        # pr_curve calculations
        cpu_labels = dxs.detach().cpu()
        cpu_predictions = torch.sigmoid(pred_classes).detach().cpu()
        # self.logger.experiment.add_pr_curve(
        #     f"train_pr_curve",
        #     cpu_labels,
        #     cpu_predictions,
        #     global_step=self.global_step,
        # )

        log = {
            "train_loss": loss,
            "train_emb_loss": emb_loss,
            "train_class_loss": class_loss,
        }
        return {
            "loss": loss,
            "log": log,
            "train_labels": cpu_labels,
            "train_predictions": cpu_predictions,
            "train_raw_labels": str_code_dxs,
        }

    def training_epoch_end(self, outputs):
        cpu_labels = torch.cat([x["train_labels"] for x in outputs])
        cpu_predictions = torch.cat([x["train_predictions"] for x in outputs])
        raw_labels = []
        for x in outputs:
            raw_labels += x["train_raw_labels"]

        self.logger.experiment.add_pr_curve(
            f"train_pr_curve",
            cpu_labels,
            cpu_predictions,
            global_step=self.global_step,
        )

        # labelwise Receiver Operating Characteristic Curves
        num_samples, num_labels = cpu_labels.shape
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        label_thresholds = np.empty(num_labels)
        for label_idx in range(num_labels):
            ct_code = str(self.data_config["classes"][label_idx])
            class_abbrv = self.data_config["snomed_map"].get(ct_code, (ct_code, None))[
                0
            ]
            labels = cpu_labels[:, label_idx]
            predictions = cpu_predictions[:, label_idx]
            fpr, tpr, thresholds = roc_curve(labels, predictions)
            tpr_no_nan = np.nan_to_num(tpr, 0)
            fpr_no_nan = np.nan_to_num(fpr, 0)
            threshold_rank = tpr_no_nan + (1 - fpr_no_nan)
            best_threshold_idx = np.argmax(threshold_rank)
            best_threshold = thresholds[best_threshold_idx]
            ax.plot(fpr, tpr, label=f"{class_abbrv} ({best_threshold:.2e})")
            label_thresholds[label_idx] = best_threshold
        self.label_thresholds = label_thresholds
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"Training ROC curve (Epoch: {self.current_epoch + 1})")
        plt.legend(loc="best")
        self.logger.experiment.add_figure("train_roc_curve", fig, self.global_step)

        predicted_labels = cpu_predictions.numpy() > label_thresholds
        for sample_idx, has_one in enumerate(predicted_labels.any(axis=1)):
            if has_one:
                continue
            predicted_labels[sample_idx, cpu_predictions[sample_idx].argmax()] = True

        # calculate training metrics
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
            predicted_classes=self.data_config["classes"],
            predicted_labels=predicted_labels,
            predicted_probabilities=cpu_predictions.numpy(),
            raw_ground_truth_labels=raw_labels,
        )

        # class_output_string = "Classes,{}\nAUROC,{}\nAUPRC,{}\nF-measure,{}".format(
        #     ",".join(classes),
        #     ",".join("{:.3f}".format(x) for x in auroc_classes),
        #     ",".join("{:.3f}".format(x) for x in auprc_classes),
        #     ",".join("{:.3f}".format(x) for x in f_measure_classes),
        # )
        # print(class_output_string)
        log = {
            "train_auroc": auroc,
            "train_auprc": auprc,
            "train_accuracy": accuracy,
            "train_f_measure": f_measure,
            "train_f_beta_measure": f_beta_measure,
            "train_g_beta_measure": g_beta_measure,
            "train_challenge_metric": challenge_metric,
        }

        return {"log": log}

    def validation_step(self, batch, batch_idx):
        beat_windows, seq_lens, dxs, _str_abbrv_dxs, str_code_dxs = batch
        pred_classes, _, x_source, x_hat = self(beat_windows, seq_lens)
        emb_loss = self.emb_loss_function(x_hat, x_source)
        class_loss = self.class_loss_function(
            pred_classes, dxs, self.data_config["val_weights"]
        )
        loss = emb_loss + class_loss
        cpu_labels = dxs.detach().cpu()
        cpu_predictions = torch.sigmoid(pred_classes).detach().cpu()
        return {
            "val_loss": loss,
            "val_emb_loss": emb_loss,
            "val_class_loss": class_loss,
            "val_labels": cpu_labels,
            "val_raw_labels": str_code_dxs,
            "val_predictions": cpu_predictions,
        }

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        emb_loss = torch.stack([x["val_emb_loss"] for x in outputs]).mean()
        class_loss = torch.stack([x["val_class_loss"] for x in outputs]).mean()

        cpu_labels = torch.cat([x["val_labels"] for x in outputs])
        cpu_predictions = torch.cat([x["val_predictions"] for x in outputs])
        self.logger.experiment.add_pr_curve(
            f"val_pr_curve",
            cpu_labels,
            cpu_predictions,
            global_step=self.global_step,
        )

        # labelwise Receiver Operating Characteristic Curves
        # num_samples, num_labels = cpu_labels.shape
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # for label_idx in range(num_labels):
        #     ct_code = str(self.data_config["classes"][label_idx])
        #     class_abbrv = self.data_config["snomed_map"].get(ct_code, (ct_code, None))[
        #         0
        #     ]
        #     labels = cpu_labels[:, label_idx]
        #     predictions = cpu_predictions[:, label_idx]
        #     fpr, tpr, thresholds = roc_curve(labels, predictions)
        #     tpr_no_nan = np.nan_to_num(tpr, 0)
        #     fpr_no_nan = np.nan_to_num(fpr, 0)
        #     threshold_rank = tpr_no_nan + (1 - fpr_no_nan)
        #     best_threshold_idx = np.argmax(threshold_rank)
        #     best_threshold = thresholds[best_threshold_idx]
        #     ax.plot(fpr, tpr, label=f"{class_abbrv} ({best_threshold:.2e})")

        # plt.xlabel("False positive rate")
        # plt.ylabel("True positive rate")
        # plt.title(f"Validation ROC curve (Epoch: {self.current_epoch + 1})")
        # plt.legend(loc="best")
        # self.logger.experiment.add_figure("val_roc_curve", fig, self.global_step)

        log = {
            "val_avg_loss": loss,
            "val_avg_emb_loss": emb_loss,
            "val_avg_class_loss": class_loss,
        }

        # calculate metrics
        if self.label_thresholds is not None:
            raw_labels = []
            for x in outputs:
                raw_labels += x["val_raw_labels"]
            predicted_labels = cpu_predictions.numpy() > self.label_thresholds
            for sample_idx, has_one in enumerate(predicted_labels.any(axis=1)):
                if has_one:
                    continue
                predicted_labels[
                    sample_idx, cpu_predictions[sample_idx].argmax()
                ] = True

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
                predicted_classes=self.data_config["classes"],
                predicted_labels=predicted_labels,
                predicted_probabilities=cpu_predictions.numpy(),
                raw_ground_truth_labels=raw_labels,
            )

            # class_output_string = "Classes,{}\nAUROC,{}\nAUPRC,{}\nF-measure,{}".format(
            #     ",".join(classes),
            #     ",".join("{:.3f}".format(x) for x in auroc_classes),
            #     ",".join("{:.3f}".format(x) for x in auprc_classes),
            #     ",".join("{:.3f}".format(x) for x in f_measure_classes),
            # )
            # print(class_output_string)
            log.update(
                {
                    "val_auroc": auroc,
                    "val_auprc": auprc,
                    "val_accuracy": accuracy,
                    "val_f_measure": f_measure,
                    "val_f_beta_measure": f_beta_measure,
                    "val_g_beta_measure": g_beta_measure,
                    "val_challenge_metric": challenge_metric,
                }
            )
        else:
            print("skipping validation metrics, no thresholds set yet")
            challenge_metric = -1

        return {"log": log, "val_loss": loss, "val_challenge_metric": challenge_metric}

    def test_step(self, batch, batch_idx):
        beat_windows, seq_lens, dxs, str_abbrv_dxs, str_code_dxs = batch
        pred_classes, bottleneck, x_source, x_hat = self(beat_windows, seq_lens)
        emb_loss = self.emb_loss_function(x_hat, x_source)
        class_loss = self.class_loss_function(
            pred_classes, dxs, self.data_config["test_weights"]
        )
        loss = emb_loss + class_loss

        # embed_x_source = torch.stack([s[:2, :] for s in x_source]).flatten(1)
        # embed_x_reco = torch.stack([s[:2, :] for s in x_hat]).flatten(1)
        cpu_labels = dxs.detach().cpu()
        cpu_predictions = torch.sigmoid(pred_classes).detach().cpu()

        return {
            "test_loss": loss,
            "test_emb_loss": emb_loss,
            "test_class_loss": class_loss,
            "bottleneck_lstm": bottleneck,
            # "embed_x_source": embed_x_source,
            # "embed_x_reco": embed_x_reco,
            "test_labels": cpu_labels,
            "test_raw_labels": str_code_dxs,
            "test_predictions": cpu_predictions,
        }

    def test_epoch_end(self, outputs):
        loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        emb_loss = torch.stack([x["test_emb_loss"] for x in outputs]).mean()
        class_loss = torch.stack([x["test_class_loss"] for x in outputs]).mean()
        # cpu_labels = torch.cat([x["test_labels"] for x in outputs])
        cpu_predictions = torch.cat([x["test_predictions"] for x in outputs])

        # embed_x_source = torch.cat([x["embed_x_source"] for x in outputs], dim=0)
        # embed_x_reco = torch.cat([x["embed_x_reco"] for x in outputs], dim=0)
        bottleneck = torch.cat([x["bottleneck_lstm"] for x in outputs], dim=0)
        meta = []
        for x in outputs:
            meta += x["test_raw_labels"]
        self.logger.experiment.add_embedding(
            bottleneck,
            tag="bottleneck_embedding",
            metadata=meta,
            global_step=self.global_step,
        )

        log = {
            "test_loss": loss,
            "test_emb_loss": emb_loss,
            "test_class_loss": class_loss,
        }

        # calculate metrics
        raw_labels = []
        for x in outputs:
            raw_labels += x["test_raw_labels"]
        predicted_labels = cpu_predictions.numpy() > self.label_thresholds
        for sample_idx, has_one in enumerate(predicted_labels.any(axis=1)):
            if has_one:
                continue
            predicted_labels[sample_idx, cpu_predictions[sample_idx].argmax()] = True

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
            predicted_classes=self.data_config["classes"],
            predicted_labels=predicted_labels,
            predicted_probabilities=cpu_predictions.numpy(),
            raw_ground_truth_labels=raw_labels,
        )

        log.update(
            {
                "test_auroc": auroc,
                "test_auprc": auprc,
                "test_accuracy": accuracy,
                "test_f_measure": f_measure,
                "test_f_beta_measure": f_beta_measure,
                "test_g_beta_measure": g_beta_measure,
                "test_challenge_metric": challenge_metric,
            }
        )
        # print(self.label_thresholds)

        log_dir = self.logger.experiment.get_logdir()
        with open(os.path.join(log_dir, "test_output.json"), "w") as f:
            test_output = {}
            for k, v in log.items():
                test_output[k] = float(v)
            test_output["thresholds"] = self.label_thresholds.tolist()
            json.dump(test_output, f)
        with open(os.path.join(log_dir, "labelwise_metrics.csv"), "w") as f:
            class_output_string = "Classes,{}\nAUROC,{}\nAUPRC,{}\nF-measure,{}".format(
                ",".join(classes),
                ",".join("{:.3f}".format(x) for x in auroc_classes),
                ",".join("{:.3f}".format(x) for x in auprc_classes),
                ",".join("{:.3f}".format(x) for x in f_measure_classes),
            )
            f.write(class_output_string)

        return {"log": log, "test_loss": loss}

    def configure_optimizers(self):
        # opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        opt = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.base_lr,
            momentum=self.momentum,
        )
        sched = torch.optim.lr_scheduler.CyclicLR(
            opt, base_lr=self.base_lr, max_lr=self.max_lr
        )
        return [opt,], [
            {
                "scheduler": sched,
                "interval": "step",
            }
        ]


if __name__ == "__main__":
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "bae_checkpoint", type=str, help="Path to LinearBeatAutoencoder checkpoint"
    )
    parser.add_argument(
        "--seq_len",
        default=20,
        type=int,
        help="Maximum number of beats per record",
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="Dataloader batch size"
    )
    parser.add_argument(
        "--train_workers", default=64, type=int, help="Train dataloader workers"
    )
    parser.add_argument(
        "--val_workers", default=16, type=int, help="Val dataloader workers"
    )
    parser.add_argument(
        "--test_workers", default=16, type=int, help="Test dataloader workers"
    )

    parser.add_argument(
        "--num_layers", default=2, type=int, help="Number of LSTM layers"
    )
    parser.add_argument(
        "--base_lr", default=1e-5, type=float, help="Cyclic base learning rate"
    )
    parser.add_argument(
        "--max_lr", default=1e-3, type=float, help="Cyclic max learning rate"
    )
    parser.add_argument(
        "--lstm_dropout",
        default=0.1,
        type=float,
        help="Autoencoder LSTM dropout",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Autoencoder classifier dropout"
    )
    parser.add_argument(
        "--hidden_size", default=256, type=int, help="Classifier hidden dimension"
    )
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum")

    args = parser.parse_args()

    state_dict = torch.load(args.bae_checkpoint)
    window_size = state_dict["hyper_parameters"]["pqrst_window_size"]
    data_config = state_dict["hyper_parameters"]["data_config"]

    cinc2020seq = SequenceZarrDataModule(
        window_size=window_size,
        record_splits=data_config,
        sequence_length=args.seq_len,
        batch_size=args.batch_size,
        train_workers=args.train_workers,
        val_workers=args.val_workers,
        test_workers=args.test_workers,
    )
    model = SequenceAutoEncoder(
        args.bae_checkpoint,
        base_lr=args.base_lr,
        max_lr=args.max_lr,
        num_layers=args.num_layers,
        lstm_dropout=args.lstm_dropout,
        momentum=args.momentum,
        sequence_length=args.seq_len,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        data_config=cinc2020seq.data_config(),
    )

    # ==== Callbacks ====
    # Custom logger output directory
    logger = pl.loggers.TensorBoardLogger(
        save_dir=os.getcwd(), name="log_sequence_autoencoder_100_long"
    )
    # Early stopping on validation classification metric
    early_stopping = EarlyStopping(
        monitor="val_challenge_metric", patience=30, verbose=True, mode="max"
    )
    checkpointer = ModelCheckpoint(
        filepath=logger.experiment.get_logdir(),
        prefix="",
        monitor="val_challenge_metric",
        mode="max",
        verbose=True,
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        checkpoint_callback=checkpointer,
        callbacks=[LearningRateLogger(), early_stopping],
        max_epochs=200,
    )
    trainer.logger.log_hyperparams(args)
    trainer.fit(model, cinc2020seq)
    trainer.test(model)
