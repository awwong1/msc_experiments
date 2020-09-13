#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import kl_div
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from datasets import CinC2020BeatsDataModule
from utils import View


class MetricsCallback(pl.Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


class Encoder(nn.Module):
    def __init__(
        self, seq_len=400, n_features=12, hidden_dim=512, embedding_dim=128, dropout=0.1
    ):
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        self.enc = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(self.seq_len * self.n_features, self.hidden_dim),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
        )

    def forward(self, x):
        return self.enc(x)


class Decoder(nn.Module):
    def __init__(self, seq_len=400, n_features=12, hidden_dim=512, embedding_dim=128):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.dec = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.seq_len * self.n_features),
            View((-1, self.seq_len, self.n_features)),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.dec(x)


class BeatAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-2,
        num_leads: int = 12,
        pqrst_window_size: int = 400,
        hidden_dim: int = 512,
        embedding_dim: int = 128,
    ):
        super().__init__()

        self.lr = lr
        self.num_leads = num_leads
        self.pqrst_window_size = pqrst_window_size
        self.example_input_array = torch.rand(1, pqrst_window_size, num_leads)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # encoding
        self.enc = Encoder(
            seq_len=self.pqrst_window_size,
            n_features=self.num_leads,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
        )

        # decoding
        self.dec = Decoder(
            seq_len=self.pqrst_window_size,
            n_features=self.num_leads,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
        )

    def loss_function(self, recon_x, x):
        # WARNING! x may contain NaNs
        _x = x[~torch.isnan(x)]
        _recon_x = recon_x[~torch.isnan(x)]

        # return F.binary_cross_entropy_with_logits(
        #     _recon_x, _x, reduction="sum"
        # )  # no sigmoid()
        # return F.binary_cross_entropy(_recon_x, _x, reduction="sum")  # set sigmoid()
        return F.mse_loss(_recon_x, _x, reduction="sum")  # set sigmoid()
        # return F.l1_loss(_recon_x, _x, reduction="sum")

    @staticmethod
    def sum_kl_divergence(recon_x, x):
        _x = torch.sigmoid(x).detach().cpu().numpy()
        _x_reco = torch.sigmoid(recon_x).detach().cpu().numpy()
        return kl_div(_x, _x_reco).sum()

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        # x_hat = x_hat.view(-1, 400, 12)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, *_ = batch
        x_hat = self(x)
        loss = self.loss_function(x_hat, x)
        kl_div = self.sum_kl_divergence(x_hat, x)

        # if torch.isnan(loss):
        #     x_recon = x_hat.detach().cpu().numpy()
        #     x_source = x_spec.detach().cpu().numpy()
        #     with open("nan_loss.npy", "wb") as f:
        #         np.save(f, x_recon)
        #         np.save(f, x_source)
        #     raise Exception(f"NaN at epoch {self.current_epoch} batch_idx: {batch_idx}")

        if self.logger and batch_idx % 37 == 0:
            _x = x.detach().cpu().numpy()
            _x_reco = x_hat.detach().cpu().numpy()

            fig, axs = plt.subplots(12, 2, figsize=(10, 8))
            axs[0, 0].set_title("Source")
            axs[0, 1].set_title("Generated")
            for lead_idx in range(12):
                axs[lead_idx, 0].plot(_x[0, :, lead_idx])
                axs[lead_idx, 1].plot(_x_reco[0, :, lead_idx])
            fig.suptitle(f"Epoch {self.current_epoch}, Batch {batch_idx}")
            self.logger.experiment.add_figure("train_spec", fig, self.global_step)
            plt.close(fig)

        log = {
            "train_loss": loss,
            "train_kl_div": kl_div,
        }
        return {"loss": loss, "log": log}  # "kl_div": kl_div,

    # def train_epoch_end(self, outputs):
    #     loss = torch.stack([x["loss"] for x in outputs]).mean()
    #     kl_div = np.stack([x["kl_div"] for x in outputs]).mean()
    #     log = {"avg_loss/train": loss, "avg_kl_div/train": kl_div}
    #     return {"log": log}

    def validation_step(self, batch, batch_idx):
        x, *_ = batch
        x_hat = self(x)
        loss = self.loss_function(x_hat, x)
        kl_div = self.sum_kl_divergence(x_hat, x)
        return {"val_loss": loss, "val_kl_div": kl_div}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        kl_div = np.stack([x["val_kl_div"] for x in outputs]).mean()
        log = {"val_avg_loss": loss, "val_avg_kl_div": kl_div}
        return {"log": log, "val_loss": loss}

    def configure_optimizers(self):
        # opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        sched = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=1e-5, max_lr=self.lr)
        return [opt,], [
            {
                "scheduler": sched,
                "interval": "step",
            }
        ]


def objective(trial):
    DIR = os.getcwd()
    MODEL_DIR = os.path.join(DIR, "result")

    lr = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    pqrst_window_size = trial.suggest_categorical('pqrst_window_size', [300, 400, 500])
    hidden_dim = trial.suggest_int('hidden_dim', 64, 1024)
    embedding_dim = trial.suggest_int('embedding_dim', 64, 1024)

    # Filenames for each trial must be made unique in order to access each checkpoint.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "trial_{}".format(trial.number), "{epoch}"), monitor="val_loss"
    )

    cinc2020beat = CinC2020BeatsDataModule(
        pqrst_window_size=pqrst_window_size,
        batch_size=128,
        train_workers=16,
        val_workers=16,
    )
    model = BeatAutoEncoder(
        lr=lr,
        pqrst_window_size=pqrst_window_size,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim
    )

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We don't use any logger here as it requires us to implement several abstract
    # methods. Instead we setup a simple callback, that saves metrics from each validation step.
    metrics_callback = MetricsCallback()
    trainer = pl.Trainer(
        # fast_dev_run=True,
        logger=False,
        checkpoint_callback=checkpoint_callback,
        max_epochs=100,
        gpus=4,
        callbacks=[metrics_callback],
        # distributed_backend="ddp",
        early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="val_loss"),
    )

    trainer.fit(model, cinc2020beat)

    return metrics_callback.metrics[-1]["val_loss"].item()


if __name__ == "__main__":
    # from argparse import ArgumentParser

    # parser = ArgumentParser(description="Beat Autoencoder (Linear Symmetric)")
    # parser = pl.Trainer.add_argparse_args(parser)
    # parser.add_argument(
    #     "--pqrst_window_size", default=400, type=int, help="default: 400"
    # )
    # parser.add_argument("--batch_size", default=128, type=int, help="default: 128")
    # parser.add_argument("--train_workers", default=8, type=int, help="default: 8")
    # parser.add_argument("--val_workers", default=4, type=int, help="default: 4")
    # parser.add_argument("--lr", default=1e-3, type=float, help="default 1e-3")
    # parser.add_argument("--hidden_dim", default=512, type=int, help="default 512")
    # parser.add_argument("--embedding_dim", default=128, type=int, help="default 128")

    # args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner()  # optuna.pruners.NopPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_jobs=8, n_trials=1, timeout=None)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # shutil.rmtree(MODEL_DIR)
