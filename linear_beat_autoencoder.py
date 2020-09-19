#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import BeatsZarrDataModule
from utils import View


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
    def __init__(
        self, seq_len=400, n_features=12, hidden_dim=512, embedding_dim=128, dropout=0.1
    ):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        self.dec = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(True),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.seq_len * self.n_features),
            View((-1, self.seq_len, self.n_features)),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.dec(x)


class BeatAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        base_lr: float = 1e-2,
        max_lr: float = 1e-3,
        num_leads: int = 12,
        pqrst_window_size: int = 400,
        hidden_dim: int = 1024,
        embedding_dim: int = 700,
        dropout: float = 0.1,
        momentum: float = 0.5,
        batch_size: int = 512,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.num_leads = num_leads
        self.pqrst_window_size = pqrst_window_size
        self.example_input_array = torch.rand(1, pqrst_window_size, num_leads)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.momentum = momentum
        self.batch_size = batch_size

        # encoding
        self.enc = Encoder(
            seq_len=self.pqrst_window_size,
            n_features=self.num_leads,
            hidden_dim=hidden_dim,
            embedding_dim=self.embedding_dim,
            dropout=self.dropout,
        )

        # decoding
        self.dec = Decoder(
            seq_len=self.pqrst_window_size,
            n_features=self.num_leads,
            hidden_dim=hidden_dim,
            embedding_dim=self.embedding_dim,
            dropout=self.dropout,
        )

    def loss_function(self, recon_x, x):
        return (
            F.mse_loss(recon_x, x, reduction="sum") / self.batch_size
        )
        # WARNING! x may contain NaNs
        # _x = x[~torch.isnan(x)]
        # _recon_x = recon_x[~torch.isnan(x)]

        # return F.binary_cross_entropy_with_logits(
        #     _recon_x, _x, reduction="sum"
        # )  # no sigmoid()
        # return F.binary_cross_entropy(_recon_x, _x, reduction="sum")  # set sigmoid()

        # return F.l1_loss(_recon_x, _x, reduction="sum")

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        # x_hat = x_hat.view(-1, 400, 12)
        return z, x_hat

    def training_step(self, batch, batch_idx):
        x = batch
        _, x_hat = self(x)
        loss = self.loss_function(x_hat, x)

        if torch.isnan(loss):
            x_recon = x_hat.detach().cpu().numpy()
            x_source = x.detach().cpu().numpy()
            with open("nan_loss.npy", "wb") as f:
                np.save(f, x_recon)
                np.save(f, x_source)
            raise Exception(f"NaN at epoch {self.current_epoch} batch_idx: {batch_idx}")

        if batch_idx % 101 == 0:
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
        }
        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_idx):
        x = batch
        _, x_hat = self(x)
        loss = self.loss_function(x_hat, x)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        log = {"val_avg_loss": loss}
        return {"log": log, "val_loss": loss}

    def test_step(self, batch, batch_idx):
        x = batch
        _, x_hat = self(x)
        loss = self.loss_function(x_hat, x)
        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        log = {"test_avg_loss": loss}
        return {"log": log, "test_loss": loss}

    def configure_optimizers(self):
        # opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        opt = torch.optim.SGD(
            self.parameters(), lr=self.base_lr, momentum=self.momentum
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
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--window_size", default=400, type=int, help="default: 400")
    parser.add_argument(
        "--batch_size", default=1024, type=int, help="Dataloader batch size"
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
        "--base_lr", default=1e-5, type=float, help="Cyclic base learning rate"
    )
    parser.add_argument(
        "--max_lr", default=1e-3, type=float, help="Cyclic max learning rate"
    )
    parser.add_argument(
        "--hidden_dim", default=1024, type=int, help="Autoencoder hidden dimension"
    )
    parser.add_argument(
        "--embedding_dim",
        default=768,
        type=int,
        help="Autoencoder bottleneck dimension",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Autoencoder dropout"
    )
    parser.add_argument("--momentum", default=0.5, type=float, help="SGD momentum")

    args = parser.parse_args()

    cinc2020beat = BeatsZarrDataModule(
        window_size=args.window_size,
        batch_size=args.batch_size,
        train_workers=args.train_workers,
        val_workers=args.val_workers,
        test_workers=args.test_workers,
    )
    model = BeatAutoEncoder(
        pqrst_window_size=args.window_size,
        base_lr=args.base_lr,
        max_lr=args.max_lr,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        momentum=args.momentum,
        batch_size=args.batch_size,
        data_config=cinc2020beat.data_config(),
    )

    # log the learning rate
    from pytorch_lightning.callbacks.lr_logger import LearningRateLogger

    lr_logger = LearningRateLogger()

    trainer = pl.Trainer.from_argparse_args(args, callbacks=[lr_logger])
    trainer.logger.log_hyperparams(args)
    trainer.fit(model, cinc2020beat)
    trainer.test(model)
