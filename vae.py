#!/usr/bin/env python3
import pytorch_lightning as pl
import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from datasets import CinC2020


class BasicAutoEncoder(pl.LightningModule):
    def __init__(
        self, lr: float = 1e-3, seq_len=4000, n_fft: int = 50, power: float = 2.0
    ):
        super().__init__()

        self.lr = lr
        self.seq_len = seq_len

        self.to_spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, power=power, normalized=True
        )
        self.to_waveform = torchaudio.transforms.GriffinLim(
            n_fft=n_fft, power=power, normalized=True
        )

        # encoding
        self.enc = nn.Sequential(
            # torch.Size([b, 12, 26, 201])
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3, 2)),
            nn.ReLU(),
            # torch.Size([b, 24, 24, 200])
            nn.MaxPool2d(2),
            # torch.Size([b, 24, 12, 100])
        )
        # decoding
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=24, out_channels=24, kernel_size=1, stride=1
            ),
            nn.ReLU(),
            # torch.Size([b, 24, 12, 100])
            nn.Upsample(scale_factor=2),
            # torch.Size([b, 24, 24, 200])
            nn.ConvTranspose2d(
                in_channels=24, out_channels=12, kernel_size=(3, 2), stride=1
            ),
            nn.ReLU(),
            # torch.Size([b, 12, 26, 201])
        )

    def encode(self, x):
        x = torch.transpose(x, 1, 2)
        x_spec = self.to_spectrogram(x)
        return x_spec, self.enc(x_spec)

    def decode(self, z):
        x_hat = self.dec(z)
        # x_hat = torch.transpose(x_hat, 1, 2)
        return x_hat

    def loss_function(self, recon_x, x):
        # return F.binary_cross_entropy_with_logits(recon_x, x, reduction="sum")
        return F.mse_loss(recon_x, x, reduction="mean")

    def forward(self, z):
        return self.decode(z)

    def training_step(self, batch, batch_idx):
        x, *_ = batch

        x_spec, z = self.encode(x)
        x_hat = self(z)
        loss = self.loss_function(x_hat, x_spec)

        log = {"train_loss": loss}
        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_idx):
        x, *_ = batch

        x_spec, z = self.encode(x)
        x_hat = self(z)
        val_loss = self.loss_function(x_hat, x_spec)

        recon = torch.transpose(self.to_waveform(x_hat), 1, 2)
        return {"val_loss": val_loss, "recon": recon}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        log = {"avg_val_loss": val_loss}
        return {"log": log, "val_loss": val_loss}

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.lr)
        return torch.optim.SGD(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--seq_len", default=5000, type=int, help="default: 5000")
    parser.add_argument("--batch_size", default=32, type=int, help="default: 32")
    parser.add_argument("--fs", default=500, type=int, help="default: 500")
    parser.add_argument("--train_dl_workers", default=0, type=int, help="default: 0")
    parser.add_argument("--val_dl_workers", default=0, type=int, help="default: 0")
    parser.add_argument(
        "--learning_rate", default=1e-3, type=float, help="default 1e-3"
    )

    args = parser.parse_args()

    dataset = CinC2020(set_seq_len=args.seq_len, fs=args.fs)
    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len
    train, val = random_split(dataset, [train_len, val_len])

    model = BasicAutoEncoder(lr=args.learning_rate, seq_len=args.seq_len)
    trainer = pl.Trainer.from_argparse_args(
        args,
        # fast_dev_run=True
    )
    trainer.fit(
        model,
        DataLoader(
            train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.train_dl_workers,
        ),
        DataLoader(val, batch_size=args.batch_size, num_workers=args.val_dl_workers),
    )
