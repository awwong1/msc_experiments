#!/usr/bin/env python3
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import random_split

from datasets import CinC2020


class CinC2020DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        seq_len: int = 5000,
        fs: int = 500,
        batch_size: int = 32,
        train_workers: int = 8,
        val_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.fs = fs
        self.batch_size = batch_size
        self.train_workers = train_workers
        self.val_workers = val_workers

    def setup(self, stage=None):
        dataset = CinC2020(set_seq_len=self.seq_len, fs=self.fs)
        train_len = int(len(dataset) * 0.8)  # 80% train, 20% validation
        val_len = len(dataset) - train_len
        train, val = random_split(dataset, [train_len, val_len])

        self.train_ds = train
        self.val_ds = val

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.train_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.val_workers,
        )


class BasicAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        seq_len=5000,
        n_fft: int = 50,
        power: float = 0.1,
        normalized: bool = True,
    ):
        super().__init__()

        self.lr = lr
        self.seq_len = seq_len

        self.to_spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, power=power, normalized=True
        )
        self.to_waveform = torchaudio.transforms.GriffinLim(
            n_fft=n_fft, power=power, normalized=normalized
        )

        # encoding
        self.enc = nn.Sequential(
            # torch.Size([b, 12, 26, 201])
            nn.Conv2d(
                in_channels=12,
                out_channels=24,
                kernel_size=(26, 3),
                stride=2,
                padding=0,
                groups=1,
            ),
            nn.Softplus(),
            # torch.Size([b, 24, 1, 100])
            nn.Flatten(2, -1),
            # torch.Size([b, 24, 100])
            nn.Conv1d(
                in_channels=24,
                out_channels=48,
                kernel_size=4,
                stride=2,
                padding=0,
                groups=1,
            ),
            nn.Softplus()
            # torch.Size([b, 48, 49])
        )
        # decoding
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=48,
                out_channels=12,
                kernel_size=32,
                stride=2,
                padding=0,
                groups=1,
            ),
            nn.Softplus(),
            # torch.Size([b, 12, 128])
            nn.Linear(128, 26 * 201),
            # torch.Size([3, 12, 5226]),
            # Unflatten(2, torch.Size((26, 201)))
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = torch.transpose(x, 1, 2)
        x_spec = self.to_spectrogram(x).detach()
        return x_spec, self.enc(x_spec)

    def decode(self, z):
        x_hat = self.dec(z)
        x_hat = x_hat.view(-1, 12, 26, 201)
        # x_hat = torch.transpose(x_hat, 1, 2)
        return x_hat

    def loss_function(self, recon_x, x):
        # WARNING! x may contain NaNs
        _x = x[~torch.isnan(x)]
        _recon_x = recon_x[~torch.isnan(x)]

        # return F.binary_cross_entropy(recon_x, x, reduction="sum")
        # return F.binary_cross_entropy_with_logits(recon_x, x, reduction="sum")
        return F.mse_loss(_recon_x, _x, reduction="sum")

    def forward(self, z):
        return self.decode(z)

    def training_step(self, batch, batch_idx):
        x, *_ = batch

        x_spec, z = self.encode(x)
        x_hat = self(z)
        loss = self.loss_function(x_hat, x_spec)

        # only works with batch size 1, debugging
        # source_grid = torchvision.utils.make_grid(torch.transpose(x_spec, 0, 1), nrow=1)
        # self.logger.experiment.add_image("source_images", source_grid, batch_idx)
        # gen_grid = torchvision.utils.make_grid(torch.transpose(x_hat, 0, 1), nrow=1)
        # self.logger.experiment.add_image("generated_images", gen_grid, batch_idx)

        # if torch.isnan(loss):
        #     x_recon = x_hat.detach().cpu().numpy()
        #     x_source = x_spec.detach().cpu().numpy()
        #     with open("nan_loss.npy", "wb") as f:
        #         np.save(f, x_recon)
        #         np.save(f, x_source)
        #     raise Exception(f"NaN at epoch {self.current_epoch} batch_idx: {batch_idx}")

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
        return torch.optim.Adam(self.parameters(), lr=self.lr)
        # return torch.optim.SGD(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--seq_len", default=5000, type=int, help="default: 5000")
    parser.add_argument("--batch_size", default=32, type=int, help="default: 32")
    parser.add_argument("--fs", default=500, type=int, help="default: 500")
    parser.add_argument("--train_workers", default=8, type=int, help="default: 8")
    parser.add_argument("--val_workers", default=4, type=int, help="default: 4")
    parser.add_argument("--lr", default=1e-3, type=float, help="default 1e-3")

    args = parser.parse_args()

    cinc2020 = CinC2020DataModule(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        fs=args.fs,
        train_workers=args.train_workers,
        val_workers=args.val_workers,
    )
    model = BasicAutoEncoder(lr=args.lr, seq_len=args.seq_len)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, cinc2020)
