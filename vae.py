#!/usr/bin/env python3
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from datasets import CinC2020DataModule
from utils import View


class BasicAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-2,
        seq_len=5000,
        n_fft: int = 50,
        power: float = 0.1,
        normalized: bool = False,
    ):
        super().__init__()

        self.lr = lr

        self.seq_len = seq_len
        self.example_input_array = torch.rand(1, 12, 26, 201)

        self.to_spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, power=power, normalized=normalized
        )
        self.to_waveform = torchaudio.transforms.GriffinLim(
            n_fft=n_fft, power=power, normalized=normalized
        )

        # encoding
        self.enc = nn.Sequential(
            # torch.Size([b, 12, 26, 201])
            nn.Flatten(2, -1),
            # torch.Size([b, 12, 5226])
            nn.Linear(5226, 256),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            # torch.Size([b, 12, 256])
            nn.Linear(256, 128),
            nn.Dropout(p=0.1),
            nn.ReLU()
            # torch.Size([b, 12, 128])
        )

        # decoding
        self.dec = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            # torch.Size([b, 12, 256])
            nn.Linear(256, 5226),
            # torch.Size([b, 12, 5226])
            View((-1, 12, 26, 201)),
            # torch.Size([b, 12, 26, 201])
            nn.Sigmoid(),
        )

    def loss_function(self, recon_x, x):
        # WARNING! x may contain NaNs
        _x = x[~torch.isnan(x)]
        _recon_x = recon_x[~torch.isnan(x)]

        # return F.binary_cross_entropy_with_logits(
        #     _recon_x, _x, reduction="sum"
        # )  # no sigmoid()
        return F.binary_cross_entropy(_recon_x, _x, reduction="sum")  # set sigmoid()
        # return F.mse_loss(_recon_x, _x, reduction="sum")  # set sigmoid()

    def forward(self, x_spec):
        z = self.enc(x_spec)
        x_hat = self.dec(z)
        x_hat = x_hat.view(-1, 12, 26, 201)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, *_ = batch

        x[torch.isnan(x)] = 0.0

        x = torch.transpose(x, 1, 2)
        x_spec = self.to_spectrogram(x).detach()
        x_hat = self(x_spec)
        loss = self.loss_function(x_hat, x_spec)

        # if torch.isnan(loss):
        #     x_recon = x_hat.detach().cpu().numpy()
        #     x_source = x_spec.detach().cpu().numpy()
        #     with open("nan_loss.npy", "wb") as f:
        #         np.save(f, x_recon)
        #         np.save(f, x_source)
        #     raise Exception(f"NaN at epoch {self.current_epoch} batch_idx: {batch_idx}")

        if batch_idx % 7 == 0:
            # x_hat = torch.sigmoid(x_hat)  # only when bce_with_logits
            # Spectrograms
            _x_spec = x_spec.detach().cpu().numpy()
            _x_reco_spec = x_hat.detach().cpu().numpy()

            _x_spec_img = np.concatenate(
                [_x_spec[0, i, :, :].squeeze() for i in range(12)]
            )
            _x_reco_img = np.concatenate(
                [_x_reco_spec[0, i, :, :].squeeze() for i in range(12)]
            )

            fig, ax = plt.subplots(1, 2, figsize=(10, 10))
            ax[0].set_title("Source")
            im = ax[0].imshow(
                _x_spec_img,
                cmap="Greys",
            )
            ax[0].set_yticklabels([])
            fig.colorbar(im, ax=ax[0], orientation="horizontal")
            ax[1].set_title("Generated")
            im = ax[1].imshow(
                _x_reco_img,
                cmap="Greys",
                vmin=_x_spec_img.min(),
                vmax=_x_spec_img.max(),
            )
            ax[1].set_yticklabels([])
            fig.colorbar(im, ax=ax[1], orientation="horizontal")
            fig.tight_layout()
            fig.suptitle(f"Epoch {self.current_epoch}, Batch {batch_idx}")
            self.logger.experiment.add_figure("train_spec", fig, self.global_step)
            plt.close(fig)

            # Signal
            _x_sig = self.to_waveform(x_spec).detach().cpu().numpy()
            _x_reco_sig = self.to_waveform(x_hat).detach().cpu().numpy()

            sig_fig = plt.figure(constrained_layout=True, figsize=(10, 8))
            spec = gridspec.GridSpec(ncols=2, nrows=12, figure=sig_fig)
            ax = None
            for i in range(12):
                ax_sig = sig_fig.add_subplot(spec[i, 0])
                ax_sig.set_xticklabels([])
                ax_reco = sig_fig.add_subplot(spec[i, 1])
                ax_reco.set_xticklabels([])
                if i == 0:
                    ax_sig.set_title("Source")
                    ax_reco.set_title("Generated")
                ax_sig.plot(_x_sig[0, i])
                ax_reco.plot(_x_reco_sig[0, i])
            spec.tight_layout(sig_fig)
            sig_fig.suptitle(f"Epoch {self.current_epoch}, Batch {batch_idx}")
            self.logger.experiment.add_figure("train_sig", sig_fig, self.global_step)
            plt.close(sig_fig)

        log = {"train_loss": loss}
        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_idx):
        x, *_ = batch

        x[torch.isnan(x)] = 0.0

        # convert signal to spectrogram input
        x = torch.transpose(x, 1, 2)
        x_spec = self.to_spectrogram(x).detach()
        x_hat = self(x_spec)
        val_loss = self.loss_function(x_hat, x_spec)
        # recon = torch.transpose(self.to_waveform(x_hat), 1, 2)

        payload = {"val_loss": val_loss}
        if batch_idx == 0:
            payload["sample"] = (x_spec, x_hat)
        return payload

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        log = {"avg_val_loss": val_loss}

        return {"log": log, "val_loss": val_loss}

    def configure_optimizers(self):
        # opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        sched = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=0.0001, max_lr=self.lr)
        return [opt,], [
            {
                "scheduler": sched,
                "interval": "step",
            }
        ]


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

    # log the learning rate
    from pytorch_lightning.callbacks.lr_logger import LearningRateLogger

    lr_logger = LearningRateLogger()

    trainer = pl.Trainer.from_argparse_args(args, callbacks=[lr_logger])
    trainer.fit(model, cinc2020)
