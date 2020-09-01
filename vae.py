#!/usr/bin/env python3
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from datasets import CinC2020


class VAE(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()

        self.lr = lr

        self.fc1 = nn.Linear(5000, 500)
        self.fc21 = nn.Linear(500, 100)
        self.fc22 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 500)
        self.fc4 = nn.Linear(500, 5000)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy_with_logits(
            recon_x, x.view(-1, 5000), reduction="sum"
        )

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def forward(self, z):
        return self.decode(z)

    def training_step(self, batch, batch_idx):
        x, *_ = batch

        mu, logvar = self.encode(x.view(-1, 5000))
        z = self.reparameterize(mu, logvar)
        x_hat = self(z)
        loss = self.loss_function(x_hat, x, mu, logvar)

        log = {"train_loss": loss}
        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_idx):
        x, *_ = batch

        mu, logvar = self.encode(x.view(-1, 5000))
        z = self.reparameterize(mu, logvar)
        x_hat = self(z)
        val_loss = self.loss_function(x_hat, x, mu, logvar)

        return {"val_loss": val_loss, "x_hat": x_hat}

    def validation_epoch_end(self, outputs):

        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        log = {"avg_val_loss": val_loss}
        return {"log": log, "val_loss": val_loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--batch_size", default=32, type=int, help="default: 32")
    parser.add_argument(
        "--learning_rate", default=1e-3, type=float, help="default 1e-3"
    )

    args = parser.parse_args()

    dataset = CinC2020(set_seq_len=5000)
    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len
    train, val = random_split(dataset, [train_len, val_len])

    model = VAE(lr=args.learning_rate).double()
    trainer = pl.Trainer.from_argparse_args(
        args,
        # fast_dev_run=True
    )
    trainer.fit(
        model,
        DataLoader(train, batch_size=args.batch_size),
        DataLoader(val, batch_size=args.batch_size),
    )
