#!/usr/bin/env python3
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        train_workers: int = 8,
        val_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_workers = train_workers
        self.val_workers = val_workers

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )

    def setup(self, stage=None):
        self.train_ds = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=self.transform
        )
        self.val_ds = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=self.transform
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.train_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=self.val_workers
        )


class CIFAR10AutoEncoder(pl.LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()

        self.lr = lr

        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),
            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),
            # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            # [batch, 48, 4, 4]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def training_step(self, batch, batch_idx):
        x, _ = batch
        encoded, reconstructed = self(x)

        loss = F.binary_cross_entropy(reconstructed, x, reduction="sum")
        log = {"train_loss": loss}

        if batch_idx % 7 == 0:
            gen_grid = torchvision.utils.make_grid(reconstructed)
            self.logger.experiment.add_image("generated_images", gen_grid, self.global_step)
            source_grid = torchvision.utils.make_grid(x)
            self.logger.experiment.add_image("source_images", source_grid, self.global_step)

        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        encoded, reconstructed = self(x)

        loss = F.binary_cross_entropy(reconstructed, x, reduction="sum")
        return {"val_loss": loss, "recon": reconstructed}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        log = {"mean_val_loss": val_loss}
        return {"log": log, "mean_val_loss": val_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--batch_size", default=64, type=int, help="default: 32")
    parser.add_argument("--lr", default=1e-3, type=float, help="default: 0.001")
    parser.add_argument("--train_workers", default=8, type=int, help="default: 8")
    parser.add_argument("--val_workers", default=4, type=int, help="default: 4")

    args = parser.parse_args()

    cifar10 = CIFAR10DataModule(
        batch_size=args.batch_size,
        train_workers=args.train_workers,
        val_workers=args.val_workers,
    )
    model = CIFAR10AutoEncoder(lr=args.lr)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, cifar10)
