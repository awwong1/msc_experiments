#!/usr/bin/env python3
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence

from datasets import SequenceZarrDataModule
from linear_beat_autoencoder import BeatAutoEncoder


class SequenceAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        bae_checkpoint_path: str,
        base_lr: float = 1e-2,
        max_lr: float = 1e-3,
        num_layers: int = 2,
        dropout: float = 0.1,
        momentum: float = 0.9,
        sequence_length: int = 10,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.num_layers = num_layers
        state_dict = torch.load(bae_checkpoint_path)
        self.embedding_dim = state_dict["hyper_parameters"]["embedding_dim"]
        self.momentum = momentum

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
            dropout=dropout,
        )
        self.lstm_dec = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            num_layers=self.num_layers,
            dropout=dropout,
        )

    def loss_function(self, recon_x, source_x):
        # recon_x and x are lists of tensors, each element is a sample
        # the entire list being the batch
        sq_err = torch.sum(
            torch.stack(
                [
                    F.mse_loss(r_x, s_x, reduction="sum")  # Sum over sample
                    for r_x, s_x in zip(recon_x, source_x)
                ]
            )
        )
        msq_err = sq_err / len(recon_x)  # Average over batches
        return msq_err

    def forward(self, x):
        # x is arbitrary length beat window sequences
        bw, lstm_input_lengths = x

        # Convert into beat autoencoder embedding
        raw_seq_enc_input, _recon = zip(*map(self.beat_autoencoder, bw))
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
        bw_outputs = []
        for idx, length in enumerate(lstm_input_lengths):
            bw_outputs.append(recon_seq[0:length, idx, :])
        #         print(len(bw_outputs))
        #         print([bwo.shape for bwo in bw_outputs])

        return seq_bottleneck, raw_seq_enc_input, bw_outputs

    def training_step(self, batch, batch_idx):
        x = batch
        _, x_source, x_hat = self(x)
        loss = self.loss_function(x_hat, x_source)
        log = {"train_loss": loss}
        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_idx):
        x = batch
        _, x_source, x_hat = self(x)
        loss = self.loss_function(x_hat, x_source)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        log = {"val_avg_loss": loss}
        return {"log": log, "val_loss": loss}

    def test_step(self, batch, batch_idx):
        x = batch
        _, x_source, x_hat = self(x)
        loss = self.loss_function(x_hat, x_source)
        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        log = {"test_avg_loss": loss}
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
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "bae_checkpoint", type=str, help="Path to LinearBeatAutoencoder checkpoint"
    )
    parser.add_argument(
        "--seq_len",
        default=10,
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
        "--dropout", default=0.1, type=float, help="Autoencoder dropout"
    )
    parser.add_argument("--momentum", default=0.5, type=float, help="SGD momentum")

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
        dropout=args.dropout,
        momentum=args.momentum,
        sequence_length=args.seq_len,
        data_config=cinc2020seq.data_config(),
    )

    # Custom logger output directory
    logger = pl.loggers.TensorBoardLogger(
        save_dir=os.getcwd(), name="log_sequence_autoencoder"
    )

    # log the learning rate
    from pytorch_lightning.callbacks.lr_logger import LearningRateLogger

    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, callbacks=[LearningRateLogger()]
    )
    trainer.logger.log_hyperparams(args)
    trainer.fit(model, cinc2020seq)
    trainer.test(model)
