#!/usr/bin/env python3
import optuna
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback

from datasets import BeatsZarrDataModule

from linear_beat_autoencoder import BeatAutoEncoder


class MetricsCallback(pl.Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def objective(trial):
    # trial.suggest_int()
    hidden_dim = trial.suggest_int("hidden_dim", 64, 2048)
    embedding_dim = trial.suggest_int("embedding_dim", 96, 768)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    momentum = trial.suggest_float("momentum", 0.1, 0.9)

    cinc2020beat = BeatsZarrDataModule(
        window_size=400,
        batch_size=1024,
        train_workers=64,
        val_workers=16,
        test_workers=16,
    )
    model = BeatAutoEncoder(
        pqrst_window_size=400,
        base_lr=1e-5,
        max_lr=1e-3,
        dropout=dropout,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        momentum=momentum,
        batch_size=1024,
        data_config=cinc2020beat.data_config(),
    )

    # log the learning rate
    from pytorch_lightning.callbacks.lr_logger import LearningRateLogger

    lr_logger = LearningRateLogger()
    metrics_callback = MetricsCallback()

    trainer = pl.Trainer(
        max_epochs=40,
        gpus=1,
        callbacks=[lr_logger, metrics_callback],
        early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="val_loss")
    )
    trainer.fit(model, cinc2020beat)
    trainer.test(model)

    return metrics_callback.metrics[-1]["val_loss"].item()


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="Symmetric_Linear_BeatAutoencoder",
        storage="sqlite:///optuna_symmetric_linear_bae.db",
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),  # optuna.pruners.NopPruner()
        load_if_exists=True,
    )
    study.optimize(objective, n_jobs=1, n_trials=100, timeout=None)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # shutil.rmtree(MODEL_DIR)
