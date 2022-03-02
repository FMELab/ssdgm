import pytorch_lightning as pl

import torch as T

from src.models.vae import VariationalAutoencoder
from torchmetrics import MetricCollection, MetricTracker, MeanSquaredError
from typing import Any, List

class ProbabilisticPredictor(pl.LightningModule):
    def __init__(
        self,
        vae_checkpoint_path,
        regressor,
        regressor_stddev,
        lr,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.feature_extractor = VariationalAutoencoder.load_from_checkpoint(vae_checkpoint_path)
        self.feature_extractor.freeze()

        self.regressor = regressor


        metrics = MetricCollection(
            {
                "MSE": MeanSquaredError(),
                "RMSE": MeanSquaredError(squared=False),
            }
        )

        self.train_metrics = MetricTracker(metrics.clone(prefix='train/'), maximize=[False, False])
        self.valid_metrics = MetricTracker(metrics.clone(prefix='val/'), maximize=[False, False])
        self.test_metrics = MetricTracker(metrics.clone(prefix='test/'), maximize=[False, False])

    def pdf_normal(self, x, mu, sigma):
        p = 1.0 / (sigma * T.sqrt(T.tensor(2.0 * T.pi))) * T.exp(-0.5 * T.square((x - mu) / sigma))

        return p
    
    def calc_loss(self, y_hat, y):
        p = self.pdf_normal(y, y_hat, self.hparams.regressor_stddev)

        return -T.mean(T.log(p))


    def forward(self, x):
        _, _, z = self.feature_extractor(x)
        y_hat = self.regressor(z)

        return y_hat

    def on_train_epoch_start(self) -> None:
        self.train_metrics.increment()

    def training_step(self, batch, batch_idx):
        x_labeled, y = batch["labeled"]


        y_hat = self.forward(x_labeled)

        loss = self.calc_loss(y_hat, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        self.train_metrics(y_hat, y)

        return loss

    def training_epoch_end(self, outputs):
        self.log_dict(self.train_metrics.compute())

    def on_validation_epoch_start(self):
        self.valid_metrics.increment()

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)

        self.valid_metrics(y_hat, y)

    def validation_epoch_end(self, outputs: List[Any]):
        self.log_dict(self.valid_metrics.compute())
        best_metrics, _ = self.valid_metrics.best_metric(return_step=True)
        best_metrics = {f"{key}_best": val for key, val in best_metrics.items()}
        self.log_dict(best_metrics)


    def on_test_epoch_start(self) -> None:
        self.test_metrics.increment()        

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)

        self.test_metrics(y_hat, y)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self.log_dict(self.test_metrics.compute())
        

    def configure_optimizers(self):
        optimizer = T.optim.Adam(
            params=self.regressor.parameters(),
            lr = self.hparams.lr,
        )

        return optimizer