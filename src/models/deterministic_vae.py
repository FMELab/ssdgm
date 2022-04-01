import pytorch_lightning as pl

import torch as T
from src.models.modules.dense import Fcn

from src.models.vae import VariationalAutoencoder

from torchmetrics import MetricCollection, MetricTracker, MeanSquaredError, ExplainedVariance, MeanAbsoluteError, MeanAbsolutePercentageError, R2Score

from typing import Any, List

class VAEDeterministicPredictor(pl.LightningModule):
    def __init__(
        self,
        vae_checkpoint_path,
        reg_in_features,
        reg_hidden_features,
        reg_out_features,
        leaky_relu_slope,
        dropout_proba,
        lr,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.feature_extractor = VariationalAutoencoder.load_from_checkpoint(vae_checkpoint_path)
        self.feature_extractor.freeze()

        self.regressor = Fcn(
            in_features=reg_in_features,
            hidden_features=reg_hidden_features,
            out_features=reg_out_features,
            leaky_relu_slope=leaky_relu_slope,
            dropout_proba=dropout_proba
        )
        self.lr = lr

        self.criterion = T.nn.MSELoss()
        
        
        metrics = MetricCollection(
            {
                "MSE": MeanSquaredError(),
                "RMSE": MeanSquaredError(squared=False),
                "ExplainedVariance": ExplainedVariance(),
                "MAE": MeanAbsoluteError(),
                "MAPE": MeanAbsolutePercentageError(),
                "R2": R2Score(),
            }
        )

        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

    def forward(self, x):
        z_mean, _, _ = self.feature_extractor(x)
        y_hat = self.regressor(z_mean)

        return y_hat


    def training_step(self, batch, batch_idx):
        x_labeled, y = batch["labeled"]

        y_hat = self.forward(x_labeled)

        loss = self.criterion(y_hat, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        self.train_metrics(y_hat, y)

        return loss

    def training_epoch_end(self, outputs):
        self.log_dict(self.train_metrics)



    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)

        self.valid_metrics(y_hat, y)

    def validation_epoch_end(self, outputs: List[Any]):
        self.log_dict(self.valid_metrics)


    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)

        self.test_metrics(y_hat, y)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self.log_dict(self.test_metrics)

    def configure_optimizers(self):
        optimizer = T.optim.Adam(
            params=self.regressor.parameters(),
            lr = self.lr,
        )

        return optimizer

