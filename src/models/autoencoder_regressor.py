from typing import Any, List
import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from torch import nn

from torchmetrics import MetricCollection, MetricTracker, MeanSquaredError, ExplainedVariance, MeanAbsoluteError, MeanAbsolutePercentageError, R2Score

from src.models.autoencoder import Autoencoder
from src.models.modules.dense import Fcn


class AutoencoderRegressor(pl.LightningModule):
    def __init__(
        self,
        #feature_extractor: Autoencoder,
        checkpoint_path,  
        reg_in_features,
        reg_hidden_features,
        reg_out_features,
        leaky_relu_slope,
        dropout_proba,
        lr: float = 0.001,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.feature_extractor = Autoencoder.load_from_checkpoint(checkpoint_path)
        self.feature_extractor.freeze()
        
        self.regressor = Fcn(
            in_features=reg_in_features,
            hidden_features=reg_hidden_features,
            out_features=reg_out_features,
            leaky_relu_slope=leaky_relu_slope,
            dropout_proba=dropout_proba
        )


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

    def forward(self, x: torch.Tensor):
        _, embedding = self.feature_extractor(x)
        y_hat = self.regressor(embedding)

        return y_hat

    #def _get_loss(self, batch, stage):
        #if stage == "fit":
            #x, y = batch["labeled"]
        
        #if stage in ("val", "test"):
            #x, y = batch
        
        #y = y.view(-1, 1)
        #y_hat = self.forward(x)
        #loss = self.criterion(y, y_hat)

        #return loss, y, y_hat


    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch["labeled"]

        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        self.train_metrics(y_hat, y)

        return loss

    def training_epoch_end(self, outputs):
        self.log_dict(self.train_metrics)


    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch

        y_hat = self.forward(x)

        self.valid_metrics(y_hat, y)

    def validation_epoch_end(self, outputs: List[Any]):
        self.log_dict(self.valid_metrics)

    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch

        y_hat = self.forward(x)

        self.test_metrics(y_hat, y)

    def test_step_end(self, outputs: List[Any]):
        self.log_dict(self.test_metrics)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr = self.hparams.lr,
        )
        return optimizer