from typing import Any, List
import pytorch_lightning as pl

from torchmetrics import MetricCollection, MetricTracker, MeanSquaredError

import torch

from src.models.modules.dense import Decoder, Encoder, Fcn

class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        enc_in_features,
        enc_hidden_features,
        enc_out_features,
        dec_in_features,
        dec_hidden_features,
        dec_out_features,
        leaky_relu_slope,
        dropout_proba,
        lr: float = 0.001,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.encoder = Fcn(
            in_features=enc_in_features,
            hidden_features=enc_hidden_features,
            out_features=enc_out_features,
            leaky_relu_slope=leaky_relu_slope,
            dropout_proba=dropout_proba
        ) 
        
        self.decoder = Fcn(
            in_features=dec_in_features,
            hidden_features=dec_hidden_features,
            out_features=dec_out_features,
            leaky_relu_slope=leaky_relu_slope,
            dropout_proba=dropout_proba
        )

        self.criterion = torch.nn.MSELoss()

        metrics = MetricCollection(
            {
                "MSE": MeanSquaredError(),
                "RMSE": MeanSquaredError(squared=False),
            }
        )

        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')
        #self.train_mse = MeanSquaredError() 
        #self.val_mse = MeanSquaredError()
        #self.test_mse = MeanSquaredError()

        #self.val_mse_best = MinMetric()

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    #def _get_reconstruction_loss(self, batch, stage):

        #if stage == "fit":
            #x_labeled, _ = batch["labeled"]
            #x_unlabeled, _ = batch["unlabeled"]

            #x = torch.cat((x_labeled, x_unlabeled), dim=0)
        
        #if stage in ("val", "test"):
            #x, _ = batch

        #x_hat, _ = self.forward(x)
        #loss = self.criterion(x, x_hat)

        #return loss, x, x_hat


    #def on_train_epoch_start(self) -> None:
    #    self.train_metrics.increment()

    def training_step(self, batch: Any, batch_idx: int):
        x_labeled, _ = batch["labeled"]
        x_unlabeled, _ = batch["unlabeled"]

        x = torch.cat((x_labeled, x_unlabeled), dim=0)
        
        x_hat, _ = self.forward(x)
        loss = self.criterion(x, x_hat)

        # Log the loss
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        self.train_metrics(x_hat, x)

        return loss

    def training_epoch_end(self, outputs):
        # Log the metrics at the end of each training epoch
        self.log_dict(self.train_metrics)


    def validation_step(self, batch: Any, batch_idx: int):
        x, _ = batch
        x_hat, _ = self.forward(x)

        self.valid_metrics(x_hat, x)

    def validation_epoch_end(self, outputs: List[Any]):
        self.log_dict(self.valid_metrics)
        

    def test_step(self, batch: Any, batch_idx: int):
        x, _ = batch
        x_hat, _ = self.forward(x)

        self.test_metrics(x_hat, x)

    def test_epoch_end(self, outputs: List[Any]):
        self.log_dict(self.test_metrics)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
        )
        return optimizer