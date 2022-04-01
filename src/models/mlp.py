import pytorch_lightning as pl
import torch as T
import torch.nn.functional as F

from src.models.modules.dense import Fcn
from torchmetrics import MetricCollection, MetricTracker, MeanSquaredError, ExplainedVariance, MeanAbsoluteError, MeanAbsolutePercentageError, R2Score

class MultiLayerPerceptron(pl.LightningModule):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        leaky_relu_slope,
        dropout_proba,
        lr,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.net = Fcn(
            in_features,
            hidden_features,
            out_features,
            leaky_relu_slope,
            dropout_proba,
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

        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x):
        y_hat = self.net(x)
        return y_hat
    



    def training_step(self, batch, batch_idx):
        x_labeled, y = batch["labeled"]

        y_hat = self.forward(x_labeled)

        loss = F.mse_loss(y_hat, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        self.train_metrics(y_hat, y)

        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics)


    def validation_step(self, batch, batch_idx):
        x_labeled, y = batch

        y_hat = self.forward(x_labeled)        
        self.valid_metrics(y_hat, y)

    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics)
    

    def test_step(self, batch, batch_idx):
        x_labeled, y = batch

        y_hat = self.forward(x_labeled)        
        self.test_metrics(y_hat, y)

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics)

    def configure_optimizers(self):
        optimizer = T.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
        )

        return optimizer