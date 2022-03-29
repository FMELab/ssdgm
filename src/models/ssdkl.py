from src.models.modules.dkl import DeepKernelLearning
from src.models.modules.dense import Fcn

import pytorch_lightning as pl
import torch
import gpytorch

from torchmetrics import MetricCollection, MetricTracker, MeanSquaredError
from typing import Any, List



class SemiSupervisedDeepKernelLearning(pl.LightningModule):
    def __init__(
        self,
        enc_in_features,
        enc_hidden_features,
        enc_out_features,
        variance_loss_multiplier,
        leaky_relu_slope,
        dropout_proba,
        lr=0.01,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.encoder = Fcn(
            enc_in_features,
            enc_hidden_features,
            enc_out_features,
            leaky_relu_slope,
            dropout_proba,
        )

        metrics = MetricCollection(
            {
                "MSE": MeanSquaredError(),
                "RMSE": MeanSquaredError(squared=False),
            }
        )

        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')


    def forward(self, x):
        with gpytorch.settings.cholesky_jitter(1e-1):
            output_distribution = self.dkl_model(x) 

        return output_distribution



    def training_step(self, batch, batch_idx):
        x_labeled, y = batch["labeled"]
        y = y.flatten()

        x_unlabeled, _ = batch["unlabeled"]

        if self.current_epoch == 0 and batch_idx == 0:
            self.dkl_model = DeepKernelLearning(x_labeled, y, self.likelihood, self.encoder)
            #if torch.cuda.is_available():
            #    self.dkl_model = self.dkl_model.cuda()
        


        # the output of `dkl_model` is a distribution
        output_distribution_labeled = self.forward(x_labeled)

        log_marginal_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.dkl_model)
        with gpytorch.settings.cholesky_jitter(1e-1):
            likelihood_loss = -log_marginal_likelihood(output_distribution_labeled, y)

        self.dkl_model.eval()
        with gpytorch.settings.fast_pred_var(False):
            output_distribution_unlabeled = self.forward(x_unlabeled)
        self.dkl_model.train()

        variance_loss = torch.mean(output_distribution_unlabeled.variance)

        loss = likelihood_loss + self.hparams.variance_loss_multiplier * variance_loss


        self.log("train/likelihood_loss", likelihood_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/variance_loss", variance_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)


        return loss



    def on_validation_epoch_start(self):
        self.dkl_model.eval()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.flatten()

        output_distribution = self.forward(x)
        y_hat = output_distribution.mean

        self.valid_metrics(y_hat, y)

    def validation_epoch_end(self, outputs: List[Any]):
        self.log_dict(self.valid_metrics)

        self.dkl_model.train()

    def on_test_epoch_start(self) -> None:
        self.dkl_model.eval()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.flatten()

        output_distribution = self.forward(x)
        y_hat = output_distribution.mean

        self.test_metrics(y_hat, y)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self.log_dict(self.test_metrics)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr = self.hparams.lr,
        )

        return optimizer