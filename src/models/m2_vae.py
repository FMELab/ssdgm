import torch 
import torch.nn.functional as F

import pytorch_lightning as pl
from src.models.modules.dense import Fcn, VaeEncoder

from src.models.vae import VariationalAutoencoder

from torchmetrics import MetricCollection, MetricTracker, MeanSquaredError, ExplainedVariance, MeanAbsoluteError, MeanAbsolutePercentageError, R2Score
from typing import Any, List

class M2Predictor(pl.LightningModule):
    def __init__(
        self,
        enc_in_features,
        enc_hidden_features,
        enc_out_features,
        dec_in_features,
        dec_hidden_features,
        dec_out_features,
        reg_in_features,
        reg_hidden_features,
        reg_out_features,
        regressor_loss_multiplier,  # the multiplier for the regressor loss
        leaky_relu_slope,
        dropout_proba,
        lr,
        ) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.encoder = VaeEncoder(
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


    def reparameterize(self, z_mean, z_log_var):
        """
        Samples latent code from variational distribution.

        Args:
            mean: mean of the variational distribution
            log_var: log variance of the the variation distribution

        Returns:
            Latent code
        """
        eps = torch.normal(mean=0, std=1, size=z_mean.size())
        eps = eps.type_as(z_mean)

        return z_mean + torch.exp(z_log_var * 0.5) * eps  # μ + exp[log(σ²) * 0.5] * ε = μ + exp[log(σ)] * ε = μ + σ * ε

    def negative_elbo(self, x, x_hat, z_mean, z_log_var):
        """
        Calculates the negative ELBO
        """
        
        # Reconstruction loss if output distribution is Gaussian
        rec_loss = 0.5 * (torch.log(2 * torch.pi * F.mse_loss(x_hat, x)) + 1)

        # Regularization term (negative KL divergence KL( q(z|x,y) || p(z) ), where p(z) ~ N(0, 1) )
        mu_squared = z_mean ** 2   # shape --> (bsz, latent_dim)
        var = torch.exp(z_log_var)   # shape --> (bsz, latent_dim)
        latent_dim = z_mean.size(1)  
        log_var = z_log_var   # shape --> (bsz, latent_dim)

        reg_loss = torch.mean(0.5 * torch.sum(mu_squared + var - 1 - log_var, dim=1))

        elbo = rec_loss + reg_loss
        return rec_loss, reg_loss, elbo


    def forward(self, x):
        y_hat = self.regressor(x)

        return y_hat


    def training_step(self, batch, batch_idx):
        x_labeled, y = batch["labeled"]
        
        x_unlabeled, _ = batch["unlabeled"]

        # calculate the labeled loss
        z_mean_labeled, z_logvar_labeled = self.encoder(torch.cat((x_labeled, y), dim=1))  # z_labeled.shape = (bsz, latent_dim)
        z_labeled = self.reparameterize(z_mean_labeled, z_mean_labeled)
        x_recon_labeled = self.decoder(torch.cat((z_labeled, y), dim=1))


        _, _, labeled_loss = self.negative_elbo(x_labeled, x_recon_labeled, z_mean_labeled, z_logvar_labeled)


        # calculate the regressor loss
        y_hat_labeled = self.forward(x_labeled)
        regressor_loss = F.mse_loss(y_hat_labeled, y)

        self.train_metrics(y_hat_labeled, y)
    

        # calculate the unlabeled loss
        y_hat_unlabeled = self.forward(x_unlabeled)
        z_mean_unlabeled, z_logvar_unlabeled = self.encoder(torch.cat((x_unlabeled, y_hat_unlabeled), dim=1))
        z_unlabeled = self.reparameterize(z_mean_unlabeled, z_logvar_unlabeled)
        x_recon_unlabeled = self.decoder(torch.cat((z_unlabeled, y_hat_unlabeled), dim=1))
        
        _, _, unlabeled_loss = self.negative_elbo(x_unlabeled, x_recon_unlabeled, z_mean_unlabeled, z_logvar_unlabeled)

        # add them all up
        loss = labeled_loss + unlabeled_loss + self.hparams.regressor_loss_multiplier * regressor_loss

        self.log("train/labeled_loss", labeled_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/unlabeled_loss", unlabeled_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/regressor_loss", regressor_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

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
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr = self.hparams.lr,
        )

        return optimizer
