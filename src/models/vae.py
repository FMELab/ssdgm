from typing import Any
import pytorch_lightning as pl

import torch as T
import torch.nn.functional as F
from torch.optim import Adam

from torchmetrics import MetricCollection, MeanSquaredError

class VariationalAutoencoder(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        lr,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.encoder = encoder
        self.decoder = decoder

        metrics = MetricCollection(
            [
                MeanSquaredError(),
                #MeanSquaredError(squared=False),
            ]
        )

        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def reparameterize(self, z_mean, z_log_var):
        """
        Samples latent code from variational distribution.

        Args:
            mean: mean of the variational distribution
            log_var: log variance of the the variation distribution

        Returns:
            Latent code
        """
        eps = T.normal(mean=0, std=1, size=z_mean.size())
        eps = eps.type_as(z_mean)

        return z_mean + T.exp(z_log_var * 0.5) * eps  # μ + exp[log(σ²) * 0.5] * ε = μ + exp[log(σ)] * ε = μ + σ * ε

    
    def negative_elbo(self, x, x_hat, z_mean, z_log_var):
        """
        Calculates the negative ELBO
        """
        
        # Reconstruction loss
        rec_loss = 0.5 * (T.log(2 * T.pi * F.mse_loss(x_hat, x)) + 1)

        # Regularization term (negative KL divergence)
        mu_squared = z_mean ** 2
        var = T.exp(z_log_var)
        latent_dim = z_mean.size(1)
        log_var = z_log_var

        reg_loss = T.mean(0.5 * (T.sum(mu_squared, dim=1) + T.sum(var, dim=1) - latent_dim - T.sum(log_var, dim=1)))

        return rec_loss + reg_loss


    def forward(self, x: T.Tensor):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)

        return z_mean, z_log_var, z

    def training_step(self, batch: Any, batch_idx: int):

        x_labeled, _ = batch["labeled"]
        x_unlabeled, _ = batch["unlabeled"]

        x = T.cat((x_labeled, x_unlabeled), dim=0)

        z_mean, z_log_var, z = self.forward(x)

        x_hat = self.decoder(z)

        # We minimize the negative ELBO
        loss = self.negative_elbo(x, x_hat, z_mean, z_log_var)

        return {"loss": loss, "x_hat": x_hat}
    
    def training_epoch_end(self, outputs):
        o = outputs
        print(o)



    def validation_step(self, batch: Any, batch_idx: int):
        x, _ = batch

        _, _, z = self.forward(x)
        x_hat = self.decoder(z)

        self.log

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def configure_optimizers(self):
        optimizer = Adam(
            params=self.parameters(),
            lr = self.hparams.lr
        )

        return optimizer


