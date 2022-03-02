from typing import Any, List
import pytorch_lightning as pl

import torch as T
import torch.nn.functional as F
from torch.optim import Adam

from torchmetrics import MetricCollection, MetricTracker, MeanSquaredError

class VariationalAutoencoder(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        lr,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)#, ignore=["encoder", "decoder"])

        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        
        metrics = MetricCollection(
            {
                "MSE": MeanSquaredError(),
                "RMSE": MeanSquaredError(squared=False),
            }
        )

        self.train_metrics = MetricTracker(metrics.clone(prefix='train/'), maximize=[False, False])
        self.valid_metrics = MetricTracker(metrics.clone(prefix='val/'), maximize=[False, False])
        self.test_metrics = MetricTracker(metrics.clone(prefix='test/'), maximize=[False, False])

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
        mu_squared = z_mean ** 2   # shape --> (bsz, latent_dim)
        var = T.exp(z_log_var)   # shape --> (bsz, latent_dim)
        latent_dim = z_mean.size(1)  
        log_var = z_log_var   # shape --> (bsz, latent_dim)

        reg_loss = T.mean(0.5 * T.sum(mu_squared + var - 1 - log_var, dim=1))

        elbo = rec_loss + 0.05 * reg_loss
        return rec_loss, reg_loss, elbo


    def forward(self, x: T.Tensor):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)

        return z_mean, z_log_var, z

    def on_train_epoch_start(self) -> None:
        self.train_metrics.increment()
        
    def training_step(self, batch: Any, batch_idx: int):

        x_labeled, _ = batch["labeled"]
        x_unlabeled, _ = batch["unlabeled"]

        x = T.cat((x_labeled, x_unlabeled), dim=0)

        z_mean, z_log_var, z = self.forward(x)

        x_hat = self.decoder(z)

        # We minimize the negative ELBO
        rec_loss, reg_loss, elbo = self.negative_elbo(x, x_hat, z_mean, z_log_var)

        self.log("train/rec_loss", rec_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/reg_loss", reg_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/elbo", elbo, on_step=False, on_epoch=True, prog_bar=False)

        self.train_metrics(x_hat, x)

        return elbo
    
    def training_epoch_end(self, outputs):
        self.log_dict(self.train_metrics.compute())

    ####################### VAL #######################

    def on_validation_epoch_start(self):
        self.valid_metrics.increment()
        
    def validation_step(self, batch: Any, batch_idx: int):
        x, _ = batch

        _, _, z = self.forward(x)
        x_hat = self.decoder(z)
        
        self.valid_metrics(x_hat, x)

    def validation_epoch_end(self, outputs: List[Any]):
        self.log_dict(self.valid_metrics.compute())
        best_metrics, _ = self.valid_metrics.best_metric(return_step=True)
        best_metrics = {f"{key}_best": val for key, val in best_metrics.items()}
        self.log_dict(best_metrics)

    ####################### TEST ##########################

    def on_test_epoch_start(self) -> None:
        self.test_metrics.increment()        

    def test_step(self, batch: Any, batch_idx: int):
        x, _ = batch

        _, _, z = self.forward(x)
        x_hat = self.decoder(z)

        # log the test metrics
        self.test_metrics(x_hat, x)
    
    def test_epoch_end(self, outputs: List[Any]) -> None:
        self.log_dict(self.test_metrics.compute())
    ####################### OPT ##########################

    def configure_optimizers(self):
        optimizer = Adam(
            params=self.parameters(),
            lr = self.hparams.lr
        )

        return optimizer


