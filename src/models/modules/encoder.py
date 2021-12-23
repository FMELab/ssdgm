import torch.nn as nn

class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        lin1_size: int,
        lin2_size: int,
        lin3_size: int,
        latent_size: int,
        ) -> None:
        super().__init__()

        self.latent_size = latent_size

        self.net = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, self.latent_size),

            #nn.Linear(hparams["input_size"], hparams["lin1_size"]),
            #nn.BatchNorm1d(hparams["lin1_size"]),
            #nn.ReLU(),
            #nn.Linear(hparams["lin1_size"], hparams["lin2_size"]),
            #nn.BatchNorm1d(hparams["lin2_size"]),
            #nn.ReLU(),
            #nn.Linear(hparams["lin2_size"], hparams["lin3_size"]),
            #nn.BatchNorm1d(hparams["lin3_size"]),
            #nn.ReLU(),
            #nn.Linear(hparams["lin3_size"], hparams["latent_size"]),
        )



    def forward(self, x):
        return self.net(x)
