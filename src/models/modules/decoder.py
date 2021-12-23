import torch.nn as nn

class Decoder(nn.Module):
    def __init__(
        self,
        latent_size: int,
        lin1_size: int,
        lin2_size: int,
        lin3_size: int,
        output_size: int,
        ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),

            #nn.Linear(hparams["latent_size"], hparams["lin4_size"]),
            #nn.BatchNorm1d(hparams["lin4_size"]),
            #nn.ReLU(),
            #nn.Linear(hparams["lin4_size"], hparams["lin5_size"]),
            #nn.BatchNorm1d(hparams["lin5_size"]),
            #nn.ReLU(),
            #nn.Linear(hparams["lin5_size"], hparams["lin6_size"]),
            #nn.BatchNorm1d(hparams["lin6_size"]),
            #nn.ReLU(),
            #nn.Linear(hparams["lin6_size"], hparams["output_size"]),
        )


    def forward(self, x):
        return self.net(x)

