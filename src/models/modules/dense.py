from typing import List
import torch
import torch.nn as nn
from torch.nn import Parameter


class FCN(nn.Module):
    def __init__(
        self,
        input_size,
        lin1_size,
        lin2_size,
        lin3_size,
        lin4_size,
        output_size,
        ) -> None:
        super().__init__()

        
        def block(in_features, out_features, normalize=True, activate=True):
            layers = [nn.Linear(in_features, out_features)]

            if normalize:
                layers.append(nn.BatchNorm1d(out_features))

            if activate:
                layers.append(nn.LeakyReLU(0.2))

            return layers

        self.first = nn.Sequential(*block(input_size, lin1_size))

        self.hidden = nn.Sequential(
            *block(lin1_size, lin2_size),
            *block(lin2_size, lin3_size),
            *block(lin3_size, lin4_size),
        )

        self.last = nn.Sequential(nn.Linear(lin4_size, output_size))


    def forward(self, x):
        x = self.first(x)
        x = self.hidden(x)
        x = self.last(x)
        return x 

class FcBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        use_batch_norm: bool = False,
        use_weight_norm: bool = True,
        use_activation: bool = True,
        leaky_relu_slope: float = 0.2,
        use_dropout: bool = True,
        dropout_proba: float = 0.5,
    ) -> None:
        super().__init__()

        lin_layer = nn.Linear(in_features, out_features)

        if use_weight_norm:
            layers = [WeightNorm(lin_layer, ["weight"])]
        else:
            layers = [lin_layer]

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
        
        if use_activation:
            layers.append(nn.LeakyReLU(negative_slope=leaky_relu_slope))

        if use_dropout:
            layers.append(nn.Dropout(p=dropout_proba))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Fcn(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        #batch_norm: bool = False,
        #weight_norm: bool = False,
        #activate: bool = True,
        #relu_slope: float = 0.01
    ) -> None:
        super().__init__()

        #self.sizes = [in_features, *hidden_features, out_features]
        self.first = FcBlock(
            in_features, hidden_features[0],
            )

        self.hidden = nn.Sequential(*[FcBlock(in_f, out_f) for (in_f, out_f) in zip(hidden_features, hidden_features[1:])])

        self.last = FcBlock(
            hidden_features[-1], out_features,
            use_activation=False,
            use_dropout=False)

    def forward(self, x):
        x = self.first(x)
        x = self.hidden(x)

        # save the output after the last hidden layer to facilitate loss calculation of SRGAN
        self.features = x

        x = self.last(x)

        return x


class VaeEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: List[int],
        latent_features,
    ) -> None:
        super().__init__()

        self.first = FcBlock(in_features, hidden_features[0])

        self.hidden = nn.Sequential(*[FcBlock(in_f, out_f) for (in_f, out_f) in zip(hidden_features, hidden_features[1:])])
        
        self.mean = FcBlock(
            hidden_features[-1], latent_features,
            use_activation=False,
            use_dropout=False,
        )

        self.log_var = FcBlock(
            hidden_features[-1], latent_features,
            use_activation=False,
            use_dropout=False,
        )

    def forward(self, x):
        x = self.first(x)
        x = self.hidden(x)

        mean = self.mean(x)
        log_var = self.log_var(x)

        return mean, log_var 


class DenseNetFcn(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features: List[int],
        out_features,
    ) -> None:
        super().__init__()


        self.layer1 = FcBlock(in_features, hidden_features[0])

        self.layer2 = FcBlock(in_features + hidden_features[0], hidden_features[1])

        self.layer3 = FcBlock(in_features + hidden_features[0] + hidden_features[1], hidden_features[2])

        self.layer4 = FcBlock(hidden_features[-1], out_features)

    def forward(self, x):
        x1 = self.layer1(x)

        x = torch.cat((x, x1), dim=1)
        x2 = self.layer2(x)

        x = torch.cat((x, x2), dim=1)
        x3 = self.layer3(x)

        out = self.layer4(x3)

        return out

class WeightNorm(nn.Module):
    append_g = '_g'
    append_v = '_v'

    def __init__(self, module, weights):
        super(WeightNorm, self).__init__()
        self.module = module
        self.weights = weights
        self._reset()

    def _reset(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)

            # construct g,v such that w = g/||v|| * v
            g = torch.norm(w)
            v = w/g.expand_as(w)
            g = Parameter(g.data)
            v = Parameter(v.data)
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v

            # remove w from parameter list
            del self.module._parameters[name_w]

            # add g and v as new parameters
            self.module.register_parameter(name_g, g)
            self.module.register_parameter(name_v, v)

    def _setweights(self):
        for name_w in self.weights:
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v
            g = getattr(self.module, name_g)
            v = getattr(self.module, name_v)
            w = v*(g/torch.norm(v)).expand_as(v)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


if __name__ == "__main__":
    Fcn(18, [64, 64], 1)
    VaeEncoder(18, [64, 64], 4)
    print("Success.")