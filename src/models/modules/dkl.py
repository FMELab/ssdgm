import gpytorch
from gpytorch.utils.grid import choose_grid_size

class DeepKernelLearning(gpytorch.models.ExactGP):
    def __init__(
        self,
        x_train,
        y_train,
        likelihood,
        encoder
    ) -> None:
        super().__init__(x_train, y_train, likelihood)
        
        self.encoder = encoder
        enc_out_features = self.encoder.last.out_features


        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=enc_out_features)
        )

            

        #self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        projected_x = self.encoder(x)
        #projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"    
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
