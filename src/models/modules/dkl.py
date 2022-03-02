import gpytorch

class DeepKernelLearning(gpytorch.models.ExactGP):
    def __init__(
        self,
        x_train,
        y_train,
        likelihood,
        feature_extractor
    ) -> None:
        super().__init__(x_train, y_train, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
            num_dims=2, grid_size=100
        )

        self.feature_extractor = feature_extractor

    def forward(self, x):
        projected_x = self.feature_extractor(x)

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
