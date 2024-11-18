import gpytorch, torch
from gpytorch.lazy import AddedDiagLazyTensor, RootLazyTensor, NonLazyTensor, ConstantDiagLazyTensor

class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, kernel="RBF-ARD"):

        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if kernel == "RBF-ARD":
            print("Initialising with RBF-ARD Kernel")
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.keops.RBFKernel(ard_num_dims=train_x.size(-1))
            )
        elif kernel == "RBF":
            print("Initialising with RBF Kernel")
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.keops.RBFKernel()
            )
        elif kernel == "Matern32":
            print("Initialising with Matern 1.5 Kernel")
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.keops.MaternKernel(nu=1.5, ard_num_dims=train_x.size(-1))
            )
        elif kernel == "Matern52":
            print("Initialising with Matern 2.5 Kernel")
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.keops.MaternKernel(nu=2.5, ard_num_dims=train_x.size(-1))
            )
        else:
            raise NotImplementedError("Please choose a valid kernel function")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class InducingGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, inducing_points, kernel="RBF-ARD"):
        super(InducingGPModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        if kernel == "RBF-ARD":
            print("Initialising with RBF-ARD Kernel")
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.keops.RBFKernel(ard_num_dims=train_x.size(-1))
            )
        elif kernel == "RBF":
            print("Initialising with RBF Kernel")
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.keops.RBFKernel()
            )
        elif kernel == "Matern32":
            print("Initialising with Matern 1.5 Kernel")
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.keops.MaternKernel(nu=1.5, ard_num_dims=train_x.size(-1))
            )
        elif kernel == "Matern52":
            print("Initialising with Matern 2.5 Kernel")
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.keops.MaternKernel(nu=2.5, ard_num_dims=train_x.size(-1))
            )
        else:
            raise NotImplementedError("Please choose a valid kernel function")
        
        self.inducing_points = torch.nn.Parameter(
            inducing_points
        )
        
    def forward(self, x):

        mean_x = self.mean_module(x)

        tall = self.covar_module(x, self.inducing_points)
        small = self.covar_module(self.inducing_points, self.inducing_points)
        L = gpytorch.utils.cholesky.psd_safe_cholesky(small.evaluate())
        P = torch.linalg.solve_triangular(L, tall.t().evaluate(), upper=False)

        return gpytorch.distributions.MultivariateNormal(mean_x, RootLazyTensor(P.T))

