import torch, os, argparse

import gpytorch
import gpytorch.settings as settings

import uci_data_loader.data as data
import uci_data_loader.uci as uci

from time import time


def train(
    model, likelihood, train_x, train_y,
    eta=0.1, maxiter=20, save_loc=None,
):
    """
    Simple routine to train & save model.
    """
    if not os.path.isdir(save_loc):
        print("creating folder \'{}\'".format(save_loc))
        os.makedirs(save_loc)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=eta)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(maxiter):
        t1 = time()
        optimizer.zero_grad()
        settings.record_residual.lst_residual_norm = []

        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()

        optimizer.step()
        iter_time = time() - t1

        with torch.no_grad():
            print(
                "iter {:3d}/{:3d},".format(i + 1, maxiter),
                "loss {:.6f},".format(loss.item()),
                "outputscale {:.4f},".format(model.covar_module.outputscale.item()),
                "avg lengthscale {:.4f},".format(model.covar_module.base_kernel.lengthscale.mean().item()),
                "noise {:.4f},".format(model.likelihood.noise.item()),
                "CG/Altproj iterations {:4d}".format(len(settings.record_residual.lst_residual_norm)),
            )

            torch.save(
                {
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'lengthscale': model.covar_module.base_kernel.lengthscale,
                    'noise': model.likelihood.noise.item(),
                    'lst_residual_norm': settings.record_residual.lst_residual_norm,
                    'iteration_time': iter_time,
                }, "{}/epoch_{}.tar".format(save_loc, i)
            )

    model.eval()
    likelihood.eval()

    return model

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, 
                        choices=["AirQuality", "BikeSharing", "GasSensors", "HouseholdPower",
                                 "KEGGUndir", "Parkinsons", "Protein", "RoadNetwork",
                                 "SGEMMGPU", "Song"], default="Parkinsons")
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kernel", type=str, choices=['RBF-ARD', 'RBF', 'Matern32', 'Matern52'], default='RBF-ARD')
    parser.add_argument("--preconditioner", type=str, choices=['pivchol', 'rpchol', 'nyssvd'], default='pivchol')
    parser.add_argument("--noise_constraint", type=float, default=1e-4)

    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--precond_size", type=int, default=15)
    parser.add_argument("--max_cg_iterations", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=0.1)
    parser.add_argument("--save_loc", type=str, default='./tmp')
    parser.add_argument("--batch", type=int, default=1000)

    return parser.parse_args()

def train_data(data_name, seed, device):
    func = getattr(uci, data_name)
    ds = data.UCI_Dataset(func, seed=seed, device=device)
    ds.preprocess()
    return ds.train_x, ds.train_y

def solve_system(
    model, likelihood, train_x, train_y, 
    save_loc=None, rand_rhs=False, trials=10
    ):
    """
    Simple routine to solve the linear system Kv=b
    """
    n, d = train_x.shape
    device = train_x.device
    
    if rand_rhs:
        train_y = torch.randn(n, device=device)
        train_y /= train_y.norm()
    
    
    mvn = likelihood(model(train_x))
    K_hat = mvn.lazy_covariance_matrix

    res = torch.zeros(trials, gpytorch.settings.max_cg_iterations.value())
    times = torch.zeros(trials)
    
    for i in range(trials):
        gpytorch.settings.record_residual.lst_residual_norm = []
        t1 = time()
        sol = K_hat.inv_matmul(b)
        times[i] = time() - t1
        res[i, 0:len(gpytorch.settings.record_residual.lst_residual_norm)] = torch.tensor(gpytorch.settings.record_residual_lst_residual_norm)

    avgs = res.sum(0) / (res != 0).sum(0)
    avgs = avgs[:torch.where(torch.isnan(avgs))[0][0]]
    times = times.mean()

    torch.save(
                {
                    'average_error': avgs,
                    'run_times': times
                }, "{}/numtrials_{}.tar".format(save_loc, trials)
            )


