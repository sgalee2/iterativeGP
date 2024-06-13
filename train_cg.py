import torch, gpytorch, argparse, time

import gpytorch.settings as settings
import uci_data_loader.data as data
import uci_data_loader.uci as uci

from model import ExactGPModel
from utils import train

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, 
                        choices=["AirQuality", "BikeSharing", "GasSensors", "HouseholdPower",
                                 "KEGGUndir", "Parkinsons", "Protein", "RoadNetwork",
                                 "SGEMMGPU", "Song"])
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kernel", type=str, choices=['RBF-ARD', 'RBF', 'Matern32', 'Matern52'])
    parser.add_argument("--preconditioner", type=str, choices=['pivchol', 'rpchol', 'nyssvd'])
    parser.add_argument("--noise_constraint", type=float, default=1e-4)

    parser.add_argument("--eta", type=float)
    parser.add_argument("--maxiter", type=int)
    parser.add_argument("--precond_size", type=int)
    parser.add_argument("--max_cg_iterations", type=int)
    parser.add_argument("--tol", type=float)
    parser.add_argument("--save_loc", type=str)

    return parser.parse_args()

def train_data(data_name, seed, device):
    func = getattr(uci, data_name)
    ds = data.UCI_Dataset(func, seed=seed, device=device)
    ds.preprocess()
    return ds.train_x, ds.train_y

if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = parse_args()

    train_x, train_y = train_data(args.dataset, args.seed, device)

    precon_func = getattr(settings, args.preconditioner)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPModel(train_x, train_y, likelihood, kernel = args.kernel).to(device)

    with settings.max_preconditioner_size(args.precond_size), \
         settings.cg_tolerance(args.tol), \
         settings.max_cg_iterations(args.max_cg_iterations), \
         precon_func():

            if args.save_loc is not None:
                train(model, likelihood, train_x, train_y,
                eta = args.eta, maxiter=args.maxiter, save_loc = save_loc)
            else:
                print("No save location specified.")
