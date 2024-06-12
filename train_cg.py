import torch, gpytorch, argparse, time

import gpytorch.settings as settings
import gpytorch.uci_data_loader.data as data
import gpytorch.uci_data_loader.uci as uci

from model import ExactGPModel
from utils import train

if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, 
                        choices=["AirQuality", "BikeSharing", "GasSensors", "HouseholdPower",
                                 "KEGGUndir", "Parkinsons", "Protein", "RoadNetwork",
                                 "SGEMMGPU", "Song"])
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kernel", type=str, choices=['matern15', 'matern', 'rbf'])
    parser.add_argument("--noise_constraint", type=float, default=1e-4)

    parser.add_argument("--eta", type=float)
    parser.add_argument("--maxiter", type=int)
    parser.add_argument("--precond_size", type=int)
    parser.add_argument("--max_cg_iterations", type=int)
    parser.add_argument("--tol", type=float)
    parser.add_argument("--save_loc", type=str, default="./checkpoints")