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
    parser.add_argument("--kernel", type=str, choices=['matern15', 'matern', 'rbf'])
    parser.add_argument("--preconditioner", type=str, choices=['piv_chol', 'rp_chol', 'r_nys'])
    parser.add_argument("--noise_constraint", type=float, default=1e-4)

    parser.add_argument("--eta", type=float)
    parser.add_argument("--maxiter", type=int)
    parser.add_argument("--precond_size", type=int)
    parser.add_argument("--max_cg_iterations", type=int)
    parser.add_argument("--tol", type=float)
    parser.add_argument("--save_loc", type=str, default="./checkpoints")

    return parser.parse_args()

    def run():
        pass

if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = parse_args()
    print(args)

    if args.preconditioner == "piv_chol":
        with gpytorch.settings.use_pivchol_preconditioner():
            run()
    elif args.preconditioner == "rp_chol":
        with gpytorch.settings.use_rpchol_preconditioner():
            run()
    elif args.preconditioner == "r_nys":
        with gpytorch.settings.use_nyssvd_preconditioner():
            run()
    else:
        assert 0