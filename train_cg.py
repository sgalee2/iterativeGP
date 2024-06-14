import torch, gpytorch

import gpytorch.settings as settings

from model import ExactGPModel
from utils import train, parse_args, train_data

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
         settings.max_cholesky_size(0), \
         precon_func():

            if args.save_loc is not None:
                train(model, likelihood, train_x, train_y,
                eta = args.eta, maxiter=args.maxiter, save_loc = args.save_loc)
            else:
                print("No save location specified.")
