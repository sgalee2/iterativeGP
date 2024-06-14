import torch, gpytorch

import gpytorch.settings as settings

from model import ExactGPModel
from utils import train, parse_args, train_data

if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = parse_args()

    train_x, train_y = train_data(args.dataset, args.seed, device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood, kernel=args.kernel)

    with settings.max_cholesky_size(0), \
         settings.max_preconditioner_size(0), \
         settings.skip_logdet_forward(), \
         settings.use_alternating_projection(), \
         settings.altproj_batch_size(args.batch), \
         settings.cg_tolerance(args.tol), \
         settings.max_cg_iterations(args.max_cg_iterations):
        
        if args.save_loc is not None:
            train(model, likelihood, train_x, train_y, 
                  eta = args.eta, maxiter = args.maxiter, save_loc = args.save_loc)
        else:
            print("No save location specified.")