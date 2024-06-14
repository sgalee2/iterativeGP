import torch, gpytorch

import gpytorch.settings as settings

from model import ExactGPModel
from utils import train, parse_args, train_data

if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = parse_args()

    train_x, train_y = train_data(args.dataset, args.seed, device)