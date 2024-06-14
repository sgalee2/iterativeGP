# Recommendations for Iterative Gaussian Process models
This repository contains experiments demonstrating recommended settings and algorithms for efficient Gaussian Process models using iterative methods.

**It is highly recommended that the scripts are run with CUDA available. Otherwise it will run, but very slowly.**

## Dependency
The following packages are required:
  - python == 3.8
  - pytorch == 2.3.0+cu121
  - pykeops
  - pandas

The GPyTorch branch which contains that implements the necessary methods can be installed with
```
pip install git+https://www.github.com/sgalee2/gpytorch.git@altproj
```
and is a branch of this [repository][link].

[link]: https://github.com/cornellius-gp/gpytorch/tree/altproj

## Example
A simple example of GP training with Conjugate Gradients can be done by running
```
python train_cg.py
```
Checkpoints are automatically saved into `.\tmp`.
