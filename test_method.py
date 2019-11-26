# coding: utf8
"""
Convenience file for testing implementation of a sufficient dimension reduction
method.
"""

import numpy as np

from problem_factory.problems import get_problem
from problem_factory.sampling import sample_data_uniform_ball


if __name__ == '__main__':
    D = 10
    N = 5000
    n_levelsets = 50
    sigma_eps = 1e-2
    f, basis = get_problem('exp', D)
    X, Y = sample_data_uniform_ball(N, D, f, sigma_eps)
    from estimators.nndr import nndr
    # sdrknn = SDRKnn('RCLR', n_neighbors = 1, n_components = basis.shape[1], n_levelsets = 10)
    # sdrknn = sdrknn.fit(X, Y)
    # Y = sdrknn.predict(X)
    vecs = nndr(X, Y, d  = basis.shape[1], L = 1, W = 2 * D, n_epochs = 3000, rescale = False, reg_l2 = 0.01, return_mat = False, verbose = True)
    import pdb; pdb.set_trace()
