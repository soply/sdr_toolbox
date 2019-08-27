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
    N = 100
    n_levelsets = 5
    sigma_eps = 1e-2
    problem_id = 'linreg'
    f, basis = get_problem('non_monoton_SIM', D)
    X, Y = sample_data_uniform_ball(N, D, f, sigma_eps)
    from estimators.iht import iht
    vecs = iht(X, Y, n_levelsets = n_levelsets, d = 1)
    import pdb; pdb.set_trace()
