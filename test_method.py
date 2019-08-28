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
    problem_id = 'linreg'
    f, basis = get_problem('non_monoton_SIM', D)
    X, Y = sample_data_uniform_ball(N, D, f, sigma_eps)
    from estimators.rclr import rclr, rclr_parameterfree
    vecs = rclr_parameterfree(X, Y, max_n_levelsets = 40, d = 1, whiten = True, return_proxy = True)
