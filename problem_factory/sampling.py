# coding: utf8
""" Methods for sampling data synthethic data sets. """


import numpy as np


def sample_data_uniform_ball(N, D, f, sigma_eps):
    # Sample detachment coefficients from ||k||_2 < noise_level * sqrt(D - 1)
    rand_sphere = np.random.normal(size = (D, N))
    rand_sphere = rand_sphere/np.linalg.norm(rand_sphere, axis = 0)
    radii = np.random.uniform(0, 1, size = N)
    radii = np.power(radii, 1.0/(D))
    X = rand_sphere * radii
    Y = f(X)
    scale = np.std(Y)
    Y = Y + np.random.normal(scale = sigma_eps * scale, size = N)
    X = X.T
    return X, Y


def sample_data_normal(N, D, f, sigma_eps):
    # Sample detachment coefficients from ||k||_2 < noise_level * sqrt(D - 1)
    X = np.random.multivariate_normal(np.zeros(D), np.eye(D), size = (N)).T
    Y = f(X)
    scale = np.std(Y)
    Y = Y + np.random.normal(scale = sigma_eps * scale, size = N)
    X = X.T
    return X, Y


def sample_data_uniform_box(N, D, f, sigma_eps):
    # Sample detachment coefficients from ||k||_2 < noise_level * sqrt(D - 1)
    X = np.random.uniform(low = -0.5, high = 0.5, size = (D, N))
    Y = f(X)
    scale = np.std(Y)
    Y = Y + np.random.normal(scale = sigma_eps * scale, size = N)
    X = X.T
    return X, Y
