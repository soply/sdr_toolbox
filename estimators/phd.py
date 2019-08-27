# coding: utf8
"""
SDR-method: Principal Hessian Directions introduced in [1]

[1] Li, Ker-Chau. On principal Hessian directions for data visualization and dimension
reduction: Another application of Stein's lemma

"""

import numpy as np
from scipy.linalg import eig
from sklearn.covariance import empirical_covariance
from sklearn.linear_model import LinearRegression


def phd(X, Y, **kwargs):
    """
    Parameters
    ----------
    X : array-like, shape = [N, D]
        Training data, where N is the number of samples and
        D is the number of features.
    Y : array-like, shape = [N]
        Response variable, where n_samples is the number of samples


    Argument dictionary should contain:
    kwargs = {
        'd' : intrinsic dimension (int)
        'use_residuals' : If True, creates PHDs from the use_residuals of linear regression
                    (defaults to False)
        'return_mat' : Boolean whether key PHD matrix should be returned (defaults
                    to False).
    }

    Returns
    -----------
    proj_vecs : array-like, shape = [n_features, d]
        Orthonormal system spanning the sufficient dimension subspace, where
        d refers to the intrinsic dimension.
    }
    """
    # Extract arguments from dictionary
    d = kwargs['d']
    use_residuals = kwargs.get('use_residuals', False)
    return_mat = kwargs.get('return_mat', False)
    N, D = X.shape
    # Calculate covariance matrix and empirical covariance matrix
    mean_all = np.mean(X, axis = 0)
    cov_all = empirical_covariance(X)
    weighted_cov = np.zeros(cov_all.shape)
    if use_residuals:
        linreg = LinearRegression()
        linreg = linreg.fit(X, Y)
        res = Y - linreg.predict(X)
        Y = res
    else:
        Ymean = np.mean(Y)
        Y = Y - Ymean
    for i in range(N):
        weighted_cov += Y[i] * np.outer(X[i,:] - mean_all, X[i,:] - mean_all)
    weighted_cov = weighted_cov/float(N)
    vals, vecs = eig(weighted_cov, cov_all)
    order = np.argsort(np.abs(vals))[::-1]
    proj_vecs = vecs[:,order[:d]]
    if return_mat:
        return proj_vecs, weighted_cov
    else:
        return proj_vecs
