# coding: utf8
"""
SDR-method: Iterative Hessian transformations as introduced in the rejoinder of [1]

[1] Cook, R. Dennis, and Bing Li. "Dimension reduction for conditional mean in
    regression." The Annals of Statistics 30.2 (2002): 455-474.

"""

import numpy as np

from sklearn.covariance import empirical_covariance
from sklearn.linear_model import LinearRegression

from utils.whitening import whiten_data
from utils.partitioning import split


def iht(X, Y, **kwargs):
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
    n_iter = kwargs.get('n_iter', 20)
    use_residuals = kwargs.get('use_residuals', False)
    return_mat = kwargs.get('return_mat', False)
    N, D = X.shape
    # Standardize X
    Z, cov_all_sqrtinv = whiten_data(X)
    # Compute OLS vector
    ols_vector = np.mean((Z.T * (Y - np.mean(Y))).T, axis = 0)
    # Compute Hessian matrix
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
    # Apply iterative transformations
    M = np.zeros((D, D)) # critical mat for IHT
    iterative_matrix = np.eye(D)
    for i in range(d):
        M += np.outer(iterative_matrix.dot(ols_vector), iterative_matrix.dot(ols_vector))
        iterative_matrix = iterative_matrix.dot(iterative_matrix)
    # Compute eigendecomposition
    U, S, V = np.linalg.svd(M)
    # Apply inverse transformation
    vecs = cov_all_sqrtinv.dot(U[:,:d])
    proj_vecs, dummy = np.linalg.qr(vecs)
    if return_mat:
        return proj_vecs, M
    else:
        return proj_vecs
