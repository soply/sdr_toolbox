# coding: utf8
"""
SDR-method: Principal Hessian Directions introduced in [1]

[1] Li, Ker-Chau. On principal Hessian directions for data visualization and dimension
reduction: Another application of Stein's lemma

"""

import numpy as np
import scipy
from scipy.linalg import sqrtm, eig
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from ..partitioning.single_layer_levelset import \
    split_statistically_equivalent_blocks


def phd(X, Y, **kwargs):
    """
    Parameters
    ----------
    X : array-like, shape = [n_features, n_samples]
        Training data, where n_samples is the number of samples and
        n_features is the number of features.
    Y : array-like, shape = [n_samples]
        Response variable, where n_samples is the number of samples


    Argument dictionary should contain:
    kwargs = {
        'd' : intrinsic dimension (int)
        'residuals' : If True, creates PHDs from the residuals of linear regression
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
    residuals = kwargs.get('residuals', False)
    return_mat = kwargs.get('return_mat', False)
    D, N = X.shape
    # Calculate covariance matrix and empirical covariance matrix
    emc = EmpiricalCovariance()
    emc = emc.fit(X.T) # Covariance of all samples
    cov_all = emc.covariance_
    weighted_cov = np.zeros(cov_all.shape)
    if residuals:
        linreg = LinearRegression()
        linreg = linreg.fit(X.T, Y)
        res = Y - linreg.predict(X.T)
        Y = res
    Ymean = np.mean(Y)
    mean_all = np.mean(X, axis = 1)
    for i in range(N):
        weighted_cov += (Y[i] - Ymean) * np.outer(X[:,i] - mean_all, X[:,i] - mean_all)
    weighted_cov = weighted_cov/float(N)
    vals, vecs = eig(weighted_cov, cov_all)
    order = np.argsort(np.abs(vals))[::-1]
    proj_vecs = vecs[:,order[:d]]
    if return_mat:
        return proj_vecs, weighted_cov
    else:
        return proj_vecs
