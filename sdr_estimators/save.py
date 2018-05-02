# coding: utf8
"""
SDR-method: SAVE introduced in [1]

[1] Dennis Cook, R. "SAVE: a method for dimension reduction and graphics in regression."
Communications in statistics-Theory and methods 29.9-10 (2000): 2109-2121.

"""

import numpy as np
import scipy
from scipy.linalg import sqrtm
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler

from ..partitioning.single_layer_levelset import \
    split_statistically_equivalent_blocks


def save(X, Y, **kwargs):
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
        'n_levelsets' : number of slices to use (int)
        'rescale' : Boolean whether standardization should be performed (True
                    for yes).
        'return_mat' : Boolean whether key SIR matrix should be returned (defaults
                    to False).

    Returns
    -----------
    proj_vecs : array-like, shape = [n_features, d]
        Orthonormal system spanning the sufficient dimension subspace, where
        d refers to the intrinsic dimension.
    }
    """
    # Extract arguments from dictionary
    d = kwargs['d']
    n_levelsets = kwargs['n_levelsets']
    rescale = kwargs['rescale']
    return_mat = kwargs.get('return_mat', False)


    D, N = X.shape
    # Standardize X
    emc = EmpiricalCovariance()
    emc = emc.fit(X.T) # Covariance of all samples
    mean_all = np.mean(X, axis = 0)
    cov_all = emc.covariance_
    scaler = StandardScaler()
    if rescale:
        Z = scaler.fit_transform(X.T).T
    labels, n_levelsets = split_statistically_equivalent_blocks(X, Y, n_levelsets)
    M = np.zeros((D, D)) # Key matrix in SAVE
    empirical_probabilities = np.zeros(n_levelsets)
    for i in range(n_levelsets):
        empirical_probabilities[i] = float(len(np.where(labels == i)[0]))/float(N)
        if rescale:
            emc = emc.fit(Z[:,labels == i].T) # Covariance of all samples
            cov_sub = emc.covariance_
            M += empirical_probabilities[i] * (np.eye(D) - cov_sub).dot((cov_all - cov_sub))
        else:
            emc = emc.fit(X[:,labels == i].T) # Covariance of all samples
            cov_sub = emc.covariance_
            M += empirical_probabilities[i] * (cov_all - cov_sub).dot((cov_all - cov_sub))
    U, S, V = np.linalg.svd(M)
    if rescale:
        # Apply inverse transformation
        vecs = sqrtm(scipy.linalg.inv(cov_all)).dot(U[:,:d])
        proj_vecs, dummy = np.linalg.qr(vecs)
    else:
        proj_vecs = U[:,:d]
    if return_mat:
        return proj_vecs, M
    else:
        return proj_vecs
