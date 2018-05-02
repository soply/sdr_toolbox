# coding: utf8
"""
SDR-method: SIR introduced in [1]

[1] Li, Ker-Chau. "Sliced inverse regression for dimension reduction."
Journal of the American Statistical Association 86.414 (1991): 316-327.

"""

import numpy as np
import scipy
from scipy.linalg import sqrtm
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler

from ..partitioning.single_layer_levelset import \
    split_statistically_equivalent_blocks


def sir(X, Y, **kwargs):
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
    mean_all = np.mean(X, axis = 1)
    scaler = StandardScaler()
    if rescale:
        emc = EmpiricalCovariance()
        emc = emc.fit(X.T) # Covariance of all samples
        cov_all = emc.covariance_
        Z = scaler.fit_transform(X.T).T
    labels, n_levelsets = split_statistically_equivalent_blocks(X, Y, n_levelsets)
    M = np.zeros((D, D)) # Key matrix in SIR
    empirical_probabilities = np.zeros(n_levelsets)
    for i in range(n_levelsets):
        empirical_probabilities[i] = float(len(np.where(labels == i)[0]))/float(N)
        if rescale:
            slice_mean = np.mean(Z[:,labels == i], axis = 1)
            M += empirical_probabilities[i] * np.outer(slice_mean, slice_mean)
        else:
            slice_mean = np.mean(X[:,labels == i], axis = 1)
            M += empirical_probabilities[i] * np.outer(slice_mean, slice_mean)
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
