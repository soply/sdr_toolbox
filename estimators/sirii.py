# coding: utf8
"""
SDR-method: SIRII as introduced in the rejoinder of [1]

[1] Li, Ker-Chau. "Sliced inverse regression for dimension reduction."
Journal of the American Statistical Association 86.414 (1991): 316-327.

"""

import numpy as np
from sklearn.covariance import empirical_covariance

from utils.whitening import whiten_data
from utils.partitioning import split


def sirii(X, Y, **kwargs):
    """

    Parameters
    ----------
    X : array-like, shape = [N, D]
        Training data, where N is the number of samples and
        D is the number of features.
    Y : array-like, shape = [N]
        Response variable, where N is the number of samples


    Argument dictionary should contain:
    kwargs = {
        'd' : intrinsic dimension (int)
        'n_levelsets' : number of slices to use (int)
        'split_by' : 'dyadic' (dyadic decomposition) or 'stateq' (statistically equivalent blocks) (default: 'dyadic')
        'return_mat' : Boolean whether SIR matrix should be returned (default: False)

    Returns
    -----------
    proj_vecs : array-like, shape = [n_features, d]
        Orthonormal system spanning the sufficient dimension subspace, where
        d refers to the intrinsic dimension.

    M : SIR matrix, only if return_mat option is True
    }
    """
    # Extract arguments from dictionary
    d = kwargs['d']
    n_levelsets = kwargs['n_levelsets']
    split_by = kwargs.get('split_by', 'dyadic')
    return_mat = kwargs.get('return_mat', False)
    N, D = X.shape
    # Standardize X
    Z, cov_all_sqrtinv = whiten_data(X)
    # Create partition
    labels = split(Y, n_levelsets, split_by)
    M1 = np.zeros((D, D)) # Container for E[Cov(X|Y)]
    M2 = np.zeros((D, D)) # Container for key matrix in SIRII
    empirical_probabilities = np.zeros(n_levelsets)
    for i in range(n_levelsets):
        empirical_probabilities[i] = float(len(np.where(labels == i)[0]))/float(N)
        if empirical_probabilities[i] == 0:
            continue
        M1 += empirical_probabilities[i] * empirical_covariance(X[labels == i, :])
    # Compute SIRII matrix
    for i in range(n_levelsets):
        if empirical_probabilities[i] == 0:
            continue
        slice_mean = np.mean(Z[labels == i, :], axis = 0)
        M2 += empirical_probabilities[i] * (np.outer(slice_mean, slice_mean) - M1).dot(np.outer(slice_mean, slice_mean) - M1)
    U, S, V = np.linalg.svd(M2)
    # Apply inverse transformation
    vecs = cov_all_sqrtinv.dot(U[:,:d])
    proj_vecs, dummy = np.linalg.qr(vecs)
    if return_mat:
        return proj_vecs, M2
    else:
        return proj_vecs
