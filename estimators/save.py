# coding: utf8
"""
SDR-method: SAVE introduced in [1]

[1] Dennis Cook, R. "SAVE: a method for dimension reduction and graphics in regression."
Communications in statistics-Theory and methods 29.9-10 (2000): 2109-2121.

"""

import numpy as np
from sklearn.covariance import empirical_covariance

from utils.whitening import whiten_data
from utils.partitioning import split


def save(X, Y, **kwargs):
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
        'n_levelsets' : number of slices to use (int)
        'split_by' : 'dyadic' (dyadic decomposition) or 'stateq' (statistically equivalent blocks) (default: 'dyadic')
        'return_mat' : Boolean whether key SIR matrix should be returned (defaults
                    to False).
    }

    Returns
    -----------
    proj_vecs : array-like, shape = [n_features, d]
        Orthonormal system spanning the sufficient dimension subspace, where
        d refers to the intrinsic dimension.

    M : SAVE matrix, only if return_mat option is True
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
    M = np.zeros((D, D)) # Container for key matrix in SIR
    # Compute SAVE matrix
    empirical_probabilities = np.zeros(n_levelsets)
    for i in range(n_levelsets):
        empirical_probabilities[i] = float(len(np.where(labels == i)[0]))/float(N)
        if empirical_probabilities[i] == 0:
            continue
        cov_sub = empirical_covariance(Z[labels == i,:]) # Covariance of all samples
        M += empirical_probabilities[i] * (np.eye(D) - cov_sub).dot(np.eye(D) - cov_sub)
    U, S, V = np.linalg.svd(M)
    # Apply inverse transformation
    vecs = cov_all_sqrtinv.dot(U[:,:d])
    proj_vecs, dummy = np.linalg.qr(vecs)
    if return_mat:
        return proj_vecs, M
    else:
        return proj_vecs
