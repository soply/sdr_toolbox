# coding: utf8
import numpy as np
import scipy

from sklearn.covariance import empirical_covariance
from scipy.linalg import sqrtm

from utils.whitening import whiten_data
from utils.partitioning import split


"""
References:

[1] Li, Bing, and Shaoli Wang.
    "On directional regression for dimension reduction."
    Journal of the American Statistical Association 102.479 (2007): 997-1008.


[2] Yu, Zhou, Yuexiao Dong, and Mian Huang.
    "General directional regression."
    Journal of Multivariate Analysis 124 (2014): 94-104.
"""


def directional_regression(X, Y, **kwargs):
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
        't1' : scaling parameter 1 for generalized directional regression (see [2])
        't2' : scaling parameter 2 for generalized directional regression (see [2])

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
    split_by = kwargs.get('split_by', 'dyadic')
    return_mat = kwargs.get('return_mat', False)
    t1 = kwargs.get('t1', 0.5) # Generalized directional regression parameter 1 (default is DR)
    t2 = kwargs.get('t2', 1.0) # Generalized directional regression parameter 1 (default is DR)
    N, D = X.shape
    # Standardize X
    Z, cov_all_sqrtinv = whiten_data(X)
    # Create partition
    labels = split(Y, n_levelsets, split_by)
    # Containers for DR matrices
    sum_1 = np.zeros((D, D))
    sum_2 = np.zeros((D, D))
    sum_3 = 0
    empirical_probabilities = np.zeros(n_levelsets)
    for i in range(n_levelsets):
        empirical_probabilities[i] = float(len(np.where(labels == i)[0]))/float(N)
        if empirical_probabilities[i] == 0:
            continue
        U_h = np.mean(Z[labels == i, :], axis = 0)
        cov_local = empirical_covariance(Z[labels == i, :])
        V_h = cov_local - np.eye(D)
        # Compute sums
        sum_1 += empirical_probabilities[i] * V_h.dot(V_h)
        sum_2 += empirical_probabilities[i] * np.outer(U_h, U_h)
        sum_3 += empirical_probabilities[i] * np.dot(U_h, U_h)
    F = t1 * sum_1 + (1.0 - t1) * sum_2.dot(sum_2) + (1.0 - t1) * t2 * sum_3 * sum_2
    U, S, V = np.linalg.svd(F)
    # Apply inverse transformation
    vecs = cov_all_sqrtinv.dot(U[:,:d])
    # Get Projection from vecs (don't need to be norm 1 anymore)
    proj_vecs, dummy = np.linalg.qr(vecs)
    if return_mat:
        return proj_vecs, F
    else:
        return proj_vecs
