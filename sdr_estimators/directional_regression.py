# coding: utf8
import numpy as np
import scipy

from ..partitioning.single_layer_levelset import \
    split_statistically_equivalent_blocks
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
from scipy.linalg import sqrtm

"""
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
    rescale = kwargs['rescale']
    return_mat = kwargs.get('return_mat', False)
    t1 = kwargs.get('t1', 0.5) # Generalized directional regression parameter 1 (default is DR)
    t2 = kwargs.get('t2', 1.0) # Generalized directional regression parameter 1 (default is DR)

    D, N = X.shape
    # Standardize X
    scaler = StandardScaler()
    if rescale:
        emc = EmpiricalCovariance()
        emc = emc.fit(X.T) # Covariance of all samples
        mean_all = np.mean(X, axis = 1)
        cov_all = emc.covariance_
        Z = scaler.fit_transform(X.T).T
    labels, n_levelsets = split_statistically_equivalent_blocks(X, Y, n_levelsets)
    sum_1 = np.zeros((D, D))
    sum_2 = np.zeros((D, D))
    sum_3 = 0
    empirical_probabilities = np.zeros(n_levelsets)
    for i in range(n_levelsets):
        empirical_probabilities[i] = float(len(np.where(labels == i)[0]))/float(N)
        if rescale:
            U_h = np.mean(Z[:, labels == i], axis = 1)
            emc = emc.fit(Z[:, labels == i].T)
            V_h = emc.covariance_ - np.eye(D)
            # Compute sums
            sum_1 += empirical_probabilities[i] * V_h.dot(V_h)
            sum_2 += empirical_probabilities[i] * np.outer(U_h, U_h)
            sum_3 += empirical_probabilities[i] * np.dot(U_h, U_h)
        else:
            # Not sure if the non-scaled verison works
            U_h = np.mean(X[:, labels ==  i], axis = 1) - mean_all
            emc = emc.fit(X[:, labels == i].T)
            V_h = emc.covariance_ - cov_all
            # Compute sums
            sum_1 += empirical_probabilities[i] * V_h.dot(V_h)
            sum_2 += empirical_probabilities[i] * np.outer(U_h, U_h)
            sum_3 += empirical_probabilities[i] * np.dot(U_h, U_h)
    F = t1 * sum_1 + (1.0 - t1) * sum_2.dot(sum_2) + (1.0 - t1) * t2 * sum_3 * sum_2
    U, S, V = np.linalg.svd(F)
    # Apply inverse transformation
    if rescale:
        vecs = sqrtm(scipy.linalg.inv(cov_all)).dot(U[:,:d])
        # Get Projection from vecs (don't need to be norm 1 anymore)
        proj_vecs, dummy = np.linalg.qr(vecs)
    else:
        proj_vecs = U[:,:d]
    if return_mat:
        return proj_vecs, F
    else:
        return proj_vecs
