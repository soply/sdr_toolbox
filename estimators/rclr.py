# coding: utf8
"""
SDR-method: Response conditional linear regression
"""

import numpy as np
import copy

from sklearn.linear_model import LinearRegression
from sklearn.covariance import empirical_covariance

from utils.whitening import whiten_data
from utils.partitioning import split


def rclr(X, Y, **kwargs):
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
        'whiten' : If true, the data is whitened before applying the method (default: False)
        'return_proxy' : If true, a data-driven guess for the projection error is returned.

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
    return_proxy = kwargs.get('return_proxy', False)
    whiten = kwargs.get('whiten', False)
    N, D = X.shape
    data_driven_proxy = 0
    if whiten:
        # Standardize X
        Z, cov_all_sqrtinv = whiten_data(X)
        copy_kwargs = copy.deepcopy(kwargs)
        copy_kwargs['whiten'] = False
        if return_proxy:
            transformed_vecs, data_driven_proxy = rclr(Z, Y, **copy_kwargs)
        else:
            transformed_vecs = rclr(Z, Y, **copy_kwargs)
        # Apply inverse transformation
        vecs = cov_all_sqrtinv.dot(transformed_vecs)
        proj_vecs, dummy = np.linalg.qr(vecs)
    else:
        # Create partition
        labels = split(Y, n_levelsets, split_by)
        M = np.zeros((D, D)) # Container for key matrix in SIR
        # Compute SIR matrix
        empirical_probabilities = np.zeros(n_levelsets)
        for i in range(n_levelsets):
            empirical_probabilities[i] = float(len(np.where(labels == i)[0]))/float(N)
            if empirical_probabilities[i] == 0:
                continue
            sigma_j = empirical_covariance(X[labels == i, :])
            sigma_j_inv = np.linalg.pinv(sigma_j)
            rhs = np.mean((X[labels == i,:] - np.mean(X[labels == i,:])).T * (Y[labels == i] -  np.mean(Y[labels == i])), axis = 1)
            local_ols = sigma_j_inv.dot(rhs)
            M += empirical_probabilities[i] * np.outer(local_ols, local_ols)
            # Compute proxy error if desired
            if return_proxy:
                if empirical_probabilities[i] * N > 5.0 * D and np.linalg.norm(local_ols) > 0.0:
                    # Projections
                    normalized_ols = local_ols/np.linalg.norm(local_ols)
                    P = np.outer(normalized_ols, normalized_ols)
                    Q = np.eye(D) - P
                    # Compute spectral norms
                    PSP = P.dot(sigma_j).dot(P)
                    PSDP = P.dot(sigma_j_inv).dot(P)
                    QSQ = Q.dot(sigma_j).dot(Q)
                    QSDQ = Q.dot(sigma_j_inv).dot(Q)
                    # Compute variance
                    local_var_y = np.var(Y[labels == i])
                    n_PSP = np.linalg.norm(PSP, 2)
                    n_QSQ = np.linalg.norm(QSQ, 2)
                    n_PSDP = np.linalg.norm(PSDP, 2)
                    n_QSDQ = np.linalg.norm(QSDQ, 2)
                    local_kappa = np.maximum(n_PSP * n_PSDP, n_QSQ * n_QSDQ)
                    local_eta_q = np.sqrt(local_var_y * n_QSDQ)
                    # Removing some uncertainty by using only sufficiently populated level sets
                    data_driven_proxy += np.sqrt(empirical_probabilities[i] * local_kappa) * np.linalg.norm(local_ols) * local_eta_q
        U, S, V = np.linalg.svd(M)
        if return_proxy:
            if data_driven_proxy == 0:
                # This just means that on no level set there we enough samples to compute the proxy quantity, thus
                # we effectively do not have a proxy
                data_driven_proxy = 1e16
            data_driven_proxy = data_driven_proxy * np.sqrt(1 + np.log(n_levelsets)) * 1.0/S[d-1]
        # Apply inverse transformation
        proj_vecs = U[:,:d]
    if return_proxy:
        return proj_vecs, data_driven_proxy
    else:
        return proj_vecs


def rclr_parameterfree(X, Y, **kwargs):
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
        'max_n_levelsets' : Maximum number of slices to use (int)
        'split_by' : 'dyadic' (dyadic decomposition) or 'stateq' (statistically equivalent blocks) (default: 'dyadic')
        'whiten' : If true, the data is whitened before applying the method (default: False)

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
    max_n_levelsets = kwargs['max_n_levelsets']
    split_by = kwargs.get('split_by', 'dyadic')
    whiten = kwargs.get('whiten', False)
    N, D = X.shape
    proxy = np.zeros(max_n_levelsets)
    vecs = np.zeros((D, d, max_n_levelsets))
    copy_kwargs = copy.deepcopy(kwargs)
    del copy_kwargs['max_n_levelsets']
    for i in range(max_n_levelsets):
        copy_kwargs['n_levelsets'] = i + 1
        copy_kwargs['return_proxy'] = True
        vecs[:,:,i], proxy[i] = rclr(X, Y, **copy_kwargs)
    ind = np.argmin(proxy[d:])
    return vecs[:,:,ind], proxy
