# coding: utf8
"""
Peak SSV Method:

Take small ball in the response domain around the maximum response value Y
and take the d singular vectors to the smallest d eigenvalues of the corresponding
sample covariance matrix.
"""

import numpy as np
import scipy
from scipy.linalg import sqrtm
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def peakSSV(X, Y, **kwargs):
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
        'n_samples' : Number of samples around the maximum Y to take.
        'rescale' : Boolean whether standardization should be performed (True
                    for yes).
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
    n_samples = kwargs['n_samples']
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
    pca = PCA()
    order = np.argsort(Y)
    XO = X[:,order]
    pca = pca.fit(X[:,-n_samples:].T)
    U = pca.components_[-d:,:].T
    if rescale:
        # Apply inverse transformation
        vecs = sqrtm(scipy.linalg.inv(cov_all)).dot(U[:,:d])
        proj_vecs, dummy = np.linalg.qr(vecs)
    else:
        proj_vecs = U[:,:d]
    return proj_vecs
