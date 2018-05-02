# coding: utf8
"""
Simple PCA method that is not using responses. This is implemented
just for completeness and convenience since it may be used along
with the other dimension reduction techniques.
"""

import numpy as np
import scipy
from scipy.linalg import sqrtm
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca(X, **kwargs):
    """
    Parameters
    ----------
    X : array-like, shape = [n_features, n_samples]
        Training data, where n_samples is the number of samples and
        n_features is the number of features.


    Argument dictionary should contain:
    kwargs = {
        'd' : intrinsic dimension (int)
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
    d = kwargs['d']
    return_mat = kwargs['return_mat']
    rescale = kwargs['rescale']
    scaler = StandardScaler()
    if rescale:
        emc = EmpiricalCovariance()
        emc = emc.fit(X.T) # Covariance of all samples
        cov_all = emc.covariance_
        Z = scaler.fit_transform(X.T).T
        pca = PCA(svd_solver = 'full')
        pca = pca.fit(Z.T)
        proj_vecs = pca.components_[:d,:].T
        # Apply inverse transformation
        vecs = sqrtm(scipy.linalg.inv(cov_all)).dot(proj_vecs)
        proj_vecs, dummy = np.linalg.qr(vecs)
    else:
        pca = PCA(svd_solver = 'full')
        pca = pca.fit(X.T)
        proj_vecs = pca.components_[:d,:].T
    if return_mat:
        return proj_vecs, X.dot(X.T)
    else:
        return proj_vecs
