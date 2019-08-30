# coding: utf8
""" Methods for standardizing a data set """

import numpy as np
from scipy.linalg import sqrtm
from sklearn.covariance import empirical_covariance


def whiten_data(X):
    mean_all = np.mean(X, axis = 0)
    cov_all = empirical_covariance(X, assume_centered = False)
    cov_all_sqrtinv = sqrtm(np.linalg.pinv(cov_all))
    Z = (X-mean_all).dot(cov_all_sqrtinv)
    return Z, cov_all_sqrtinv
