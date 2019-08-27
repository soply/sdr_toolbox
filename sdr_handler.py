# coding: utf8

"""
Handler to call any of the sufficient dimension reduction estimator.
"""

from estimators.sir import sir
from estimators.save import save
from estimators.pca import pca
from estimators.phd import phd
from estimators.directional_regression import directional_regression


__methods__ = ['SIR', 'SAVE', 'PCA', 'PHD', 'DR']


def estimate_sdr(X, Y, method, **kwargs):
    if method not in __methods__:
        raise NotImplementedError('Method {0} not implemented yet.'.format(method))
    elif method == 'SIR':
        return sir(X, Y, **kwargs)
    elif method == 'SAVE':
        return save(X, Y, **kwargs)
    elif method == 'PCA':
        return pca(X, **kwargs)
    elif method == 'PHD':
        return phd(X, Y, **kwargs)
    elif method == 'DR':
        return directional_regression(X, Y, **kwargs)
