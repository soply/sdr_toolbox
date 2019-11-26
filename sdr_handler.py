# coding: utf8

"""
Handler to call any of the sufficient dimension reduction estimator.
"""

from estimators.sir import sir
from estimators.sirii import sirii
from estimators.save import save
from estimators.pca import pca
from estimators.phd import phd
from estimators.directional_regression import directional_regression
from estimators.rclr import rclr, rclr_parameterfree
from estimators.iht import iht
from estimators.phd import phd
from estimators.nndr import nndr



__methods__ = ['SIR', 'SIRII', 'SAVE', 'PCA', 'PHD', 'DR', 'RCLR','RCLR_proxy','IHT', 'PHD', 'NNDR']


def estimate_sdr(X, Y, method, **kwargs):
    if method not in __methods__:
        raise NotImplementedError('Method {0} not implemented yet.'.format(method))
    elif method == 'SIR':
        return sir(X, Y, **kwargs)
    elif method == 'SIRII':
        return sirii(X, Y, **kwargs)
    elif method == 'SAVE':
        return save(X, Y, **kwargs)
    elif method == 'PCA':
        return pca(X, **kwargs)
    elif method == 'PHD':
        return phd(X, Y, **kwargs)
    elif method == 'DR':
        return directional_regression(X, Y, **kwargs)
    elif method == 'RCLR':
        return rclr(X, Y, **kwargs)
    elif method == 'RCLR_proxy':
        return rclr_parameterfree(X, Y, **kwargs)
    elif method == 'IHT':
        return iht(X, Y, **kwargs)
    elif method == 'PHD':
        return iht(X, Y, **kwargs)
    elif method == 'NNDR':
        return nndr(X, Y, **kwargs)
