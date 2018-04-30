# coding: utf8

"""
Handler to call any of the SDR estimators.
"""

from sdr_estimators.sir import sir
from sdr_estimators.save import save

__methods__ = ['SIR', 'SAVE']


def estimate_sdr(X, Y, method, **kwargs):
    if method not in __methods__:
        raise NotImplementedError('Method {0} not implemented yet.'.format(
            method))
    elif method == 'SIR':
        return sir(X, Y, **kwargs)
    elif method == 'SAVE':
        return save(X, Y, **kwargs)
    
