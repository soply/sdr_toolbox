# coding: utf8
import inspect
import os
import sys

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KNeighborsRegressor

# Full path
from sdr_handler import estimate_sdr



class SDRKnn(BaseEstimator, RegressorMixin):
    """
    Performs kNN regression after applying SDR method.
    """

    def __init__(self,
                 method,
                 n_neighbors,
                 n_components,
                 n_levelsets,
                 whiten = False,
                 use_residuals = False,
                 split_by = 'dyadic',
                 rescale = False):
        # Set attributes of object to the same name as given in the argument
        # list.
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)


    def fit(self, X, y=None):
        """
        """
        if self.n_components > self.n_levelsets:
            raise RuntimeError("n_components = {0} > {1} = n_levelsets".format(self.n_components, self.n_levelsets))
        n_samples, n_features = X.shape
        self.SDR_space = estimate_sdr(X, y, self.method,
                                      d = self.n_components,
                                      n_levelsets = self.n_levelsets,
                                      split_by = self.split_by,
                                      use_residuals = self.use_residuals,
                                      whiten = self.whiten,
                                      return_mat = False)
        self.PX_ = X.dot(self.SDR_space)
        self.knn_ = KNeighborsRegressor(n_neighbors = self.n_neighbors)
        self.knn_ = self.knn_.fit(self.PX_, y)
        return self


    def predict(self, X, y=None):
        try:
            getattr(self, "knn_")
        except AttributeError:
            raise RuntimeError("You must train estimator before predicting data!")

        return self.knn_.predict(X.dot(self.SDR_space))
