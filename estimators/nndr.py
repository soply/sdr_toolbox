# coding:utf8
from keras import backend as K
import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2, Regularizer

from estimators.utils.whitening import whiten_data

class RegressionNetwork:
    """
    Wrapper for simple fully connected regression network. Last layer is regression
    layer without activation function.
    """

    def __init__(self, W = None, L = 0, act ='relu', reg_l2 = 0.00, **kwargs):
        """
        D: Input dimension
        W: Layer width
        L: Number of hidden (nonlinear) layers
        """
        self.W = W
        self.L = L
        self.act = act
        self.reg_l2 = reg_l2
        layers = []
        for i in range(L):
            layers.append(Dense(W, use_bias = True,
                          activation = act,
                          kernel_regularizer=l2(reg_l2),
                          bias_regularizer=l2(reg_l2)))
        layers.append(Dense(1))
        self.net = Sequential(layers)

    def train(self, X, Y, batch_size = None, n_epochs = 400, optimizer = 'adam', verbose = 0, **kwargs):
        self.net.compile(optimizer=optimizer, loss="mean_squared_error")
        N, D = X.shape
        if batch_size is None:
            batch_size = N
        self.history = self.net.fit(X, Y,
                                    epochs=n_epochs,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    verbose=verbose)

    def predict(self,data):
        return self.net.predict(data)

def nndr(X, Y, **kwargs):
    N, D = X.shape
    d = kwargs['d']
    return_mat = kwargs.get('return_mat', False)
    whiten = kwargs.get('whiten', False)
    if whiten:
        Z, cov_all_sqrtinv = whiten_data(X)
        reg_net = RegressionNetwork(**kwargs)
        reg_net.train(Z, Y, **kwargs)
        weight_matrix = reg_net.net.layers[0].get_weights()[0]
        opm_matrix = np.zeros((D, D))
        for i in range(weight_matrix.shape[1]):
            opm_matrix += np.outer(weight_matrix[:,i], weight_matrix[:,i])
        U, S, V = np.linalg.svd(opm_matrix)
        # Apply inverse transformation
        vecs = cov_all_sqrtinv.dot(U[:,:d])
        proj_vecs, dummy = np.linalg.qr(vecs)
    else:
        reg_net = RegressionNetwork(**kwargs)
        reg_net.train(X, Y, **kwargs)
        weight_matrix = reg_net.net.layers[0].get_weights()[0]
        opm_matrix = np.zeros((D, D))
        for i in range(weight_matrix.shape[1]):
            opm_matrix += np.outer(weight_matrix[:,i], weight_matrix[:,i])
        U, S, V = np.linalg.svd(opm_matrix)
        proj_vecs = U[:,:d]
    if return_mat:
        return proj_vecs, opm_matrix
    else:
        return proj_vecs
