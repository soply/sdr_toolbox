# # coding: utf8
# import numpy as np
# import scipy
#
# from partitioning.level_set_based import split_statistically_equivalent_blocks
# from sklearn.covariance import EmpiricalCovariance
# from sklearn.preprocessing import StandardScaler
# from visualisation.vis_nD import *
# from scipy.linalg import sqrtm
#
#
# def directional_regression(X, Y, d, n_levelsets, proj_A, scaled = False):
#     D, N = X.shape
#     # Standardize X
#     scaler = StandardScaler()
#     emc = EmpiricalCovariance()
#     emc = emc.fit(X.T) # Covariance of all samples
#     mean_all = np.mean(X, axis = 1)
#     cov_all = emc.covariance_
#     Z = scaler.fit_transform(X.T).T
#     labels = split_statistically_equivalent_blocks(X, Y, n_levelsets)
#     sum1 = np.zeros((D, D))
#     sum2 = np.zeros((D, D))
#     sum3 = 0
#     empirical_probabilities = np.zeros(n_levelsets)
#     for i in range(n_levelsets):
#         empirical_probabilities[i] = float(len(np.where(labels == i)[0]))/float(N)
#         if scaled:
#             # Sum 1
#             emc = emc.fit(Z[:, labels == i].T)
#             cov = emc.covariance_.T - np.eye(D)
#             sum1 += cov * cov
#             # Sum 2
#             mean = np.mean(Z[:,labels == i], axis = 1)
#             sum2 += np.outer(mean, mean)
#             # Sum 3
#             mean = np.mean(Z[:,labels == i], axis = 1)
#             sum3 += np.dot(mean, mean)
#         else:
#             # Sum 1
#             emc = emc.fit(X[:, labels == i].T)
#             cov = emc.covariance_.T - cov_all
#             sum1 += cov * cov
#             # Sum 2
#             mean = np.mean(X[:,labels == i], axis = 1)
#             sum2 += np.outer(mean - mean_all, mean-mean_all)
#             # Sum 3
#             mean = np.mean(X[:,labels == i], axis = 1)
#             sum3 += np.dot(mean-mean_all, mean-mean_all)
#     F = 2.0 * (sum1 + sum2.dot(sum2) + sum3 * sum2)
#     U, S, V = np.linalg.svd(F)
#     # Apply inverse transformation
#     if scaled:
#         vecs = sqrtm(scipy.linalg.inv(cov_all)).dot(U[:,:d])
#         # Get Projection from vecs (don't need to be norm 1 anymore)
#         proj_vecs, dummy = np.linalg.qr(vecs)
#     else:
#         proj_vecs = U[:,:d]
#     return np.linalg.norm(proj_vecs.dot(proj_vecs.T) - proj_A.dot(proj_A.T), 2) ** 2
