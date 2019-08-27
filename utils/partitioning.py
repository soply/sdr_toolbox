# coding: utf8
import numpy as np

"""
Several methods to create a level set based partitioning with a single layer.
"""


def split(Y, n_splits, split_by):
    """ Handler for different partitioning methods """
    if split_by == 'dyadic':
        return split_dyadically(Y, n_splits)
    elif split_by == 'stateq':
        return split_statistically_equivalent_blocks(Y, n_splits)
    else:
        raise NotImplementedError("{0} is not implemented".format(split_by))


def split_statistically_equivalent_blocks(Y, n_splits):
    """
    Splits the given data Y into statistically equivalent blocks
    (i.e. #points of two blocks differs at most by 1) based on the order
    of Y. Returns vector with labels from 0 to n_splits - 1.
    """
    N = Y.shape
    order = np.argsort(Y)
    pieces = np.array_split(order, n_splits)
    labels = np.zeros(N)
    for i, piece in enumerate(pieces):
        labels[piece] = i
    return labels


def split_dyadically(Y, n_splits):
    """
    Partitions given data Y based on level set partitioning on the Y-values.
    To split the Y values, we create n_splits disjoint intervals spanning the
    range of Y and having equal width. Afterwards, we assign each sample
    based on its Y value to one of the intervals.
    Returns vector with labels from 0 to n_splits - 1.
    """
    hist, edges = np.histogram(Y, bins = n_splits)
    # Correct for upper and lower edge to include all samples
    edges[0] -= 1e-10
    edges[-1] += 1e-10
    labels = np.digitize(Y, edges) - 1
    return labels
