# coding: utf8
import numpy as np

"""
Several methods to create a level set based partitioning with a single layer.


All methods return:
-------------------------

"""





def based_on_df(X, Y, deltaf):
    """
    Partitions given data (X,Y) based on level set partitioning on the Y-values.
    The Y-values are split according to a partition of the function range into
    intervals of length deltaf. If some level set contains less than
    n_features + 1 data points afterwards, two consecutive level sets are merged.
    This is repeated until each level set contains at least
    n_features + 1 data points.

    Parameters
    -------------


    Returns
    ------------
    labels: np.array of size (n_samples) with ints, containing the label of the
            level set a data point was assigned to.

    n_level_sets: # of resulting level sets

    """
    n_samples, n_features = X.shape
    labels = np.zeros(n_samples).astype('int')
    level_sets = np.arange(np.min(Y) - 1e-10, np.max(Y) + 1e-10, deltaf)
    # Sort the Y values inside the bins
    labels = np.digitize(Y, level_sets)
    # Check if all labels occur at least n_features + 1 times
    counts = np.bincount(labels)
    while any(counts < n_features + 1):
        erase_boundaries = np.array()
        for i in range(len(counts)):
            if counts[i] < n_features + 1 and i < len(counts) - 1:
                erase_boundaries = np.append(erase_boundaries, i)
                print "Merged {0} and {1}".format(i, i+1)
            elif counts[i] < n_features + 1:
                erase_boundaries = np.append(erase_boundaries, i-1)
                print "Merged {0} and {1}".format(i-1, i)
        # Adjust level sets
        level_sets = np.delete(level_sets, erase_boundaries)
        labels = np.digitize(Y, level_sets)
        counts = np.bincount(labels)
    n_level_sets = len(set(labels))
    return labels, n_level_sets


def split_statistically_equivalent_blocks(X, Y, n_splits):
    """ Splits the given data (X,Y) into statistically equivalent blocks
    (i.e. #points of two blocks differs at most by 1) based on the order
    of Y. Returns 1 vector with labels from 0 to n_splits - 1."""
    D, N = X.shape
    order = np.argsort(Y)
    pieces = np.array_split(order, n_splits)
    labels = np.zeros(N)
    for i, piece in enumerate(pieces):
        labels[piece] = i
    return labels, n_splits
