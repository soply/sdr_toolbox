# coding: utf8
import numpy as np

"""
Several methods to create a level set based partitioning with multiple
layers.

All methods return:
-------------------------


"""

def dyadic_tree_partition(X, Y, var_f = 0.0, n_layers = None):
    """ Creates a dyadic tree partition on the function values Y to assign the
    points to into a level set partition of X at different scales. The maximum
    depth is given by n_layers. If n_layers = None, the finest scale is the
    maximum scale such that each partition contains at least
    (n_features + 1) points. If n_layers is larger than this scale, partitions
    will only be refined where it is possible, i.e. where a refinement still
    produces partitions with at lest (n_features + 1) data points.

    Parameters
    -------------


    Returns
    ------------

    """
    return tree_partition(X, Y, var_f, split_by = 'mid',
                          n_layers = n_layers)


def median_tree_partition(X, Y, var_f = 0.0, n_layers = None):
    """ Creates a dyadic tree partition on the function values Y to assign the
    points to into a level set partition of X at different scales. The maximum
    depth is given by n_layers. If n_layers = None, the finest scale is the
    maximum scale such that each partition contains at least
    (n_features + 1) points. If n_layers is larger than this scale, partitions
    will only be refined where it is possible, i.e. where a refinement still
    produces partitions with at lest (n_features + 1) data points.

    Parameters
    -------------


    Returns
    ------------

    """
    return tree_partition(X, Y, var_f, split_by = 'med',
                          n_layers = n_layers)


def tree_partition(X, Y, var_f = 0.0, split_by = 'mid', n_layers = None):
    """ Creates a tree partition on the function values Y to assign the
    points to into a level set partition of X at different scales. The maximum
    depth is given by n_layers. If n_layers = None, the finest scale is the
    maximum scale such that each partition contains at least
    (n_features + 1) points. If n_layers is larger than this scale, partitions
    will only be refined where it is possible, i.e. where a refinement still
    produces partitions with at lest (n_features + 1) data points. The split_by
    argument decides how a partition is refined. Currently implemented are:

        'mid': split partition at Y_mid = 0.5 * (Y_min + Y_max)
        'med': split partition at Y_mid = med(Y)

    Parameters
    -------------


    Returns
    ------------

    """
    n_features, n_samples = X.shape
    labels = {}
    current_layer = 0
    bin_edges = np.array([np.min(Y) - 1e-10, np.max(Y) + 1e-10])
    all_bin_edges = {} # Storing level set partition to return it
    stop = False
    while n_layers is None or current_layer < n_layers:
        indices = np.digitize(Y, bin_edges, right=False) - 1
        counts = np.bincount(indices)
        labels[current_layer] = indices
        all_bin_edges[current_layer] = bin_edges
        # Prepare next layer
        bins_to_refine, bin_edges_to_insert = [], []
        for i in range(len(bin_edges) - 1):
            if split_by == 'mid':
                Y_mid = 0.5 * (bin_edges[i] + bin_edges[i+1])
            elif split_by == 'med':
                Y_mid = np.median(Y[indices == i])
            if len(np.where(Y[indices == i] < Y_mid)[0]) > n_features and \
                    len(np.where(Y[indices == i] > Y_mid)[0]) > n_features and \
                    (Y_mid - bin_edges[i]) > np.sqrt(var_f) and \
                    (bin_edges[i+1] - Y_mid) > np.sqrt(var_f):
                # Add bin for refinement
                bins_to_refine.append(i+1)
                bin_edges_to_insert.append(Y_mid)
            elif n_layers is None:
                # Stop adding layers in this case
                stop = True
                break
        if len(bins_to_refine) == 0 or stop:
            # Stop in this case as well because we can refine no bin further
            break
        bin_edges = np.insert(bin_edges, bins_to_refine, bin_edges_to_insert)
        current_layer += 1
    return labels, current_layer, all_bin_edges


def split_n_equidistantly(X, Y, n_splits, tol = 0.0):
    """ Split into two level sets for n_split Y values. n_split is
    chosen equidistantly between minY and maxY. Tol can be set to reduce
    chance of overlapping regions."""
    n_features, n_samples = X.shape
    labels = {}
    all_bin_edges = {}
    boundary_vals = np.linspace(np.min(Y), np.max(Y), num = n_splits + 2)
    boundary_vals = boundary_vals[1:-1] # Cut of both endpoints
    for current_layer, boun in enumerate(boundary_vals):
        labels{current_layer} = np.ones(n_samples).astype('int')
        idx1 = np.where(Y < boun - tol)[0]
        idx2 = np.where(Y >= boun + tol)[0]
        labels{current_layer}[idx1] = 0
        labels{current_layer}[idx2] = 1
        all_bin_edges{current_layer} = np.array([np.min(Y), boun, np.max(Y)])
    return labels, n_splits, all_bin_edges
