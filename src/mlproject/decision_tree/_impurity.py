import numpy as np


def gini_impurity(y):
    """Calculates the gini impurity of a given node

    Parameters
    ----------
    y : 2d ndarray
        array of y labels

    Returns
    -------
    float
        gini impurity score for the node
    """
    # flatten the array only because np.bincount expects a 1 dimensional array
    y = y.flatten()
    counts = np.bincount(y)
    N = np.sum(counts)
    p = counts/N
    return 1 - np.sum(p**2)

def entropy_impurity(y):
    """Calculates the entropy of a given node

    Parameters
    ----------
    y : 2d ndarray
        array of y labels

    Returns
    -------
    float
        entropy impurity of a given node
    """
    epsilon = 1e-07
    # flatten the array only because np.bincount expects a 1 dimensional array
    y = y.flatten()
    counts = np.bincount(y)
    N = np.sum(counts)
    p = counts / N
    return np.sum(-p*np.log2(p+epsilon))
