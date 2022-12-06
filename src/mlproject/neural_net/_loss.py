import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """Compute the categorical cross entropy loss
    from the given true labels and predicted labels.

    adding 1e-7 to avoid taking the log of 0 - Inspired by keras implemenation:
    https://github.com/keras-team/keras/blob/master/keras/backend.py#L5487-L5547 &
    https://github.com/keras-team/keras/blob/master/keras/backend_config.py#L34-L44


    Parameters
    ----------
    y_true : 1d ndarray
        true class labels of size 1 x n where n is the number of data points.
    y_pred : 1d ndarray
        predicted class labels of size 1 x n where n is the number of data points.

    Returns
    -------
    float
        Cross entropy score for the given prediction
    """
    epsilon = 1e-07
    return -np.sum(np.log(y_pred+epsilon) * y_true)
