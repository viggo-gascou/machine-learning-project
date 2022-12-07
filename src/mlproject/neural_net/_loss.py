import numpy as np

def cross_entropy_loss(y_true, y_pred):
    r"""Compute the categorical cross entropy loss
    from the given true labels and predicted labels.

    We add $1\mathrm{e}{-7}$ (epsilon) to the prediction to avoid taking the log of $0$ 
    
    - Inspired by keras implemenation:
    [Keras implementation](https://github.com/keras-team/keras/blob/master/keras/backend.py#L5487-L5547) where
    epsilon is defined [here](https://github.com/keras-team/keras/blob/master/keras/backend_config.py#L34-L44)


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
