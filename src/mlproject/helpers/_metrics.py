import numpy as np


def accuracy_score(y_true, y_pred, normalize=True):
    """Calculate the accuracy score from a given array of true labels
    and a given array of predicted labels.

    Inspired by [https://stackoverflow.com/a/64680660](https://stackoverflow.com/a/64680660)

    Parameters
    ----------
    y_true : 2d ndarray
        array of shape (n_samples, 1) of true labels
    y_pred : 2d ndarray
        array of shape (n_samples, 1) of predicted labels

    Returns
    -------
    accuracy_scores : float
        calculated accuracy score

    Raises
    ------
    ValueError
        if y_true and y_pred are not of the same shape
    """

    if y_true.shape[0] != y_pred.shape[0] and y_true.shape[1] != y_pred.shape[1]:
        raise ValueError(
            f"Length of y_true: ({len(y_true)}) and y_pred: ({len(y_pred)}) should be the same!"
        )

    accuracy = []
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            accuracy.append(1)
        else:
            accuracy.append(0)
    if normalize == True:
        return np.mean(accuracy)
    if normalize == False:
        return sum(accuracy)
