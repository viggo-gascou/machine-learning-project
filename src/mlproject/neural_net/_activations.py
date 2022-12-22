import numpy as np


def leaky_relu(z):
    """Leaky relu activation function

    Parameters
    ----------
    z : 2d ndarray
        input to the leaky relu activation function

    Returns
    -------
    2d ndarray
        leaky relu 'activated' version of the input `z`
    """
    return np.where(z > 0, z, z * 0.01)


def leaky_relu_der(z):
    """Derivative of the leaky relu activation function

    Parameters
    ----------
    z : 2d ndarray
        input to calculate the derivative of

    Returns
    -------
    2d ndarray
        derivative of the specific neuron with a leaky relu activation function
    """
    return np.where(z > 0, 1, 0.01)


def stable_softmax(z):
    """Numerically stable softmax activation function

    Inspired by https://stackoverflow.com/a/50425683 &
    https://github.com/scipy/scipy/blob/v1.9.3/scipy/special/_logsumexp.py#L130-L223

    Parameters
    ----------
    z : 2d ndarray
        input to the softmax activation function

    Returns
    -------
    2d ndarray
        softmax 'activated' version of the input `z`
    """
    # When keepdims is set to True we keep the original dimensions/shape of the input.
    # axis = 1 means that we find the maximum value along the first axis i.e. the rows.
    e_max = np.amax(z, axis=1, keepdims=True)
    e = np.subtract(z, e_max)
    e = np.exp(e)
    return e / np.sum(e, axis=1, keepdims=True)
