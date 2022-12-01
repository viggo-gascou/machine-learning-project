import numpy as np


class DenseLayer:
    """Fully connected layer of a neural network

    Parameters
    ----------
    input_n : int
        The amount of inputs to the DenseLayer
    output_n : int
        The amount of outputs to the DenseLayer
    activation : str
        The activation function to use for all of the neurons in the DenseLayer
    """
    def __init__(self, input_n: int, output_n: int, activation: str):
        # he initiliasation of weights and biases
        n_avg = 1/2 * (input_n + output_n)
        sigma = 2 / n_avg
        self.weights = np.random.normal(loc = 0, scale = sigma, size=(input_n, output_n))

        self.biases = np.zeros(shape=(output_n))

        if activation == 'leaky_relu':
            self.activation = leaky_relu
        elif activation == 'softmax':
            self.activation = softmax
        else:
            raise NotImplementedError(f"{activation} not implemented yet. Choose from ['leaky_relu', 'softmax']")

            
    def forward(self, X):
        """Computes a single forward pass of the DenseLayer

        Parameters
        ----------
        X : 2d ndarray
            An n x p matrix of data points
            where n is the number of data points and p is the number of features.

        Returns
        -------
        2d ndarray
            An n x output_n numpy array where n is the number of samples
            and output_n is the number of neurons in the DenseLayer
        """
        return self.activation(X @ self.weights + self.biases)

    def activation_function(self):
        if self.activation == softmax:
            return 'softmax'
        elif self.activation == leaky_relu:
            return 'leaky_relu'


def leaky_relu(z):
    return np.where(z > 0, z, z * 0.01)

def softmax(z):
    e = np.exp(z)
    return e / e.sum()
    