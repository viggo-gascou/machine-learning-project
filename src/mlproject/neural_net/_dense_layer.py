import numpy as np
from mlproject.neural_net._activations import leaky_relu, stable_softmax


class DenseLayer:
    """Fully connected layer of a neural network

    The weight initialisations are dependant on the activation function.

    * If the activation function is LeakyReLU, then the weights are initialised from a He-Normal distribution - see: http://proceedings.mlr.press/v9/glorot10a.html

    * If the activation function is softmax, then the weights are initialised from a Glorot/Xavier normal distribution - see: https://arxiv.org/abs/1502.01852

    Parameters
    ----------
    input_n : int
        The amount of inputs to the DenseLayer
    output_n : int
        The amount of outputs to the DenseLayer
    activation : str
        The activation function to use for all of the neurons in the DenseLayer, either 'leaky_relu' or 'softmax'
    """

    def __init__(self, input_n: int, output_n: int, activation: str):

        self.output_n, self.input_n = output_n, input_n

        self.z = None

        self.biases = np.zeros(shape=(output_n))

        if activation == "leaky_relu":
            self.activation = leaky_relu
            # He initiliasation of weights
            # see https://keras.io/api/layers/initializers/#henormal-class
            stddev = np.sqrt(2 / input_n)
            self.weights = np.random.normal(
                loc=0, scale=stddev, size=(input_n, output_n)
            )

        elif activation == "softmax" or activation == "stable_softmax":
            # Glorot/Xavier initiliasation of weights
            # see https://keras.io/api/layers/initializers/#glorotnormal-class
            stddev = np.sqrt(2 / (input_n + output_n))
            self.weights = np.random.normal(
                loc=0, scale=stddev, size=(input_n, output_n)
            )
            self.activation = stable_softmax

        else:
            raise NotImplementedError(
                f"{activation} not implemented yet. Choose from ['leaky_relu', 'stable_softmax']"
            )

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
        self.z = X.dot(self.weights) + self.biases
        return self.activation(self.z)

    def _out_neurons(self):
        """Return the number of output neurons in the DenseLayer

        Returns
        -------
        int
            The total number of output neurons in the DenseLayer
        """
        return self.output_n

    def _in_neurons(self):
        """Return the number of input neurons in the DenseLayer

        Returns
        -------
        int
            The total number of input neurons in the DenseLayer
        """
        return self.input_n

    def activation_function(self):
        """Return a string representing the activation function of the given DenseLayer

        Returns
        -------
        string
            string representing the activation function of the given DenseLayer
        """
        if self.activation == stable_softmax:
            return "softmax"
        elif self.activation == leaky_relu:
            return "leaky_relu"
