import numpy as np


class DenseLayer:
    """_summary_

    Parameters
    ----------
    input_n : _type_
        _description_
    output_n : _type_
        _description_
    activation : _type_
        _description_
    """
    def __init__(self, input_n, output_n, activation):
        self.weights = np.random.rand(input_n, output_n)

        self.biases = np.random.rand(output_n)

        if activation == 'leaky_relu':
            self.activation = leaky_relu
        elif activation == 'softmax':
            self.activation = softmax
        else:
            raise NotImplementedError("Cannot find Activation Function. Choose from ['leaky_relu', 'softmax']")

    def forward(self, X):
        """_summary_

        Parameters
        ----------
        X : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return self.activation(X @ self.weights + self.biases)


def leaky_relu(z):
    return np.where(z > 0, z, z * 0.1)

def softmax(z):
    e = np.exp(z)
    return e / e.sum()