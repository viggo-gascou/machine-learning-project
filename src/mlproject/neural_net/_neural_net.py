import numpy as np
from mlproject.neural_net._dense_layer import DenseLayer
from mlproject.helpers import accuracy_score
from mlproject.neural_net._loss import cross_entropy_loss
from mlproject.neural_net._activations import leaky_relu_der
from sklearn.preprocessing import OneHotEncoder

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)

progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    SpinnerColumn(),
    BarColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)


class NeuralNetworkClassifier:
    """NeuralNetworkClassifier

    Feed Forward Neural Network Classifier with however many
    dense layers (fully connected layers) of class [`DenseLayer`][mlproject.neural_net._dense_layer.DenseLayer] each with own
    activation function and a network wide loss function.
    The layers of the network can either be added when initilizing the network, as a list
    or added individually with the [`add`][mlproject.neural_net._neural_net.NeuralNetworkClassifier.add] method after initialization.

    Parameters
    ----------
    layers : list, optional
        A list of class [`DenseLayer`][mlproject.neural_net._dense_layer.DenseLayer]
    loss : str, optional
        The loss function to be used, currently only [`cross_entropy`][mlproject.neural_net._loss.cross_entropy_loss] is supported.

    Attributes
    ----------
    X : 2d ndarray
        Data points to use for training the neural network
    y : 1d ndarray
        Target classes
    n : int
        Number of data points (X.shape[0])
    p : int
        Number of features (X.shape[1])
    """

    def __init__(self, layers=[], loss="cross_entropy"):

        self.X = None
        self.n, self.p = None, None
        self.y = None
        self.k = None
        self.layers = layers
        self.activations, self.sums = [], []

        if loss == "cross_entropy":
            self.loss_str = "cross_entropy_loss"
            self.loss = cross_entropy_loss
        else:
            raise NotImplementedError(
                f"{loss} not implemented yet. Choose from ['cross_entropy']"
            )

    def add(self, layer: DenseLayer):
        """Add a new layer to the network, after the current layer.

        Parameters
        ----------
        layer : DenseLayer
            Fully connected layer.

        Example
        -------
        ``` py
        >>> NN = NeuralNetworkClassifier(loss='cross_entropy')
        >>> NN.add(DenseLayer(784,128,"leaky_relu"))
        >>> NN.add(DenseLayer(128,5,"softmax"))
        >>> print(NN)

        NeuralNetworkClassifier
        --------------------------------
        Loss function: cross_entropy_loss

        Input layer:
                Input: 784, Output: 128 , Activation: leaky_relu

        Output layer:
                Input: 128, Output: 5 , Activation: softmax
        ```
        """
        self.layers.append(layer)

    def forward(self, X):
        """Compute a single forward pass of the network.

        Parameters
        ----------
        X : 2d ndarray
            The data to use for the forward pass.
            Must be of size n x input_n
            where input_n must come from the first [`DenseLayer`][mlproject.neural_net._dense_layer.DenseLayer] in the network.

        Returns
        -------
        2d ndarray
            An n x output_n array
            where output_n corresponds to the output_n of the last [`DenseLayer`][mlproject.neural_net._dense_layer.DenseLayer] in the network
            and n is the number of data points.
        """
        self.activations.append(X)
        for layer in self.layers:
            X = layer.forward(X)
            self.activations.append(X)
            self.sums.append(layer.z)

        return X

    def predict(self, X):
        """Predict class labels for the given data.

        Parameters
        ----------
        X : 2d ndarray
            The data that we want to use to make predictions.

        Returns
        -------
        1d ndarray
            All predicted class labels with size n, where n is the number of data points.
        """
        probabilities = self.predict_proba(X)

        return np.array(
            [self.label[pred] for pred in np.argmax(probabilities, axis=1).astype(int)]
        )

    def predict_proba(self, X):
        """Predict class probabilities for the given data

        Parameters
        ----------
        X : 2d ndarray
            The data that we want to use to make predictions

        Returns
        -------
        2d ndarray
            All probabilites with size n x k, where n is the number of data points and k is the number classes
        """
        return self.forward(X)

    def fit(self, X, y, batches: int = 1, epochs: int = 1000, lr: float = 0.01):
        r"""The actual training of the network to the given data

        Parameters
        ----------
        X : 2d ndarray
            An $N \times P$ matrix of data points
            where n is the number of data points and p is the number of features.
        y : 1d ndarray
            $N \times 1$ vector of target class labels
        batches : int, optional
            The number of batches to use for training in each epoch,
            an integer indicating the number of splits to split the data into,
            by default $1$ which corresponds to training on the entire dataset
            in every epoch.
        epochs : int, optional
            The number of iterations to train for
        lr : float, optional
            The learning rate for gradient descent
        """

        self.X = X
        self.n, self.p = self.X.shape
        self.y = y
        self.learning_rate = lr

        unique_classes = np.unique(y)
        self.k = len(unique_classes)

        one_hot = OneHotEncoder(categories=[unique_classes])
        self.y_hot_encoded = one_hot.fit_transform(self.y).toarray()

        if self.layers[-1]._out_neurons() != self.k:
            raise ValueError(
                f"The number of neurons in the output layer, output_n: ({self.layers[-1].out_neurons()}) must be equal to the number of classes, k: ({self.k})"
            )
        if self.layers[0]._in_neurons() != self.X.shape[1]:
            raise ValueError(
                f"The number of neurons in the input layer, input_n: ({self.layers[0].in_neurons()}) must be equal to the number features in the dataset: ({self.X.shape[1]})"
            )

        # populate label-intcode dictionaries
        self.label = {k: unique_classes[k] for k in range(self.k)}
        self.intcode = {unique_classes[k]: k for k in range(self.k)}

        self.loss_history = []
        self.accuracy_history = []

        # get indices of every data point
        idxs = np.arange(self.n)

        with progress as pb:
            t1 = pb.add_task("[blue]Training", total=epochs)

            for epoch in range(epochs):

                # randomly shuffle the data --> split it into number of batches
                # here np.array_split returns an array of arrays of indices
                # of the different splits
                np.random.shuffle(idxs)
                batch_idxs = np.array_split(idxs, batches)

                for batch in batch_idxs:

                    X_batch = self.X[batch]
                    y_batch = self.y_hot_encoded[batch]

                    # compute the initial class probabilities by doing a single forward pass
                    # note: this should come 'automatically' from defining the last layer
                    # in the model as a layer with output_n = k with softmax activation
                    # where k is the number of classes.
                    init_probs = self.forward(X_batch)
                    if np.isnan(init_probs).any() or np.isinf(init_probs).any():
                        raise ValueError(
                            f"Unexpected value for init_probs, please try different parameters for either `batches`, `epocs` or `lr`"
                        )

                    # dividide by the number of data points in this specific batch to get the average loss.
                    loss = self.loss(y_batch, init_probs) / len(y_batch)
                    if np.isnan(loss) or np.isinf(loss):
                        raise ValueError(
                            f"Unexpected value for loss, please try different parameters for either `batches`, `epocs` or `lr`"
                        )

                    self._backward(y_batch)

                # add the latest loss to the history
                self.loss_history.append(loss)

                # predict with the current weights and biases on the whole data set
                batch_predict = self.predict(self.X)

                # calculate the accuracy score of the prediction
                train_accuracy = accuracy_score(self.y, batch_predict)

                # add accuracy to the history
                self.accuracy_history.append(train_accuracy)

                # update rich progress bar for each epoch
                pb.update(t1, advance=1)

                if progress.finished:
                    pb.update(t1, description="[bright_green]Training complete!")

    def _backward(self, y_batch):
        """Computes a single backward pass all the way through the network.
        as well as updating the weights and biases.

        Parameters
        ----------
        y_batch : 2d ndarray
            array of one-hot encoded ground_truth labels
        """

        delta = self.activations[-1] - y_batch

        grad_bias = delta.sum(0)

        grad_weight = self.activations[-2].T @ delta

        grad_biases, grad_weights = [], []
        grad_weights.append(grad_weight)
        grad_biases.append(grad_bias)

        for i in range(2, len(self.layers) + 1):
            layer = self.layers[-i + 1]
            dzda = delta @ layer.weights.T
            delta = dzda * leaky_relu_der(self.sums[-i])

            grad_bias = delta.sum(0)
            grad_weight = self.activations[-i - 1].T @ delta
            grad_weights.append(grad_weight)
            grad_biases.append(grad_bias)

        # reverse the gradient lists so we can index them normally.
        grad_biases_rev = list(reversed(grad_biases))
        grad_weights_rev = list(reversed(grad_weights))

        for i in range(0, len(self.layers)):
            self.layers[i].weights -= self.learning_rate * grad_weights_rev[i]
            self.layers[i].biases -= self.learning_rate * grad_biases_rev[i]

    def __str__(self):
        s = "\nNeuralNetworkClassifier \n"
        s += "--------------------------------\n"
        s += f"Loss function: {self.loss_str}\n\n"
        layers = [self.layers[i] for i in range(0, len(self.layers))]
        layers_neu = [
            f"\tInput: {i.input_n}, Output: {i.output_n} , Activation: {i.activation_function()}"
            for i in layers
        ]
        layer_num = 0
        for layer in layers_neu:
            if layer_num == 0:
                s += "Input layer: \n" + layer + "\n\n"
            elif layer_num == len(self.layers) - 1:
                s += f"Output layer: \n" + layer
            else:
                s += f"Layer: {layer_num}\n" + layer + "\n\n"
            layer_num += 1

        return s
