import numpy as np
from ._dense_layer import DenseLayer
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from typing import Union



# is this allowed???
from sklearn.metrics import accuracy_score 

progress = Progress(TextColumn("[progress.description]{task.description}"),MofNCompleteColumn(), BarColumn(),
        TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn())



class NeuralNetworkClassifier:
    """NeuralNetworkClassifier

    Feed Forward Neural Network Classifier with however many
    dense layers (fully connected layers) of class DenseLayer each with own
    activation function and finally the loss function can be chosen.

    Parameters
    ----------
    layers : list, optional
        A list of class DenseLayer, by default empty list

    loss : str, optional
        The loss function to be used, by default 'cross_entropy'

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
    def __init__(self, layers = [], loss = 'cross_entropy'):
        
        self.X = None
        self.n, self.p = None, None
        self.y = None
        self.k = None
        self.layers = layers

        if loss == 'cross_entropy':
            self.loss = cross_entropy_loss
        else:
            raise NotImplementedError(f"{loss} not implemented yet. Choose from ['cross_entropy']") 

    def add(self, layer):
        """Add a new layer to the network, after the current layer.

        Parameters
        ----------
        layer : DenseLayer
            Fully connected layer.
        """
        self.layers.append(layer)

    def forward(self, X):
        """Compute a single forward pass of the network.

        Parameters
        ----------
        X : 2d ndarray
            The data to use for the forward pass.
            Must be of size n x input_n 
            where input_n must come from the first DenseLayer in the network.

        Returns
        -------
        2d ndarray
            An n x output_n array
            where output_n corresponds to the output_n of the last DenseLayer in the network
            and n is the number of data points.
        """
        for layer in self.layers:
            X = layer.forward(X)

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

        return np.array([self.label[pred] for pred in np.argmax(probabilities, axis=1).astype(int)])

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

    def fit(self, X, y, batches: Union[float,int] = 1, epochs:int = 1000, lr:float = 0.01):
        """The actual training of the network to the given data

        Parameters
        ----------
        X : 2d ndarray
            An n x p matrix of data points
            where n is the number of data points and p is the number of features.

        y : 1d ndarray
            n x 1 vector of target class labels

        batches : float or int, optional
            The number of batches to use for training in each epoch
            can either be a proportion of the dataset i.e., 0.5 for 50 % of the dataset
            or an integer indicating the number of splits to split the data into,
            by default 1 which corresponds to training on the entire dataset
            in every epoch.

        epochs : int, optional
            The number of iterations to train for, by default 1000

        lr : float, optional
            The learning rate for gradient descent, by default 0.01
        """        
        
        self.X = X
        self.n, self.p = self.X.shape
        self.y = y
        self.k = len(np.unique(self.y))

        # populate label-intcode dictionaries
        unique_classes = np.unique(y)

        self.label = {k: unique_classes[k] for k in range(self.k)}
        self.intcode = {unique_classes[k]:k for k in range(self.k)}

        """need check here for if batch is float or int and handle accordingly"""

        
        self.loss_history = []
        self.accuracy_history = []
        self._batch_num = 0

        # get indices of every data point
        idxs = np.arange(self.n)

        with progress as pb:
            t2 = pb.add_task('Epoch', total=epochs) # outer
            t1 = pb.add_task('Batch', total=batches) # inner

            for epoch in range(epochs):
                
                # randomly shuffle the data --> split it into number of batches 
                    # here np.array_split returns an array of arrays of indices 
                    # of the different splits
                np.random.shuffle(idxs)
                batch_idxs = np.array_split(idxs, batches)

                for batch in batch_idxs:
                    
                    X_batch = self.X[batch]
                    y_batch = self.y[batch]

                    # compute the initial class probabilities by doing a single forward pass
                    # and softmax to get a single prediction for each data point
                        # note: this should come 'automatically' from defining the last layer
                        # in the model as a layer with output_n = k with softmax activation
                        # where k is the number of classes.
                    # maybe a better name is 'init_guess' ?
                    init_probs = self.forward(X_batch)
                    loss = self.loss(y_batch, init_probs)

                    # DO BACKWARD PASS
                        # fancy stuff here
                    # UPDATE WEIGHTS
                        # less fancy here

                    
                    # update rich progress bar for each batch
                    pb.update(task_id=t1, advance=1)

                        
                # reset the progress bar after each batch
                pb.update(task_id=t1, completed=0)

                # add the latest loss to the history
                self.loss_history.append(loss)

                # predict with the current weights and biases on the whole data set
                batch_predict = self.predict(self.X)
                
                train_accuracy = accuracy_score(self.y, batch_predict)

                # add accuracy to the history
                self.accuracy_history.append(train_accuracy)

                # update rich progress bar for each epoch
                pb.update(task_id=t2, advance=1)
                
                
                


    
def cross_entropy_loss(y_true, y_pred):
    """Compute the categorical cross entropy loss
    from the given true labels and predicted labels.

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
    return -np.sum(np.log(y_pred) * y_true)

