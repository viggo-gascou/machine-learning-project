import numpy as np

def data_loader():
    """Loads the fashion_mnist training and test data from the data directory.
    The class returns four numpy arrays containing the training and test data
    respectively. 

    Training data:                
        `X_train`: (10000 x 784)
        `y_train`: (10000 x 1)
    Test data:  
        `X_test`:  (5000 x 784)
        `y_test`:  (5000 x 1)

    Returns
    -------
    2d ndarrays
        numpy data arrays of shape in order of how it is returned:
            (10000 x 784), (5000 x 784), (10000 x 1) & (5000 x 1)
    """
   
    X_train, y_train = np.hsplit(np.load("data/fashion_train.npy"),[-1])
    X_test, y_test = np.hsplit(np.load("data/fashion_test.npy"),[-1])
    return X_train, X_test, y_train, y_test









