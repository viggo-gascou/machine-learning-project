import os
import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../..'))

def data_loader(raw=True, scaled=False, pca=False):
    """Loads the fashion_mnist training and test data from the data directory.
    The class returns four numpy arrays containing the training and test data
    respectively. 
    If specified it can also return the standard scaled version of the data or 
    the first 10 principal components of the data.

    Raw:
        Training data:                
            `X_train`: (10000 x 784)
            `y_train`: (10000 x 1)
        Test data:  
            `X_test`:  (5000 x 784)
            `y_test`:  (5000 x 1)
    Scaled:
        Training data:                
            `X_train`: (10000 x 784)
            `y_train`: (10000 x 1)
        Test data:  
            `X_test`:  (5000 x 784)
            `y_test`:  (5000 x 1)
    PCA:
        Training data:                
            `X_train`: (10000 x 10)
            `y_train`: (10000 x 1)
        Test data:  
            `X_test`:  (5000 x 10)
            `y_test`:  (5000 x 1)

    Returns
    -------
    2d ndarrays
        numpy data arrays of shape in order of how it is returned.
    """
    if raw:
        X_train, y_train = np.hsplit(np.load(f"{ROOT_DIR}/data/fashion_train.npy"),[-1])
        X_test, y_test = np.hsplit(np.load(f"{ROOT_DIR}/data/fashion_test.npy"),[-1])
    elif scaled and not raw:
        X_train, y_train = np.hsplit(np.load(f"{ROOT_DIR}/data/fashion_train_scaled.npy"),[-1])
        X_test, y_test = np.hsplit(np.load(f"{ROOT_DIR}/data/fashion_test_scaled.npy"),[-1])
    elif pca and not raw and not scaled:
        X_train, y_train = np.hsplit(np.load(f"{ROOT_DIR}/data/fashion_train_pca.npy"),[-1])
        X_test, y_test = np.hsplit(np.load(f"{ROOT_DIR}/data/fashion_test_pca.npy"),[-1])
        y_train, y_test = y_train.astype(int), y_test.astype(int)
    return X_train, X_test, y_train, y_test









