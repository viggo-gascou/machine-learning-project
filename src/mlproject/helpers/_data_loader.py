import os
import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../..'))

def data_loader(raw=True, scaled=False, pca=False):
    r"""Loads the fashion_mnist training and test data from the data directory.
    
    The function returns four numpy arrays containing the training and test data
    respectively. 
    
    If specified it can also return the standard scaled version of the data or 
    the first 10 principal components of the data.
    
    The different dimensions of the returned data is below:

    |              |          Raw          |         Scaled        |          PCA         |
    |:------------:|:---------------------:|:---------------------:|:--------------------:|
    | **Training** |                       |                       |                      |
    |      $X$     | $(10.000 \times 784)$ | $(10.000 \times 784)$ | $(10.000 \times 10)$ |
    |      $Y$     |  $(10.000 \times 1)$  |  $(10.000 \times 1)$  |  $(10.000 \times 1)$ |
    |   **Test**   |                       |                       |                      |
    |      $X$     |  $(5.000 \times 784)$ |  $(5.000 \times 784)$ |  $(5.000 \times 10)$ |
    |      $Y$     |   $(5.000 \times 1)$  |   $(5.000 \times 1)$  |  $(5.000 \times 1)$  |

    Returns
    -------
    2d ndarrays
        numpy data arrays in the order X_train, X_test, y_train, y_test.
    """
    if raw:
        X_train, y_train = np.hsplit(np.load(f"{ROOT_DIR}/data/fashion_train.npy"),[-1])
        X_test, y_test = np.hsplit(np.load(f"{ROOT_DIR}/data/fashion_test.npy"),[-1])

    elif scaled and not raw:
        X_train, y_train = np.hsplit(np.load(f"{ROOT_DIR}/data/fashion_train_scaled.npy"),[-1])
        X_test, y_test = np.hsplit(np.load(f"{ROOT_DIR}/data/fashion_test_scaled.npy"),[-1])
        # converting the y_labels back to integers from floats to avoid issues
        y_train, y_test = y_train.astype(int), y_test.astype(int)
        
    elif pca and not raw and not scaled:
        X_train, y_train = np.hsplit(np.load(f"{ROOT_DIR}/data/fashion_train_pca.npy"),[-1])
        X_test, y_test = np.hsplit(np.load(f"{ROOT_DIR}/data/fashion_test_pca.npy"),[-1])
        # converting the y_labels back to integers from floats to avoid issues
        y_train, y_test = y_train.astype(int), y_test.astype(int)

    return X_train, X_test, y_train, y_test









