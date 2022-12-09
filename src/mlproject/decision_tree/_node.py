import numpy as np

class Node:
    """Node object for building a decision tree.

    Parameters
    ----------
    feature : int index, optional
        index of the best feature for splitting this Node, by default None
    threshold : float, optional
        the threshold for the best split of the data, by default None
    left : Node, optional
        the left child of this Node also of class Node, by default None
    right : Node, optional
        the right child of this Node also of class Node, by default None
    majority_class : int, optional
        The majority class in this node, only if this Node is a leaf, by default None
    class_probs : 1d ndarray, optional
        An array of class probabilities for this node, only if this Node is a leaf, by default None
    """
    def __init__(self, left=None, right=None, feature=None, threshold=None,*,majority_class=None,class_probs=None):
        self.feature = feature
        self.threshold = threshold
        self.left, self.right = left, right
        self.majority_class = majority_class
        self.class_probs = class_probs
        

    def is_leaf(self):
        """Returns True if this Node is a leaf node, otherwise False

        Returns
        -------
        bool
            True if this Node is a leaf node, otherwise False
        """        
        return self.majority_class is not None