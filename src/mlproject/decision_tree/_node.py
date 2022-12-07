import numpy as np

class Node:
    """_summary_

    Parameters
    ----------
    feature : _type_, optional
        _description_, by default None
    threshold : _type_, optional
        _description_, by default None
    left : _type_, optional
        _description_, by default None
    right : _type_, optional
        _description_, by default None
    majority_class : _type_, optional
        _description_, by default None
    """
    def __init__(self, left=None, right=None, feature=None, threshold=None,*,majority_class=None):
        self.feature = feature
        self.threshold = threshold
        self.left, self.right = left, right
        self.majority_class = majority_class
        

    def is_leaf(self):
        return self.majority_class is not None