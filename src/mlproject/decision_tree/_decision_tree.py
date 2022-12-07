import numpy as np
from mlproject.decision_tree._impurity import gini_impurity, entropy_impurity
from mlproject.decision_tree._node import Node

class DecisionTreeClassifier:
    """Decision Tree Classifier

    Simple decision tree classifier with user specific impurity, max depth and 
    minimum number of samples in leaf nodes.

    Parameters
    ----------
    criterion : str, optional
        The impurity criterion to use when splitting nodes, by default 'gini'
    max_depth : int, optional
        The maximum depth of the decision tree, by default 100
    min_samples_in_leaf : int, optional
        The minimum number of samples that need to be in a leaf, by default 2
    """
    def __init__(self, criterion='gini', max_depth=100, min_samples_in_leaf=2):

        self.max_depth = max_depth
        self.min_samples_in_leaf = min_samples_in_leaf
        self.root = None

        if criterion.lower() == 'gini':
            self.criterion = gini_impurity
        elif criterion.lower() == 'entropy':
            self.criterion = entropy_impurity

    def fit(self, X, y):

        # calls the _grow function
        assert NotImplementedError("not implemented yet")

    def predict(self, X):

        # calls the _traverse function
        assert NotImplementedError("not implemented yet")
    
    def predict_proba(self, X):

        assert NotImplementedError("not implemented yet")
        
    def _grow(self, X, y, cur_depth=0):
        
        self.n, self.p = self.X.shape
        node_unique_classes = np.unique(y)
        self.node_k = len(node_unique_classes)
        
        if (cur_depth >= self.max_depth or self.node_k == 1 or self.n <= self.min_samples_in_leaf):
            most_common = self._most_common_label(y)
            return Node(majority_class=most_common)
        
        cur_depth += 1

        best_feature, best_threshold = self._best_split(X, y)

        left_idxs = np.argwhere(best_feature >= best_threshold).flatten()
        right_idxs = np.argwhere(best_feature < best_threshold).flatten()

        left = self._grow(X[left_idxs, :], y[left_idxs], cur_depth)
        right = self._grow(X[right_idxs, :], y[right_idxs], cur_depth)
        return Node(left, right, best_feature, best_threshold)


    def _best_split(self, X, y):
        best_gain = -np.inf

        for feat_idx in range(X.shape[1]):
            feature_col = X[:, feat_idx]
            possible_splits = np.unique(feature_col)

            for split in possible_splits:
                cur_gain = self._information_gain(y, feature_col, split)

                if cur_gain > best_gain:
                    best_gain = cur_gain
                    split_idx = feat_idx
                    split_thresh = split

        return split_idx, split_thresh

    def _information_gain(self, y, feature_col, split_thresh):

        parent_impurity = self.criterion(y)

        left_idxs = np.argwhere(feature_col >= split_thresh).flatten()
        right_idxs = np.argwhere(feature_col < split_thresh).flatten()

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(y)
        left_prob = len(left_idxs) / n
        right_prob = len(right_idxs) / n

        left_impurity = self.criterion(y[left_idxs])
        right_impurity = self.criterion(y[right_idxs])

        weighted_impurity = (left_prob * left_impurity + right_prob * right_impurity)

        information_gain = parent_impurity - weighted_impurity
        return information_gain
 
    def _traverse(self, X, node):

        # traverses the tree until it reaches a leaf node and return the majority
        # class of that node.
        assert NotImplementedError("not implemented yet")

    def _most_common_label(self, y):
        uniques, counts = np.unique(y, return_counts=True)
        label_counts = dict(zip(uniques, counts))
        sorted_keys = list(sorted(label_counts.items(), key = lambda x : x[1], reverse=True))
        return sorted_keys[0][0]
