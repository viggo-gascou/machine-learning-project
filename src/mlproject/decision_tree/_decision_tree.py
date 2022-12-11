import numpy as np
from mlproject.decision_tree._impurity import gini_impurity, entropy_impurity
from mlproject.decision_tree._node import Node

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

    def __init__(self, criterion="gini", max_depth=100, min_samples_in_leaf=2):

        self.max_depth = max_depth
        self.min_samples_in_leaf = min_samples_in_leaf
        self.root = None

        if criterion.lower() == "gini":
            self.criterion = gini_impurity
        elif criterion.lower() == "entropy":
            self.criterion = entropy_impurity

    def fit(self, X, y):
        """Fit the decision tree to the given data

        Parameters
        ----------
        X : 2d ndarray
            The data to be used for fitting the decision tree
        y : 2d ndarray
            An array of the true labels for the data points
        """
        with progress as pb:
            t1 = pb.add_task("[blue]Training", total=1)

            self.root = self._grow(X, y)
            pb.update(t1, advance=1)
            if progress.finished:
                pb.update(t1, description="[bright_green]Training complete!")

    def predict(self, X):
        """Predict class labels for the given data.

        For all data points in the dataset traverse the decision tree until it reaches a leaf node
        and return the majority class of that leaf node.

        Parameters
        ----------
        X : 2d ndarray
            The data that we want to use to make predictions.

        Returns
        -------
        1d ndarray
            All predicted class labels with size n, where n is the number of data points.
        """
        return np.array([self._traverse(datapoint, self.root) for datapoint in X])

    def predict_proba(self, X):
        """Predict class probabilities for the given data

        For all data points in the dataset traverse the decision tree until it reaches a leaf node
        and return the class probabilities of that leaf node.

        Parameters
        ----------
        X : 2d ndarray
            The data that we want to use to make predictions

        Returns
        -------
        2d ndarray
            All probabilites with size n x k, where n is the number of data points and k is the number classes
        """

        return np.array(
            [self._traverse(datapoint, self.root, prob=True) for datapoint in X]
        )

    def _grow(self, X, y, cur_depth=0):
        """Grows a decision tree from the given data.
        This is the part that is doing the actual fitting of the decision tree.

        Parameters
        ----------
        X : 2d ndarray
            The data to use when growing the decision tree
        y : 2d ndarray
            array of the true class labels
        cur_depth : int, optional
            The current depth of the decision tree, by default 0

        Returns
        -------
        Node
            A new node of class Node with new left and right children.
        """

        self.n, self.p = X.shape
        node_unique_classes = np.unique(y)
        self.node_k = len(node_unique_classes)

        if (
            cur_depth >= self.max_depth
            or self.n < self.min_samples_in_leaf
            or self.node_k == 1
        ):

            most_common = self._most_common_label(y, prob=False)
            class_probs = self._most_common_label(y, prob=True)
            return Node(majority_class=most_common, class_probs=class_probs)

        cur_depth += 1

        best_feature, best_threshold = self._best_split(X, y)

        left_idxs = np.argwhere(X[:, best_feature] <= best_threshold).flatten()
        right_idxs = np.argwhere(X[:, best_feature] > best_threshold).flatten()

        left = self._grow(X[left_idxs, :], y[left_idxs], cur_depth)
        right = self._grow(X[right_idxs, :], y[right_idxs], cur_depth)

        return Node(left, right, best_feature, best_threshold)

    def _best_split(self, X, y):
        """Calculates the best split of a node with the given data points

        Parameters
        ----------
        X : 2d ndarray
            The data points to consider for splitting this node
        y : 2d ndarray
            The true labels to consider for splitting this node

        Returns
        -------
        tuple
            A tuple containing the best index and threshold for the split
        """
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
        """Calculates the information gain of a node with the given data labels

        Parameters
        ----------
        y : 2d ndarray
            array of true labels for this node
        feature_col : 2d ndarray
            Column of dataset containing the data points of the best feature for this split
        split_thresh : float or int
            the threshold for the best split of the data

        Returns
        -------
        float
            The information gain from this node compared to it's parent
        """

        parent_impurity = self.criterion(y)

        left_idxs = np.argwhere(feature_col <= split_thresh).flatten()
        right_idxs = np.argwhere(feature_col > split_thresh).flatten()

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        left_prob = len(left_idxs) / n
        right_prob = len(right_idxs) / n

        left_impurity = self.criterion(y[left_idxs])
        right_impurity = self.criterion(y[right_idxs])

        weighted_impurity = left_prob * left_impurity + right_prob * right_impurity

        information_gain = parent_impurity - weighted_impurity
        return information_gain

    def _traverse(self, X, node, prob=False):
        """Traverses the tree until it reaches a leaf node and returns either the majority
        class of that node or the class probabilities if prob is True.

        Parameters
        ----------
        X : 2d ndarray
            The data points to use for traversing the tree
        node : Node
            The node to start the traversal from.
        prob : bool, optional
            used to specify whether or not to return class probabilities, by default False

        Returns
        -------
        int or bool
            Depending on `prob` it either returns the majority class or the class probabilities
        """

        if node.is_leaf():
            if prob:
                return np.argmax(node.class_probs)
            return node.majority_class

        if X[node.feature] <= node.threshold:
            return self._traverse(X, node.left)

        elif X[node.feature] > node.threshold:
            return self._traverse(X, node.right)

    def _most_common_label(self, y, prob=False):
        """Calculates the most common label of a leaf node or the class probabilities

        Parameters
        ----------
        y : 2d ndarray
            Array of true labels for this particular node
        prob : bool, optional
            used to specify whether or not to return class probabilities, by default False

        Returns
        -------
        int or bool
            Depending on `prob` it either returns the majority class or the class probabilities
        """

        uniques, counts = np.unique(y, return_counts=True)
        label_counts = dict(zip(uniques, counts))
        sorted_keys = list(
            sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        )

        if prob:
            n = np.sum(counts)
            return np.array([label_counts[i] / n for i in uniques])

        return sorted_keys[0][0]
