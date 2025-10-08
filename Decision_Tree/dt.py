from __future__ import annotations

from collections import Counter
from typing import Literal

import numpy as np


class _DecisionNode:
    def __init__(
        self,
        feature_index: int,
        threshold: np.floating,
        left: _DecisionNode | _LeafNode,
        right: _DecisionNode | _LeafNode,
    ) -> None:
        self.feature_index: int = feature_index
        self.threshold: np.floating = threshold
        self.left: _DecisionNode | _LeafNode = left
        self.right: _DecisionNode | _LeafNode = right


class _LeafNode:
    def __init__(self, value: np.floating | str) -> None:
        self.value: np.floating | str = value


class DecisionTree:
    def __init__(
        self,
        max_depth: int,
        min_samples_split: int,
        criterion: Literal["gini", "entropy", "mse"],
        task: Literal["classification", "regression"],
    ) -> None:
        if task == "regression" and criterion != "mse":
            raise ValueError("For regression task 'mse' criterion must be used")
        if task == "classification" and criterion == "mse":
            raise ValueError("For classification task 'mse' criterion cannot be used")

        self.max_depth: int = max_depth
        self.min_samples_split: int = min_samples_split
        self.criterion: Literal["gini", "entropy", "mse"] = criterion
        self.task: Literal["classification", "regression"] = task
        self.root: _DecisionNode | _LeafNode | None = None

    def _calculate_entropy(self, y: np.ndarray) -> np.floating:
        labels_counter = Counter(y)
        total = labels_counter.total()

        probabilities = np.array(list(labels_counter.values())) / total
        entropy = -(np.sum(probabilities * np.log2(probabilities)))

        return entropy

    def _calculate_gini(self, y: np.ndarray) -> np.floating:
        labels_counter = Counter(y)
        total = labels_counter.total()

        probabilities = np.array(list(labels_counter.values())) / total
        gini = 1 - np.sum(probabilities**2)

        return gini

    def _calculate_mse(self, y: np.ndarray) -> np.floating:
        return np.mean(np.sum((y - np.mean(y)) ** 2))

    def _grow_tree(
        self, X: np.ndarray, y: np.ndarray, cur_depth: int
    ) -> _LeafNode | _DecisionNode:
        n_instances, n_features = X.shape
        if (
            (cur_depth >= self.max_depth)
            or (n_instances < self.min_samples_split)
            or (len(np.unique(y)) == 1)
        ):
            if self.task == "classification":
                return _LeafNode(Counter(y).most_common(1)[0][0])
            elif self.task == "regression":
                return _LeafNode(np.mean(y))
            else:
                raise ValueError(f"Task {self.task} is not supported")

        if self.criterion == "gini":
            calculate_criterion = self._calculate_gini
        elif self.criterion == "entropy":
            calculate_criterion = self._calculate_entropy
        elif self.criterion == "mse":
            calculate_criterion = self._calculate_mse
        else:
            raise ValueError(f"Unknown Criterion {self.criterion}")

        best_split_gain: np.floating = np.float64(0)
        initial_criterion = calculate_criterion(y)
        best_feature_to_split: int | None = None
        best_threshold: np.floating | None = None

        for feature in range(n_features):
            sorted_mask = np.argsort(X[:, feature])
            sorted_X_feature = X[:, feature][sorted_mask]
            sorted_y = y[sorted_mask]

            for i in range(n_instances - 1):
                if sorted_X_feature[i] == sorted_X_feature[i + 1]:
                    continue

                threshold = (sorted_X_feature[i] + sorted_X_feature[i + 1]) / 2
                left_candidate = sorted_X_feature[: i + 1]
                right_candidate = sorted_X_feature[i + 1 :]

                left_candidate_labels = sorted_y[: i + 1]
                right_candidate_labels = sorted_y[i + 1 :]

                left_candidate_total, right_candidate_total = (
                    left_candidate.shape[0],
                    right_candidate.shape[0],
                )

                left_criterion = calculate_criterion(left_candidate_labels)
                right_criterion = calculate_criterion(right_candidate_labels)

                gain = initial_criterion - (
                    left_criterion * (left_candidate_total / n_instances)
                    + right_criterion * (right_candidate_total / n_instances)
                )

                if gain > best_split_gain:
                    best_split_gain = gain
                    best_feature_to_split = feature
                    best_threshold = threshold

        if best_feature_to_split is None or best_threshold is None:
            raise ValueError("Best split was not found")

        if not best_split_gain > 0:
            if self.task == "classification":
                return _LeafNode(Counter(y).most_common(1)[0][0])
            elif self.task == "regression":
                return _LeafNode(np.mean(y))
            else:
                raise ValueError(f"Task {self.task} is not supported")

        best_left_mask = X[:, best_feature_to_split] < best_threshold
        best_right_mask = X[:, best_feature_to_split] >= best_threshold
        best_left = X[best_left_mask]
        best_right = X[best_right_mask]
        best_left_labels = y[best_left_mask]
        best_right_labels = y[best_right_mask]

        left_node = self._grow_tree(best_left, best_left_labels, cur_depth + 1)
        right_node = self._grow_tree(best_right, best_right_labels, cur_depth + 1)
        return _DecisionNode(
            best_feature_to_split, best_threshold, left_node, right_node
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.root = self._grow_tree(X, y, 0)

    def _traverse_tree(
        self, row: np.ndarray, node: _DecisionNode | _LeafNode
    ) -> np.floating | str:
        if isinstance(node, _LeafNode):
            return node.value

        if row[node.feature_index] < node.threshold:
            return self._traverse_tree(row, node.left)
        return self._traverse_tree(row, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise ValueError("The model is not fitted yet. Call fit() method")
        predictions = []
        for row in range(X.shape[0]):
            prediction = self._traverse_tree(X[row, :], self.root)
            predictions.append(prediction)

        return np.array(predictions)
