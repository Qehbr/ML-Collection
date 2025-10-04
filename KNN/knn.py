import numpy as np
from typing import Literal


class KNN:
    def __init__(self, k: int):
        self.k: int = k
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def _euclidian_distance(self, X_predict: np.ndarray) -> np.ndarray:
        if self.X_train is None:
            raise ValueError("Model is not fitted yet. Call fit().")

        differences = np.expand_dims(X_predict, axis=1) - np.expand_dims(
            self.X_train, axis=0
        )
        return np.sqrt(np.sum(differences**2, axis=2))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train

    def predict(
        self, X_predict: np.ndarray, task: Literal["classification", "regression"]
    ) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model is not fitted yet. Call fit().")

        distances = self._euclidian_distance(X_predict)
        k_nearest_neighbors_indices = np.argpartition(distances, kth=self.k, axis=1)[
            :, : self.k
        ]
        neighbors_labels = self.y_train[k_nearest_neighbors_indices]

        if task == "classification":
            return np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=1, arr=neighbors_labels
            )
        if task == "regression":
            return np.mean(neighbors_labels, axis=1)
