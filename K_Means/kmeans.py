import numpy as np


class K_Means:
    def __init__(self, k: int, max_iterations: int) -> None:
        self.k: int = k
        self.max_iterations: int = max_iterations
        self.centroids: np.ndarray | None = None

    def _euclidian_distance(self, X: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            raise ValueError("Model is not fitter yet. Use fit().")
        # X (n_instances, n_features)
        # centroids (k, n_features)
        differences = np.expand_dims(X, axis=1) - np.expand_dims(
            self.centroids, axis=0
        )  # (n_instances, k, n_features)
        return np.sqrt(np.sum(differences**2, axis=2))

    def fit(self, X: np.ndarray):
        n_instances = X.shape[0]

        random_indices = np.random.choice(n_instances, size=self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iterations):
            distances_to_centroids = self._euclidian_distance(X)
            cluster_labels = np.argmin(distances_to_centroids, axis=1)

            new_centroids = np.zeros_like(self.centroids)
            for cluster_i in range(self.k):
                points_in_cluster = X[cluster_labels == cluster_i]
                new_centroids[cluster_i] = np.mean(points_in_cluster, axis=0)

            if np.allclose(self.centroids, new_centroids):
                print(f"Converged at iteration {i + 1}")
                break

            self.centroids = new_centroids

    def predict(self, X: np.ndarray):
        distances_to_centroids = self._euclidian_distance(X)
        return np.argmin(distances_to_centroids, axis=1)
