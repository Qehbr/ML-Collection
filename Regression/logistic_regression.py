import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate: float, epochs: int) -> None:
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs

        self.weights: None | np.ndarray = None
        self.bias: np.floating = np.float64(0)

    def _sigmoid(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_instances, n_features = X.shape
        self.weights = np.random.randn(n_features)

        for i in range(self.epochs):
            y_pred = self._sigmoid(X @ self.weights + self.bias)
            y_pred_class = (y_pred >= 0.5).astype(int)

            # accuracy
            correct = np.sum(y_pred_class == y)
            accuracy = correct / n_instances

            # BCE Loss
            y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
            bce_loss = -np.mean(
                y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped)
            )

            delta = y_pred - y
            dw = (X.T @ delta) / n_instances
            db = np.mean(delta)

            self.weights -= dw
            self.bias -= db

            print(f"Epoch {i + 1}, BCE Loss: {bce_loss}, Accuracy: {accuracy}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.weights + self.bias)
