import numpy as np


class LinearRegression:
    def __init__(self, learning_rate: float, n_iterations: int) -> None:
        self.learning_rate: float = learning_rate
        self.n_iterations: int = n_iterations

        self.weights: np.ndarray | None = None
        self.bias: np.floating = np.float64(0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        rows, cols = X.shape

        self.weights = np.random.randn(cols)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        for i in range(self.n_iterations):
            pred = (X @ self.weights) + self.bias
            delta = y - pred
            delta_squared = delta**2

            mse = np.mean(delta_squared)
            ss_res = np.sum(delta_squared)
            r_squared = 1 - (ss_res / ss_tot)

            # mse = L = (y-y^)^2
            # y^ = wx + b
            # dL/dw = dL/dy^ * dy^/dw = -2*(y-y^) * x = -2x*(y-y^)
            # dL/db = dL/dy^ * dy^/db = -2*(y-y^)

            dw = -2 * ((X.T @ delta) / rows)
            db = np.mean(-2 * delta)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            print(f"Epoch {i + 1}, MSE Loss: {mse}, R^2: {r_squared}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X @ self.weights) + self.bias
