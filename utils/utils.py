import numpy as np
import pandas as pd


def train_test_split(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    ratio: float = 0.8,
    random_state: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = X.shape[0]

    if random_state != -1:
        rng = np.random.default_rng(seed=random_state)
        shuffled_indices = rng.permutation(rows)
    else:
        shuffled_indices = np.random.permutation(rows)
    X_shuffled, y_shuffled = X.iloc[shuffled_indices], y.iloc[shuffled_indices]

    split_idx = int(rows * ratio)

    X_train = X_shuffled.iloc[:split_idx]
    X_test = X_shuffled.iloc[split_idx:]
    y_train = y_shuffled.iloc[:split_idx]
    y_test = y_shuffled.iloc[split_idx:]
    return (
        X_train.to_numpy(),
        X_test.to_numpy(),
        y_train.to_numpy(),
        y_test.to_numpy(),
    )


def calculate_mse_loss(y_pred: np.ndarray, y_act: np.ndarray) -> np.floating:
    return np.mean((y_pred - y_act) ** 2)


def calculate_accuracy(y_pred: np.ndarray, y_act: np.ndarray) -> np.floating:
    y_pred_class = (y_pred >= 0.5).astype(int)
    return np.mean(y_pred_class == y_act)
