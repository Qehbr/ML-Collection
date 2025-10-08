import pandas as pd

from K_Means.kmeans import K_Means
from utils.utils import train_test_split

if __name__ == "__main__":
    data_name = "housing"

    data = pd.read_csv(f"Data/{data_name}.csv")

    if data_name == "breast_cancer":
        data.dropna(inplace=True, axis=1)
        X = data.drop(["diagnosis", "id"], axis=1)
        y = data["diagnosis"].map({"M": 1, "B": 0})
        task = "classification"

    elif data_name == "housing":
        data.dropna(inplace=True)
        X = data.drop(["median_house_value", "ocean_proximity"], axis=1)
        y = data["median_house_value"]
        task = "regression"

    else:
        raise ValueError("Correct data_name should be specified.")

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = K_Means(k=10, max_iterations=100)
    model.fit(X_train)

    y_pred = model.predict(X_test)
