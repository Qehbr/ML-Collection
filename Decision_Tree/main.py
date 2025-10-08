import pandas as pd

from Decision_Tree.dt import DecisionTree
from utils.utils import calculate_accuracy, calculate_mse_loss, train_test_split

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = DecisionTree(max_depth=10, min_samples_split=10, criterion="mse", task=task)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if task == "regression":
        mse_loss = calculate_mse_loss(y_pred, y_test)
        print(f"Final MSE loss for test set: {mse_loss}")

    if task == "classification":
        accuracy = calculate_accuracy(y_pred, y_test)
        print(f"Final accuracy for test set: {accuracy}")
