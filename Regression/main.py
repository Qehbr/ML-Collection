import pandas as pd
from Regression.logistic_regression import LogisticRegression
from utils.utils import calculate_accuracy, train_test_split

if __name__ == "__main__":
    data = pd.read_csv("../Data/breast_cancer.csv")
    print(data.head())
    data.dropna(inplace=True, axis=1)

    # X = data.drop(["median_house_value", "ocean_proximity"], axis=1)
    # y = data["median_house_value"]

    X = data.drop(["diagnosis", "id"], axis=1)
    y = data["diagnosis"].replace({"M": 1, "B": 0})

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = LogisticRegression(0.1, 100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # mse_loss = calculate_mse_loss(y_pred, y_test)
    # print(f"Final MSE loss for test set: {mse_loss}")

    accuracy = calculate_accuracy(y_pred, y_test)
    print(f"Final accuracy for test set: {accuracy}")
