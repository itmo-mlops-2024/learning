import pandas as pd
from catboost import CatBoostRegressor

target = "totalFare"


def train_model():
    df_train = pd.read_csv("./data/processed/train.csv")
    df_test = pd.read_csv("./data/processed/test.csv")

    x_train = df_train.drop(target, axis=1)
    y_train = df_train[target]
    x_test = df_test.drop(target, axis=1)
    y_test = df_test[target]

    cb_model = CatBoostRegressor(
        cat_features=("startingAirport", "destinationAirport")
    ).fit(x_train, y_train, eval_set=(x_test, y_test))

    cb_model.save_model("./models/model.cbm")


if __name__ == "__main__":
    train_model()
