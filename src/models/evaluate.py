import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score

target = "totalFare"


def evaluate_model():
    df = pd.read_csv("./data/processed/test.csv")
    x_test = df.drop(target, axis=1)
    y_test = df[target]

    cb_model = CatBoostRegressor().load_model("./models/model.cbm")
    y_pred = cb_model.predict(x_test)

    score = pd.DataFrame(
        {
            "r2": r2_score(y_test, y_pred),
            "mape": mean_absolute_percentage_error(y_test, y_pred),
        },
        index=[0],
    )

    score.to_csv("./reports/score.csv", index=False)


if __name__ == "__main__":
    evaluate_model()
