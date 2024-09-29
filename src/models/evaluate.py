import click
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score

target = "totalFare"


@click.command()
@click.argument("model", type=click.Path(exists=True))
@click.argument("input", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
def evaluate_model(model, input, output):
    df = pd.read_csv(input)
    x_test = df.drop(target, axis=1)
    y_test = df[target]

    cb_model = CatBoostRegressor().load_model(model)
    y_pred = cb_model.predict(x_test)

    score = pd.DataFrame(
        {
            "r2": r2_score(y_test, y_pred),
            "mape": mean_absolute_percentage_error(y_test, y_pred),
        },
        index=[0],
    )

    score.to_csv(output, index=False)


if __name__ == "__main__":
    evaluate_model()
