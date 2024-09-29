import click
import pandas as pd
from catboost import CatBoostRegressor

target = "totalFare"


@click.command()
@click.argument("in_train", type=click.Path(exists=True))
@click.argument("in_test", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
def train_model(in_train, in_test, output):
    df_train = pd.read_csv(in_train)
    df_test = pd.read_csv(in_test)

    x_train = df_train.drop(target, axis=1)
    y_train = df_train[target]
    x_test = df_test.drop(target, axis=1)
    y_test = df_test[target]

    cb_model = CatBoostRegressor(
        cat_features=("startingAirport", "destinationAirport")
    ).fit(x_train, y_train, eval_set=(x_test, y_test))

    cb_model.save_model(output)


if __name__ == "__main__":
    train_model()
