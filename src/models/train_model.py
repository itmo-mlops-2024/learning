import os

import click
import mlflow
import pandas as pd
from catboost import CatBoostRegressor
from dotenv import load_dotenv
from mlflow.catboost import log_model
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_absolute_percentage_error, r2_score

load_dotenv()
remote_server_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(remote_server_uri)

target = "totalFare"


@click.command()
@click.argument("in_train", type=click.Path(exists=True))
@click.argument("in_test", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
def train_model(in_train, in_test, output):
    with mlflow.start_run():
        mlflow.get_artifact_uri()

        df_train = pd.read_csv(in_train)
        df_test = pd.read_csv(in_test)

        x_train = df_train.drop(target, axis=1)
        y_train = df_train[target]
        x_test = df_test.drop(target, axis=1)
        y_test = df_test[target]

        params = {
            'cat_features': ("startingAirport", "destinationAirport")
        }
        cb_model = CatBoostRegressor(
            **params
        ).fit(x_train, y_train, eval_set=(x_test, y_test))

        y_pred = cb_model.predict(x_test)
        score = {
            "r2": r2_score(y_test, y_pred),
            "mape": mean_absolute_percentage_error(y_test, y_pred),
        }
        signature = infer_signature(x_test, y_pred)

        mlflow.log_params(params)
        mlflow.log_metrics(score)
        log_model(lgb_model=cb_model,
                                  artifact_path="model",
                                  registered_model_name="plane_price_catboost",
                                  signature=signature)


        cb_model.save_model(output)


if __name__ == "__main__":
    train_model()
