import click
import pandas as pd


@click.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("airport_coords", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
def build_features(input, airport_coords, output):
    df = pd.read_csv(input)
    coords = pd.read_csv(airport_coords).set_index("iata")

    df["stops"] = df["segmentsArrivalAirportCode"].str.count(r"\|\|")

    # prepare dates
    dates = pd.to_datetime(df["flightDate"])
    df["day_of_week"] = dates.dt.day_of_week
    df["month"] = dates.dt.month
    df = df.drop("flightDate", axis=1)

    # remove excess segment data
    df = df.drop(
        [
            "segmentsDepartureTimeEpochSeconds",
            "segmentsArrivalTimeEpochSeconds",
            "segmentsArrivalAirportCode",
            "segmentsDepartureAirportCode",
            "segmentsAirlineCode",
            "segmentsDurationInSeconds",
            # 'travelDuration',
        ],
        axis=1,
    )

    # encode airports to lat/lon
    airport_columns = ["startingAirport", "destinationAirport"]
    for col in airport_columns:
        df = df.set_index(col).join(coords, rsuffix=f"_{col}").reset_index()

    df.to_csv(output, index=False)


if __name__ == "__main__":
    build_features()
