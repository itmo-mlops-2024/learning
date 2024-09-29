import click
import pandas as pd


@click.command()
@click.argument("iata_path", type=click.Path(exists=True))
@click.argument("lat_lon_path", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
def prepare_airport_coords(iata_path, lat_lon_path, output):
    airports_iata = pd.read_csv(
        iata_path,
        header=None,
        sep=":",
        usecols=(1,),
        names=("iata",),
    )
    airport_latlon = pd.read_csv(
        lat_lon_path, header=None, names=("lat", "lon")
    )

    df = pd.concat((airports_iata, airport_latlon), axis=1).dropna()

    df.to_csv(output, index=False)


if __name__ == "__main__":
    prepare_airport_coords()
