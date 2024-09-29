import pandas as pd


def prepare_airport_coords():
    df = pd.read_csv("./data/interim/flights.csv")
    airports_iata = pd.read_csv(
        "./data/external/GlobalAirportDatabase.txt",
        header=None,
        sep=":",
        usecols=(1,),
        names=("iata",),
    )
    airport_latlon = pd.read_csv(
        "./data/external/gadb_declatlon.csv", header=None, names=("lat", "lon")
    )

    df = pd.concat((airports_iata, airport_latlon), axis=1).dropna()

    df.to_csv("./data/interim/airport_coords.csv", index=False)


if __name__ == "__main__":
    prepare_airport_coords()
