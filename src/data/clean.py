import pandas as pd


def clean_data():
    df = pd.read_csv("./data/raw/itineraries_random_2M.csv")

    # remove low-frequency cases
    is_premium = (
        df["segmentsCabinCode"].str.contains("business")
        | df["segmentsCabinCode"].str.contains("premium")
        | df["segmentsCabinCode"].str.contains("first")
    )
    df = df[~is_premium & ~df["isRefundable"]].copy()
    df = df.drop(["segmentsCabinCode", "isRefundable"], axis=1)
    df = df.dropna()

    # drop duplicate cols
    df = df.drop(
        [
            "segmentsAirlineName",
            "segmentsDepartureTimeRaw",
            "segmentsArrivalTimeRaw",
            "isBasicEconomy",
            "isNonStop",
            "totalTravelDistance",
        ],
        axis=1,
    )
    # drop irrelevans cols
    df = df.drop(
        [
            "legId",
            "fareBasisCode",
            "searchDate",
            "seatsRemaining",
            "baseFare",
            "segmentsEquipmentDescription",
            "segmentsDistance",
        ],
        axis=1,
    )

    # parse duration to minutes
    def parse_duration(i):
        if not i[2]:
            return int(i[1])
        return 60 * int(i[1]) + int(i[2])

    df["travelDuration"] = (
        df["travelDuration"].str.split(r"[^0-9]+").map(parse_duration)
    )

    df.to_csv("./data/interim/flights.csv", index=False)


if __name__ == "__main__":
    clean_data()
