import pandas as pd

def build_features():
    df = pd.read_csv('./data/interim/flights.csv')

    df['stops'] = df['segmentsArrivalAirportCode'].str.count(r'\|\|')

    # prepare dates
    dates = pd.to_datetime(df['flightDate'])
    df['day_of_week'] = dates.dt.day_of_week
    df['month'] = dates.dt.month
    df = df.drop('flightDate', axis=1)

    # remove excess segment data
    df = df.drop([
        'segmentsDepartureTimeEpochSeconds',
        'segmentsArrivalTimeEpochSeconds',
        'segmentsArrivalAirportCode',
        'segmentsDepartureAirportCode',
        'segmentsAirlineCode',
        'segmentsDurationInSeconds',
        # 'travelDuration',
    ], axis=1)

    # one-hot encode airports
    categorical_columns = ['startingAirport', 'destinationAirport']
    for col in categorical_columns:
        col_ohe = pd.get_dummies(df[col], prefix=col).astype(int)
        df = pd.concat((df, col_ohe), axis=1).drop(col, axis=1)

    df.to_csv('./data/processed/flights.csv', index=False)


if __name__ == "__main__":
    build_features()
