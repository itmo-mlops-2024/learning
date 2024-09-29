import pandas as pd

def build_features():
    df = pd.read_csv('./data/interim/flights.csv')
    coords = pd.read_csv('./data/interim/airport_coords.csv').set_index('iata')

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

    # encode airports to lat/lon
    airport_columns = ['startingAirport', 'destinationAirport']
    for col in airport_columns:
        df = df.set_index(col).join(coords, rsuffix=f'_{col}').reset_index()

    df.to_csv('./data/processed/flights.csv', index=False)


if __name__ == "__main__":
    build_features()
