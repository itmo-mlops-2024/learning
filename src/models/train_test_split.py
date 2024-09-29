import pandas as pd

def train_test_split():
    df = pd.read_csv('./data/processed/flights.csv')

    train = df.sample(frac=0.75, random_state=200)
    test = df.drop(train.index)

    train.to_csv('./data/processed/train.csv', index=False)
    test.to_csv('./data/processed/test.csv', index=False)


if __name__ == "__main__":
    train_test_split()
