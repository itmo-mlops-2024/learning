import click
import pandas as pd


@click.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("out_train", type=click.Path())
@click.argument("out_test", type=click.Path())
def train_test_split(input, out_train, out_test):
    df = pd.read_csv(input)

    train = df.sample(frac=0.75, random_state=200)
    test = df.drop(train.index)

    train.to_csv(out_train, index=False)
    test.to_csv(out_test, index=False)


if __name__ == "__main__":
    train_test_split()
