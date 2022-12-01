import pandas as pd


def window(df: pd.DataFrame, period: int):
    last = len(df) - period + 1
    for i in range(0, last):
        yield df.iloc[i:i + period]
