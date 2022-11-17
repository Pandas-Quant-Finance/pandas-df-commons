import pandas as pd


def cumpct_change(df):
    return ((df.pct_change().fillna(0) + 1).cumprod() - 1)


def cumapply(df, func: callable, start_value=None, **kwargs):
    last = [start_value]

    def exec(x):
        val = func(last[0], x)
        last[0] = val
        return val

    return df.apply(exec, **kwargs)
