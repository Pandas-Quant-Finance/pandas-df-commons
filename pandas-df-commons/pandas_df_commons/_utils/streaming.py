import random

import pandas as pd

from pandas_df_commons.indexing.intersection import intersection_of_index


def window(df: pd.DataFrame, period: int, shuffle: bool = False):
    last = len(df) - period + 1
    all_windows = range(0, last)

    if shuffle:
        all_windows = list(all_windows)
        random.shuffle(all_windows)

    for i in all_windows:
        yield df.iloc[i:i + period]


def frames_at_common_index(*dfs: pd.DataFrame, level=None):
    dfs = [f for f in dfs if f is not None]
    idx = intersection_of_index(*dfs, level=level)
    for i in idx:
        if len(dfs) > 1:
            yield tuple(df.loc[i] for df in dfs)
        else:
            yield dfs[0].loc[i]

