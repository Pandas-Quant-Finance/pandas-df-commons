import random

import pandas as pd

from pandas_df_commons.indexing.intersection import intersection_of_index


class Window(object):

    def __init__(self, df: pd.DataFrame, period: int, shuffle: bool = False):
        super().__init__()
        self.df = df
        self.period = period
        self.shuffle = shuffle

        self.last = len(df) - period + 1
        self.all_windows = list(range(0, self.last))
        self._index = -1

        if shuffle:
            random.shuffle(self.all_windows)

    def __iter__(self):
        return self

    def __next__(self):
        self._index += 1
        if self._index >= self.last:
            raise StopIteration()
        else:
            return self[self._index]

    def __getitem__(self, item):
        i = self.all_windows[item]
        return self.df.iloc[i:i + self.period]

    def __len__(self):
        return self.last

    def reset(self):
        self._index = -1
        if self.shuffle:
            random.shuffle(self.all_windows)


def window(df: pd.DataFrame, period: int, shuffle: bool = False, all_windows = None):
    if all_windows is None:
        # last = len(df) - period + 1
        all_windows = range(0, len(df) - period + 1)

    if shuffle:
        if not isinstance(all_windows, list):
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

