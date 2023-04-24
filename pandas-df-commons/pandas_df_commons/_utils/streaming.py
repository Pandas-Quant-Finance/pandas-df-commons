from __future__ import annotations
import random
from typing import Iterator

import pandas as pd

from pandas_df_commons.indexing.intersection import intersection_of_index


class Window(object):

    def __init__(self, df: pd.DataFrame, period: int | None, shuffle: bool = False):
        super().__init__()
        self.df = df
        self.period = period
        self.shuffle = shuffle

        self.last = len(df) - (period or 1) + 1
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
        if isinstance(item, slice):
            return [self.__getitem__(i) for i in range(0, len(self)).__getitem__(item)]
        else:
            i = self.all_windows[item]

            # if period is None we just stream as it we would have called `iterrows` or `items`
            return self.df.iloc[i:i + self.period] if self.period else self.df.iloc[i]

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


class IterRows(object):

    def __init__(self, iterable):
        super().__init__()
        self.iterable = iterable
        self.length = len(iterable)
        self.skip_index = False

        self._iterable = self.iterable

    def __iter__(self):
        return self

    def __getitem__(self, item):
        if getattr(self.iterable, 'iloc', False):
            return self.iterable.iloc[item]
        else:
            return self.iterable[item]

    def __next__(self):
        if getattr(self._iterable, 'iterrows', False):
            self._iterable = self._iterable.iterrows()
            self.skip_index = True
        elif getattr(self._iterable, 'items', False):
            self._iterable = self._iterable.items()
            self.skip_index = True
        elif getattr(self._iterable, 'iteritems', False):
            self._iterable = self._iterable.iteritems()
            self.skip_index = True
        elif not isinstance(self._iterable, Iterator):
            self._iterable = iter(self._iterable)

        n = next(self._iterable)
        return n[1] if self.skip_index else n
