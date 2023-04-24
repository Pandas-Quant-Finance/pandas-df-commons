from functools import lru_cache
from typing import Iterator

import numpy as np

from pandas_df_commons._utils.streaming import IterRows


class Batch(object):

    def __init__(self, iterable, batch_size=1):
        super().__init__()
        self.iterable = iterable
        self.batch_size = batch_size
        self.batch = []

        self.iter_length = len(iterable)
        self._iterable = iterable

    def __iter__(self):
        return self

    def __getitem__(self, item):
        assert 0 <= item < len(self), "index out of bounds"
        pos = item * self.batch_size
        if getattr(self.iterable, 'iloc', False):
            return self.iterable.iloc[pos:pos+self.batch_size]
        else:
            return self.iterable[pos:pos+self.batch_size]

    def __next__(self):
        if not isinstance(self._iterable, Iterator):
            self._iterable = IterRows(self._iterable)

        if self.batch is None:
            raise StopIteration()

        try:
            while len(self.batch) < self.batch_size:
                self.batch.append(next(self._iterable))

            batch = self.batch
            self.batch = []
            return batch

        except StopIteration as si:
            batch = self.batch
            self.batch = None
            return batch

    def __len__(self):
        return int(np.ceil(self.iter_length / self.batch_size))
