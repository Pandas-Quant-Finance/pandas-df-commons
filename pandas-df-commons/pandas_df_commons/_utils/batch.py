from copy import deepcopy
from functools import lru_cache
from typing import Iterator

import numpy as np

from pandas_df_commons._utils.streaming import IterRows


class Batch(object):

    def __init__(self, iterable, batch_size=1, copy=False, **kwargs):
        super().__init__()
        self.iterable = iterable
        self.batch_size = batch_size
        self.copy = copy
        self.kwargs = kwargs

        self.has_kwargs = len(kwargs) > 0
        self.batch, self._iterable, self._index = self.__reset_iteration()
        self.iter_length = None

    def __iter__(self):
        return self

    def __getitem__(self, item):
        assert 0 <= item < len(self), "index out of bounds"
        pos = item * self.batch_size
        if getattr(self.iterable, 'iloc', False):
            return self.__return__(self.iterable.iloc[pos:pos+self.batch_size])
        else:
            return self.__return__(self.iterable[pos:pos+self.batch_size])

    def __next__(self):
        # check if we emitted the last batch already
        if self.batch is None:
            self._end_iter_and_reset()

        # slicing is supposed to be faster than iterating, we want to try to slice first
        if self._index is not None:
            try:
                self._index += 1
                if self._index >= len(self):
                    self._end_iter_and_reset()
                else:
                    return self[self._index]
            except TypeError as e:
                self._index = None

        # slicing did not work for this iterable, so we stream into batches
        if not isinstance(self._iterable, Iterator):
            self._iterable = IterRows(self._iterable)

        try:
            while len(self.batch) < self.batch_size:
                self.batch.append(next(self._iterable))

            batch = self.batch
            self.batch = []
            return self.__return__(batch)

        except StopIteration as si:
            batch = self.batch
            self.batch = None
            return self.__return__(batch)

    @lru_cache(1)
    def __len__(self):
        self.iter_length = self.__estimate_length__(self.iterable)
        return int(np.ceil(self.iter_length / self.batch_size))

    def __return__(self, value):
        return value if not self.has_kwargs else (value, self.kwargs)

    def __estimate_length__(self, iterable):
        try:
            return len(iterable)
        except Exception as e:
            i = -1
            for i, _ in enumerate(deepcopy(iterable) if self.copy else iterable): pass
            return i

    def _end_iter_and_reset(self):
        self.__reset_iteration()
        raise StopIteration()

    def __reset_iteration(self):
        self.batch = []
        self._iterable = deepcopy(self.iterable) if self.copy else self.iterable
        self._index = -1
        return self.batch, self._iterable, self._index
