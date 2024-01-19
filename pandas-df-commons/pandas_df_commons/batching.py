import math
from typing import List

import cachetools
import pandas as pd


class BatchList(object):

    def __init__(self, df: pd.DataFrame, batch_size: int, window: int = 1, overlapping_windows: bool = True):
        self.df = df
        self.batch_size = batch_size
        self.window = window
        self.overlapping_windows = overlapping_windows

    def __getitem__(self, item: int) -> List[pd.DataFrame]:
        all_data_len = len(self.df)
        last_batch = len(self)
        item = last_batch - item

        if self.overlapping_windows:
            start_idx = all_data_len - (self.window + item * self.batch_size - 1)
        else:
            start_idx = all_data_len - self.window * self.batch_size * item

        start_indices = [start_idx + (i * (1 if self.overlapping_windows else self.window)) for i in range(self.batch_size)]

        batches = [self.df.iloc[si:si + self.window] for si in start_indices if si + self.window <= all_data_len and si >= 0]
        return batches

    @cachetools.cached(cache=cachetools.FIFOCache(maxsize=1))
    def __len__(self) -> int:
        nr = len(self.df)
        nr = (nr - self.window + 1) if self.overlapping_windows else (nr // self.window)

        assert nr > 0, f"not enough data {len(self.df)} < {self.window}"

        return int(math.ceil(nr / self.batch_size))
