from __future__ import annotations
from functools import partial
from typing import Callable

import pandas as pd

from pandas_df_commons._utils.streaming import window
from pandas_df_commons.indexing.decorators import convert_series_as_data_frame


def cumpct_change(df):
    return ((df.pct_change().fillna(0) + 1).cumprod() - 1)


def cumapply(df, func: callable, start_value=None, **kwargs):
    last = [start_value]

    def exec(x):
        val = func(last[0], x)
        last[0] = val
        return val

    return df.apply(exec, **kwargs)


@convert_series_as_data_frame
def rolling_apply(df: pd.DataFrame, period: int, func: Callable[[pd.DataFrame], pd.DataFrame | pd.Series], parallel=False):
    if parallel:
        from pandas_df_commons._utils.multiprocessing import streaming_parallel
        res = streaming_parallel(partial(func), lambda: window(df, period))
    else:
        res = [func(w) for w in window(df, period)]

    if isinstance(res[-1], pd.DataFrame):
        return pd.concat(res, axis=0, keys=df.index[period-1:])
    elif isinstance(res[-1], pd.Series):
        return pd.DataFrame(res, index=df.index[period - 1:])
    else:
        return pd.DataFrame(res, index=df.index[period-1:])
