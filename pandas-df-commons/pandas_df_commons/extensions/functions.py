from __future__ import annotations
from functools import partial
from typing import Callable, Tuple, Any, List

import pandas as pd

from pandas_df_commons._utils.rescaler import ReScaler
from pandas_df_commons._utils.streaming import window, frames_at_common_index_generator
from pandas_df_commons.indexing.decorators import convert_series_as_data_frame
from pandas_df_commons.indexing.intersection import intersection_of_index


def rescale(df, range: Tuple[float, float], clip=False, axis=None):
    if axis is None:
        if df.ndim > 1:
            return pd.DataFrame(
                ReScaler((df.min().min(), df.max().max()), range, clip)(df),
                index=df.index,
                columns=df.columns
            )
        else:
            return pd.Series(ReScaler((df.min(), df.max()), range, clip)(df), name=df.name, index=df.index)
    else:
        return df.apply(lambda x: ReScaler((x.min(), x.max()), range, clip)(x), axis=axis, result_type='broadcast')


def cumpct_change(df):
    return ((df.pct_change().fillna(0) + 1).cumprod() - 1)


def sateful_apply(df, func: Callable[[Any, pd.DataFrame], Tuple[Any, pd.DataFrame]], start_state=None, **kwargs):
    state = [start_state]

    def exec(x):
        state[0], val = func(state[0], x)
        return val

    return df.apply(exec, **kwargs)


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
        res = streaming_parallel(func, lambda: window(df, period))
    else:
        res = [func(w) for w in window(df, period)]

    if isinstance(res[-1], pd.DataFrame):
        return pd.concat(res, axis=0, keys=df.index[period-1:])
    elif isinstance(res[-1], pd.Series):
        return pd.DataFrame(res, index=df.index[period - 1:])
    else:
        return pd.DataFrame(res, index=df.index[period-1:])


def joint_apply(*df: pd.DataFrame, func: Callable[[Tuple[pd.DataFrame]], pd.Series], level=None, parallel=False):
    if parallel:
        from pandas_df_commons._utils.multiprocessing import streaming_parallel
        res = streaming_parallel(func, lambda: frames_at_common_index_generator(*df, level=level))
    else:
        res = [func(w) for w in frames_at_common_index_generator(*df, level=level)]

    return pd.DataFrame(res, index=intersection_of_index(*df, level=level))