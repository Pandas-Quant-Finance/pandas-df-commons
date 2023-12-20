"""Augment pandas DataFrame with commonly used methods and utilities"""
__version__ = open(f"{__file__.replace('__init__.py', '')}VERSION").read()

import logging
from typing import Callable, Tuple

import numpy as np
import pandas as pd

from pandas_df_commons.indexing.decorators import foreach_top_level_row

_log = logging.getLogger(__name__)
_log.debug(f"numpy version {np.__version__}")
_log.debug(f"pandas version {pd.__version__}")


def _extender(df):
    import pandas_df_commons.indexing as indexing
    from pandas_df_commons.extensions.functions import cumapply, cumpct_change, rolling_apply, rescale, joint_apply

    class Extender(object):

        def __init__(self, df):
            self.df = df

        def __getitem__(self, item):
            return indexing.get_columns(self.df, item)

        def rescale(self, range: Tuple[float, float], clip=False, axis=None):
            return rescale(self.df, range, clip, axis)

        def cumpct_change(self):
            return cumpct_change(self.df)

        def cumapply(self, func: callable, start_value=None, **kwargs):
            return cumapply(self.df, func, start_value, **kwargs)

        def rolling_apply(self, period: int, func: Callable[[pd.DataFrame], pd.DataFrame], parallel=False):
            return rolling_apply(self.df, period, func, parallel)

        def joint_apply(self, *dfs: pd.DataFrame, func: Callable[[Tuple[pd.DataFrame]], pd.Series], level=None, parallel=False):
            return joint_apply(self.df, *dfs, func=func, level=level, parallel=parallel)

        def for_toplevel_row(self, func: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
            return foreach_top_level_row(func)(df)

    return Extender(df)


# IMPORTANT call this function after the import
def monkey_patch_dataframe(extender='X'):
    from pandas.core.base import PandasObject

    existing = getattr(PandasObject, extender, None)
    if existing is not None:
        if not isinstance(existing, property)\
        or not str(existing.fget(None)).startswith("<pandas_df_commons._extender.<locals>.Extender object"):
            raise ValueError(f"field already exists as {existing.fget(None)}")

    setattr(PandasObject, extender, property(lambda self: _extender(self)))
    setattr(pd.DataFrame, "to_frame", lambda self: self)

    #setattr(pd.DataFrame, "flatten_columns", flatten_multi_column_index)
    #setattr(pd.DataFrame, "unique_level_columns", unique_level_columns)
    #setattr(pd.DataFrame, "has_indexed_columns", lambda self: has_indexed_columns(self))
    #setattr(pd.DataFrame, "add_multi_index", lambda self, *args, **kwargs: add_multi_index(self, *args, **kwargs))
    #setattr(pd.Series, "add_multi_index", lambda self, *args, **kwargs: add_multi_index(self, *args, **kwargs))
    #
    #setattr(pd.MultiIndex, "unique_level", lambda self, *args: unique_level(self, *args))


