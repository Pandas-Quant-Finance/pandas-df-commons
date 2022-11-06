"""Augment pandas DataFrame with commonly used methods and utilities"""
__version__ = '0.1.0'

import logging

import numpy as np
import pandas as pd

from pandas.core.base import PandasObject


_log = logging.getLogger(__name__)
_log.debug(f"numpy version {np.__version__}")
_log.debug(f"pandas version {pd.__version__}")


def _extender(df):
    import pandas_df_commons.indexing as indexing
    from pandas_df_commons.extensions.functions import cumapply, cumpct_change

    class Extender(object):

        def __init__(self, df):
            self.df = df

        def __getitem__(self, item):
            return indexing.get_columns(self.df, item)

        def cumpct_change(self):
            return cumpct_change(self.df)

        def cumapply(self, func: callable, start_value=None, **kwargs):
            return cumapply(self.df, func, start_value, **kwargs)

    return Extender(df)


# IMPORTANT call this function after the import
def monkey_patch_dataframe(extender='X'):
    setattr(PandasObject, extender, property(extender))
    setattr(pd.DataFrame, "to_frame", lambda self: self)

    #setattr(pd.DataFrame, "flatten_columns", flatten_multi_column_index)
    #setattr(pd.DataFrame, "unique_level_columns", unique_level_columns)
    #setattr(pd.DataFrame, "has_indexed_columns", lambda self: has_indexed_columns(self))
    #setattr(pd.DataFrame, "add_multi_index", lambda self, *args, **kwargs: add_multi_index(self, *args, **kwargs))
    #setattr(pd.Series, "add_multi_index", lambda self, *args, **kwargs: add_multi_index(self, *args, **kwargs))
    #
    #setattr(pd.MultiIndex, "unique_level", lambda self, *args: unique_level(self, *args))
