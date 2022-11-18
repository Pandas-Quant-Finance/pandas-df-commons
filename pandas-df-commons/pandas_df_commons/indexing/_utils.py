from __future__ import annotations

import logging
from typing import Dict

import pandas as pd

from pandas_df_commons.indexing import unique_level_values
from pandas_df_commons.indexing.multiindex_utils import add_to_multi_index
from pandas_ta.pandas_ta_utils.index_utils import same_columns_after_level

_log = logging.getLogger(__name__)


class _Utils():

    @staticmethod
    def row_agg(results: Dict):
        return pd.concat(results.values(), keys=results.keys(), axis=0)

    @staticmethod
    def col_agg(results, level):
        groups = [add_to_multi_index(res, group, inplace=True, level=level) for group, res in results.items()]
        return pd.concat(groups, axis=1)

    @staticmethod
    def get_top_level_columns(df, level=0):
        if df.ndim > 1 and isinstance(df.columns, pd.MultiIndex):
            # check if the shape of the 2nd level is identical else threat as if not multi index
            if same_columns_after_level(df, level):
                return unique_level_values(df, level, axis=1)
            else:
                _log.warning(f"columns in further levels do not follow the same structure! Treat as normal Index")

        return None

    @staticmethod
    def get_top_level_rows(df, level=0):
        if isinstance(df.index, pd.MultiIndex):
            top_level = unique_level_values(df, level=level, axis=0)
            if len(top_level) > 1:
                return top_level

        return None

