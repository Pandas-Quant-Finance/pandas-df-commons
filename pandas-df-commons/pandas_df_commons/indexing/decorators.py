from __future__ import annotations

import logging
from functools import wraps
from typing import Dict, Callable, Any, T

import numpy as np
import pandas as pd

from pandas_df_commons.indexing._utils import row_agg, col_agg, get_top_level_rows, get_top_level_columns, \
    loc_with_name, top_level_separator_generator
from pandas_df_commons._utils.multiprocessing import blocking_parallel, streaming_parallel

_log = logging.getLogger(__name__)


def convert_series_as_data_frame(func):
    @wraps(func)
    def to_dataframe(df: pd.DataFrame | pd.Series, *args, **kwargs):
        return func(df.to_frame() if df.ndim < 2 else df, *args, **kwargs)

    return to_dataframe


def foreach_column(func):
    @wraps(func)
    def exec_on_each_column(df: pd.DataFrame, *args, **kwargs):
        if df.ndim > 1 and df.shape[1] > 0:
            results = [func(df[col], *args, **kwargs) for col in df.columns]  # theoretically could be done parallel
            if results[0].ndim > 1 and results[0].shape[1] > 1:
                for i, col in enumerate(df.columns):
                    results[i].columns = pd.MultiIndex.from_product([[col], results[i].columns.tolist()])

            return pd.concat(results, axis=1, join='inner')
        else:
            return func(df, *args, **kwargs)

    return exec_on_each_column



# Top Level decorator Handling
def foreach_top_level(
        parallel=False,
        row_aggregator=row_agg,
        column_aggregator=col_agg,
        col_level=0,
        progress_bar=False,
):
    def decorator(func):
        @wraps(func)
        def wrapper(df, *args, **kwargs):
            top_level_rows = get_top_level_rows(df) if row_aggregator is not None else None
            top_level_columns = get_top_level_columns(df, level=col_level) if column_aggregator is not None else None
            index_generator = top_level_separator_generator(df, top_level_rows, top_level_columns, col_level)
            parts = len(top_level_columns) if top_level_columns is not None else 0
            parts += len(top_level_rows) if top_level_rows is not None else 0

            if parallel and parts > 1:
                if progress_bar:
                    from tqdm import tqdm
                    index_generator = tqdm(index_generator, total=parts)

                results = streaming_parallel(
                    lambda indexes_sub_df: (indexes_sub_df[0], func(indexes_sub_df[1], *args, **kwargs)),
                    lambda: index_generator
                )
            else:
                results = [(idx, func(sub_df, *args, **kwargs)) for idx, sub_df in index_generator]

            # fix names
            for _, r in results:
                r.index.name = df.index.name
                r.columns.name = df.columns.name

            # aggregate results[(col_idx, row_idx,), df)
            return (row_aggregator or row_agg)(
                {row_tl_idx: (column_aggregator or col_agg)(
                    {c: f for (c, r), f in results if r == row_tl_idx},
                    col_level
                ) for row_tl_idx in (top_level_rows or [None])}
            )

        return wrapper
    return decorator


def foreach_top_level_row_and_column(parallel=False, row_aggregator=row_agg, column_aggregator=col_agg, progress_bar=False):
    return foreach_top_level(parallel, row_agg, col_agg, progress_bar=progress_bar)


def foreach_top_level_column(func):
    return foreach_top_level_column_aggregate(col_agg, parallel=False)(func)


def foreach_top_level_column_aggregate(aggregator: Callable[[Dict[Any, T], int], T] = col_agg, level=0, parallel=False):
    return foreach_top_level(parallel, None, aggregator, level)


def foreach_top_level_row(func):
    return foreach_top_level_row_aggregate(row_agg)(func)


def foreach_top_level_row_aggregate(aggregator: Callable[[Dict[Any, T]], T] = row_agg, parallel=False):
    return foreach_top_level(parallel, aggregator, None)


def is_time_consuming(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._is_timeconsuming = True
    return wrapper


def rename_with_parameters(function_name, parameter_names, output_names=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            columns, result = func(*args, **kwargs)
            length = max([len(r) for r in result]) if isinstance(result, (tuple, list)) else len(result)

            argvalues = list(args[1:])
            if len(argvalues) < len(parameter_names):
                argvalues += [kwargs.get(pn, "None") for pn in parameter_names[len(args) - 1:]]
            argvaluestring = f', {str(argvalues)[1:-1]}' if len(argvalues) > 0 else ""

            if output_names is None or len(output_names) < 2:
                column_names = [f"{function_name}({str(columns)[1:-1]}{argvaluestring})"]
            else:
                column_names = pd.MultiIndex.from_tuples([
                    (f"{function_name}({str(columns)[1:-1]}{argvaluestring})", out_name) for out_name in output_names
                ])

            index = args[0].index[-length:]
            df = pd.DataFrame(
                result if isinstance(result, np.ndarray) else np.array(result).T,
                index=index, columns=column_names
            )

            return df
        return wrapper
    return decorator

