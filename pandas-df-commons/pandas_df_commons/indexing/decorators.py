from __future__ import annotations

import logging
from functools import wraps
from typing import Dict, Callable, Any, T

import numpy as np
import pandas as pd

from pandas_df_commons.indexing._utils import row_agg, col_agg, get_top_level_rows, get_top_level_columns, loc_with_name
from pandas_df_commons._utils.multiprocessing import blocking_parallel


_log = logging.getLogger(__name__)


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


def convert_series_as_data_frame(func):
    @wraps(func)
    def to_dataframe(df: pd.DataFrame | pd.Series, *args, **kwargs):
        return func(df.to_frame() if df.ndim < 2 else df, *args, **kwargs)

    return to_dataframe


def foreach_top_level_row_and_column(parallel=False, row_aggregator=row_agg, column_aggregator=col_agg):
    def decorator(func):
        @wraps(func)
        def wrapper(df, *args, **kwargs):
            tl_rows = get_top_level_rows(df) if parallel else None
            tl_columns = get_top_level_columns(df) if parallel else None
            nr_rows = len(tl_rows) if tl_rows else 0
            nr_columns = len(tl_columns) if tl_columns else 0
            pr = nr_rows > nr_columns

            @foreach_top_level_row_aggregate(row_aggregator, parallel=parallel and pr)
            @foreach_top_level_column_aggregate(column_aggregator, parallel=parallel and not pr)
            def executor(df, *args, **kwargs):
                return func(df, *args, **kwargs)

            return executor(df, *args, **kwargs)

        return wrapper
    return decorator


def foreach_top_level_column(func):
    return foreach_top_level_column_aggregate(col_agg, parallel=False)(func)


def foreach_top_level_column_aggregate(aggregator: Callable[[Dict[Any, T], int], T] = col_agg, level=0, parallel=False):
    def decorator(func):
        @wraps(func)
        def exec_on_each_tl_column(df: pd.DataFrame, *args, **kwargs):
            top_level = get_top_level_columns(df, level=level)
            if top_level:
                if parallel:
                    # multiProcessing
                    results = dict(
                        zip(
                            top_level,
                            blocking_parallel(
                                lambda sub_df: func(sub_df, *args, **kwargs),
                                [df.xs(tl, axis=1, level=level).copy() for tl in top_level]
                            )
                        )
                    )
                else:
                    # sequential
                    results = {group: func(df.xs(group, axis=1, level=level), *args, **kwargs) for group in top_level}

                return aggregator(results, level)
            else:
                return func(df, *args, **kwargs)

        return exec_on_each_tl_column
    return decorator


def foreach_top_level_row(func):
    return foreach_top_level_row_aggregate(row_agg)(func)


def foreach_top_level_row_aggregate(aggregator: Callable[[Dict[Any, T]], T] = row_agg, parallel=False):
    def decorator(func):
        @wraps(func)
        def exec_on_each_tl_row(df: pd.DataFrame, *args, **kwargs):
            top_level = get_top_level_rows(df)

            if top_level:
                # multiProcessing
                if parallel:
                    results = dict(
                        zip(
                            top_level,
                            blocking_parallel(
                               lambda sub_df: func(sub_df, *args, **kwargs),
                               [loc_with_name(df, tl) for tl in top_level]
                            )
                        )
                    )
                else:
                    # sequential
                    results = {group: func(df.loc[group], *args, **kwargs) for group in top_level}

                return aggregator(results)
            else:
                return func(df, *args, **kwargs)

        return exec_on_each_tl_row
    return decorator


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

