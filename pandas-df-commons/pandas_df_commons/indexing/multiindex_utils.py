from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Tuple, Any

import pandas as pd


def get_top_level_of_multi_index(df: pd.DataFrame):
    def get_top_level_columns():
        if hasattr(df, "columns") and isinstance(df.columns, pd.MultiIndex):
            return unique_level_values(df, axis=1)
        else:
            return None

    if isinstance(df.index, pd.MultiIndex):
        if df.ndim > 1:
            return unique_level_values(df, axis=0), get_top_level_columns()
        else:
            return unique_level_values(df, axis=0)
    else:
        return None, get_top_level_columns()


def unique_level_values(df: pd.DataFrame | pd.Index, level=0, axis=0):
    idx = df if isinstance(df, pd.Index) else (df.index if axis == 0 else df.columns)
    return unique(idx.get_level_values(level)) if isinstance(idx, pd.MultiIndex) else idx


def unique(items):
    return list(OrderedDict.fromkeys(items))


def add_to_multi_index(df, head, inplace=False, axis=1, level=0):
    df = df if inplace else df.copy()

    if axis == 0:
        df.index = pd.MultiIndex.from_tuples([(head, *(t if isinstance(t, tuple) else (t, ))) for t in df.index.tolist()]).swaplevel(0, level)
    elif axis == 1:
        if df.ndim > 1:
            df.columns = pd.MultiIndex.from_tuples([(head, *(t if isinstance(t, tuple) else (t, ))) for t in df.columns.tolist()]).swaplevel(0, level)
        else:
            df.name = (head, df.name)
    else:
        raise ValueError("illegal axis, expected 0|1")

    return df


def make_top_level_row_iterator(frames: pd.DataFrame | Iterable[Tuple[Any, pd.DataFrame]], level: int = 0) -> Tuple[Any, pd.DataFrame]:
    if isinstance(frames, pd.DataFrame):
        frames = [(None, frames)]

    for name, frame in frames:
        if isinstance(frame.index, pd.MultiIndex):
            for idx in unique_level_values(frame.index, level=level):
                yield (name, idx) if name is not None else idx, frame.loc[idx]
        else:
            yield name, frame
