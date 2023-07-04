from __future__ import annotations

import itertools
from collections import OrderedDict, defaultdict
from functools import partial
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


def loc_at_level(df: pd.DataFrame, locs, level, axis=0):
    if locs is None:
        return None
    elif not isinstance(locs, slice):
        if hasattr(locs, '__len__') and len(locs) < 1:
            return df.iloc[:0]
    elif isinstance(locs, Iterable):
        locs = pd.unique(locs)

    if level is None:
        return df.loc[locs, :] if axis == 0 else df.loc[:, locs]

    levels = (df.index if axis == 0 else df.columns).nlevels

    filter = [slice(None) for _ in range(levels)]
    filter[level] = locs
    if len(filter) == 1:
        filter = filter[0]
    else:
        filter = tuple(filter)

    return df.loc[filter, :] if axis == 0 else df.loc[:, filter]


def last_index_value(df: pd.DataFrame | pd.Index, level=None, axis=0):
    idx = df if isinstance(df, pd.Index) else (df.index if axis == 0 else df.columns)
    if len(idx) < 1: return None
    return idx.get_level_values(level)[-1] if level is not None else idx[-1]


def nth(df: pd.DataFrame, n: int, include_last: bool = False, level=None, axis=0):
    idx = df.index if axis == 0 else df.columns
    keys = pd.unique(idx.get_level_values(level) if level is not None else idx)

    indexes = [i for i in range(0, len(keys), n)]
    if include_last and indexes[-1] < len(keys) - 1:
        indexes.append(len(keys) - 1)

    return df.loc[keys[indexes]]


def index_counts(df: pd.DataFrame | pd.Index, axis=0) -> dict:
    idx = df if isinstance(df, pd.Index) else (df.index if axis == 0 else df.columns)
    if idx.nlevels <= 1: return {0: len(idx)}

    def make_nested_dd(level) -> defaultdict:
        return defaultdict(partial(make_nested_dd, level - 1)) if level > 0 else defaultdict(lambda:0)

    counts = make_nested_dd(idx.nlevels - 1)
    if len(idx) <= 0:
        c = counts
        for i in range(idx.nlevels): c = c[0]
        return counts

    for i in idx:
        c = counts
        for j in i[:-1]: c = c[j]
        c[0] += 1

    return counts


def index_shape(df: pd.DataFrame | pd.Index, axis=0) -> Tuple[int, ...]:
    idx = df if isinstance(df, pd.Index) else (df.index if axis == 0 else df.columns)
    if idx.nlevels <= 1: return (len(idx), )
    if len(idx) <= 0: return tuple(0 for _ in range(idx.nlevels))

    counts = index_counts(df, axis=axis)

    def is_nd(values):
        if len(set(len(v) if isinstance(v, defaultdict) else v for v in values)) > 1:
            return False
        else:
            if isinstance(next(iter(values)), int):
                return True
            else:
                return is_nd(list(itertools.chain(*[v.values() for v in values])))


    if not is_nd(counts.values()):
        return (len(df), )

    def calc_shape(counts, shape):
        next_value = next(iter(counts.values()))

        if isinstance(next_value, int):
            return shape[:-1] + (next_value, )
        else:
            return calc_shape(next_value, shape + (len(next_value),))

    return calc_shape(counts, (len(counts), ))
