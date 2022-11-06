from __future__ import annotations

from collections import OrderedDict

import pandas as pd


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