from __future__ import annotations
import pandas as pd


def intersection_of_index(*dfs: pd.DataFrame | pd.Index, level=None):
    intersect_index = dfs[0] if isinstance(dfs[0], pd.Index) else dfs[0].index
    if isinstance(intersect_index, pd.MultiIndex) and level is not None:
        intersect_index = intersect_index.get_level_values(level)

    for i in range(1, len(dfs)):
        if dfs[i] is not None:
            other_index = dfs[i] if isinstance(dfs[i], pd.Index) else dfs[i].index
            if isinstance(other_index, pd.MultiIndex) and level is not None:
                other_index = other_index.get_level_values(level)

            intersect_index = intersect_index.intersection(other_index)

    return intersect_index.sort_values()


def aligend_frames(*dfs: pd.DataFrame, level=None, dropna=False):
    idx = intersection_of_index(*[df.dropna() if dropna else df for df in dfs if df is not None], level=level)
    return (df.loc[idx] if df is not None else None for df in dfs)
