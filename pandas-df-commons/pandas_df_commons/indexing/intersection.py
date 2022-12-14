from __future__ import annotations
import pandas as pd


def intersection_of_index(*dfs: pd.DataFrame | pd.Index, level=None):
    intersect_index = dfs[0] if level is not None and isinstance(dfs[0], pd.Index) else dfs[0].index
    if isinstance(intersect_index, pd.MultiIndex):
        intersect_index = pd.unique(intersect_index.get_level_values(level))

    for i in range(1, len(dfs)):
        if dfs[i] is not None:
            other_index = dfs[i] if level is not None and isinstance(dfs[i], pd.Index) else dfs[i].index
            if isinstance(other_index, pd.MultiIndex):
                other_index = pd.unique(other_index.get_level_values(level))

            intersect_index = intersect_index.intersection(other_index)

    return intersect_index.sort_values()
