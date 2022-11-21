import pandas as pd


def intersection_of_index(*dfs: pd.DataFrame):
    intersect_index = dfs[0].index

    for i in range(1, len(dfs)):
        if dfs[i] is not None:
            intersect_index = intersect_index.intersection(dfs[i].index)

    return intersect_index.sort_values()
