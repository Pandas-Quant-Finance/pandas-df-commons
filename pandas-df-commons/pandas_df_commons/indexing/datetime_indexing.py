from __future__ import annotations

from datetime import timedelta, datetime, date
from functools import partial
from itertools import chain
from typing import List, Tuple

import numpy as np
import pandas as pd


def forecast_time_index(
        from_tst: pd.Timestamp | datetime | date,
        nr_of_timesteps: int,
        timestep: timedelta = timedelta(days=1),
        only_weekdays=True,
        include_start_date=False
):
    tst = from_tst if isinstance(from_tst, pd.Timestamp) else pd.Timestamp(from_tst)
    forecast_tst = [tst] if include_start_date else []

    while len(forecast_tst) < nr_of_timesteps:
        tst += timestep

        if only_weekdays and tst.isoweekday() > 5:
            continue

        forecast_tst.append(tst)

    return pd.DatetimeIndex(forecast_tst, tz=tst.tz).sort_values()


def extend_time_indexed_dataframe(
        df: pd.DataFrame,
        nr_of_timesteps: int,
        timestep: timedelta = timedelta(days=1),
        only_weekdays: bool = True,
        return_level_counts: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, List[int]]:
    assert nr_of_timesteps > 0, f"{nr_of_timesteps} not > 0"

    make_index = partial(
        forecast_time_index, only_weekdays=only_weekdays
    )

    if df.index.nlevels > 1:
        # create the index at level 0 and get all counts of sub indexes needed
        indexes = df.index[-1]
        idx_level_0 = make_index(indexes[0], nr_of_timesteps, timestep, include_start_date=False)
        nr_lvl_timesteps = [len(df.loc[indexes[:l]]) for l in range(1, df.index.nlevels)]

        # construct an index tree
        idx_at_lvl = [[l] for l in idx_level_0]
        index_tree = [idx_level_0.tolist()]

        for nr in nr_lvl_timesteps:
            idx_at_lvl = [make_index(i, nr, -timestep, include_start_date=True).tolist() for j in idx_at_lvl for i in j]
            index_tree.append(list(chain(*idx_at_lvl)))

        # expand all nodes to have the same length but keep order
        for i, idx in enumerate(index_tree[:-1]):
            index_tree[i] = np.sort(idx * int(np.prod(np.array(nr_lvl_timesteps[i-1:]))))

        # create multi index tree
        extra_index = pd.MultiIndex.from_tuples(zip(*index_tree))
    else:
        nr_lvl_timesteps = []
        extra_index = make_index(df.index[-1], nr_of_timesteps, timestep, include_start_date=False)

    res = pd.concat([df, pd.DataFrame({}, index=extra_index)], axis=0, )
    return (res, nr_lvl_timesteps) if return_level_counts else res
