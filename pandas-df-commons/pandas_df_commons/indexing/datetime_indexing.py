from datetime import timedelta, datetime, date
from functools import partial

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

    return pd.DatetimeIndex(forecast_tst, tz=tst.tz)


def extend_time_indexed_dataframe(
        df: pd.DataFrame,
        nr_of_timesteps: int,
        timestep: timedelta = timedelta(days=1),
        only_weekdays: bool = True
) -> pd.DataFrame:
    assert nr_of_timesteps > 1, f"{nr_of_timesteps} not > 1"

    make_index = partial(
        forecast_time_index, timestep=timestep, only_weekdays=only_weekdays, include_start_date=False
    )

    if df.index.nlevels > 1:
        indexes = df.index[-1]
        extra_index = pd.MultiIndex.from_product(
            [make_index(indexes[0], len(df.loc[indexes[l - 1]]) if l > 0 else nr_of_timesteps) for l in range(df.index.nlevels)]
        )
    else:
        extra_index = make_index(df.index[-1], nr_of_timesteps)

    res = pd.concat([df, pd.DataFrame({}, index=extra_index)], axis=0, )
    return res
