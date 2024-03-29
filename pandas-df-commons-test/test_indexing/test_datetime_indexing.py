from datetime import datetime
from unittest import TestCase

import pandas as pd
from pandas import Timestamp as ts

from pandas_df_commons.extensions.functions import rolling_apply
from pandas_df_commons.indexing.datetime_indexing import forecast_time_index, extend_time_indexed_dataframe


class TestDateTimeIndexing(TestCase):

    def test_extending_timestamps(self):
        res = forecast_time_index(datetime.fromisoformat("2020-01-01"), 12, only_weekdays=True)
        pd.testing.assert_index_equal(
            res,
            pd.DatetimeIndex([
                '2020-01-02', '2020-01-03', '2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09', '2020-01-10',
                '2020-01-13', '2020-01-14', '2020-01-15', '2020-01-16', '2020-01-17'
            ]),
        )

    def test_extend_time_indexed_dataframe(self):
        df = pd.DataFrame({1: range(13)}, index=forecast_time_index(datetime.fromisoformat("2020-01-01"), 13, only_weekdays=True))
        res = extend_time_indexed_dataframe(df, 3)

        self.assertEquals(len(df) + 3, len(res))

    def test_extend_time_indexed_dataframe_multi_index(self):
        df = pd.DataFrame({1: range(13)}, index=forecast_time_index(datetime.fromisoformat("2020-01-01"), 13, only_weekdays=True))
        df = rolling_apply(df, 4, lambda x: x)

        res = extend_time_indexed_dataframe(df, 3)
        # print(res.loc[df.index[-1][0]:].index)

        self.assertListEqual(
            res.loc[df.index[-1][0]:].index.tolist(),
            pd.MultiIndex.from_tuples(
                [
                    (ts('2020-01-20'), ts('2020-01-15')),
                    (ts('2020-01-20'), ts('2020-01-16')),
                    (ts('2020-01-20'), ts('2020-01-17')),
                    (ts('2020-01-20'), ts('2020-01-20')),
                    (ts('2020-01-21'), ts('2020-01-16')),
                    (ts('2020-01-21'), ts('2020-01-17')),
                    (ts('2020-01-21'), ts('2020-01-20')),
                    (ts('2020-01-21'), ts('2020-01-21')),
                    (ts('2020-01-22'), ts('2020-01-17')),
                    (ts('2020-01-22'), ts('2020-01-20')),
                    (ts('2020-01-22'), ts('2020-01-21')),
                    (ts('2020-01-22'), ts('2020-01-22')),
                    (ts('2020-01-23'), ts('2020-01-20')),
                    (ts('2020-01-23'), ts('2020-01-21')),
                    (ts('2020-01-23'), ts('2020-01-22')),
                    (ts('2020-01-23'), ts('2020-01-23')),
                ]
           ).tolist()
        )

        #self.assertEquals(len(df) + 3, len(res))
