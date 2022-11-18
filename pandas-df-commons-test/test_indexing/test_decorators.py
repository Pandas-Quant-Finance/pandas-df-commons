from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_df_commons.indexing.decorators import for_each_top_level_column, for_each_top_level_row, \
    for_each_top_level_row_and_column_parallel


def _frame(l):
    return pd.DataFrame({col: np.random.random(l) for col in ["open", "high", "low", "close"]})


class TestDecorators(TestCase):

    def test_multiindex_columns(self):
        keys = ["A", "B", "C"]
        frames = [_frame(100) for _ in keys]
        df = pd.concat(frames, axis=1, keys=keys)

        @for_each_top_level_column
        def compute_parallel(df, y, z):
            return compute(df, y, z)

        def compute(df, y, z):
            return df.pct_change().fillna(y) + z

        expected = {
            "A": compute(df["A"], 0.1, 12),
            "B": compute(df["B"], 0.1, 12),
            "C": compute(df["C"], 0.1, 12),
        }

        res = compute_parallel(df, 0.1, 12)

        pd.testing.assert_frame_equal(res["A"], expected["A"])
        pd.testing.assert_frame_equal(res["B"], expected["B"])
        pd.testing.assert_frame_equal(res["C"], expected["C"])

    def test_multiindex_rows(self):
        keys = ["A", "B", "C"]
        frames = [_frame(100) for _ in keys]
        df = pd.concat(frames, axis=0, keys=keys)

        @for_each_top_level_row
        def compute_parallel(df, y, z):
            return compute(df, y, z)

        def compute(df, y, z):
            return df.pct_change().fillna(y) + z

        expected = {
            "A": compute(df.loc["A"], 0.1, 12),
            "B": compute(df.loc["B"], 0.1, 12),
            "C": compute(df.loc["C"], 0.1, 12),
        }

        res = compute_parallel(df, 0.1, 12)

        pd.testing.assert_frame_equal(res.loc["A"], expected["A"])
        pd.testing.assert_frame_equal(res.loc["B"], expected["B"])
        pd.testing.assert_frame_equal(res.loc["C"], expected["C"])

    def test_multiindex_rows_and_columns_parallel(self):
        keys = ["A", "B", "C"]
        frames = [_frame(100) for _ in keys]
        df = pd.concat(
            [pd.concat(frames, axis=0, keys=keys) for _ in keys],
            axis=1,
            keys=keys
        )

        def compute(df, y, z):
            return df.pct_change().fillna(y) + z

        @for_each_top_level_row_and_column_parallel(parallel=True)
        def compute_parallel(df, y, z):
            return compute(df, y, z)

        dfres = compute_parallel(df, 0.1, 12)

        for x in keys:
            for y in keys:
                pd.testing.assert_frame_equal(dfres.loc[x, y], compute(df.loc[x, y],  0.1, 12))
