from unittest import TestCase

import os
import numpy as np
import pandas as pd

from pandas_df_commons.indexing.decorators import foreach_top_level_column, foreach_top_level_row, \
    foreach_top_level_row_and_column, foreach_column


def _frame(l):
    return pd.DataFrame({col: np.random.random(l) for col in ["open", "high", "low", "close"]})


class TestDecorators(TestCase):

    def test_multiindex_columns(self):
        keys = ["A", "B", "C"]
        frames = [_frame(100) for _ in keys]
        df = pd.concat(frames, axis=1, keys=keys)

        @foreach_top_level_column
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

        @foreach_top_level_row
        def compute_parallel(df, y, z):
            self.assertIsNotNone(df.index.name)
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
        this_pid = os.getpid()
        keys = ["A", "B", "C"]
        frames = [_frame(100) for _ in keys]
        df = pd.concat(
            [pd.concat(frames, axis=0, keys=keys) for _ in keys],
            axis=1,
            keys=keys
        )

        def compute(df, y, z):
            # should raise at least once
            # assert os.getpid() == this_pid
            return df.pct_change().fillna(y) + z

        @foreach_top_level_row_and_column(parallel=True)
        def compute_parallel(df, y, z):
            assert df.index.name is not None
            return compute(df, y, z)

        dfres = compute_parallel(df, 0.1, 12)

        for x in keys:
            for y in keys:
                pd.testing.assert_frame_equal(dfres.loc[x, y], compute(df.loc[x, y],  0.1, 12))

    def test_for_each_column(self):
        keys = ["A", "B", "C"]
        frames = [_frame(100) for _ in keys]
        df = pd.concat(frames, axis=1, keys=keys)

        @foreach_column
        def compute(x):
            return x

        pd.testing.assert_frame_equal(df, compute(df))

    def test_for_each_column_3lvl(self):
        keys = ["A", "B", "C"]
        frames = [_frame(100) for _ in keys]
        df = pd.concat(frames, axis=1, keys=keys)
        df = pd.concat([df] * 2, axis=1, keys=["1", "2"])

        @foreach_column
        def compute(x):
            return x

        pd.testing.assert_frame_equal(df, compute(df))

    def test_only_single_levels(self):
        df = pd.DataFrame({"A": range(10)})

        @foreach_top_level_row_and_column()
        def compute(x):
            return x

        pd.testing.assert_frame_equal(df, compute(df))
