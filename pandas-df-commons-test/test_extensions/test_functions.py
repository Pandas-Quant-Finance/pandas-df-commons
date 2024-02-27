from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_df_commons import _extender
from pandas_df_commons.extensions.functions import cumpct_change, cumapply, rolling_apply, sateful_apply, joint_apply, \
    p1cumprod, row_apply

df = pd.DataFrame({"A": range(1, 11)})


class TestExtensionFunctions(TestCase):

    def test_cumpct_change(self):
        np.testing.assert_almost_equal(cumpct_change(df).values.squeeze(), np.array(range(0, 10), dtype='float'))

    def test_cumapply(self):
        np.testing.assert_almost_equal(
            cumapply(df, lambda last, x: (last + x) / 2, 0).values.squeeze(),
            np.arange(0.5, 5.1, 0.5)
        )

    def test_cumapply_1pcumprod(self):
        _df = df.copy()
        _df["B"] = _df["A"]

        np.testing.assert_almost_equal(
            row_apply(_df, p1cumprod).values,
            np.array(
                [[2,  4],
                 [3,  9],
                 [4, 16],
                 [5, 25],
                 [6, 36],
                 [7, 49],
                 [8, 64],
                 [9, 81],
                 [10, 100],
                 [11, 121]]
            )
        )

    def test_rolling_apply(self):
        df = pd.DataFrame({"A": range(1, 11), "B": range(1, 11)})
        res = rolling_apply(df, 3, lambda x: x.sum().sum())
        res1 = rolling_apply(df, 3, lambda x: x.sum(axis=1).reset_index(drop=True))
        res4 = rolling_apply(df, 3, lambda x: x.sum(axis=0))
        res2 = rolling_apply(df, 3, lambda x: x * 2)
        res3 = rolling_apply(df, 3, lambda x: pd.concat([x * 2, x * 3]))
        res5 = rolling_apply(df, 3, lambda x: p1cumprod(x))

        self.assertEquals(res.shape, (8, 1))
        self.assertEquals(res1.shape, (8, 3))
        self.assertEquals(res4.shape, (8, 2))
        self.assertEquals(res2.shape, (24, 2))
        self.assertEquals(res3.shape, (48, 2))
        self.assertEquals(res5.shape, (24, 2))

    def test_rolling_apply_parallel(self):
        df = pd.DataFrame({"A": range(1, 11), "B": range(1, 11)})
        _ = rolling_apply(df, 3, lambda x: pd.concat([x * 2, x * 3]), parallel=True)
        res = rolling_apply(df, 3, lambda x: pd.concat([x * 2, x * 3]), parallel=True)

        self.assertEquals(res.shape, (48, 2))

    def test_stateful_apply(self):
        df = pd.DataFrame({"A": range(1, 11), "B": range(1, 11)})
        def calc(s, x):
            return s - 1, x+s

        res = sateful_apply(df, calc, 12, axis=1)
        self.assertEquals(res["A"].tolist(), [13] * 10)
        self.assertEquals(res["B"].tolist(), [13] * 10)

    def test_joint_apply(self):
        df1 = pd.DataFrame({"A": range(1, 11), "B": range(1, 11)}, index=range(1, 11))
        df2 = pd.DataFrame({"A": range(0, 10), "B": range(0, 10)}, index=range(0, 10))

        res = joint_apply(df1, df2, func=lambda x: x[0].sum() + x[1].sum(), parallel=True)
        self.assertListEqual(res.index.tolist(), list(range(1, 10)))
        self.assertListEqual(res[0].tolist(), [4, 8, 12, 16, 20, 24, 28, 32, 36])

    def test_for_toplevel_row(self):
        df = pd.concat([pd.DataFrame({"A": range(1, 11)})] * 2, axis=0, keys=[0, 1])
        res = _extender(df).for_toplevel_row(lambda x: x + 1)
        pd.testing.assert_frame_equal(res, df + 1)
