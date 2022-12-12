from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_df_commons.extensions.functions import cumpct_change, cumapply, rolling_apply, sateful_apply

df = pd.DataFrame({"A": range(1, 11)})


class TestExtensionFunctions(TestCase):

    def test_cumpct_change(self):
        np.testing.assert_almost_equal(cumpct_change(df).values.squeeze(), np.array(range(0, 10), dtype='float'))

    def test_cumapply(self):
        np.testing.assert_almost_equal(
            cumapply(df, lambda last, x: (last + x) / 2, 0).values.squeeze(),
            np.arange(0.5, 5.1, 0.5)
        )

    def test_rolling_apply(self):
        df = pd.DataFrame({"A": range(1, 11), "B": range(1, 11)})
        res = rolling_apply(df, 3, lambda x: x.sum().sum())
        res1 = rolling_apply(df, 3, lambda x: x.sum(axis=1).reset_index(drop=True))
        res4 = rolling_apply(df, 3, lambda x: x.sum(axis=0))
        res2 = rolling_apply(df, 3, lambda x: x * 2)
        res3 = rolling_apply(df, 3, lambda x: pd.concat([x * 2, x * 3]))

        self.assertEquals(res.shape, (8, 1))
        self.assertEquals(res1.shape, (8, 3))
        self.assertEquals(res4.shape, (8, 2))
        self.assertEquals(res2.shape, (24, 2))
        self.assertEquals(res3.shape, (48, 2))

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
