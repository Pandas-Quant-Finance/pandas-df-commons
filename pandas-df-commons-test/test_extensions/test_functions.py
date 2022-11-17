from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_df_commons.extensions.functions import cumpct_change, cumapply

df = pd.DataFrame({"A": range(1, 11)})


class TestExtensionFunctions(TestCase):

    def test_cumpct_change(self):
        np.testing.assert_almost_equal(cumpct_change(df).values.squeeze(), np.array(range(0, 10), dtype='float'))

    def test_cumapply(selfs):
        np.testing.assert_almost_equal(
            cumapply(df, lambda last, x: (last + x) / 2, 0).values.squeeze(),
            np.arange(0.5, 5.1, 0.5)
        )