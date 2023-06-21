from unittest import TestCase

import pandas as pd

from pandas_df_commons.math import col_div


class TestMath(TestCase):

    def test_div(self):
        a = pd.DataFrame({"a": range(10), "b": range(10)})

        pd.testing.assert_frame_equal(
            col_div(a, pd.DataFrame({"c": range(1, 11)}, index=range(1, 11))),
            col_div(a, pd.Series(range(1, 11), index=range(1, 11)))
        )

        print(col_div(a, 12))