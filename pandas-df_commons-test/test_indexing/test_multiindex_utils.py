from unittest import TestCase

import pandas as pd

from pandas_df_commons.indexing.multiindex_utils import add_to_multi_index


class TestMultiIndexUtils(TestCase):

    def test_add_multi_index(self):
        df = pd.DataFrame({}, index=[1, 2, 3, 4])
        df1 = add_to_multi_index(df, "A", axis=0)
        df2 = add_to_multi_index(df1, "B", axis=0, level=1)
        df3 = add_to_multi_index(df1, "B", axis=0, level=2)
        # print(df3)

        self.assertListEqual(df1.index.to_list(), [("A", 1), ("A", 2), ("A", 3), ("A", 4)])
        self.assertListEqual(df2.index.to_list(), [("A", "B", 1), ("A", "B", 2), ("A", "B", 3), ("A", "B", 4)])
        self.assertListEqual(df3.index.to_list(), [(1, "A", "B"), (2, "A", "B"), (3, "A", "B"), (4, "A", "B")])
