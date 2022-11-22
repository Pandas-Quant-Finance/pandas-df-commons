from unittest import TestCase

import pandas as pd

from pandas_df_commons.indexing._utils import same_columns_after_level
from pandas_df_commons.indexing.multiindex_utils import add_to_multi_index, get_top_level_of_multi_index


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

    def test_similar_columns_multi_index(self):
        df1 = pd.DataFrame({}, index=[1, 2, 3, 4], columns=pd.MultiIndex.from_product([["a", "b"], range(3)]))
        df2 = pd.DataFrame({}, index=[1, 2, 3, 4], columns=pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 3)]))

        self.assertTrue(same_columns_after_level(df1))
        self.assertFalse(same_columns_after_level(df2))

    def test_get_top_level_values(self):
        df = pd.DataFrame({"a": range(3), "b": range(3)})

        self.assertEquals(get_top_level_of_multi_index(df), (None, None))
        self.assertEquals(get_top_level_of_multi_index(pd.concat([df, df], axis=1, keys=[1, 2])), (None, [1, 2]))
        self.assertEquals(get_top_level_of_multi_index(pd.concat([df, df], axis=0, keys=[1, 2])), ([1, 2], None))
        self.assertEquals(
            get_top_level_of_multi_index(pd.concat([
                pd.concat([df, df], axis=1, keys=[3, 4]),
                pd.concat([df, df], axis=1, keys=[3, 4]),
            ], axis=0, keys=[1, 2])),
            ([1, 2], [3, 4])
        )

        # test corner case series
        self.assertEquals(get_top_level_of_multi_index(pd.concat([df["a"], df["a"]], axis=0, keys=[1, 2])), [1, 2])
