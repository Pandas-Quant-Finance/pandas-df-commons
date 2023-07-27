from unittest import TestCase

import pandas as pd

from pandas_df_commons.indexing._utils import same_columns_after_level
from pandas_df_commons.indexing.multiindex_utils import add_to_multi_index, get_top_level_of_multi_index, nth, \
    index_shape, index_counts, last_index


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

    def test_nth(self):
        df=pd.DataFrame({"a": range(8)})
        self.assertEquals(len(nth(df, 3)), 3)
        self.assertEquals(len(nth(df, 3, include_last=True)), 4)

        dfmi = pd.DataFrame({"a": range(8 * 2)}, index=pd.MultiIndex.from_product([range(8), range(2)]))
        self.assertEquals(len(nth(dfmi, 3, level=0)), 3*2)
        self.assertEquals(len(nth(dfmi, 3, level=0, include_last=True)), 4*2)

    def test_multiindex_counts(self):
        self.assertDictEqual(index_counts(pd.Index(range(4))), {0: 4})
        self.assertDictEqual(
            index_counts(pd.MultiIndex.from_product([range(4), range(3)])),
            {i: {0: 3} for i in range(4)}
        )
        self.assertDictEqual(
            index_counts(pd.MultiIndex.from_product([range(4), range(2), range(3)])),
            {i: {j: {0: 3} for j in range(2)} for i in range(4)}
        )

        self.assertEquals(index_counts(pd.Index(range(0))), {0: 0})
        self.assertEquals(index_counts(pd.MultiIndex.from_product([[], []])), {0: {0: 0}},)
        self.assertEquals(
            index_counts(pd.MultiIndex.from_tuples([(1, 0), (1, 1), (0, 0), (0, 1), (0, 2)])),
            {
                1: {0: 2},
                0: {0: 3},
            }
        )
        self.assertEquals(
            index_counts(pd.MultiIndex.from_tuples([(1, 0, 0), (1, 1, 0), (0, 0, 0), (0, 1, 0), (0, 2, 0)])),
            {
                1:{ 0: {0: 1}, 1: {0: 1} },
                0:{ 0: {0: 1}, 1: {0: 1}, 2: {0: 1} },
            }
        )
        self.assertDictEqual(
            index_counts(pd.MultiIndex.from_tuples([(1, 0, 0), (1, 1, 0), (0, 0, 0), (0, 1, 0), (0, 1, 2)])),
            {
                1: {0: {0: 1}, 1: {0: 1}},
                0: {0: {0: 1}, 1: {0: 2}}
            }
        )

    def test_multiindex_shape(self):
        self.assertEquals(index_shape(pd.MultiIndex.from_tuples([(1,0), (1,1), (0,0), (0,1), (0,2)])), (5,))
        self.assertEquals(index_shape(pd.MultiIndex.from_product([range(4), range(3)])), (4, 3))

        self.assertEquals(index_shape(pd.Index(range(4))), (4, ))
        self.assertEquals(index_shape(pd.MultiIndex.from_product([range(4), range(3)])), (4, 3))
        self.assertEquals(index_shape(pd.MultiIndex.from_product([range(4), range(2), range(3)])), (4, 2, 3))
        self.assertEquals(index_shape(pd.MultiIndex.from_product([range(4), range(2), range(3), range(5)])), (4, 2, 3, 5))

        self.assertEquals(index_shape(pd.Index(range(0))), (0,))
        self.assertEquals(index_shape(pd.MultiIndex.from_product([[], []])), (0, 0))

        self.assertEquals(index_shape(pd.MultiIndex.from_tuples([(1,0), (1,1), (0,0), (0,1), (0,2)])), (5,))
        self.assertEquals(index_shape(pd.MultiIndex.from_tuples([(1,0,0), (1,1,0), (0,0,0), (0,1,0), (0,2,0)])), (5,))
        self.assertEquals(index_shape(pd.MultiIndex.from_tuples([(1,0,0), (1,1,0), (0,0,0), (0,1,0), (0,1,2)])), (5,))

        df = pd.DataFrame({"a": range(4*3*2), "b": range(4*3*2)}, index=pd.MultiIndex.from_product([range(4), range(2), range(3)]))
        self.assertEquals(index_shape(df), (4, 2, 3))
        df.values.reshape((4, 3, 2, 2))

    def test_last_index(self):
        df = pd.DataFrame({"a": range(24)}, pd.MultiIndex.from_product([range(4), range(3), range(2)]))
        idx = last_index(df)

        pd.testing.assert_index_equal(idx, pd.MultiIndex.from_product([range(4), range(3), [1]]), check_names=False)
        self.assertEquals(len(df.loc[idx]), 4 * 3)
