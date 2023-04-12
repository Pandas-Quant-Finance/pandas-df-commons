from unittest import TestCase
from flaky import flaky

import pandas as pd

from pandas_df_commons.indexing._utils import get_top_level_rows, top_level_separator_generator, get_top_level_columns


class TestIndexingUtils(TestCase):

    @flaky(max_runs=3)
    def test_top_level_shuffling(self):
        df1 = pd.DataFrame({}, index=pd.MultiIndex.from_product([range(50), range(50)]))
        df2 = pd.DataFrame({}, columns=pd.MultiIndex.from_product([range(50), range(50)]))
        df3 = pd.DataFrame({}, index=pd.MultiIndex.from_product([range(10), range(10)]), columns=pd.MultiIndex.from_product([range(10), range(10)]))

        top_level_rows = get_top_level_rows(df1)
        top_level_columns = get_top_level_columns(df1)
        self.assertNotEqual(
            [x for (_, x), _ in top_level_separator_generator(df1, top_level_rows, top_level_columns, shuffle_top_level_rows=True, shuffle_level_columns=True)],
            [x for (_, x), _ in top_level_separator_generator(df1, top_level_rows, top_level_columns, shuffle_top_level_rows=True, shuffle_level_columns=True)]
        )

        top_level_rows = get_top_level_rows(df2)
        top_level_columns = get_top_level_columns(df2)
        self.assertNotEqual(
            [y for (y, _), _ in top_level_separator_generator(df2, top_level_rows, top_level_columns, shuffle_top_level_rows=True, shuffle_level_columns=True)],
            [y for (y, _), _ in top_level_separator_generator(df2, top_level_rows, top_level_columns, shuffle_top_level_rows=True, shuffle_level_columns=True)]
        )

        top_level_rows = get_top_level_rows(df3)
        top_level_columns = get_top_level_columns(df3)
        self.assertNotEqual(
            [z for z, _ in top_level_separator_generator(df3, top_level_rows, top_level_columns, shuffle_top_level_rows=True, shuffle_level_columns=True)],
            [z for z, _ in top_level_separator_generator(df3, top_level_rows, top_level_columns, shuffle_top_level_rows=True, shuffle_level_columns=True)]
        )

        top_level_rows = get_top_level_rows(df3)
        top_level_columns = get_top_level_columns(df3)
        self.assertNotEqual(
            [x for (_, x), _ in top_level_separator_generator(df3, top_level_rows, top_level_columns, shuffle_top_level_rows=True, shuffle_level_columns=True)],
            [x for (_, x), _ in top_level_separator_generator(df3, top_level_rows, top_level_columns, shuffle_top_level_rows=True, shuffle_level_columns=True)]
        )
        self.assertNotEqual(
            [y for (y, _), _ in top_level_separator_generator(df3, top_level_rows, top_level_columns, shuffle_top_level_rows=False, shuffle_level_columns=True)],
            [y for (y, _), _ in top_level_separator_generator(df3, top_level_rows, top_level_columns, shuffle_top_level_rows=False, shuffle_level_columns=True)]
        )
        self.assertListEqual(
            [x for (_, x), _ in top_level_separator_generator(df3, top_level_rows, top_level_columns, shuffle_top_level_rows=False, shuffle_level_columns=True)],
            [x for (_, x), _ in top_level_separator_generator(df3, top_level_rows, top_level_columns, shuffle_top_level_rows=False, shuffle_level_columns=True)]
        )
        self.assertListEqual(
            [y for (y, _), _ in top_level_separator_generator(df3, top_level_rows, top_level_columns, shuffle_top_level_rows=True, shuffle_level_columns=False)],
            [y for (y, _), _ in top_level_separator_generator(df3, top_level_rows, top_level_columns, shuffle_top_level_rows=True, shuffle_level_columns=False)]
        )
