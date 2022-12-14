from unittest import TestCase

import pandas as pd

from pandas_df_commons.indexing.intersection import intersection_of_index


class TestIndexIntersect(TestCase):

    def test_intersection_of_index(self):
        df1 = pd.DataFrame({}, index=[1, 2, 3, 4])
        df2 = pd.DataFrame({}, index=[   2, 3, 4])
        df3 = pd.DataFrame({}, index=[1,    3, 4])

        index = intersection_of_index(df1, df2, df3)

        self.assertListEqual([3, 4], index.tolist())

    def test_intersection_and_lavel(self):
        df1 = pd.DataFrame({}, index=[1, 2, 3, 4])
        df2 = pd.DataFrame({}, index=pd.MultiIndex.from_product([[1, 2], [1, 2]]))

        index = intersection_of_index(df1, df2, level=0)
        self.assertListEqual([1, 2], index.tolist())