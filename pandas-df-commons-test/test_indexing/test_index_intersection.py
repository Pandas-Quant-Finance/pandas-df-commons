from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_df_commons.indexing.intersection import intersection_of_index, aligend_frames


class TestIndexIntersect(TestCase):

    def test_intersection_of_index(self):
        df1 = pd.DataFrame({}, index=[1, 2, 3, 4])
        df2 = pd.DataFrame({}, index=[   2, 3, 4])
        df3 = pd.DataFrame({}, index=[1,    3, 4])

        index = intersection_of_index(df1, df2, df3)

        self.assertListEqual([3, 4], index.tolist())

    def test_intersection_multiindex_0(self):
        df1 = pd.DataFrame({}, index=pd.MultiIndex.from_product([[1, 2], [1, 3]]))
        df2 = pd.DataFrame({}, index=pd.MultiIndex.from_product([[1, 2], [1, 2]]))

        index = intersection_of_index(df1, df2)
        self.assertListEqual([(1, 1), (2, 1)], index.tolist())

    def test_intersection_mixed_multiindex_0(self):
        df1 = pd.DataFrame({}, index=[1, 2, 3, 4])
        df2 = pd.DataFrame({}, index=pd.MultiIndex.from_product([[1, 2], [1, 2]]))

        index = intersection_of_index(df1, df2, level=0)
        self.assertListEqual([1, 2], index.tolist())

    def test_intersection_mixed_multiindex_last(self):
        df1 = pd.DataFrame({}, index=[1, 2, 3, 4])
        df2 = pd.DataFrame({}, index=pd.MultiIndex.from_product([[1, 2], [1, 2]]))

        index = intersection_of_index(df1, df2, level=-1)
        self.assertListEqual([1, 2], index.tolist())

    def test_alligned_frames(self):
        expected_returns = pd.DataFrame({"A": np.random.normal(0, 2, 20), "B": np.random.random(20)})
        covariances = expected_returns.rolling(5).cov().dropna()
        risk_free_rate = pd.DataFrame({"risk_free_rate": 1e-3}, index=expected_returns.index)
        expected_returns, covariances, rf = aligend_frames(expected_returns, covariances, risk_free_rate, level=0)

        #print(expected_returns.shape, covariances.shape, rf.shape)
        self.assertListEqual(
            [expected_returns.shape, covariances.shape, rf.shape],
            [(16, 2), (32, 2), (16, 1)]
        )