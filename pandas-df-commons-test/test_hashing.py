from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_df_commons.hashing import hash_df


class TestHashingDf(TestCase):

    def test_hashes(self):
        df1 = pd.DataFrame(np.random.randint(0, 100, (100, 12)))
        df2 = pd.DataFrame(np.random.randint(0, 100, (100, 12)))

        self.assertEquals(hash_df(df1), hash_df(df1.apply(lambda x: x)))
        self.assertEquals(hash_df(df2), hash_df(df2.apply(lambda x: x)))
        self.assertNotEquals(hash_df(df1), hash_df(df2))
