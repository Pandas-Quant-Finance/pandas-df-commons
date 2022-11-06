from unittest import TestCase

import pandas as pd
import numpy as np

from pandas_df_commons.indexing.find_index import get_columns


class TestFindIndex(TestCase):

    def test_find_index(self):
        df = pd.DataFrame({"a": [1, 2, 3], 12: [1, 2, 3]})

        # regular access
        self.assertListEqual(get_columns(df, ["a"]).tolist(), [1, 2, 3])
        self.assertListEqual(get_columns(df, [12]).tolist(), [1, 2, 3])
        self.assertListEqual(get_columns(df, ["a", 12]).values.flatten().tolist(), [1, 1, 2, 2, 3, 3])
        self.assertListEqual(get_columns(df, [False, True, False]).values.flatten().tolist(), [2, 2])

        print(get_columns(df, "a"))
        print(get_columns(df, ["a", 12]))
        print(get_columns(df, ["a", 12, df["a"] * 2]))
        print(get_columns(df, [lambda f: [12] * len(f)]))
        # multi index
        # regex
