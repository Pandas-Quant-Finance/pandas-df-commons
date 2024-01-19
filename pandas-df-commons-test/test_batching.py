from unittest import TestCase

import pandas as pd

from pandas_df_commons.batching import BatchList


class TestBatchList(TestCase):

    def test_batchlist_len(self):
        sample = pd.DataFrame([range(11)]).T

        batch = BatchList(sample, 3, 9)
        self.assertEqual(len(batch), 1)  # one batch of len=3
        self.assertEqual(len(batch[0]), 3)

        batch = BatchList(sample, 3, 8)
        self.assertEqual(len(batch), 2)  # one batch of len=2 and one of len=1
        self.assertEqual(len(batch[0]), 1)
        self.assertEqual(len(batch[1]), 3)

    def test_batchlist_len_non_overlap(self):
        sample = pd.DataFrame([range(11)]).T

        batch = BatchList(sample, 4, 2, overlapping_windows=False)
        self.assertEqual(len(batch), 2)  # one batch of len=3
        self.assertEqual(len(batch[0]), 1)
        self.assertEqual(len(batch[1]), 4)

        batch = BatchList(sample, 2, 2, overlapping_windows=False)
        self.assertEqual(len(batch), 3)  # one batch of len=2 and one of len=1
        self.assertEqual(len(batch[0]), 1)
        self.assertEqual(len(batch[1]), 2)
        self.assertEqual(len(batch[2]), 2)
