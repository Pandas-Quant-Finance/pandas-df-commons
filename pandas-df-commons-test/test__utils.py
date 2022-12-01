from unittest import TestCase

from pandas_df_commons._utils.multiprocessing import streaming_parallel


class Test_Utils(TestCase):

    def test_streaming_parallel(self):
        l = 100
        results = streaming_parallel(
            lambda x: x * 2,
            lambda: range(l)
        )

        self.assertListEqual(
            results,
            [x * 2 for x in range(l)]
        )
