import os
from unittest import TestCase

import numpy as np
import pandas as pd
from flaky import flaky

from pandas_df_commons._utils.batch import Batch
from pandas_df_commons._utils.multiprocessing import streaming_parallel
from pandas_df_commons._utils.patching import _add_functions
from pandas_df_commons._utils.rescaler import ReScaler
from pandas_df_commons._utils.streaming import window, Window, IterRows
from pandas_df_commons.extensions.functions import rescale


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

    def test_streaming_parallel_pids(self):
        l = 100
        results = set(streaming_parallel(
            lambda x: os.getpid(),
            lambda: range(l)
        ))

        print(results)
        self.assertGreater(len(results), 1)

    def test_Patch(self):
        p1 = _add_functions('pandas_df_commons._utils.patching', filter=lambda _, x: x[1:] if x.startswith("_add") else None)(None)
        self.assertIsNotNone(getattr(p1, 'add_functions', None))

        p2 = _add_functions('pandas_df_commons._utils.patching', filter=lambda _, x: "foo." + x[1:] if x.startswith("_add") else None)(None)
        self.assertIsNotNone(getattr(p2.foo, 'add_functions', None))

        df = pd.DataFrame({1: np.linspace(10, 11, 5)})
        p3 = _add_functions('pandas_df_commons.extensions.functions', filter=lambda _, x: "foo.bar.baz." + x)(df)

        self.assertIsNotNone(getattr(p3.foo.bar.baz, 'cumpct_change', None))
        np.testing.assert_almost_equal(
            p3.foo.bar.baz.cumpct_change().values.squeeze(),
            np.array([0.0, 0.0249, 0.04999, 0.07499, 0.09999]),
            1.e-04
        )

        p4 = _add_functions('pandas_df_commons._utils.patching', filter=lambda _, x: "foo." + x[1:] if x.startswith("_add") else x)(None)
        self.assertIsNotNone(getattr(p4.foo, 'add_functions', None))
        self.assertIsNotNone(getattr(p4, '_monkey_patch_dataframe', None))

    def test_rescaling_rows(self):
        self.assertEquals(ReScaler((10, 8), (1, -1))(10), 1)
        self.assertEquals(ReScaler((10, 8), (1, -1))(8), -1)
        self.assertEquals(ReScaler((10, 8), (1, -1))(9), 0)

        self.assertEquals(ReScaler((8, 10), (1, -1))(10), -1)
        self.assertEquals(ReScaler((8, 10), (1, -1))(8), 1)
        self.assertEquals(ReScaler((8, 10), (1, -1))(9), 0)

        self.assertEquals(ReScaler((8, 10), (1, -1))(10), -1)
        self.assertEquals(ReScaler((8, 10), (1, -1))(8), 1)
        self.assertEquals(ReScaler((8, 10), (1, -1))(9), 0)

        df = pd.DataFrame({"a": range(10), "b": range(10)})
        print(rescale(df, (0, 1)))
        print(rescale(df["a"], (0, 1)))
        print(rescale(df, (0, 1), axis=0))
        print(rescale(df.assign(b=5), (0, 1), axis=1))

    @flaky(max_runs=3)
    def test_window(self):
        df = pd.DataFrame({"a": range(10), "b": range(10)})
        ordered_windows = [w for w in window(df, 3)]
        # print(ordered_windows)
        self.assertEqual(len(ordered_windows), 10-3+1)
        self.assertListEqual(ordered_windows[0].index.tolist(), [0, 1, 2])
        self.assertListEqual(ordered_windows[-1].index.tolist(), [7, 8, 9])

        shuffled_windows = [w for w in window(df, 3, shuffle=True)]
        # print(shuffled_windows)
        self.assertEqual(len(shuffled_windows), 10 - 3 + 1)
        self.assertNotEqual(shuffled_windows[0].index.tolist()[0], 0)
        self.assertNotEqual(shuffled_windows[-1].index.tolist()[-1], 9)

    def test_IterRows(self):
        df = pd.DataFrame({"a": range(10), "b": range(10)})
        it = IterRows(df)
        pd.testing.assert_frame_equal(df, pd.DataFrame([r for r in it]))
        pd.testing.assert_frame_equal(df, pd.DataFrame([it[i] for i in range(len(df))]))

    def test_Window(self):
        df = pd.DataFrame({"a": range(10), "b": range(10)})
        windows = Window(df, 3)
        for i, w in enumerate(windows):
            pd.testing.assert_frame_equal(w, windows[i])

        self.assertEquals(2, len(windows[:2]))

    def test_batch(self):
        batches = list(Batch(range(21), 10))
        self.assertEqual(len(batches), 3)
        self.assertListEqual(batches[0], list(range(10)))
        self.assertListEqual(batches[1], list(range(10, 20)))
        self.assertListEqual(batches[2], list(range(20, 21)))

        self.assertListEqual(batches[2], list(Batch(range(21), 10)[2]))

        df = pd.DataFrame({"a": range(10), "b": range(10)})
        batches = Batch(df, 3)
        for i, b in enumerate(batches):
            print(b)