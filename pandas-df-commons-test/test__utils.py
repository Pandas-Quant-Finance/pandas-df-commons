from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_df_commons._utils.multiprocessing import streaming_parallel
from pandas_df_commons._utils.patching import _add_functions


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

    def test_Patch(self):
        p1 = _add_functions('pandas_df_commons._utils.patching', filter=lambda x: x[1:] if x.startswith("_add") else None)(None)
        self.assertIsNotNone(getattr(p1, 'add_functions', None))

        p2 = _add_functions('pandas_df_commons._utils.patching', filter=lambda x: "foo." + x[1:] if x.startswith("_add") else None)(None)
        self.assertIsNotNone(getattr(p2.foo, 'add_functions', None))

        df = pd.DataFrame({1: np.linspace(10, 11, 5)})
        p3 = _add_functions('pandas_df_commons.extensions.functions', filter=lambda x: "foo.bar.baz." + x)(df)

        self.assertIsNotNone(getattr(p3.foo.bar.baz, 'cumpct_change', None))
        np.testing.assert_almost_equal(
            p3.foo.bar.baz.cumpct_change().values.squeeze(),
            np.array([0.0, 0.0249, 0.04999, 0.07499, 0.09999]),
            1.e-04
        )

        p4 = _add_functions('pandas_df_commons._utils.patching', filter=lambda x: "foo." + x[1:] if x.startswith("_add") else x)(None)
        self.assertIsNotNone(getattr(p4.foo, 'add_functions', None))
        self.assertIsNotNone(getattr(p4, '_monkey_patch_dataframe', None))