from unittest import TestCase

import pandas as pd

import pandas_df_commons as pdc

# patch dataframe
pdc.monkey_patch_dataframe()


class TestExtensionFunctions(TestCase):

    def test_cumprod(self):
        pass

    def test_extensions(self):
        df = pd.DataFrame({})
        self.assertEquals(str(type(df.X)), "<class 'pandas_df_commons.monkey_patch_dataframe.<locals>.Extendor'>")