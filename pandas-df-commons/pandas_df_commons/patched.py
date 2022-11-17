import pandas as pd
from pandas_df_commons import monkey_patch_dataframe

print("pandas version", pd.__version__)
monkey_patch_dataframe()