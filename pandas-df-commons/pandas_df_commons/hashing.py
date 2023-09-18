import hashlib

import pandas as pd


def hash_df(df: pd.DataFrame):
    return int(hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest(), 16)
