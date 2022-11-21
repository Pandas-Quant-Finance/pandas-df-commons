from __future__ import annotations

import re
from collections import abc
from typing import (
    TYPE_CHECKING,
    Hashable,
)

import numpy as np
import pandas as pd
from pandas.core.indexes.api import (
    Index,
)
from pandas.core.indexes.multi import (
    MultiIndex,
)
from pandas.core.series import Series

if TYPE_CHECKING:
    pass


def get_columns(frame: pd.DataFrame | pd.Series, keys) -> pd.DataFrame | pd.Series | None:
    """
    Alternative way to df[x] or df.loc[:, x] where __get_item__ behaves in a similar way as `df.set_index` or `df.groupby`.
    But additionally allows to partially match a column by a part of a MultiIndex or a regex embedded in ~/:the regex:/.
    Constant values could be modelled using a lambda like df[lambda f: [12] * len(f)]

    :param frame: the DataFrame (Series will be converted to DataFrame)
    :param keys: the keys or values to be found or returned
    :return: returns the columns of the dataframe of the values p
    """

    if keys is None:
        return None

    ## NOTICE !!!
    ## This is a modified copy of DataFrame.set_index()
    if not isinstance(keys, list):
        keys = [keys]
    elif isinstance(keys, slice):
        # if it is a slices we operate on the index not the column
        return frame[keys]

    # if all elements are boolean we just use as index mask
    all_bool = True
    for k in keys:
        if not isinstance(k, bool):
            all_bool = False
            break
    if all_bool and len(keys) == len(frame):
        return frame[pd.Index(keys)]

    # for convenience, we need to convert to frame
    if hasattr(frame, "to_frame"):
        frame = frame.to_frame()

    # if key is a callable we need to replace it first
    keys = [key(frame) if callable(key) else key for key in keys]

    err_msg = (
        'The parameter "keys" may be a column key, one-dimensional '
        "array, or a list containing only valid column keys and "
        "one-dimensional arrays."
    )

    missing: list[Hashable] = []
    search_keys = []
    for col in keys:
        if isinstance(col, (Index, Series, np.ndarray, list, abc.Iterator)):
            # arrays are fine as long as they are one-dimensional
            # iterators get converted to list below
            if getattr(col, "ndim", 1) != 1:
                raise ValueError(err_msg)
            else:
                search_keys.append(col)
        else:
            try:
                found = True

                # everything else gets tried as a key; see GH 24969
                if col in frame.columns:
                    search_keys.append(col)

                # could be matching one level of a multiindex
                elif isinstance(frame.columns, MultiIndex):
                    for column_tuple in frame.columns.tolist():
                        if col in column_tuple:
                            search_keys.append(column_tuple)

                        # could be regex ~/<regex>/
                        elif isinstance(col, str) and (col.startswith("~") and col[1] == col[-1] and len(col) > 3):
                            column_matcher = "/".join([str(c).strip('\"') for c in column_tuple])
                            cregex = col[2:-1]
                            if re.compile(cregex).match(column_matcher):
                                search_keys.append(column_tuple)

                # could be regex ~/<regex>/
                elif isinstance(col, str) and (col.startswith("~") and col[1] == col[-1] and len(col) > 3):
                    cregex = col[2:-1]
                    for c in frame.columns.tolist():
                        if re.compile(cregex).match(str(c).strip('\"')):
                            search_keys.append(c)

                else:
                    found = False
            except TypeError as err:
                raise TypeError(
                    f"{err_msg}. Received column of type {type(col)}"
                ) from err
            else:
                if not found:
                    missing.append(col)

    if missing:
        raise KeyError(f"None of {missing} are in the columns")

    arrays = []
    names: list[Hashable] = []

    for col in search_keys:
        if isinstance(col, MultiIndex):
            for n in range(col.nlevels):
                arrays.append(col._get_level_values(n))
            names.extend(col.names)
        elif isinstance(col, (Index, Series)):
            # if Index then not MultiIndex (treated above)

            # error: Argument 1 to "append" of "list" has incompatible type
            #  "Union[Index, Series]"; expected "Index"
            arrays.append(col)  # type:ignore[arg-type]
            names.append(col.name)
        elif isinstance(col, (list, np.ndarray)):
            # error: Argument 1 to "append" of "list" has incompatible type
            # "Union[List[Any], ndarray]"; expected "Index"
            arrays.append(col)  # type: ignore[arg-type]
            names.append(None)
        elif isinstance(col, abc.Iterator):
            # error: Argument 1 to "append" of "list" has incompatible type
            # "List[Any]"; expected "Index"
            arrays.append(list(col))  # type: ignore[arg-type]
            names.append(None)
        # from here, col can only be a column label
        else:
            sub_frame = frame[[col]]
            sub_values = sub_frame._values
            arrays.extend(sub_values.T)
            names.extend(sub_frame.columns.tolist())

        if len(arrays[-1]) != len(frame):
            # check newest element against length of calling frame, since
            # ensure_index_from_sequences would not raise for append=False.
            raise ValueError(
                f"Length mismatch: Expected {len(frame)} rows, "
                f"received array of length {len(arrays[-1])}"
            )

    # just instead of an index we return a DataFrame or Series
    df = pd.DataFrame(
        arrays,
        index=pd.MultiIndex.from_tuples(names) if isinstance(frame.columns, pd.MultiIndex) and len(names) > 1 else names,
        columns=frame.index
    ).T

    if df.shape[1] > 1:
        return df
    elif len(df.columns) > 0:
        return df[df.columns[0]]
    else:
        return pd.DataFrame({}, index=df.index)
