from __future__ import annotations

import pandas as pd
from pandas.core.base import PandasObject


def _col_op(op, a: pd.DataFrame, b: pd.DataFrame | pd.Series | float):
    if isinstance(b, PandasObject):
        if b.ndim <= 1: b = b.to_frame()
        b = b.join(pd.DataFrame({}, index=a.index), how='right')
        b = b.loc[a.index.intersection(b.index)]
        bv = b.values
        if a.ndim <= 1: bv = bv.squeeze()

        a = a.copy()
        if op == '/':
            a.loc[b.index.intersection(a.index)] /= bv
        elif op == '*':
            a.loc[b.index.intersection(a.index)] *= bv
        elif op == '+':
            a.loc[b.index.intersection(a.index)] += bv
        elif op == '-':
            a.loc[b.index.intersection(a.index)] -= bv
        else:
            raise ValueError(f"Unknown operation {op}")
    else:
        if op == '/':
            a /= b
        elif op == '*':
            a *= b
        elif op == '+':
            a += b
        elif op == '-':
            a -= b
        else:
            raise ValueError(f"Unknown operation {op}")

    return a

def col_div(a: pd.DataFrame, b: pd.DataFrame | pd.Series | float):
    return _col_op('/', a, b)


def col_mul(a: pd.DataFrame, b: pd.DataFrame | pd.Series | float):
    return _col_op('*', a, b)


def col_add(a: pd.DataFrame, b: pd.DataFrame | pd.Series | float):
    return _col_op('+', a, b)


def col_sub(a: pd.DataFrame, b: pd.DataFrame | pd.Series | float):
    return _col_op('-', a, b)


