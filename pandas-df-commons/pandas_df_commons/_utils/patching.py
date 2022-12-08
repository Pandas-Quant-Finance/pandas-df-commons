from __future__ import annotations
import importlib
from functools import partial, wraps
from typing import Callable

from pandas.core.base import PandasObject


def _monkey_patch_dataframe(extension_field_name=None, extension_default_value=None, extension_class=None):
    if extension_field_name is None:
        extension_field_name = extension_default_value

    existing = getattr(PandasObject, extension_field_name, None)
    if existing is not None:
        if not isinstance(existing, property) or not isinstance(existing.fget(None), extension_class):
            raise ValueError(f"field already exists as {type(existing)}")

    setattr(PandasObject, extension_field_name, property(lambda self: extension_class(self)))


def _add_functions(*modules, filter: Callable[[str, str], str|None]):
    functions = {}
    for m in modules:
        for name, func in importlib.import_module(m).__dict__.items():
            if not callable(func):
                continue

            n = filter(m, name) if filter is not None else name
            if n is not None:
                functions[n] = func

    class _Path(object):

        def __init__(self,):
            pass

    class _Patch(object):

        def __init__(self, df):
            self.df = df

            for name, func in functions.items():
                path = name.split(".")
                props_dict = self.__dict__

                for i in range(len(path) -1):
                    if path[i] not in props_dict:
                        props_dict[path[i]] = _Path()

                    props_dict = props_dict[path[i]].__dict__

                props_dict[path[-1]] = wraps(func)(partial(func, self.df))

    return _Patch

