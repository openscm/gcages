"""
Helpers for working with [pandas_indexing](https://pandas-indexing.readthedocs.io/)

Importing pandas_indexing registers accessors on pandas objects
(e.g. `pd.DataFrame.pix`) as a side effect.
We don't want gcages to change pandas objects behind its users' backs,
so we provide a way to import pandas_indexing without this side effect.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pandas as pd

from gcages.exceptions import MissingOptionalDependencyError

PANDAS_INDEXING_ACCESSORS: tuple[str, ...] = ("pix", "idx")
"""Accessors that pandas_indexing registers on pandas objects"""

PANDAS_CLASSES_WITH_ACCESSORS: tuple[type, ...] = (pd.DataFrame, pd.Series, pd.Index)
"""Classes on which pandas_indexing registers its accessors"""

_PANDAS_INDEXING: ModuleType | None = None
"""Cached copy of pandas_indexing, imported without side effects"""


def import_pandas_indexing(callable_name: str) -> ModuleType:
    """
    Import pandas_indexing without side effects

    If pandas_indexing has already been imported
    (and hence its accessors registered), we simply return it.
    Otherwise, we import pandas_indexing,
    remove the accessors that it registered on pandas objects,
    then remove pandas_indexing from `sys.modules`.
    The last step means that, if pandas_indexing is imported again
    (e.g. by our users), it is imported as normal,
    i.e. its accessors are registered as normal.

    Parameters
    ----------
    callable_name
        Name of the callable that requires pandas_indexing

        Used to give a helpful error message if pandas_indexing isn't installed.

    Returns
    -------
    :
        pandas_indexing module

    Raises
    ------
    MissingOptionalDependencyError
        pandas_indexing is not installed

    Notes
    -----
    Only use pandas_indexing's functions (e.g. `pix.concat`, `pix.isin`)
    with the returned module, not its accessors (e.g. `df.pix.assign`),
    because the accessors may not be registered.
    """
    global _PANDAS_INDEXING  # noqa: PLW0603

    if "pandas_indexing" in sys.modules:
        return sys.modules["pandas_indexing"]

    if _PANDAS_INDEXING is not None:
        return _PANDAS_INDEXING

    registered_before = {
        (cls, name)
        for cls in PANDAS_CLASSES_WITH_ACCESSORS
        for name in PANDAS_INDEXING_ACCESSORS
        if name in cls.__dict__
    }

    try:
        import pandas_indexing  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        if exc.name != "pandas_indexing":
            # pandas_indexing is installed, but something else is wrong
            raise

        raise MissingOptionalDependencyError(
            callable_name, requirement="pandas_indexing"
        ) from exc

    for cls in PANDAS_CLASSES_WITH_ACCESSORS:
        for name in PANDAS_INDEXING_ACCESSORS:
            if (cls, name) in registered_before or name not in cls.__dict__:
                continue

            delattr(cls, name)
            cls._accessors.discard(name)  # type: ignore # private pandas API

    for module_name in [
        k
        for k in sys.modules
        if k == "pandas_indexing" or k.startswith("pandas_indexing.")
    ]:
        del sys.modules[module_name]

    _PANDAS_INDEXING = pandas_indexing

    return pandas_indexing
