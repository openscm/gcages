"""
Index manipulation

TODO: move to pandas-openscm
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd


def update_index_levels(
    ini: pd.MultiIndex, functions: dict[Any, Callable[[Any], Any]]
) -> pd.MultiIndex:
    """
    Update the levels of a [pd.MultiIndex][pandas.MultiIndex]

    Parameters
    ----------
    ini
        [pd.MultiIndex][pandas.MultiIndex] with the levels to update

    functions
        Functions to apply to levels of `ini`

        Each key of `functions` specifies the level to apply the function to,
        the values should be the functions to apply.

    Returns
    -------
    :
        `ini` with the levels in `functions` updated

    Raises
    ------
    KeyError
        `functions` refers to a level that is not in `ini.names`
    """
    levels = list(ini.levels)
    for level, level_func in functions.items():
        if level not in ini.names:
            msg = (
                f"{level} is not available in the index. Available levels: {ini.names}"
            )
            raise KeyError(msg)

        level_idx = ini.names.index(level)
        levels[level_idx] = [level_func(v) for v in levels[level_idx]]

    res = pd.MultiIndex(
        levels=levels,
        codes=ini.codes,
        names=ini.names,
    )

    return res
