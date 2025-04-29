"""
Manipulation of the index of [pd.DataFrame][pandas.DataFrame]'s
"""

# TOOD: put all of this in pandas-openscm
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar

import pandas as pd

from gcages.exceptions import MissingOptionalDependencyError

if TYPE_CHECKING:
    P = TypeVar("P", pd.DataFrame, pd.Series[Any])


def set_new_single_value_levels(  # noqa: D103
    pandas_obj: P,
    levels_to_set: dict[str, Any],  # indicate not a collection somehow
    copy: bool = True,
) -> P:
    # TODO: move to pandas-openscm
    # TODO: split out method that just works on MultiIndex
    if copy:
        pandas_obj = pandas_obj.copy()

    new_names = levels_to_set.keys()
    new_values = levels_to_set.values()

    if not isinstance(pandas_obj.index, pd.MultiIndex):
        raise TypeError(pandas_obj.index)

    pandas_obj.index = pd.MultiIndex(
        codes=[
            *pandas_obj.index.codes,  # type: ignore #  not sure why check above isn't working
            *([[0] * pandas_obj.index.shape[0]] * len(new_values)),  # type: ignore # fix when moving to pandas-openscm
        ],
        levels=[*pandas_obj.index.levels, *[pd.Index([value]) for value in new_values]],  # type: ignore # fix when moving to pandas-openscm
        names=[*pandas_obj.index.names, *new_names],  # type: ignore # fix when moving to pandas-openscm
    )

    return pandas_obj


def create_levels_based_on_existing(  # noqa: D103
    ini: pd.MultiIndex,
    create_from: dict[Any, tuple[str, Callable[[Any], Any]]],
    remove_unused_levels: bool = True,
) -> pd.MultiIndex:
    # TODO: move to pandas-openscm
    if remove_unused_levels:
        ini = ini.remove_unused_levels()  # type: ignore

    levels: list[pd.Index[Any]] = list(ini.levels)
    codes: list[list[int]] = list(ini.codes)
    names: list[str] = list(ini.names)

    for level, (source, updater) in create_from.items():
        if source not in ini.names:
            msg = (
                f"{source} is not available in the index. Available levels: {ini.names}"
            )
            raise KeyError(msg)

        source_idx = ini.names.index(source)
        new_level = ini.levels[source_idx].map(updater)
        if not new_level.has_duplicates:
            # Fast route: no clashes so no need to update the codes
            # or do anything
            new_codes = ini.codes[source_idx]

        else:
            # Slow route: have to update the codes too
            dup_level = ini.get_level_values(source).map(updater)
            new_level = new_level.unique()
            new_codes = new_level.get_indexer(dup_level)  # type: ignore

        levels.append(new_level)
        codes.append(new_codes)
        names.append(level)

    res = pd.MultiIndex(levels=levels, codes=codes, names=names)

    return res


def split_sectors(  # noqa: PLR0913
    indf: pd.DataFrame,
    dropna: bool = True,
    level_to_split: str = "variable",
    top_level: str = "table",
    middle_level: str = "species",
    bottom_level: str = "sectors",
    level_separator: str = "|",
) -> pd.DataFrame:
    """
    Split the input [pd.DataFrame][pandas.DataFrame]'s level to split into sectors

    Any levels beyond three are all left in `bottom_level`
    in the output.

    This is the inverse of [combine_sectors][(m).].

    Parameters
    ----------
    indf
        Input data

    dropna
        Should levels which have NaNs after splitting be dropped?

    level_to_split
        The level to split

    top_level
        Name of the top level after the split

    middle_level
        Name of the middle level after the split

    bottom_level
        Name of the bottom level after the split

    level_separator
        Separator between levels in `level_to_split`

    Returns
    -------
    :
        `indf` with `level_to_split` split into three levels
        with the given names

    Examples
    --------
    >>> indf = pd.DataFrame(
    ...     [
    ...         [1.0, 2.0],
    ...         [3.0, 2.0],
    ...         [1.3, 2.2],
    ...         [3.4, 2.1],
    ...     ],
    ...     columns=[2015, 2100],
    ...     index=pd.MultiIndex.from_tuples(
    ...         [
    ...             ("sa", "Emissions|CO2|sector", "t / yr"),
    ...             ("sa", "Emissions|CO2|sector|sub sector", "t / yr"),
    ...             ("sb", "Emissions|CH4|sector|sub|sub-sub|sub-sub-sub", "kg / yr"),
    ...             ("sb", "Emissions|CO2|sector", "t / yr"),
    ...         ],
    ...         names=["scenario", "variable", "unit"],
    ...     ),
    ... )
    >>> split_sectors(indf)  # doctest: +NORMALIZE_WHITESPACE
                                                                       2015  2100
    scenario unit    table     species sectors
    sa       t / yr  Emissions CO2     sector                           1.0   2.0
                                       sector|sub sector                3.0   2.0
    sb       kg / yr Emissions CH4     sector|sub|sub-sub|sub-sub-sub   1.3   2.2
             t / yr  Emissions CO2     sector                           3.4   2.1

    >>> indf_funky = pd.DataFrame(
    ...     [
    ...         [1.0, 2.0],
    ...         [3.0, 2.0],
    ...         [1.3, 2.2],
    ...         [3.4, 2.1],
    ...     ],
    ...     columns=[2015, 2100],
    ...     index=pd.MultiIndex.from_tuples(
    ...         [
    ...             ("sa", "Emissions-CO2-sector", "t / yr"),
    ...             ("sa", "Emissions-CO2-sector-sub sector", "t / yr"),
    ...             ("sb", "Emissions-CH4-sector-sub-sector-transport", "kg / yr"),
    ...             ("sb", "Emissions-CO2-sector", "t / yr"),
    ...         ],
    ...         names=["scenario", "vv", "unit"],
    ...     ),
    ... )
    >>> split_sectors(
    ...     indf_funky,
    ...     level_to_split="vv",
    ...     top_level="t",
    ...     middle_level="m",
    ...     bottom_level="b",
    ...     level_separator="-",
    ... )  # doctest: +NORMALIZE_WHITESPACE
                                                                2015  2100
    scenario unit    t         m   b
    sa       t / yr  Emissions CO2 sector                        1.0   2.0
                                   sector-sub sector             3.0   2.0
    sb       kg / yr Emissions CH4 sector-sub-sector-transport   1.3   2.2
             t / yr  Emissions CO2 sector                        3.4   2.1
    """
    try:
        from pandas_indexing.core import extractlevel
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "split_sectors", requirement="pandas_indexing"
        ) from exc

    kwargs = {
        level_to_split: level_separator.join(
            [
                "{" + f"{level}" + "}"
                for level in [top_level, middle_level, bottom_level]
            ]
        )
    }

    return extractlevel(indf, dropna=dropna, **kwargs)  # type: ignore


def split_species(  # noqa: PLR0913
    indf: pd.DataFrame,
    dropna: bool = True,
    level_to_split: str = "variable",
    top_level: str = "table",
    bottom_level: str = "species",
    level_separator: str = "|",
) -> pd.DataFrame:
    """
    Split the input [pd.DataFrame][pandas.DataFrame]'s level to split into species

    Any levels beyond two are all left in `bottom_level` in the output.

    This is the inverse of [combine_species][(m).].

    Parameters
    ----------
    indf
        Input data

    dropna
        Should levels which have NaNs after splitting be dropped?

    level_to_split
        The level to split

    top_level
        Name of the top level after the split

    bottom_level
        Name of the bottom level after the split

    level_separator
        Separator between levels in `level_to_split`

    Returns
    -------
    :
        `indf` with `level_to_split` split into two levels
        with the given names

    Examples
    --------
    >>> indf = pd.DataFrame(
    ...     [
    ...         [1.0, 2.0],
    ...         [3.0, 2.0],
    ...         [1.3, 2.2],
    ...         [3.4, 2.1],
    ...     ],
    ...     columns=[2015, 2100],
    ...     index=pd.MultiIndex.from_tuples(
    ...         [
    ...             ("sa", "Emissions|CO2", "t / yr"),
    ...             ("sa", "Emissions|CH4", "t / yr"),
    ...             ("sb", "Emissions|CH4|sector|sub", "kg / yr"),
    ...             ("sb", "Emissions|N2O", "t / yr"),
    ...         ],
    ...         names=["scenario", "variable", "unit"],
    ...     ),
    ... )
    >>> split_species(indf)  # doctest: +NORMALIZE_WHITESPACE
                                               2015  2100
    scenario unit    table     species
    sa       t / yr  Emissions CO2              1.0   2.0
                               CH4              3.0   2.0
    sb       kg / yr Emissions CH4|sector|sub   1.3   2.2
             t / yr  Emissions N2O              3.4   2.1
    >>> indf_funky = pd.DataFrame(
    ...     [
    ...         [1.0, 2.0],
    ...         [3.0, 2.0],
    ...         [1.3, 2.2],
    ...         [3.4, 2.1],
    ...     ],
    ...     columns=[2015, 2100],
    ...     index=pd.MultiIndex.from_tuples(
    ...         [
    ...             ("sa", "Emissions-CO2", "t / yr"),
    ...             ("sa", "Emissions-CO2-sector-sub sector", "t / yr"),
    ...             ("sb", "Emissions-CH4", "kg / yr"),
    ...             ("sb", "Emissions-N2O", "t / yr"),
    ...         ],
    ...         names=["scenario", "vv", "unit"],
    ...     ),
    ... )
    >>> split_species(
    ...     indf_funky,
    ...     level_to_split="vv",
    ...     top_level="t",
    ...     bottom_level="b",
    ...     level_separator="-",
    ... )  # doctest: +NORMALIZE_WHITESPACE
                                                      2015  2100
    scenario unit    t         b
    sa       t / yr  Emissions CO2                     1.0   2.0
                               CO2-sector-sub sector   3.0   2.0
    sb       kg / yr Emissions CH4                     1.3   2.2
             t / yr  Emissions N2O                     3.4   2.1
    """
    try:
        from pandas_indexing.core import extractlevel
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "split_species", requirement="pandas_indexing"
        ) from exc

    kwargs = {
        level_to_split: level_separator.join(
            ["{" + f"{level}" + "}" for level in [top_level, bottom_level]]
        )
    }

    return extractlevel(indf, dropna=dropna, **kwargs)  # type: ignore


def combine_sectors(  # noqa: PLR0913
    indf: pd.DataFrame,  # Could open this out as can also work on Series and MultiIndex
    drop: bool = True,
    combined_level: str = "variable",
    top_level: str = "table",
    middle_level: str = "species",
    bottom_level: str = "sectors",
    level_separator: str = "|",
) -> pd.DataFrame:
    """
    Combine the input [pd.DataFrame][pandas.DataFrame]'s levels

    This assumes that you want to combine three levels.

    This is the inverse of [split_sectors][(m).].

    Parameters
    ----------
    indf
        Input data

    drop
        Should the combined levels be dropped?

    combined_level
        The name of the combined level

    top_level
        Name of the top level in the combined output

    middle_level
        Name of the middle level in the combined output

    bottom_level
        Name of the bottom level in the combined output

    level_separator
        Separator between levels in `combined_level`

    Returns
    -------
    :
        `indf` with the given levels combined into `combined_level`

    Examples
    --------
    >>> indf = pd.DataFrame(
    ...     [
    ...         [1.0, 2.0],
    ...         [3.0, 2.0],
    ...         [1.3, 2.2],
    ...         [3.4, 2.1],
    ...     ],
    ...     columns=[2015, 2100],
    ...     index=pd.MultiIndex.from_tuples(
    ...         [
    ...             ("sa", "Emissions", "CO2", "sector", "t / yr"),
    ...             ("sa", "Emissions", "CO2", "sector|sub sector", "t / yr"),
    ...             (
    ...                 "sb",
    ...                 "Emissions",
    ...                 "CH4",
    ...                 "sector|sub|sub-sub|sub-sub-sub",
    ...                 "kg / yr",
    ...             ),
    ...             ("sb", "Emissions", "CO2", "sector", "t / yr"),
    ...         ],
    ...         names=["scenario", "table", "species", "sectors", "unit"],
    ...     ),
    ... )
    >>> combine_sectors(indf)  # doctest: +NORMALIZE_WHITESPACE
                                                                   2015  2100
    scenario unit    variable
    sa       t / yr  Emissions|CO2|sector                           1.0   2.0
                     Emissions|CO2|sector|sub sector                3.0   2.0
    sb       kg / yr Emissions|CH4|sector|sub|sub-sub|sub-sub-sub   1.3   2.2
             t / yr  Emissions|CO2|sector                           3.4   2.1
    >>> indf_funky = pd.DataFrame(
    ...     [
    ...         [1.0, 2.0],
    ...         [3.0, 2.0],
    ...         [1.3, 2.2],
    ...         [3.4, 2.1],
    ...     ],
    ...     columns=[2015, 2100],
    ...     index=pd.MultiIndex.from_tuples(
    ...         [
    ...             ("sa", "Emissions", "CO2", "sector", "t / yr"),
    ...             ("sa", "Emissions", "CO2", "sector-sub sector", "t / yr"),
    ...             ("sb", "Emissions", "CH4", "sector-sub-sub", "kg / yr"),
    ...             ("sb", "Emissions", "CO2", "sector", "t / yr"),
    ...         ],
    ...         names=["scenario", "prefix", "gas", "sector", "unit"],
    ...     ),
    ... )
    >>> combine_sectors(
    ...     indf_funky,
    ...     combined_level="vv",
    ...     top_level="prefix",
    ...     middle_level="gas",
    ...     bottom_level="sector",
    ...     level_separator="-",
    ... )  # doctest: +NORMALIZE_WHITESPACE
                                                      2015  2100
    scenario unit    vv
    sa       t / yr  Emissions-CO2-sector              1.0   2.0
                     Emissions-CO2-sector-sub sector   3.0   2.0
    sb       kg / yr Emissions-CH4-sector-sub-sub      1.3   2.2
             t / yr  Emissions-CO2-sector              3.4   2.1
    """
    try:
        from pandas_indexing.core import formatlevel
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "combine_sectors", requirement="pandas_indexing"
        ) from exc

    kwargs = {
        combined_level: level_separator.join(
            [
                "{" + f"{level}" + "}"
                for level in [top_level, middle_level, bottom_level]
            ]
        )
    }

    return formatlevel(indf, drop=drop, **kwargs)  # type: ignore


def combine_species(  # noqa: PLR0913
    indf: pd.DataFrame,
    drop: bool = True,
    combined_level: str = "variable",
    top_level: str = "table",
    bottom_level: str = "species",
    level_separator: str = "|",
) -> pd.DataFrame:
    """
    Combine the input [pd.DataFrame][pandas.DataFrame]'s levels

    This assumes that you want to combine two levels.

    This is the inverse of [split_species][(m).].

    Parameters
    ----------
    indf
        Input data

    drop
        Should the combined levels be dropped?

    combined_level
        The name of the combined level

    top_level
        Name of the top level in the combined output

    bottom_level
        Name of the bottom level in the combined output

    level_separator
        Separator between levels in `combined_level`

    Returns
    -------
    :
        `indf` with the given levels combined into `combined_level`

    Examples
    --------
    >>> indf = pd.DataFrame(
    ...     [
    ...         [1.0, 2.0],
    ...         [3.0, 2.0],
    ...         [1.3, 2.2],
    ...         [3.4, 2.1],
    ...     ],
    ...     columns=[2015, 2100],
    ...     index=pd.MultiIndex.from_tuples(
    ...         [
    ...             ("sa", "Emissions", "CO2", "t / yr"),
    ...             ("sa", "Emissions", "CO2", "t / yr"),
    ...             ("sb", "Emissions", "CH4", "kg / yr"),
    ...             ("sb", "Emissions", "CO2", "t / yr"),
    ...         ],
    ...         names=["scenario", "table", "species", "unit"],
    ...     ),
    ... )
    >>> combine_species(indf)  # doctest: +NORMALIZE_WHITESPACE
                                    2015  2100
    scenario unit    variable
    sa       t / yr  Emissions|CO2   1.0   2.0
                     Emissions|CO2   3.0   2.0
    sb       kg / yr Emissions|CH4   1.3   2.2
             t / yr  Emissions|CO2   3.4   2.1
    >>> indf_funky = pd.DataFrame(
    ...     [
    ...         [1.0, 2.0],
    ...         [3.0, 2.0],
    ...         [1.3, 2.2],
    ...         [3.4, 2.1],
    ...     ],
    ...     columns=[2015, 2100],
    ...     index=pd.MultiIndex.from_tuples(
    ...         [
    ...             ("sa", "Emissions", "CO2", "t / yr"),
    ...             ("sa", "Emissions", "CO2-sector-sub sector", "t / yr"),
    ...             ("sb", "Emissions", "CH4-sector-sub-sub", "kg / yr"),
    ...             ("sb", "Emissions", "CO2", "t / yr"),
    ...         ],
    ...         names=["scenario", "prefix", "gas", "unit"],
    ...     ),
    ... )
    >>> combine_species(
    ...     indf_funky,
    ...     combined_level="vv",
    ...     top_level="prefix",
    ...     bottom_level="gas",
    ...     level_separator="-",
    ... )  # doctest: +NORMALIZE_WHITESPACE
                                                      2015  2100
    scenario unit    vv
    sa       t / yr  Emissions-CO2                     1.0   2.0
                     Emissions-CO2-sector-sub sector   3.0   2.0
    sb       kg / yr Emissions-CH4-sector-sub-sub      1.3   2.2
             t / yr  Emissions-CO2                     3.4   2.1
    """
    try:
        from pandas_indexing.core import formatlevel
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "combine_species", requirement="pandas_indexing"
        ) from exc

    kwargs = {
        combined_level: level_separator.join(
            ["{" + f"{level}" + "}" for level in [top_level, bottom_level]]
        )
    }

    return formatlevel(indf, drop=drop, **kwargs)  # type: ignore
