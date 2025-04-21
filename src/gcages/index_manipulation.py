"""
Manipulation of the index of [pd.DataFrame][pandas.DataFrame]'s
"""

# TOOD: consider putting this in pandas-openscm
from __future__ import annotations

import pandas as pd

from gcages.exceptions import MissingOptionalDependencyError


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

    return extractlevel(indf, dropna=dropna, **kwargs)


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

    return extractlevel(indf, dropna=dropna, **kwargs)


def combine_sectors(  # noqa: PLR0913
    indf: pd.DataFrame,
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

    return formatlevel(indf, drop=drop, **kwargs)


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

    return formatlevel(indf, drop=drop, **kwargs)
