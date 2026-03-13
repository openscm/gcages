"""
Helpers for working with [`aneris`](https://aneris.readthedocs.io)

A lot of this should be pushed upstream at some point.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas_openscm.indexing import mi_loc

from gcages.assertions import assert_only_working_on_variable_unit_region_variations
from gcages.exceptions import MissingOptionalDependencyError


class AmbiguousHarmonisationMethod(ValueError):
    """
    Error raised when harmonisation methods are ambiguous.
    """


class MissingHistoricalError(ValueError):
    """
    Error raised when historical data is missing.
    """


class MissingHarmonisationYear(ValueError):
    """
    Error raised when the harmonisation year is missing.
    """


def _check_data(hist: pd.DataFrame, scen: pd.DataFrame, year: int) -> None:
    try:
        from pandas_indexing.core import projectlevel
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "harmonise_all", requirement="pandas_indexing"
        ) from exc

    # TODO: push back upstream
    check = ["region", "variable"]

    def downselect(df: pd.DataFrame) -> pd.Index[Any]:
        res: pd.Index[Any] = projectlevel(df.index, check)

        return res

    s = downselect(scen)
    h = downselect(hist)
    if h.empty:
        msg = f"No historical data in harmonization year {year}"
        raise MissingHarmonisationYear(msg)

    if not s.difference(h).empty:
        msg = (
            "Historical data does not match scenario data in harmonization "
            f"year for\n {s.difference(h)}"
        )
        raise MissingHistoricalError(msg)


def _convert_units_to_match(
    start: pd.DataFrame, match: pd.DataFrame, copy_on_entry: bool = True
) -> pd.DataFrame:
    try:
        from pandas_indexing.core import concat, projectlevel
        from pandas_indexing.selectors import isin
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "harmonise_all", requirement="pandas_indexing"
        ) from exc

    # TODO: push back upstream
    if copy_on_entry:
        out = start.copy()

    else:
        out = start

    differences = projectlevel(match.index, ["variable", "unit"]).difference(
        projectlevel(start.index, ["variable", "unit"])
    )
    if not differences.empty:
        updated = []
        for variable, target_unit in differences:
            v_loc = isin(variable=variable)
            updated_v = out.loc[v_loc].pix.convert_unit(target_unit)
            if updated_v.empty:
                msg = f"Can't make this match because {variable} is not in {start=}"
                raise AssertionError(msg)

            updated.append(updated_v)
            out = out.loc[~v_loc]

        out = concat([out, *updated])

    return out


def _knead_overrides(
    overrides: pd.Series[str] | None, scen: pd.DataFrame, harm_idx: pd.MultiIndex
) -> pd.Series[str] | None:
    """
    Process overrides to get a form readable by aneris

    Supports many different use cases.

    Parameters
    ----------
    overrides : pd.DataFrame or pd.Series
    scen : pyam.IamDataFrame with data for single scenario and model instance

    Returns
    -------
    :
        Something, TBD what
    """
    # TODO: push back upstream
    if overrides is None:
        return None

    # massage into a known format
    # check if no index and single value - this should be the override for everything
    if overrides.index.names == [None] and len(overrides["method"]) == 1:
        _overrides = pd.Series(
            overrides.iloc[0],
            index=pd.Index(scen.variable, name="variable"),  # only need to match 1 dim
            name="method",
        )
    # if data is provided per model and scenario, get those explicitly
    elif any(c in overrides.index.names for c in ["model", "scenario"]):
        check_cols = [c for c in overrides.index.names if c in ["model", "scenario"]]
        _overrides = mi_loc(
            overrides,
            scen.index.droplevel(
                scen.index.names.difference(check_cols)  # type: ignore # pandas-stubs confused
            ).drop_duplicates(),
        ).droplevel(check_cols)  # type: ignore # pandas-stubs confused

        # None of the overrides relevant for this scenario
        if _overrides.empty:
            return None

    else:
        _overrides = overrides

    # do checks
    if _overrides.isnull().any():
        missing: pd.Series[str] = _overrides.loc[_overrides.isnull().any(axis=1)]  # type: ignore # pandas-stubs wrong
        msg = f"Overrides are missing for provided data:\n" f"{missing}"
        raise AmbiguousHarmonisationMethod(msg)

    if _overrides.index.to_frame().isnull().any().any():
        missing = _overrides[_overrides.index.to_frame().isnull().any(axis=1)]
        msg = f"Defined overrides are missing data:\n" f"{missing}"
        raise AmbiguousHarmonisationMethod(msg)

    if _overrides.index.duplicated().any():
        msg = (
            "Duplicated values for overrides:\n"
            f"{_overrides[_overrides.index.duplicated()]}"
        )
        raise AmbiguousHarmonisationMethod(msg)

    return _overrides


def harmonise_all(
    scenarios: pd.DataFrame,
    history: pd.DataFrame,
    year: int,
    overrides: pd.Series[str] | None = None,
) -> pd.DataFrame:
    """
    Harmonise all timeseries in `scenarios` to match `history`

    This is a re-write of aneris` version of the same.
    TODO: MR upstream.

    Parameters
    ----------
    scenarios
        `pd.DataFrame` containing the timeseries to be harmonised

    history
        `pd.DataFrame` containing the historical timeseries to which
        `scenarios` should be harmonised.

    year
        The year in which `scenarios` should be harmonised to `history`

    overrides
        If not provided, the default aneris decision tree is used.

        Otherwise, `overrides` must be a `pd.DataFrame`
        containing any specifications for overriding the default aneris methods.
        Each row specifies one override.
        The override method is specified in the "method" columns.
        The other columns specify which of the timeseries in
        `scenarios` should use this override by specifying metadata to match
        ( e.g. variable, region).
        If a cell has a null value (evaluated using `pd.isnull()`)
        then that scenario characteristic will not be used for
        filtering for that override.
        For example, if you have a row with "method" equal to "constant_ratio",
        region equal to "World" and variable is null
        then all timeseries in the "World" region will use the "constant_ratio" method.
        In contrast, if you have a row with "method" equal to "constant_ratio",
        region equal to "World" and variable is "Emissions|CO2"
        then only timeseries with variable equal to "Emissions|CO2"
        and region equal to "World" will use the "constant_ratio" method.

    Returns
    -------
    :
        The harmonised timeseries

    Notes
    -----
    This interface is nowhere near as sophisticated as aneris' other interfaces.
    It simply harmonises timeseries.
    It does not check sectoral sums
    or other possible errors which can arise when harmonising.
    If you need such features, do not use this interface.

    Raises
    ------
    MissingHistoricalError
        No historical data is provided for a given timeseries

    MissingHarmonisationYear
        A value for the harmonisation year is missing or is null in `history`

    AmbiguousHarmonisationMethod
        `overrides` do not uniquely specify
        the harmonisation method for a given timeseries.
    """
    try:
        from aneris.harmonize import Harmonizer  # type: ignore
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "harmonise_all", requirement="aneris"
        ) from exc

    try:
        from pandas_indexing.core import assignlevel, concat, semijoin
        from pandas_indexing.selectors import isin
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "harmonise_all", requirement="pandas_indexing"
        ) from exc

    sidx = scenarios.index  # save in case we need to re-add extraneous indicies later

    dfs = []
    group_levels = ["model", "scenario"]
    harm_idx = ["variable", "region"]
    for (model, scenario), msdf in scenarios.groupby(group_levels):
        hist_msdf = history.loc[
            isin(region=msdf.pix.unique("region"))  # type: ignore
            & isin(variable=msdf.pix.unique("variable"))  # type: ignore
        ]
        _check_data(hist_msdf, msdf, year)

        hist_msdf = _convert_units_to_match(start=hist_msdf, match=msdf)

        # need to convert to aneris' internal datastructure
        level_order = ["model", "scenario", "region", "variable", "unit"]
        msdf_aneris = msdf.reorder_levels(level_order)
        # Drop out any years that are all nan before passing to aneris
        msdf_aneris = msdf_aneris.dropna(how="all", axis="columns")
        # Convert to format expected by aneris
        hist_msdf_aneris = hist_msdf.pix.assign(
            model="history", scenario="scen"
        ).reorder_levels(level_order)

        # Drop out the group levels
        msdf_aneris = msdf_aneris.reset_index(group_levels, drop=True)
        hist_msdf_aneris = hist_msdf_aneris.reset_index(group_levels, drop=True)

        harmoniser = Harmonizer(
            msdf_aneris,
            hist_msdf_aneris,
            # have to copy harm index as aneris modifies it for some reason
            harm_idx=harm_idx.copy(),
        )

        # knead overrides
        overrides_kneaded = _knead_overrides(overrides, msdf, harm_idx=harm_idx)  # type: ignore
        result: pd.DataFrame = harmoniser.harmonize(
            year=year, overrides=overrides_kneaded
        )

        # convert out of internal datastructure
        dfs.append(assignlevel(result, model=model, scenario=scenario))

    # realign indicies as needed
    result = concat(dfs)
    result = semijoin(result, sidx, how="right").reorder_levels(sidx.names)

    return result


def harmonise_scenario(
    indf: pd.DataFrame,
    history: pd.DataFrame,
    year: int,
    overrides: pd.Series[str] | None,
) -> pd.DataFrame:
    """
    Harmonise a single scenario

    Parameters
    ----------
    indf
        Scenario to harmonise

    history
        History to harmonise to

    year
        Year to use for harmonisation

    overrides
        Overrides to pass to aneris

    Returns
    -------
    :
        Harmonised scenario
    """
    assert_only_working_on_variable_unit_region_variations(indf)

    harmonised = harmonise_all(
        indf,
        history=history,
        year=year,
        overrides=overrides,
    )

    return harmonised
