"""
Useful assertions
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype


class DataIsNotAllNumericError(ValueError):
    """
    Raised when not all data in a [pd.DataFrame][pandas.DataFrame] is numeric
    """

    def __init__(self, df: pd.DataFrame, non_numeric_cols: Collection[Any]) -> None:
        """
        Initialise the error

        Parameters
        ----------
        df
            [pd.DataFrame][pandas.DataFrame] containing non-numeric data

        non_numeric_cols
            The columns that contain non-numeric data
        """
        # Including df in API, but not sure how to use it well right now
        # (not easy to just get the non-numeric values in a column,
        # because that's not a trivial question to ask [is "0" non-numeric or not?])
        error_msg = (
            f"The following columns contain non-numeric data: {non_numeric_cols}"
        )
        super().__init__(error_msg)


def assert_data_is_all_numeric(df: pd.DataFrame) -> None:
    """
    Assert that all data in a [pd.DataFrame][pandas.DataFrame] is numeric

    Parameters
    ----------
    df
        [pd.DataFrame][pandas.DataFrame] to check

    Raises
    ------
    DataIsNotAllNumericError
        If there are columns in `df` are not numeric
    """
    non_numeric = tuple(c for c in df if not is_numeric_dtype(df[c]))
    if non_numeric:
        raise DataIsNotAllNumericError(df=df, non_numeric_cols=non_numeric)


class MissingDataForTimesError(KeyError):
    """
    Raised when a [pd.DataFrame][pandas.DataFrame] is missing data for expected times
    """

    def __init__(
        self,
        df: pd.DataFrame,
        name: str,
        missing_times: Collection[Any],
        allow_nan: bool,
    ) -> None:
        """
        Initialise the error

        Parameters
        ----------
        df
            [pd.DataFrame][pandas.DataFrame] that is missing expected index levels

        name
            Name of `df` to display in the error message

        missing_times
            Times in `df` that are missing data

        allow_nan
            Were NaN values allowed in the values of `times` when checking the data?
        """
        if allow_nan:
            error_msg = (
                f"{name} is missing data for the following times: "
                f"{missing_times}. "
                f"Available times: {df.columns}"
            )

        else:
            tmp = df[missing_times]
            nan_view = tmp[tmp.isnull().any(axis="columns")]
            error_msg = (
                f"{name} has NaNs for the following times: {missing_times}. "
                f"Rows with Nans:\n{nan_view}"
            )

        super().__init__(error_msg)


def assert_has_data_for_times(
    df: pd.DataFrame, name: str, times: Iterable[Any], allow_nan: bool
) -> None:
    """
    Assert that a [pd.DataFrame][pandas.DataFrame] has data for the given times

    Parameters
    ----------
    df
        [pd.DataFrame][pandas.DataFrame] to check

    name
        Name of `df` to display in the error message

    times
        Times (i.e. columns) that we expect to have data in `df`

    allow_nan
        Are NaN values allowed in the values of `times` (or should all data be non-Nan)?

    Raises
    ------
    MissingDataForTimesError
        The data in `df` does not contain all times in `times`.

        If `not allow_nan`, this will also be raised if any of the data in `df`
        contains NaN for a time in `times`.
    """
    missing_times = [v for v in times if v not in df.columns]
    if missing_times:
        raise MissingDataForTimesError(
            df=df,
            name=name,
            missing_times=missing_times,
            # Failed before we even considered NaN
            allow_nan=True,
        )

    if not allow_nan:
        nan_times = [v for v in times if df[v].isnull().any()]
        if nan_times:
            raise MissingDataForTimesError(
                df=df, name=name, missing_times=nan_times, allow_nan=allow_nan
            )


class MissingIndexLevelsError(KeyError):
    """
    Raised when a [pd.DataFrame][pandas.DataFrame] is missing expected index levels
    """

    def __init__(
        self,
        df: pd.DataFrame,
        missing_levels: Collection[Any],
    ) -> None:
        """
        Initialise the error

        Parameters
        ----------
        df
            [pd.DataFrame][pandas.DataFrame] that is missing expected index levels

        missing_levels
            Levels that are missing from `df.index`
        """
        error_msg = (
            f"The DataFrame is missing the following index levels: {missing_levels}. "
            f"Available index levels: {df.index.names}"
        )
        super().__init__(error_msg)


def assert_has_index_levels(df: pd.DataFrame, levels: Iterable[Any]) -> None:
    """
    Assert that a [pd.DataFrame][pandas.DataFrame] has all the given levels in its index

    Parameters
    ----------
    df
        [pd.DataFrame][pandas.DataFrame] to check

    levels
        Levels that we expect to be in the index of `df`

    Raises
    ------
    MissingIndexLevelsError
        The index of `df` does not contain all levels in `levels`
    """
    missing_levels = [v for v in levels if v not in df.index.names]
    if missing_levels:
        raise MissingIndexLevelsError(df=df, missing_levels=missing_levels)


class IndexIsNotMultiIndexError(TypeError):
    """
    Raised when the index is not a [pd.MultiIndex][pandas.MultiIndex]
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialise the error

        Parameters
        ----------
        df
            [pd.DataFrame][pandas.DataFrame]
        """
        error_msg = (
            f"The index is not a `pd.MultiIndex`, instead we have {type(df.index)=}"
        )
        super().__init__(error_msg)


def assert_index_is_multiindex(df: pd.DataFrame) -> None:
    """
    Assert that the index is a [pd.MultiIndex][pandas.MultiIndex]

    Parameters
    ----------
    df
        [pd.DataFrame][pandas.DataFrame] to check

    Raises
    ------
    IndexIsNotMultiIndexError
        The index of `df` is not a [pd.MultiIndex][pandas.MultiIndex]
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise IndexIsNotMultiIndexError(df)


class NotAllowedMetadataValuesError(ValueError):
    """
    Raised when a [pd.DataFrame][pandas.DataFrame] contains disallowed metadata values
    """

    def __init__(
        self,
        df: pd.DataFrame,
        metadata_key: Any,
        disallowed_values: Collection[Any],
        allowed_values: Collection[Any],
    ) -> None:
        """
        Initialise the error

        Parameters
        ----------
        df
            [pd.DataFrame][pandas.DataFrame] that contains diasallowed metadata values

        metadata_key
            The metadata key which is being considered (e.g. "variable", "unit")

        disallowed_values
            The values which are not allowed but appear in `df`

        allowed_values
            The values which are allowed for `metadata_key`
        """
        error_msg = (
            f"The DataFrame contains disallowed values for {metadata_key}: "
            f"{disallowed_values}. "
            f"Allowed values: {allowed_values}"
        )
        super().__init__(error_msg)


def assert_metadata_values_all_allowed(
    df: pd.DataFrame, metadata_key: Any, allowed_values: Collection[Any]
) -> None:
    """
    Assert that a [pd.DataFrame][pandas.DataFrame] only contains allowed metadata values

    Parameters
    ----------
    df
        [pd.DataFrame][pandas.DataFrame] to check

    metadata_key
        The metadata key to check (e.g. "variable", "unit")

    allowed_values
        The values which are allowed for this metadata key

    Raises
    ------
    NotAllowedMetadataValuesError
        There is metadata for `metadata_key` in `df` that is not in `allowed_values`.
    """
    disallowed_values = [
        v
        for v in df.index.get_level_values(metadata_key).unique()
        if v not in allowed_values
    ]
    if disallowed_values:
        raise NotAllowedMetadataValuesError(
            df=df,
            metadata_key=metadata_key,
            disallowed_values=disallowed_values,
            allowed_values=allowed_values,
        )


def assert_only_working_on_variable_unit_variations(indf: pd.DataFrame) -> None:
    """
    Assert that we're only working on variations in variable and unit

    In other words, we don't have variations in scenarios, models etc.

    Parameters
    ----------
    indf
        Data to verify

    Raises
    ------
    AssertionError
        There are variations in columns other than variable and unit
    """
    variations_in_other_cols = indf.index.droplevel(["variable", "unit"]).unique()
    if len(variations_in_other_cols) > 1:
        msg = f"variations_in_other_cols=\n{variations_in_other_cols}"
        raise AssertionError(msg)


def assert_only_working_on_variable_unit_region_variations(indf: pd.DataFrame) -> None:
    """
    Assert that we're only working on variations in variable, unit and region

    In other words, we don't have variations in scenarios, models etc.

    Parameters
    ----------
    indf
        Data to verify

    Raises
    ------
    AssertionError
        There are variations in columns other than variable and unit
    """
    variations_in_other_cols = indf.index.droplevel(
        ["variable", "unit", "region"]
    ).unique()
    if len(variations_in_other_cols) > 1:
        msg = f"variations_in_other_cols=\n{variations_in_other_cols}"
        raise AssertionError(msg)
