"""General harmonisation tools"""

from __future__ import annotations

from typing import Any

import pandas as pd

from gcages.typing import NUMERIC_DATA, TIME_POINT, TimeseriesDataFrame


def add_historical_year_based_on_scaling(
    year_to_add: int,
    year_calc_scaling: int,
    emissions: pd.DataFrame,
    emissions_history: pd.DataFrame,
    ms: tuple[str, ...] = ("model", "scenario"),
) -> pd.DataFrame:
    """
    Add a historical emissions year based on scaling

    Parameters
    ----------
    year_to_add
        Year to add

    year_calc_scaling
        Year to use to calculate the scaling

    emissions
        Emissions to which to add data for `year_to_add`

    emissions_history
        Emissions history to use to calculate
        the fill values based on scaling

    ms
        Name of the model and scenario columns.

        These have to be dropped from `emissions_historical`
        before everything will line up.

    Returns
    -------
    :
        `emissions` with data for `year_to_add`
        based on the scaling between `emissions`
        and `emissions_historical` in `year_calc_scaling`.
    """
    mod_scen_unique = emissions.index.droplevel(
        emissions.index.names.difference(["model", "scenario"])  # type: ignore
    ).unique()
    if mod_scen_unique.shape[0] > 1:
        # Processing is much trickier with multiple scenarios
        raise NotImplementedError(mod_scen_unique)

    emissions_historical_common_vars = emissions_history.loc[
        emissions_history.index.get_level_values("variable").isin(
            emissions.index.get_level_values("variable")
        )
    ]

    emissions_historical_no_ms = emissions_historical_common_vars.reset_index(
        emissions_historical_common_vars.index.names.difference(  # type: ignore # pandas-stubs not up to date
            ["region", "variable", "unit"]
        ),
        drop=True,
    )

    scale_factor = emissions[year_calc_scaling].divide(
        emissions_historical_no_ms[year_calc_scaling]
    )
    fill_value = scale_factor.multiply(emissions_historical_no_ms[year_to_add])
    fill_value.name = year_to_add

    out = pd.concat([emissions, fill_value], axis="columns").sort_index(axis="columns")

    return out


# Add back in if needed
# def add_harmonisation_year_if_needed(
#     indf: pd.DataFrame,
#     harmonisation_year: int,
#     calc_scaling_year: int,
#     emissions_history: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Add data for the harmonisation year if needed
#
#     If the harmonisation year needs to be added,
#     it is added based on [add_historical_year_based_on_scaling][]
#     (this could be made more flexible of course).
#
#     Parameters
#     ----------
#     indf
#         Input data to check and potentially add data to
#
#     harmonisation_year
#         Year that is being used for harmonisation
#
#     calc_scaling_year
#         Year to use for calculating a scaling factor from historical
#
#         Only used if `harmonisation_year` has missing data in `indf`.
#
#     emissions_history
#         Emissions history to use to calculate
#         the fill values based on scaling
#
#     Returns
#     -------
#     :
#         `indf` with `harmonisation_year` data added where needed
#     """
#     if harmonisation_year not in indf:
#         emissions_to_harmonise = add_historical_year_based_on_scaling(
#             year_to_add=harmonisation_year,
#             year_calc_scaling=calc_scaling_year,
#             emissions=indf,
#             emissions_history=emissions_history,
#         )
#
#     elif indf[harmonisation_year].isnull().any():
#         null_emms_in_harm_year = indf[harmonisation_year].isnull()
#
#         dont_change = indf[~null_emms_in_harm_year]
#
#         updated = add_historical_year_based_on_scaling(
#             year_to_add=harmonisation_year,
#             year_calc_scaling=calc_scaling_year,
#             emissions=indf[null_emms_in_harm_year].drop(
#                 harmonisation_year, axis="columns"
#             ),
#             emissions_history=emissions_history,
#         )
#
#         emissions_to_harmonise = pd.concat([dont_change, updated])
#
#     else:
#         emissions_to_harmonise = indf
#
#     return emissions_to_harmonise


class NotHarmonisedError(ValueError):
    """
    Raised when a [pd.DataFrame][pandas.DataFrame] is not harmonised
    """

    def __init__(
        self,
        comparison: pd.DataFrame,
        harmonisation_time: TIME_POINT,
    ) -> None:
        """
        Initialise the error

        Parameters
        ----------
        comparison
            Results of comparing the data and history

        harmonisation_time
            Expected harmonisation time
        """
        error_msg = (
            f"The DataFrame is not harmonised in {harmonisation_time}. "
            f"comparison=\n{comparison}"
        )
        super().__init__(error_msg)


def align_history_to_data_at_time(
    df: TimeseriesDataFrame, *, history: TimeseriesDataFrame, time: Any
) -> tuple[pd.Series[NUMERIC_DATA], pd.Series[NUMERIC_DATA]]:  # type: ignore # pandas-stubs not up to date
    """
    Align history to a given set of data for a given column

    Parameters
    ----------
    df
        Data to which to align history

    history
        History data to align

    time
        Time (i.e. column) for which to align the data

    Returns
    -------
    :
        History, aligned with `df` for the given column

    Raises
    ------
    AssertionError
        `df` and `history` could not be aligned for some reason
    """
    df_year_aligned, history_year_aligned = df[time].align(history[time], join="left")

    # Implicitly assuming that people have already checked
    # that they have history values for all timeseries in `df`,
    # so any null is an obvious issue.
    if history_year_aligned.isnull().any():
        msg_l = ["history did not align properly with df"]

        if df.index.names == history.index.names:
            msg_l.append(
                "history and df have the same index levels "
                f"({list(history.index.names)}). "
                "You probably need to drop some of history's index levels "
                "so alignment can happen along the levels of interest "
                "(usually dropping everything except variable and unit (or similar)). "
            )

        # # Might be useful, pandas might handle it
        # names_only_in_hist = history.index.names.difference(df.index.names)

        for unit_col_guess in ["unit", "units"]:
            if (
                unit_col_guess in df.index.names
                and unit_col_guess in history.index.names
            ):
                df_units_guess = df.index.get_level_values(unit_col_guess)
                history_units_guess = history.index.get_level_values(unit_col_guess)

                differing_units = (
                    df_units_guess.difference(history_units_guess).unique().tolist()
                )
                msg_l.append(
                    "The following units only appear in `df`, "
                    f"which might be why the data isn't aligned: {differing_units}. "
                    f"{df_units_guess=} {history_units_guess=}"
                )

        msg = ". ".join(msg_l)
        raise AssertionError(msg)

    return df_year_aligned, history_year_aligned


def assert_harmonised(
    df: TimeseriesDataFrame,
    *,
    history: TimeseriesDataFrame,
    harmonisation_time: TIME_POINT,
    rounding: int = 10,
) -> None:
    """
    Assert that a given [TimeseriesDataFrame][(p).typing] is harmonised

    Note: currently, this does not support unit conversion
    (i.e. units have to match exactly, equivalent units e.g. "Mt CO2" and "MtCO2"
    will result in a `NotHarmonisedError`).

    Parameters
    ----------
    df
        Data to check

    history
        History to which `df` should be harmonised

    harmonisation_time
        Time at which `df` should be harmonised to `history`

    rounding
        Rounding to apply to the data before comparing

    Raises
    ------
    NotHarmonisedError
        `df` is not harmonised to `history`
    """
    df_harm_year_aligned, history_harm_year_aligned = align_history_to_data_at_time(
        df, history=history, time=harmonisation_time
    )
    comparison = df_harm_year_aligned.round(rounding).compare(  # type: ignore # pandas-stubs out of date
        history_harm_year_aligned.round(rounding), result_names=("df", "history")
    )
    if not comparison.empty:
        raise NotHarmonisedError(
            comparison=comparison, harmonisation_time=harmonisation_time
        )
