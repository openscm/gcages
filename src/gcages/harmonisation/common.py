"""
Common tools across different approaches
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from gcages.exceptions import MissingOptionalDependencyError
from gcages.testing import compare_close
from gcages.typing import NUMERIC_DATA, TIME_POINT, TimeseriesDataFrame
from gcages.units_helpers import convert_unit_like

if TYPE_CHECKING:
    import pint

    from gcages.typing import PINT_SCALAR


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

        # Might be useful, pandas might handle it
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


def assert_harmonised(  # noqa: PLR0913
    df: TimeseriesDataFrame,
    *,
    history: TimeseriesDataFrame,
    harmonisation_time: TIME_POINT,
    rounding: int = 10,
    df_unit_level: str = "unit",
    history_unit_level: str | None = None,
    ur: pint.UnitRegistry | None = None,
    species_aware_cmip7: bool | None = None,
    species_tolerances: dict[str, dict[str, float | PINT_SCALAR]] | None = None,
) -> None:
    """
    Assert that the input is harmonised

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

    df_unit_level
        Level in `df`'s index which has unit information

        Only used if unit conversion is required

    history_unit_level
        Level in `history`'s index which has unit information

        If not provided, we assume this is the same as `df_unit_level`

        Only used if unit conversion is required

    ur
        Unit registry to use for determining unit conversions

        Passed to [gcages.units_helpers.convert_unit_like][]

        Only used if unit conversion is required

    species_aware_cmip7
        Tolerances between historical and harmonised prescribed individually
        to each species

    species_tolerances
        Tolerance to apply while checking harmonisation of different species

    Raises
    ------
    NotHarmonisedError
        `df` is not harmonised to `history`
    """
    variables_df = df.index.get_level_values("variable").unique()
    history = history.loc[
        history.index.get_level_values("variable").isin(variables_df)
    ].reset_index(
        level=[lvl for lvl in ["model", "scenario"] if lvl in history.index.names],
        drop=True,
    )

    df_unit_match = convert_unit_like(
        df,
        target=history,
        df_unit_level=df_unit_level,
        target_unit_level=history_unit_level,
        ur=ur,
    )
    df_harm_year_aligned, history_harm_year_aligned = align_history_to_data_at_time(
        df_unit_match, history=history, time=harmonisation_time
    )

    species_aware = species_aware_cmip7 if species_aware_cmip7 is not None else False

    if species_aware:
        if ur is None:
            try:
                import openscm_units

                ur = openscm_units.unit_registry
            except ImportError:
                raise MissingOptionalDependencyError(  # noqa: TRY003
                    "convert_unit_like(..., ur=None, ...)", "openscm_units"
                )

        Q = ur.Quantity
        if species_tolerances is None:
            species_tolerances = {
                "BC": dict(rtol=1e-3, atol=Q(1e-3, "Mt BC/yr")),
                "CH4": dict(rtol=1e-3, atol=Q(1e-2, "Mt CH4/yr")),
                "CO": dict(rtol=1e-3, atol=Q(1e-1, "Mt CO/yr")),
                "CO2": dict(rtol=1e-3, atol=Q(1e-3, "Gt CO2/yr")),
                "NH3": dict(rtol=1e-3, atol=Q(1e-2, "Mt NH3/yr")),
                "NOx": dict(rtol=1e-3, atol=Q(1e-2, "Mt NO2/yr")),
                "OC": dict(rtol=1e-3, atol=Q(1e-3, "Mt OC/yr")),
                "Sulfur": dict(rtol=1e-3, atol=Q(1e-2, "Mt SO2/yr")),
                "VOC": dict(rtol=1e-3, atol=Q(1e-2, "Mt VOC/yr")),
                "N2O": dict(rtol=1e-3, atol=Q(1e-1, "kt N2O/yr")),
            }
        diffs = []
        for variable, scen_a_vdf in df_harm_year_aligned.groupby("variable"):
            mask = df.index.get_level_values("variable") == variable
            history_a_vdf = history_harm_year_aligned.loc[mask]
            species = str(variable).split("|")[1]
            if species in species_tolerances:
                unit_l = scen_a_vdf.index.get_level_values("unit").unique()
                if len(unit_l) != 1:
                    raise AssertionError(unit_l)
                unit = unit_l[0]

                rtol = species_tolerances[species]["rtol"]
                atol = species_tolerances[species]["atol"].to(unit).m

            else:
                rtol = 1e-4
                atol = 1e-6

            comparison = compare_close(
                scen_a_vdf.unstack("region"),
                history_a_vdf.unstack("region"),
                left_name="scenario",
                right_name="history",
                rtol=rtol,
                atol=atol,
            )
            if not comparison.empty:
                diffs.append(comparison)
        comparison = pd.concat(diffs) if diffs else pd.DataFrame()
    else:
        comparison = df_harm_year_aligned.round(rounding).compare(  # type: ignore # pandas-stubs out of date
            history_harm_year_aligned.round(rounding), result_names=("df", "history")
        )

    if not comparison.empty:
        raise NotHarmonisedError(
            comparison=comparison, harmonisation_time=harmonisation_time
        )
