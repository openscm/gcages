"""General harmonisation tools"""

from __future__ import annotations

import pandas as pd


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

    ms = ("model", "scenario")
    emissions_historical_common_vars = emissions_history.loc[
        emissions_history.index.get_level_values("variable").isin(
            emissions.index.get_level_values("variable")
        )
    ]

    emissions_historical_no_ms = emissions_historical_common_vars.reset_index(
        ms, drop=True
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
