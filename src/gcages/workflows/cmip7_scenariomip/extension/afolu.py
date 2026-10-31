"""
Extension of AFOLU CO2 emissions

Ported from https://github.com/iiasa/emissions_harmonization_historical
(commit 376987dbe3b66fcd23537519a16771e21cba6d1e,
`src/emissions_harmonization_historical/extensions/afolu_extension_functions.py`),
which is based on FLEX (https://github.com/benmsanderson/FLEX, MIT licence).
The logic is unchanged, so that we reproduce the CMIP7 ScenarioMIP extensions exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gcages.pandas_indexing_helpers import import_pandas_indexing

from .utils import interpolate_to_annual

AFOLU_sectors = [
    "Emissions|CO2|Agricultural Waste Burning",
    "Emissions|CO2|Forest Burning",
    "Emissions|CO2|Grassland Burning",
    "Emissions|CO2|Peat Burning",
    "Emissions|CO2|Agriculture",
]
Fossil_sectors = [
    "Emissions|CO2|Aircraft",
    "Emissions|CO2|International Shipping",
    "Emissions|CO2|Energy Sector",
    "Emissions|CO2|Industrial Sector",
    "Emissions|CO2|Residential Commercial Other",
    "Emissions|CO2|Solvents Production and Application",
    "Emissions|CO2|Transportation Sector",
    "Emissions|CO2|Waste",
]
global_fossil = "Emissions|CO2|Energy and Industrial Processes"
global_afolu_sector = "Emissions|CO2|AFOLU"


def get_cumulative_afolu(
    input_df: pd.DataFrame, model: str, scenario: str, emi_kind="**CO2|AFOLU"
) -> pd.DataFrame:
    """
    From yearly AFOLU DataFrame, calculate cumulative AFOLU

    Parameters
    ----------
    input_df : pd.DataFrame
        Input emissions data.
    model : str
        Model name.
    scenario : str
        Scenario name.
    emi_kind : str, optional
        Emission kind pattern, default "**CO2|AFOLU".

    Returns
    -------
    pd.DataFrame
        Cumulative AFOLU emissions.
    """
    pix = import_pandas_indexing("get_cumulative_afolu")

    emissions = interpolate_to_annual(
        input_df.loc[pix.ismatch(variable=emi_kind, model=model, scenario=scenario)]
    )
    cumulative = pd.DataFrame(
        data=np.nan_to_num(emissions.values).cumsum(axis=1),
        columns=emissions.columns,
        index=pd.MultiIndex.from_tuples(
            [
                tuple(
                    [
                        x.replace("Emissions", "Cumulative Emissions")
                        for x in tuple_index
                    ]
                )
                for tuple_index in emissions.index
            ]
        ),
    )
    return cumulative


def get_cumulative_afolu_fill_from_hist(  # noqa: PLR0913
    input_df: pd.DataFrame,
    model: str,
    scenario: str,
    hist_fill,
    emi_kind="**CO2|AFOLU",
    scenario_end_year=2100,
) -> pd.DataFrame:
    """
    Calculate cumulative AFOLU including historical AFOLU

    Parameters
    ----------
    input_df : pd.DataFrame
        Input emissions data.
    model : str
        Model name.
    scenario : str
        Scenario name.
    hist_fill : pd.DataFrame or similar
        Historical cumulative data.
    emi_kind : str, optional
        Emission kind pattern, default "**CO2|AFOLU".
    scenario_end_year : int, optional
        End year of the scenario, default 2100.

    Returns
    -------
    pd.DataFrame
        Cumulative AFOLU including historical data.
    """
    cumulative = get_cumulative_afolu(input_df, model, scenario, emi_kind=emi_kind)

    full_cumulative = np.zeros((1, scenario_end_year - int(hist_fill.columns[0]) + 1))

    # Finding the array index where scenario years start
    first_scen_year_idx = int(cumulative.columns[0] - hist_fill.columns[0])
    full_cumulative[0, :first_scen_year_idx] = hist_fill.values[0, :first_scen_year_idx]

    just_add_scen = first_scen_year_idx + hist_fill.columns[0]

    # Loop over scenario years, when the scenario cumulative goes positive,
    # we switch to adding the scenario values to the last of the history values
    # TODO: Check if this actually needs to be a loop, or if it only goes to the
    # break part directly anyways...
    for i_year in range(
        first_scen_year_idx, scenario_end_year + 1 - int(hist_fill.columns[0])
    ):
        if (
            cumulative.iloc[
                0, i_year + int(hist_fill.columns[0] - cumulative.columns[0])
            ]
            > 0
        ):
            just_add_scen = i_year + int(hist_fill.columns[0] - cumulative.columns[0])
            to_add = hist_fill.values[0, i_year] - 1
            break
        else:
            full_cumulative[0, i_year] = hist_fill.values[0, i_year]

    # After finding the switch point, add the scenario cumulative values
    full_cumulative[
        0, just_add_scen + int(cumulative.columns[0]) - int(hist_fill.columns[0]) :
    ] = to_add + cumulative.iloc[0, just_add_scen:].values
    # Construct final DataFrame
    cum_df = pd.DataFrame(
        data=full_cumulative,
        columns=np.arange(int(hist_fill.columns[0]), scenario_end_year + 1),
        index=cumulative.index,
    )
    return cum_df
