"""
Finishing of regional extensions and merging with history

Ported from https://github.com/iiasa/emissions_harmonization_historical
(commit 376987dbe3b66fcd23537519a16771e21cba6d1e,
`src/emissions_harmonization_historical/extensions/finish_regional_extensions.py`),
which is based on FLEX (https://github.com/benmsanderson/FLEX, MIT licence).
The logic is unchanged, so that we reproduce the CMIP7 ScenarioMIP extensions exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gcages.pandas_indexing_helpers import import_pandas_indexing


def extend_regional_for_missing(
    df_everything: pd.DataFrame,
    scenarios_regional: pd.DataFrame,
    fractions_list: dict,
    end_year=2500,
) -> pd.DataFrame:
    """
    Extend regional data for scenarios missing region level.

    This function extends regional emissions data for scenarios where some regions are missing.
    It uses fractional compositions to allocate gross positive emissions to missing regions.

    Parameters
    ----------
    df_everything : pd.DataFrame
        The complete dataset including existing extensions.
    scenarios_regional : pd.DataFrame
        Regional scenario data.
    fractions_list : dict
        Dictionary of fractional compositions for each model and scenario.
    end_year : int, optional
        The year to extend to, default is 2500.

    Returns
    -------
    pd.DataFrame
        Extended DataFrame with missing regions filled.
    """
    pix = import_pandas_indexing("extend_regional_for_missing")

    df_extended_list = []

    for variable in pix.uniquelevel(scenarios_regional, "variable").values:
        if not variable.startswith("Emissions|CO2"):
            continue
        unique_meta = (
            scenarios_regional.loc[pix.ismatch(variable=variable)]
            .index.droplevel(["workflow", "variable", "region", "unit"])
            .drop_duplicates()
        )
        for model, scen in unique_meta.to_list():
            data_regional = scenarios_regional.loc[
                pix.ismatch(
                    scenario=f"{scen}", model=f"{model}", variable=f"{variable}**"
                )
            ]
            data_existing = df_everything.loc[
                pix.ismatch(
                    scenario=f"{scen}", model=f"{model}", variable=f"{variable}**"
                )
            ]
            gross_pos_traj = df_everything.loc[
                pix.ismatch(
                    scenario=f"{scen}",
                    model=f"{model}",
                    variable="Emissions|CO2|Gross Positive Emissions",
                    region="World",
                )
            ]
            regions_total = pix.uniquelevel(data_regional, "region").values
            regions_existing = pix.uniquelevel(data_existing, "region").values
            if set(regions_total) == set(regions_existing):
                continue
            if variable in fractions_list[(model, scen)][
                "fractions_fossil_nocdr"
            ].index.get_level_values("variable"):
                fractions_f_em_df = fractions_list[(model, scen)][
                    "fractions_fossil_nocdr"
                ]
                for region in regions_total:
                    if region in regions_existing:
                        continue
                    data_frac_extend = (
                        gross_pos_traj
                        * fractions_f_em_df.loc[
                            pix.ismatch(variable=variable, region=f"{region}")
                        ].values[0, 0]
                    )
                    full_years = np.arange(df_everything.columns[0], end_year + 1)
                    data_extend = np.zeros((1, len(full_years)))
                    values_exist = data_regional.loc[pix.ismatch(region=region)].values[
                        0, :
                    ]
                    extend_year_idx = len(values_exist)
                    data_extend[:, :extend_year_idx] = values_exist
                    data_extend[:, extend_year_idx:] = data_frac_extend.values[
                        :, extend_year_idx:
                    ]
                    df_regional = pd.DataFrame(
                        data=data_extend,
                        columns=full_years,
                        index=data_regional.loc[pix.ismatch(region=region)].index,
                    )
                    df_extended_list.append(df_regional)
    df_extended = pix.concat(df_extended_list)
    return pix.concat([df_everything, df_extended])


def merge_historical_future_timeseries(
    history_data: pd.DataFrame, extensions_data: pd.DataFrame, overlap_year=2023
) -> pd.DataFrame:
    """
    Merge historical and future emissions data.

    This function merges historical emissions data with future extensions, replicating historical data
    for each scenario and concatenating along the time axis. It removes duplicates and filters to common variables.

    Parameters
    ----------
    history_data : pd.DataFrame
        Historical emissions data.
    extensions_data : pd.DataFrame
        Future extensions data.
    overlap_year : int, optional
        The year where historical and future data overlap, default is 2023.

    Returns
    -------
    pd.DataFrame
        Merged continuous timeseries data.
    """
    pix = import_pandas_indexing("merge_historical_future_timeseries")

    # Step 1: Clean extensions data (remove duplicates)
    extensions_clean = extensions_data[~extensions_data.index.duplicated(keep="first")]

    # Step 2: Define time splits
    hist_years = [
        col
        for col in history_data.columns
        if isinstance(col, int | float) and col <= overlap_year
    ]
    future_years = [
        col
        for col in extensions_clean.columns
        if isinstance(col, int | float) and col > overlap_year
    ]

    # Step 3: Get scenario list from extensions
    scenarios = extensions_clean.index.droplevel(
        ["region", "variable", "unit", "workflow"]
    ).drop_duplicates()

    # Step 4: Replicate historical data for each scenario
    historical_expanded = []
    for model, scenario in scenarios:
        hist_copy = history_data[hist_years].copy()

        # Update index to match scenario structure
        new_index = []
        for idx in hist_copy.index:
            new_idx = (
                model,
                scenario,
                idx[2],
                "for_scms",  # workflow
                idx[3],
                idx[4],
            )  # model, scenario, region, variable, unit
            new_index.append(new_idx)
        hist_copy.index = pd.MultiIndex.from_tuples(
            new_index,
            names=["model", "scenario", "region", "workflow", "variable", "unit"],
        )
        historical_expanded.append(hist_copy)

    historical_replicated = pd.concat(historical_expanded)
    historical_replicated.index = historical_replicated.index.reorder_levels(
        extensions_clean.index.names
    )

    # Step 5: Get future data and merge
    future_data = extensions_clean[future_years]

    # Step 6: Find common variables and merge
    hist_vars = set(historical_replicated.index.get_level_values("variable"))
    future_vars = set(future_data.index.get_level_values("variable"))
    common_vars = hist_vars & future_vars

    # Filter to common variables
    hist_common = historical_replicated.loc[
        historical_replicated.index.get_level_values("variable").isin(common_vars)
    ]
    future_common = future_data.loc[
        future_data.index.get_level_values("variable").isin(common_vars)
    ]
    # Step 7: Concatenate along time axis
    continuous_data = pd.concat([hist_common, future_common], axis=1).sort_index(axis=1)

    # Step 8: Cut superfluous global data:
    continuous_data = continuous_data.loc[~pix.ismatch(workflow="global")]

    return continuous_data
