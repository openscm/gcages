"""
Splitting and extension of CDR and gross positive fossil CO2 emissions

Ported from https://github.com/iiasa/emissions_harmonization_historical
(commit 376987dbe3b66fcd23537519a16771e21cba6d1e,
`src/emissions_harmonization_historical/extensions/cdr_and_fossil_splits.py`),
which is based on FLEX (https://github.com/benmsanderson/FLEX, MIT licence).
The logic is unchanged, so that we reproduce the CMIP7 ScenarioMIP extensions exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gcages.pandas_indexing_helpers import import_pandas_indexing


def extend_cdr_components_vectorized(
    cdr_components_dict: dict, global_cdr_ext: pd.DataFrame, baseline_year=2100
) -> dict:
    """
    Ultra-efficient CDR extension using pure vectorized operations.

    Eliminates all DataFrame fragmentation warnings by using bulk operations.

    FIXED: Handles MultiIndex alignment properly for regional fraction calculations.

    Parameters
    ----------
    cdr_components_dict : dict
        Dictionary of CDR component DataFrames.
    global_cdr_ext : pd.DataFrame
        Global CDR extension data.
    baseline_year : int, optional
        Baseline year for calculations, default is 2100.

    Returns
    -------
    dict
        Dictionary of extended CDR component DataFrames.
    """
    extension_years = [
        col
        for col in global_cdr_ext.columns
        if isinstance(col, int | float) and col > baseline_year
    ]

    extended_components = {}

    for component_name, cdr_df in cdr_components_dict.items():
        # Validate inputs and find common scenarios
        validation_result = _validate_extension_inputs(
            cdr_df, global_cdr_ext, baseline_year
        )
        if not validation_result["is_valid"]:
            extended_components[component_name] = cdr_df.copy()
            continue

        # Calculate baseline ratios
        ratios = _calculate_baseline_ratios(cdr_df, global_cdr_ext, baseline_year)

        # Build extension data
        extension_data = _build_extension_data(
            extension_years,
            global_cdr_ext,
            ratios["baseline_data"],
            ratios["component_fractions"],
            ratios["regional_fractions"],
        )

        # Construct final DataFrame
        final_df = _construct_final_dataframe(cdr_df, extension_data, baseline_year)
        extended_components[component_name] = final_df

    return extended_components


def _validate_extension_inputs(
    cdr_df: pd.DataFrame, global_cdr_ext: pd.DataFrame, baseline_year: int
) -> dict:
    """Validate inputs and find common scenarios.

    Parameters
    ----------
    cdr_df : pd.DataFrame
        CDR component DataFrame.
    global_cdr_ext : pd.DataFrame
        Global CDR extension DataFrame.
    baseline_year : int
        Baseline year to check.

    Returns
    -------
    dict
        Validation result with 'is_valid' and optionally 'common_scenarios'.
    """
    if baseline_year not in cdr_df.columns:
        return {"is_valid": False}

    cdr_scenarios = set(cdr_df.index.get_level_values("scenario"))
    global_scenarios = set(global_cdr_ext.index.get_level_values("scenario"))
    common_scenarios = cdr_scenarios & global_scenarios

    if not common_scenarios:
        return {"is_valid": False}

    return {"is_valid": True, "common_scenarios": common_scenarios}


def _calculate_baseline_ratios(
    cdr_df: pd.DataFrame, global_cdr_ext: pd.DataFrame, baseline_year: int
) -> dict:
    """Calculate baseline ratios for component and regional fractions.

    Parameters
    ----------
    cdr_df : pd.DataFrame
        CDR component DataFrame.
    global_cdr_ext : pd.DataFrame
        Global CDR extension DataFrame.
    baseline_year : int
        Baseline year for ratios.

    Returns
    -------
    dict
        Dictionary with 'baseline_data', 'component_fractions', 'regional_fractions'.
    """
    baseline_data = cdr_df[baseline_year]
    component_totals_2100 = baseline_data.groupby("scenario").sum()
    global_data_2100 = global_cdr_ext[baseline_year].groupby("scenario").first()
    component_fractions = (component_totals_2100 / global_data_2100).fillna(0)

    # Calculate regional fractions
    regional_fractions = baseline_data.copy()
    for scenario in baseline_data.index.get_level_values("scenario").unique():
        scenario_mask = baseline_data.index.get_level_values("scenario") == scenario
        scenario_data = baseline_data[scenario_mask]
        component_total = component_totals_2100[scenario]

        if component_total != 0:
            regional_fractions[scenario_mask] = scenario_data / component_total
        else:
            regional_fractions[scenario_mask] = 0

    regional_fractions = regional_fractions.fillna(0)

    return {
        "baseline_data": baseline_data,
        "component_fractions": component_fractions,
        "regional_fractions": regional_fractions,
    }


def _build_extension_data(
    extension_years: np.ndarray,
    global_cdr_ext: pd.DataFrame,
    baseline_data: pd.Series,
    component_fractions: pd.Series,
    regional_fractions: pd.DataFrame,
) -> dict:
    """
    Build extension data for all years.

    Parameters
    ----------
    extension_years : np.ndarray
        Array of years to extend to.
    global_cdr_ext : pd.DataFrame
        Global CDR extension data.
    baseline_data : pd.Series
        Baseline data at baseline year.
    component_fractions : pd.Series
        Component fractions.
    regional_fractions : pd.DataFrame
        Regional fractions.

    Returns
    -------
    dict
        Dictionary of extension data per year.
    """
    extension_data_dict = {}

    for year in extension_years:
        if year in global_cdr_ext.columns:
            global_year_totals = global_cdr_ext[year].groupby("scenario").first()
            component_year_totals = global_year_totals * component_fractions

            year_values = baseline_data.copy()
            for scenario in baseline_data.index.get_level_values("scenario").unique():
                scenario_mask = (
                    baseline_data.index.get_level_values("scenario") == scenario
                )
                if scenario in component_year_totals.index:
                    component_total_year = component_year_totals[scenario]
                    year_values[scenario_mask] = (
                        regional_fractions[scenario_mask] * component_total_year
                    )
                else:
                    year_values[scenario_mask] = 0

            extension_data_dict[year] = year_values

    return extension_data_dict


def _construct_final_dataframe(
    cdr_df: pd.DataFrame, extension_data_dict: dict, baseline_year: int
) -> pd.DataFrame:
    """
    Construct final DataFrame with original and extension data.

    Parameters
    ----------
    cdr_df : pd.DataFrame
        Original CDR DataFrame.
    extension_data_dict : dict
        Extension data dictionary.
    baseline_year : int
        Baseline year.

    Returns
    -------
    pd.DataFrame
        Final extended DataFrame.
    """
    if extension_data_dict:
        original_years = [
            col
            for col in cdr_df.columns
            if isinstance(col, int | float) and col <= baseline_year
        ]
        original_data = cdr_df[original_years]

        extension_df = pd.DataFrame(extension_data_dict, index=cdr_df.index)
        all_year_data = pd.concat([original_data, extension_df], axis=1)

        non_year_cols = [
            col for col in cdr_df.columns if not isinstance(col, int | float)
        ]

        if len(non_year_cols) == 0:
            final_df = all_year_data.copy()
        else:
            final_df = pd.concat([all_year_data, cdr_df[non_year_cols]], axis=1)
    else:
        final_df = cdr_df.copy()

    return final_df


def get_2100_compound_composition_co2(
    data_regional: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find fractional composition of values in 2100 to allocate the residual end point emissions accordingly

    Parameters
    ----------
    data_regional : pd.DataFrame
        Regional data with variables and regions.

    Returns
    -------
    tuple of pd.DataFrame
        Fractions for fossil sectors, CDR sectors, and fossil emissions sectors.
    """
    pix = import_pandas_indexing("get_2100_compound_composition_co2")

    # TODO: Clean this out to do only what we need
    sector_mapping = {
        "Emissions|CO2|AFOLU": [
            "Emissions|CO2|Peat Burning",
            "Emissions|CO2|Forest Burning",
            "Emissions|CO2|Grassland BurningEmissions|CO2|Agricultural Waste Burning",
            "Emissions|CO2|Agriculture",
        ],
        "Emissions|CO2|Energy and Industrial Processes": [
            "Emissions|CO2|Energy Sector",
            "Emissions|CO2|Industrial Sector",
            "Emissions|CO2|Solvents Production and Application",
            "Emissions|CO2|Waste",
            "Emissions|CO2|Aircraft",
            "Emissions|CO2|International Shipping",
            "Emissions|CO2|Transportation Sector",
            "Emissions|CO2|Soil Carbon Management",
            "Emissions|CO2|Biochar",
            "Emissions|CO2|Enhanced Weathering",
            "Emissions|CO2|Ocean",
            "Emissions|CO2|Other CDR",
            "Emissions|CO2|Direct Air Capture",
            "Emissions|CO2|BECCS",
            "Emissions|CO2|Residential Commercial Other",
        ],
        "Emissions|CO2|Gross Removals": [
            "Emissions|CO2|Soil Carbon Management",
            "Emissions|CO2|Biochar",
            "Emissions|CO2|Enhanced Weathering",
            "Emissions|CO2|Ocean",
            "Emissions|CO2|Other CDR",
            "Emissions|CO2|Direct Air Capture",
            "Emissions|CO2|BECCS",
        ],
        "Emissions|CO2|Gross Emissions": [
            "Emissions|CO2|Energy Sector",
            "Emissions|CO2|Industrial Sector",
            "Emissions|CO2|Solvents Production and Application",
            "Emissions|CO2|Waste",
            "Emissions|CO2|Aircraft",
            "Emissions|CO2|International Shipping",
            "Emissions|CO2|Transportation Sector",
            "Emissions|CO2|Residential Commercial Other",
        ],
    }

    data_sub_fossil = []
    data_sub_cdr = []
    data_sub_fossil_em = []
    for sector in sector_mapping["Emissions|CO2|Energy and Industrial Processes"]:
        data_here = data_regional.loc[pix.ismatch(variable=sector)]
        if len(pix.uniquelevel(data_here, "region")) > 1:
            if "World" in pix.uniquelevel(data_here, "region"):
                data_here = data_here.loc[pix.ismatch(region="World")]
                msg = "Unexpected state (`sys.exit(4)` in the original)"
                raise AssertionError(msg)
        elif data_here.shape[0] == 0:
            msg = "Unexpected state (`sys.exit(4)` in the original)"
            raise AssertionError(msg)
        data_sub_fossil.append(data_regional.loc[pix.ismatch(variable=sector)])
        if sector in sector_mapping["Emissions|CO2|Gross Emissions"]:
            data_sub_fossil_em.append(data_regional.loc[pix.ismatch(variable=sector)])
        if sector in sector_mapping["Emissions|CO2|Gross Removals"]:
            data_sub_cdr.append(data_regional.loc[pix.ismatch(variable=sector)])
    data_fossil_sectors = pd.concat(data_sub_fossil)
    sum_fossil_sectors = data_fossil_sectors.sum(axis=0)
    data_cdr_sectors = pd.concat(data_sub_cdr)
    sum_cdr_sectors = data_cdr_sectors.sum(axis=0)
    data_f_em_sectors = pd.concat(data_sub_fossil_em)
    sum_f_em_sectors = data_f_em_sectors.sum(axis=0)
    fractions = data_fossil_sectors.values / sum_fossil_sectors
    fractions_df = pd.DataFrame(
        data=fractions, columns=["fractions"], index=data_fossil_sectors.index
    )
    fractions_cdr = data_cdr_sectors.values / sum_cdr_sectors
    fractions_df_cdr = pd.DataFrame(
        data=fractions_cdr, columns=["fractions"], index=data_cdr_sectors.index
    )
    fractions_f_em = data_f_em_sectors.values / sum_f_em_sectors
    fractions_df_f_em = pd.DataFrame(
        data=fractions_f_em, columns=["fractions"], index=data_f_em_sectors.index
    )
    return fractions_df, fractions_df_cdr, fractions_df_f_em


def add_removals_and_positive_fossil_emissions_to_historical(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Make gross positive and removal variables for CO2 in historical data

    Parameters
    ----------
    history : pd.DataFrame
        Historical emissions data.

    Returns
    -------
    pd.DataFrame
        Historical data with added gross positive and removal variables.
    """
    pix = import_pandas_indexing(
        "add_removals_and_positive_fossil_emissions_to_historical"
    )

    co2_ffi_hist = history.loc[
        pix.ismatch(variable="Emissions|CO2|Energy and Industrial Processes")
    ].copy()

    # Create Gross Positive Emissions by copying AFOLU data and changing variable name
    gross_positive_hist = co2_ffi_hist.copy()
    new_index_gross = []
    for idx_tuple in gross_positive_hist.index:
        new_tuple = list(idx_tuple)
        new_tuple[3] = (
            "Emissions|CO2|Gross Positive Emissions"  # variable is at position 3
        )
        new_index_gross.append(tuple(new_tuple))
    gross_positive_hist.index = pd.MultiIndex.from_tuples(
        new_index_gross, names=gross_positive_hist.index.names
    )

    # Create Gross Removals as zeros with same structure as AFOLU
    gross_removals_hist = gross_positive_hist.copy()
    gross_removals_hist.iloc[:, :] = 0.0  # Set all values to zero
    new_index_removals = []
    for idx_tuple in gross_removals_hist.index:
        new_tuple = list(idx_tuple)
        new_tuple[3] = "Emissions|CO2|Gross Removals"  # variable is at position 3
        new_index_removals.append(tuple(new_tuple))
    gross_removals_hist.index = pd.MultiIndex.from_tuples(
        new_index_removals, names=gross_removals_hist.index.names
    )

    # Remove any previously added gross emissions variables and add the new ones
    history_clean = history.loc[
        ~history.index.get_level_values("variable").isin(
            ["Emissions|CO2|Gross Positive Emissions", "Emissions|CO2|Gross Removals"]
        )
    ]
    history = pd.concat([history_clean, gross_positive_hist, gross_removals_hist])

    return history
