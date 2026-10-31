"""
General utilities used across the extensions (interpolation, gluing etc.)

Ported from https://github.com/iiasa/emissions_harmonization_historical
(commit 376987dbe3b66fcd23537519a16771e21cba6d1e,
`src/emissions_harmonization_historical/extensions/general_utils_for_extensions.py`),
which is based on FLEX (https://github.com/benmsanderson/FLEX, MIT licence).
The logic is unchanged, so that we reproduce the CMIP7 ScenarioMIP extensions exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gcages.pandas_indexing_helpers import import_pandas_indexing


def interpolate_to_annual(
    idf: pd.DataFrame, max_supplement: float = 1e-5
) -> pd.DataFrame:
    """
    Interpolate pd.DataFrame of emissions input that might not have data for all years

    Parameters
    ----------
    idf : pd.DataFrame
        Input DataFrame with potentially missing years.
    max_supplement : float, optional
        Maximum supplement for year range, default 1e-5.

    Returns
    -------
    pd.DataFrame
        Interpolated DataFrame with all years filled.
    """
    missing_cols = np.setdiff1d(
        np.arange(idf.columns.min(), idf.columns.max() + max_supplement), idf.columns
    )

    out = idf.copy()
    out.loc[:, missing_cols] = np.nan
    out = out.sort_index(axis="columns").T.interpolate("index").T

    return out


def glue_with_historical(
    scen_df: pd.DataFrame, hist_df: pd.DataFrame, history_end=2023
) -> pd.DataFrame:
    """
    Glue historical data to the beginning of scenario data to get complete timeseries

    Parameters
    ----------
    scen_df : pd.DataFrame
        Scenario data.
    hist_df : pd.DataFrame
        Historical data.
    history_end : int, optional
        End year of history, default 2023.

    Returns
    -------
    pd.DataFrame
        Combined timeseries.
    """
    pix = import_pandas_indexing("glue_with_historical")

    out = interpolate_to_annual(scen_df.copy())
    orig_len = len(out.columns)
    missing_years = np.arange(int(hist_df.columns[0]), int(scen_df.columns[0]))
    out.loc[:, missing_years] = np.nan  # hist_df.loc[:, missing_years].values
    for index, df_row in out.iterrows():
        df_row.iloc[orig_len:] = hist_df.loc[pix.ismatch(variable=index[3])].values[
            0, : int(scen_df.columns[0] - hist_df.columns[0])
        ]
        for year in range(scen_df.columns[0], history_end + 1):
            if np.isnan(df_row.iloc[year - scen_df.columns[0]]):
                df_row.iloc[year - scen_df.columns[0]] = hist_df.loc[
                    pix.ismatch(variable=index[3])
                ].values[0, int(year - hist_df.columns[0])]
        out.loc[index, :] = df_row.values
    out = out.sort_index(axis="columns")
    return out


def standardize_year_columns(df: pd.DataFrame, startyr=2023) -> pd.DataFrame:
    """Convert all year columns to the same data type

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    startyr : int, optional
        Starting year, default 2023.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized year columns.
    """
    df_copy = df.copy()
    new_columns = []
    drop_columns = []
    for col in df_copy.columns:
        try:
            # Try to convert to target type
            if isinstance(col, str):
                # Remove '.0' suffix if present and convert
                year_val = float(col.replace(".0", ""))
            else:
                year_val = float(col)
            if year_val < startyr:
                drop_columns.append(col)
                continue
            new_columns.append(year_val)
        except (ValueError, TypeError):
            # Keep non-year columns as-is
            new_columns.append(col)
    df_copy = df_copy.drop(columns=drop_columns)
    df_copy.columns = new_columns
    return df_copy


def add_region_level_to_index(df: pd.DataFrame, region="World") -> pd.DataFrame:
    """Add a 'region' level to a DataFrame index after 'scenario'

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    region : str, optional
        Region name, default "World".

    Returns
    -------
    pd.DataFrame
        DataFrame with region level added.
    """
    if "region" in df.index.names:
        return df

    # Reset index to work with it
    df_reset = df.reset_index()

    # Add region column in the correct position (after scenario, before variable)
    cols = list(df_reset.columns)
    scenario_idx = cols.index("scenario")
    cols.insert(scenario_idx + 1, "region")
    df_reset["region"] = region
    df_reset = df_reset[cols]

    # Set the index back with the correct order
    index_cols = ["model", "scenario", "region", "variable", "unit"]
    return df_reset.set_index(index_cols)


def add_workflow_level_to_index(df: pd.DataFrame, workflow="for_scms") -> pd.DataFrame:
    """Add a 'workflow' level to a DataFrame index after 'scenario'

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    workflow : str, optional
        Workflow name, default "for_scms".

    Returns
    -------
    pd.DataFrame
        DataFrame with workflow level added.
    """
    pix = import_pandas_indexing("add_workflow_level_to_index")

    if "workflow" in df.index.names:
        return df

    # Reset index to work with it
    df_reset = df.reset_index()
    # Add region column in the correct position (after scenario, before variable)
    cols = list(df_reset.columns)
    if "region" in df.index.names:
        if len(pix.uniquelevel(df, "region")) > 1:
            workflow = "gridding"
        placement_idx = cols.index("region") + 1
        index_cols = ["model", "scenario", "region", "workflow", "variable", "unit"]
    else:
        placement_idx = cols.index("scenario") + 1
        index_cols = ["model", "scenario", "workflow", "variable", "unit"]

    cols.insert(placement_idx, "workflow")
    df_reset["workflow"] = workflow
    df_reset = df_reset[cols]

    # Set the index back with the correct order
    return df_reset.set_index(index_cols)


def fix_up_and_concatenate_extensions(extended_dfs_dict: dict) -> pd.DataFrame:
    """
    Fix up and concatenate extended DataFrames from different components.

    Ensures consistent indexing and concatenation.

    Parameters
    ----------
    extended_dfs_dict : dict
        Dictionary of extended DataFrames.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame.
    """
    pix = import_pandas_indexing("fix_up_and_concatenate_extensions")

    fixed_dfs = []
    for component_name, df in extended_dfs_dict.items():
        if component_name in ["gross_positive_extensions", "cdr_extensions"]:
            df_fixed = add_region_level_to_index(df.copy(), region="World")
        elif component_name == "afolu_extensions":
            if "model" in df.columns:
                index_cols = ["model", "scenario", "region", "variable", "unit"]
                df_fixed = df.copy().set_index(index_cols)
            else:
                df_fixed = df.copy()
        else:
            df_fixed = df.copy()
        if "workflow" not in df_fixed.index.names:
            df_fixed = add_workflow_level_to_index(df_fixed)
        df_fixed = standardize_year_columns(df_fixed)
        fixed_dfs.append(df_fixed)

    concatenated_df = pix.concat(fixed_dfs)
    return concatenated_df
