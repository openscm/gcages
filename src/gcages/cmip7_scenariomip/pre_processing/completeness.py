"""
Checks of reporting completeness
"""

from __future__ import annotations

import pandas as pd

from gcages.cmip7_scenariomip.pre_processing.constants import (
    ALL_MODEL_REGION_VARIABLES_INPUT,
    ALL_WORLD_VARIABLES_INPUT,
    INDEPENDENT_MODEL_REGION_VARIABLES_INPUT,
    INDEPENDENT_WORLD_VARIABLES_INPUT,
    REQUIRED_MODEL_REGION_VARIABLES_INPUT,
    REQUIRED_WORLD_VARIABLES_INPUT,
)


def get_required_world_index_input(
    world_region: str = "World",
    region_level: str = "region",
    variable_level: str = "variable",
    variables: tuple[str, ...] = REQUIRED_WORLD_VARIABLES_INPUT,
) -> pd.MultiIndex:
    """
    Get the required index for data reported at the world level in the input

    Parameters
    ----------
    world_region
        Name of the region which represents world i.e. global total data

    region_level
        Name of the region level in the index

    variable_level
        Name of the variable level in the index

    variables
        Variables to include in the index

    Returns
    -------
    :
        Required index
    """
    return pd.MultiIndex.from_product(
        [variables, [world_region]],
        names=[variable_level, region_level],
    )


def get_all_world_index_input(
    world_region: str = "World",
    region_level: str = "region",
    variable_level: str = "variable",
    variables: tuple[str, ...] = ALL_WORLD_VARIABLES_INPUT,
) -> pd.MultiIndex:
    """
    Get the index for all considered data reported at the world level in the input

    Parameters
    ----------
    world_region
        Name of the region which represents world i.e. global total data

    region_level
        Name of the region level in the index

    variable_level
        Name of the variable level in the index

    variables
        Variables to include in the index

    Returns
    -------
    :
        All considered data index
    """
    return pd.MultiIndex.from_product(
        [variables, [world_region]],
        names=[variable_level, region_level],
    )


def get_required_model_region_index_input(
    model_regions: tuple[str, ...],
    region_level: str = "region",
    variable_level: str = "variable",
    variables: tuple[str, ...] = REQUIRED_MODEL_REGION_VARIABLES_INPUT,
) -> pd.MultiIndex:
    """
    Get the required index for data reported at the model region level in the input

    We assume that all combinations of variables
    and model regions need to be included in the reporting.

    Parameters
    ----------
    model_regions
        The model's regions

    region_level
        Name of the region level in the index

    variable_level
        Name of the variable level in the index

    variables
        Variables to include in the index

    Returns
    -------
    :
        Required index
    """
    return pd.MultiIndex.from_product(
        [variables, model_regions],
        names=[variable_level, region_level],
    )


def get_all_model_region_index_input(
    model_regions: tuple[str, ...],
    region_level: str = "region",
    variable_level: str = "variable",
    variables: tuple[str, ...] = ALL_MODEL_REGION_VARIABLES_INPUT,
) -> pd.MultiIndex:
    """
    Get the required index for all considered data reported at the model region level

    This assumes the naming convention used in the input

    We assume that all combinations of variables
    and model regions need to be included in the reporting.

    Parameters
    ----------
    model_regions
        The model's regions

    region_level
        Name of the region level in the index

    variable_level
        Name of the variable level in the index

    variables
        Variables to include in the index

    Returns
    -------
    :
        All considered data index
    """
    return pd.MultiIndex.from_product(
        [variables, model_regions],
        names=[variable_level, region_level],
    )


def get_independent_index_input(  # noqa: PLR0913
    model_regions: tuple[str, ...],
    model_region_variables: tuple[str, ...] = INDEPENDENT_MODEL_REGION_VARIABLES_INPUT,
    world_region: str = "World",
    world_variables: tuple[str, ...] = INDEPENDENT_WORLD_VARIABLES_INPUT,
    region_level: str = "region",
    variable_level: str = "variable",
) -> pd.MultiIndex:
    """
    Get the index for all considered data reported at the world level in the input

    Parameters
    ----------
    model_regions
        The model's regions

    model_region_variables
        Variables to include at the model region level

    world_region
        Name of the region which represents world i.e. global total data

    world_variables
        Variables to include at the world level

    region_level
        Name of the region level in the index

    variable_level
        Name of the variable level in the index

    Returns
    -------
    :
        All considered data index
    """
    model_region_index = pd.MultiIndex.from_product(
        [model_region_variables, model_regions],
        names=[variable_level, region_level],
    )

    world_index = pd.MultiIndex.from_product(
        [world_variables, [world_region]],
        names=[variable_level, region_level],
    )

    res = model_region_index.append(world_index)

    return res
