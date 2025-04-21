"""
Checks of reporting completeness
"""

from __future__ import annotations

import pandas as pd

from gcages.cmip7_scenariomip.pre_processing.constants import (
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
        [REQUIRED_MODEL_REGION_VARIABLES_INPUT, [world_region]],
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
        The model regions to check

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
        [REQUIRED_MODEL_REGION_VARIABLES_INPUT, model_regions],
        names=[variable_level, region_level],
    )
