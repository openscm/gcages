"""
Assertions while doing the pre-processing
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from pandas_openscm.grouping import groupby_except
from pandas_openscm.indexing import multi_index_lookup

from gcages.cmip7_scenariomip.pre_processing.completeness import (
    get_independent_index_input,
    get_required_model_region_index_input,
    get_required_world_index_input,
)
from gcages.completeness import NotCompleteError, assert_all_groups_are_complete
from gcages.index_manipulation import (
    combine_species,
    set_new_single_value_levels,
    split_sectors,
)
from gcages.testing import compare_close

if TYPE_CHECKING:
    from gcages.cmip7_scenariomip.pre_processing import (
        CMIP7ScenarioMIPPreProcessingResult,
    )


def assert_data_is_compatible_with_pre_processing(
    indf: pd.DataFrame,
    world_region: str,
    region_level: str,
    variable_level: str,
    model_regions: tuple[str, ...],
) -> None:
    indf_drop_all_nan_cols = indf.dropna(how="all", axis="columns")
    if indf_drop_all_nan_cols.isnull().any().any():
        msg = f"NaNs after dropping unreported times:\n{indf_drop_all_nan_cols}"
        raise AssertionError(msg)

    assert_all_groups_are_complete(
        indf,
        get_required_world_index_input(
            world_region=world_region,
            region_level=region_level,
            variable_level=variable_level,
        ),
    )

    try:
        assert_all_groups_are_complete(
            indf,
            get_required_model_region_index_input(
                model_regions=model_regions,
                region_level=region_level,
                variable_level=variable_level,
            ),
        )
    except NotCompleteError as exc:
        msg = (
            "Your data is not complete at the regional level. "
            "You may need to filter "
            f"to your model regions and {world_region} before running. "
            "While checking, we assumed (maybe incorrectly) "
            f"the following model regions: {model_regions}"
        )
        raise ValueError(msg) from exc


class InternalConsistencyError(ValueError):
    """
    Raised when there is an internal consistency issue in the data

    Specifically, the sum of components doesn't match some total
    """

    def __init__(
        self,
        differences: pd.DataFrame,
        data_that_was_summed: pd.DataFrame,
    ) -> None:
        differences_variables = differences.index.get_level_values("variable").unique()
        data_that_was_summed_relevant_for_differences = data_that_was_summed[
            data_that_was_summed.index.get_level_values("variable").map(
                lambda x: any(v in x for v in differences_variables)
            )
        ].index.to_frame(index=False)

        error_msg = (
            "Summing the components does not equal the total. "
            f"Differences:\n{differences}\n"
            "This is the data we used in the sum:\n"
            f"{data_that_was_summed_relevant_for_differences}"
        )

        super().__init__(error_msg)


def assert_data_has_required_internal_consistency(  # noqa: PLR0913
    indf: pd.DataFrame,
    model_regions: tuple[str, ...],
    world_region: str,
    region_level: str,
    variable_level: str,
    tols: dict[str, dict[str, float]],
    level_separator: str = "|",
    n_levels_for_total: int = 1,
) -> None:
    totals_reported = indf.loc[
        indf.index.get_level_values(variable_level).map(
            lambda x: x.count(level_separator) == n_levels_for_total
        )
        & (indf.index.get_level_values(region_level) == world_region)
    ]

    indf_used_in_gridding = multi_index_lookup(
        indf,
        get_independent_index_input(model_regions),
    )
    for variable, totals_reported_variable in totals_reported.groupby(variable_level):
        indf_variable = indf_used_in_gridding.loc[
            indf_used_in_gridding.index.get_level_values(variable_level).str.startswith(
                f"{variable}{level_separator}"
            )
        ]
        if indf_variable.empty:
            # Nothing reported for this variable, carry on
            # (can do this here as we've checked completeness elsewhere)
            continue

        totals_exp_variable = set_new_single_value_levels(
            combine_species(
                groupby_except(
                    split_sectors(indf_variable, bottom_level="sectors"),
                    [region_level, "sectors"],
                ).sum()
            ),
            {region_level: world_region},
        ).reorder_levels(totals_reported_variable.index.names)

        comparison_variable = compare_close(
            left=totals_exp_variable,
            right=totals_reported_variable,
            left_name="derived_from_input",
            right_name="reported_total",
            **tols[variable],
        )

        if not comparison_variable.empty:
            raise InternalConsistencyError(
                differences=comparison_variable,
                data_that_was_summed=indf_variable,
            )


def assert_no_nans_in_res_ms(res_ms: CMIP7ScenarioMIPPreProcessingResult) -> None:
    for attr in [
        "global_workflow_emissions",
        "global_workflow_emissions_raw_names",
        "gridding_workflow_emissions",
    ]:
        if getattr(res_ms, attr).isnull().any().any():
            ms = (
                res_ms.index.droplevels(
                    res_ms.index.names.difference(["model", "scenario"])
                )
                .drop_duplicates()
                .tolist()
            )
            msg = f"NaNs in res.{attr} for {ms}"
            raise AssertionError(msg)


def assert_column_type_unchanged_in_res_ms(
    res_ms: CMIP7ScenarioMIPPreProcessingResult, in_column_type: type
) -> None:
    for attr in [
        "global_workflow_emissions",
        "global_workflow_emissions_raw_names",
        "gridding_workflow_emissions",
    ]:
        df = getattr(res_ms, attr)
        if df.columns.dtype != in_column_type:
            msg = "Column type has changed: " f"{df.columns.dtype=} {in_column_type}"
            raise AssertionError(msg)
