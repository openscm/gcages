"""
Integration tests of our pre-processing for CMIP7 ScenarioMIP
"""

from functools import partial

import numpy as np
import pytest
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.indexing import multi_index_lookup

from gcages.aggregation import get_region_sector_sum
from gcages.cmip7_scenariomip import (
    CMIP7ScenarioMIPPreProcessor,
)
from gcages.cmip7_scenariomip.gridding_emissions import to_global_workflow_emissions
from gcages.cmip7_scenariomip.pre_processing.reaggregation.basic import (
    get_example_input,
)
from gcages.index_manipulation import (
    split_sectors,
)
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.testing import assert_frame_equal
from gcages.units_helpers import strip_pint_incompatible_characters_from_units

pix = pytest.importorskip("pandas_indexing")


def test_output_consistency_with_input_for_non_region_sector(example_input_output):
    """
    Test consistency between the output that is not at the region-sector level
    and the input emissions for species that aren't used in gridding
    """
    gridding_species = split_sectors(
        example_input_output.output.gridding_workflow_emissions
    ).pix.unique("species")

    not_from_region_sector = [
        v
        for v in example_input_output.output.global_workflow_emissions_raw_names.pix.unique(  # noqa: E501
            "variable"
        )
        if not any(species in v for species in gridding_species)
    ]

    not_from_region_sector_res = (
        example_input_output.output.global_workflow_emissions_raw_names.loc[
            pix.isin(variable=not_from_region_sector)
        ]
    )

    not_from_region_sector_compare = strip_pint_incompatible_characters_from_units(
        example_input_output.input.loc[pix.isin(variable=not_from_region_sector)]
    )

    assert_frame_equal(not_from_region_sector_res, not_from_region_sector_compare)


def test_output_gridding_global_workflow_emissions_internal_consistency(
    example_input_output,
):
    """
    Test consistency between the gridding and global workflow emissions in the output
    """
    global_workflow_emissions_derived = to_global_workflow_emissions(
        example_input_output.output.gridding_workflow_emissions,
        global_workflow_co2_fossil_sector="Energy and Industrial Processes",
        global_workflow_co2_biosphere_sector="AFOLU",
    )

    exp_compare = multi_index_lookup(
        example_input_output.output.global_workflow_emissions_raw_names,
        global_workflow_emissions_derived.index,
    )
    assert_frame_equal(exp_compare, global_workflow_emissions_derived)


def test_output_internal_consistency_global_workflow_emissions(example_input_output):
    """
    Make sure that the two global workflow outputs are just a renaming of each other
    """
    assert_frame_equal(
        update_index_levels_func(
            example_input_output.output.global_workflow_emissions_raw_names,
            dict(
                variable=partial(
                    convert_variable_name,
                    from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                    to_convention=SupportedNamingConventions.GCAGES,
                )
            ),
        ),
        example_input_output.output.global_workflow_emissions,
    )


def test_output_vs_start_total_consistency(example_input_output):
    """
    Check that the output and the input have the same totals
    """
    gridded_emisssions_sector_regional_sum = get_region_sector_sum(
        example_input_output.output.gridding_workflow_emissions
    )
    # To avoid double counting
    drop_vars = [
        "Emissions|CH4|AFOLU",
        "Emissions|N2O|AFOLU",
        "Emissions|NOx|AFOLU",
        "Emissions|BC|AFOLU",
        "Emissions|NH3|AFOLU",
        "Emissions|OC|AFOLU",
        "Emissions|VOC|AFOLU",
        "Emissions|Sulfur|AFOLU",
        "Emissions|CO|AFOLU",
        "Emissions|CO2|AFOLU",
    ]
    df_to_sum = multi_index_lookup(
        example_input_output.input,
        example_input_output.reaggregator.get_internal_consistency_checking_index(),
    )
    df_to_sum = df_to_sum.loc[
        ~df_to_sum.index.get_level_values("variable").isin(drop_vars)
    ]

    input_emissions_sector_region_sum = get_region_sector_sum(df_to_sum)
    # The sign of the 'Carbon Removal' must be flipped to do the comparision
    # (it gets changed in the reaggregation)
    #
    mask = gridded_emisssions_sector_regional_sum.index.get_level_values(
        "variable"
    ).str.startswith("Carbon Removal")
    gridded_emisssions_sector_regional_sum.loc[mask] *= -1

    # To account for CDR in the sum
    cdr = input_emissions_sector_region_sum.xs("Carbon Removal|CO2", level="variable")
    emi = input_emissions_sector_region_sum.xs("Emissions|CO2", level="variable")
    # Sum them
    co2_sum = emi + cdr
    # Assign the result back into the original DataFrame under "Emissions|CO2"
    for idx, row in co2_sum.iterrows():
        new_idx = idx[:3] + ("Emissions|CO2",) + idx[3:]  # Rebuild full MultiIndex
        input_emissions_sector_region_sum.loc[new_idx] = row.values

    assert_frame_equal(
        gridded_emisssions_sector_regional_sum, input_emissions_sector_region_sum
    )


def test_multiple_scenarios_different_time_axes():
    model_a = "model_a"
    model_regions_a = [f"{model_a}|{r}" for r in ("China", "Pacific OECD")]
    scenario_a = get_example_input(
        model_regions=model_regions_a,
        model=model_a,
        scenario="scenario_a",
        timepoints=np.arange(2010, 2100 + 1, 10),
    )

    model_b = "model_b"
    model_regions_b = [f"{model_b}|{r}" for r in ("India", "Brazil", "North America")]
    scenario_b = get_example_input(
        model_regions=model_regions_b,
        model=model_b,
        scenario="scenario_b",
        timepoints=np.arange(2010, 2100 + 1, 10),
        global_only_variables=(
            ("Emissions|HFC|HFC245fa", "kt HFC245fa/yr"),
            ("Emissions|HFC|HFC365mfc", "kt HFC365mfc/yr"),
        ),
    )

    pre_processor = CMIP7ScenarioMIPPreProcessor(
        progress=False,
        n_processes=None,  # process serially
    )

    res_a = pre_processor(scenario_a)
    res_b = pre_processor(scenario_b)

    scenarios_combined = pix.concat([scenario_a, scenario_b]).sort_index(axis="columns")
    res_combined = pre_processor(scenarios_combined)

    for res_individual in [res_a, res_b]:
        for attr in [
            "gridding_workflow_emissions",
            "global_workflow_emissions",
            "global_workflow_emissions_raw_names",
        ]:
            res_individual_df = getattr(res_individual, attr)

            model_l = pix.uniquelevel(res_individual_df, "model")
            if len(model_l) != 1:
                raise AssertionError
            model = model_l[0]
            scenario_l = pix.uniquelevel(res_individual_df, "scenario")
            if len(scenario_l) != 1:
                raise AssertionError
            scenario = scenario_l[0]

            res_combined_df = getattr(res_combined, attr)
            res_combined_df_ms = res_combined_df.loc[
                pix.isin(model=model, scenario=scenario)
            ]
            res_combined_df_ms_nan_times_dropped = res_combined_df_ms.dropna(
                how="all", axis="columns"
            )

            assert_frame_equal(res_individual_df, res_combined_df_ms_nan_times_dropped)
