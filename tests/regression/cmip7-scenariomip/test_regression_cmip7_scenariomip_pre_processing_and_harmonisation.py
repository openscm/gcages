"""
Test pre-processing and harmonisation compared for CMIP7 ScenarioMIP
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gcages.cmip7_scenariomip import CMIP7ScenarioMIPPreProcessor
from gcages.cmip7_scenariomip.pre_processing.reaggregation import ReaggregatorBasic
from gcages.cmip7_scenariomip.pre_processing.reaggregation.basic import (
    get_required_timeseries_index,
)

# from emissions_harmonization_historical.harmonisation import harmonise
from gcages.completeness import get_missing_levels
from gcages.index_manipulation import (
    create_levels_based_on_existing,
    set_new_single_value_levels,
)

# from pandas_openscm.index_manipulation import update_index_levels_func

# from gcages.ar6 import AR6Harmoniser, AR6PreProcessor
# from gcages.renaming import SupportedNamingConventions, convert_variable_name
# from gcages.testing import (
#     KEY_TESTING_MODEL_SCENARIOS,
#     assert_frame_equal,
#     get_ar6_harmonised_emissions,
#     get_ar6_raw_emissions,
#     get_key_testing_model_scenario_parameters,
# )

pix = pytest.importorskip("pandas_indexing")

HARMONISATION_YEAR = 2023
RAW_CMIP7_DIR = Path(__file__).parents[0] / "test-data" / "cmip7_scenariomip_data"
PROCESSED_CMIP7_DIR = (
    Path(__file__).parents[0]
    / "test_regression_cmip7_scenariomip_pre_processing_and_harmonisation"
)


def test_individual_scenario(model, scenario):
    file = RAW_CMIP7_DIR / "REMIND_scenariomip_marker_input.csv"
    with open(file) as f:
        model_raw = pd.read_csv(f)

    # Preprocessing

    model_raw.loc[pix.ismatch(variable="**CO2|AFOLU", region="World")].sort_index(
        axis="columns"
    )
    model_df = model_raw.loc[:, 2015:2100].dropna(how="all", axis="columns")
    if model_df.empty:
        raise AssertionError
    model_df.columns.name = "year"
    ## Interpolate
    # (needs to be done to ensure that the CDR-Emissions correction
    # works even if Carbon Removal data is reported at different time resolutions)
    model_df = model_df.T.interpolate(method="index").T

    model_regions = [
        r for r in model_df.pix.unique("region") if r.startswith(model.split(" ")[0])
    ]
    reaggregator = ReaggregatorBasic(model_regions=model_regions)

    required_index = get_required_timeseries_index(
        model_regions=model_regions,
        world_region=reaggregator.world_region,
        region_level=reaggregator.region_level,
        variable_level=reaggregator.variable_level,
    )

    variable_unit_map = {
        "|".join(v.split("|")[:2]): u
        for v, u in model_df.index.droplevel(
            model_df.index.names.difference(["variable", "unit"])
        )
        .drop_duplicates()
        .to_list()
    }

    def guess_unit(v_in: str) -> str:
        """Guess the unit of a given variable"""
        for k, v in variable_unit_map.items():
            if v_in.startswith(f"{k}|") or v_in == k:
                return v

    tmp_l = []
    for (model_l, scenario), sdf in model_df.groupby(["model", "scenario"]):  # noqa: PLR1704
        mls = get_missing_levels(
            sdf.index,
            required_index,
            unit_col=reaggregator.unit_level,
        )

        zeros_hack = pd.DataFrame(
            np.zeros((mls.shape[0], sdf.shape[1])),
            columns=sdf.columns,
            index=create_levels_based_on_existing(
                mls, {"unit": ("variable", guess_unit)}
            ),
        )
        zeros_hack = set_new_single_value_levels(
            zeros_hack,
            {"model": model_l, "scenario": scenario},
        ).reorder_levels(sdf.index.names)
        sdf_full = pix.concat([sdf, zeros_hack])

        tmp_l.append(sdf_full)

    tmp = pix.concat(tmp_l)

    # Assuming that all worked, update the model_df
    model_df = tmp

    # ### Do the pre-processing
    reaggregator = ReaggregatorBasic(model_regions=model_regions)
    pre_processor = CMIP7ScenarioMIPPreProcessor(
        reaggregator=reaggregator,
        n_processes=None,  # run serially
    )

    # #### Temporary hack: turn off checks
    # So even if the model's reporting is inconsistent,
    # it should still pass pre-processing.
    pre_processor.run_checks = False
    pre_processing_res = pre_processor(model_df)  # noqa: F841

    # Hard override the global workflow emissions for CO2 AFOLU
    # to use globally reported numbers,
    # even if they're not consistent with region-sector reporting.

    model_pre_processed_for_global_workflow = pix.concat(
        [
            global_workflow_emissions_raw_names.loc[  # noqa: F821
                ~pix.isin(variable="Emissions|CO2|AFOLU")
            ],
            model_df.loc[pix.isin(variable="Emissions|CO2|AFOLU", region="World")],
        ]
    )

    for y in range(HARMONISATION_YEAR, 2100 + 1):
        if y not in model_pre_processed_for_global_workflow:
            model_pre_processed_for_global_workflow[y] = np.nan

    model_pre_processed_for_global_workflow = (
        model_pre_processed_for_global_workflow.sort_index(axis="columns")
    )
    model_pre_processed_for_global_workflow = (
        model_pre_processed_for_global_workflow.T.interpolate(method="index").T
    )

    ### TO ADDDD
    # history_for_harmonisation = HISTORY_HARMONISATION_DB.load(
    #     pix.ismatch(purpose="global_workflow_emissions")
    # )
    history_for_harmonisation = model_pre_processed_for_global_workflow

    # ### Harmonization

    # for key, idf, user_overrides in (
    #     ("global", model_pre_processed_for_global_workflow, user_overrides_global),
    # ):
    # if user_overrides is not None:
    #     dup_overrides = user_overrides.index.duplicated(keep=False)
    #     if dup_overrides.any():
    #         print(user_overrides.loc[dup_overrides].sort_index())
    #         msg = "There are duplicates in the overrides"
    #         raise AssertionError(msg)

    harmonised_key = harmonise(  # noqa: F821
        scenarios=model_pre_processed_for_global_workflow.reset_index(
            "stage", drop=True
        ),
        history=history_for_harmonisation,
        harmonisation_year=HARMONISATION_YEAR,
        user_overrides=None,
    )
    res = harmonised_key.timeseries
    # from_global = res["global"].timeseries.loc[~pix.isin(variable=from_
    # gridding.pix.unique("variable"))]
    ##################################
    assert_frame_equal(res, exp)  # noqa: F821
