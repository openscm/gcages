"""
Test running the whole pipeline compared to CMIP7 ScenarioMIP
"""
# TODOs: own PR

from __future__ import annotations

import multiprocessing
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas_openscm.index_manipulation import (
    set_index_levels_func,
    update_index_levels_func,
    update_levels_from_other,
)
from pandas_openscm.indexing import mi_loc
from pandas_openscm.io import load_timeseries_csv

from gcages.cmip7_scenariomip import (
    CMIP7ScenarioMIPInfiller,
    CMIP7ScenarioMIPPreProcessor,
    CMIP7ScenarioMIPSCMRunner,
    ReaggregatorBasic,
    create_cmip7_scenariomip_global_harmoniser,
    create_cmip7_scenariomip_postprocessor,
)
from gcages.cmip7_scenariomip.pre_processing.reaggregation.basic import (
    get_required_timeseries_index,
)
from gcages.completeness import get_missing_levels
from gcages.pandas_openscm_tmp import interpolate_to_annual_timesteps
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.testing import (
    assert_frame_equal,
    get_cmip7_scenariomip_complete_emissions,
    get_cmip7_scenariomip_harmonised_emissions,
    guess_magicc_exe,
)

pix = pytest.importorskip("pandas_indexing")

CMIP7_SCENARIOMIP_OUT_DIR = Path(__file__).parents[0] / "cmip7-scenariomip-output"

PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR = (
    Path(__file__).parents[0] / "cmip7-scenariomip-workflow-inputs"
)
CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE = (
    PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR / "history_cmip7_scenariomip.csv"
)

CMIP7_SCENARIOMIP_MAGICC_EXECUTABLES_DIR = (
    PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR / "magicc-v7.6.0a3/bin"
)
CMIP7_SCENARIOMIP_MAGICC_PROBABILISTIC_CONFIG_FILE = (
    PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR
    / "magicc-v7.6.0a3/configs/magicc-ar7-fast-track-drawnset-v0-3-0.json"
)

HARMONISATION_YEAR = 2023


def missing_reporting_zero_hack(reaggregator, model_df, model_regions):
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
    for (model_l, scenario), sdf in model_df.groupby(["model", "scenario"]):
        mls = get_missing_levels(
            sdf.index,
            required_index,
            unit_col=reaggregator.unit_level,
        )

        zeros_hack = pd.DataFrame(
            np.zeros((mls.shape[0], sdf.shape[1])),
            columns=sdf.columns,
            index=update_levels_from_other(mls, {"unit": ("variable", guess_unit)}),
        )
        zeros_hack = set_index_levels_func(
            zeros_hack,
            {"model": model_l, "scenario": scenario},
        ).reorder_levels(sdf.index.names)
        sdf_full = pix.concat([sdf, zeros_hack])

        tmp_l.append(sdf_full)

    res = pix.concat(tmp_l)
    return res


@pytest.mark.skip_ci_default
@pytest.mark.slow
@pytest.mark.parametrize(
    "model, scenario",
    [("REMIND-MAgPIE 3.5-4.11", "SSP1 - Very Low Emissions")],
)
def test_whole_pipeline(model, scenario):
    input_df = load_timeseries_csv(
        CMIP7_SCENARIOMIP_OUT_DIR / f"{model}_{scenario}_raw-scenario.csv",
        index_columns=["model", "scenario", "variable", "region", "unit"],
        out_columns_type=int,
        out_columns_name="year",
    )

    # This input data only has non-NaNs from 2015 onwards
    input_df = input_df.loc[:, 2015:]

    # Ensure on annual timesteps before continuing.
    # TODO: figure out where in the pipeline to put a check for this.
    input_df = interpolate_to_annual_timesteps(
        input_df.dropna(how="all", axis="columns")
    )

    model_regions = [
        r
        for r in input_df.index.get_level_values("region").unique()
        if r.startswith(model)
    ]

    reaggregator = ReaggregatorBasic(model_regions=model_regions)

    input_df = missing_reporting_zero_hack(reaggregator, input_df, model_regions)

    pre_processor = CMIP7ScenarioMIPPreProcessor(
        reaggregator=reaggregator,
        n_processes=None,  # run serially
        progress=False,
        run_checks=True,
    )
    pre_processed_res = pre_processor(input_df)

    # TODO: put this logic into the pre-processor
    # TODO: explicit tests going from raw ScenarioMIP input
    # to pre-processed emissions for the rest of the workflow
    pre_processed = pre_processed_res.global_workflow_emissions
    # Hard override the global workflow emissions for CO2 AFOLU
    # to use globally reported numbers,
    # even if they're not consistent with region-sector reporting.
    pre_processed = pix.concat(
        [
            pre_processed.loc[~pix.isin(variable="Emissions|CO2|Biosphere")],
            input_df.loc[
                pix.isin(variable="Emissions|CO2|AFOLU", region="World")
            ].pix.assign(variable="Emissions|CO2|Biosphere"),
        ]
    )
    # Need to make sure the override happens for both naming conventions
    # when we move this into pre-processing.
    # pre_processed.global_workflow_emissions_raw_names = pix.concat(
    #     [
    #         pre_processed.global_workflow_emissions_raw_names.loc[
    #             ~pix.isin(variable="Emissions|CO2|AFOLU")
    #         ],
    #         input_df.loc[pix.isin(variable="Emissions|CO2|AFOLU", region="World")],
    #     ]
    # )
    if pre_processed.empty:
        raise AssertionError

    # TODO: explicit test of pre-processed results

    pytest.importorskip("aneris")
    harmoniser = create_cmip7_scenariomip_global_harmoniser(
        cmip7_scenariomip_global_historical_emissions_file=CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE,
        aneris_global_overrides_file=PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR
        / "aneris-overrides-global.csv",
        n_processes=None,  # not parallel
        progress=False,
    )
    harmonised = harmoniser(pre_processed)

    # Get expected result
    harmonised_model_scenario_all = get_cmip7_scenariomip_harmonised_emissions(
        model=model,
        scenario=scenario,
        # TODO: check if we can get rid of the "whole_pipeline"
        # folder and just use the regression results from elsewhere
        # (answer should be yes...)
        processed_cmip7_scenariomip_output_data_dir=CMIP7_SCENARIOMIP_OUT_DIR
        / "whole_pipeline",
    )

    exp_harmonised = harmonised_model_scenario_all.loc[
        pix.ismatch(workflow="global")
    ].reset_index("workflow", drop=True)
    if exp_harmonised.empty:
        raise AssertionError

    # Convert names to gcages naming before comparing
    exp_harmonised = update_index_levels_func(
        exp_harmonised,
        {
            "variable": lambda x: convert_variable_name(
                x,
                from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                to_convention=SupportedNamingConventions.GCAGES,
            )
        },
        copy=False,
    )

    assert_frame_equal(harmonised, exp_harmonised)

    infiller = CMIP7ScenarioMIPInfiller.from_cmip7_scenariomip_config(
        cmip7_scenariomip_infilling_leader_emissions_file=PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR
        / "infilling_db_cmip7_scenariomip.csv",
        cmip7_ghg_inversions_file=PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR
        / "cmip7_ghg_inversions.csv",
        cmip7_scenariomip_global_historical_emissions_file=PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR
        / "history_cmip7_scenariomip.csv",
    )
    infilled = infiller(harmonised)

    complete_model_scenario = get_cmip7_scenariomip_complete_emissions(
        model=model,
        scenario=scenario,
        processed_cmip7_scenariomip_output_data_dir=CMIP7_SCENARIOMIP_OUT_DIR
        / "whole_pipeline",
    )
    exp_infilled = update_index_levels_func(
        complete_model_scenario,
        {
            "variable": lambda x: convert_variable_name(
                x,
                from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                to_convention=SupportedNamingConventions.GCAGES,
            )
        },
        copy=False,
    )

    assert_frame_equal(infilled, exp_infilled)

    scm_runner = CMIP7ScenarioMIPSCMRunner.from_cmip7_scenariomip_config(
        magicc_exe_path=guess_magicc_exe(CMIP7_SCENARIOMIP_MAGICC_EXECUTABLES_DIR),
        magicc_prob_distribution_path=CMIP7_SCENARIOMIP_MAGICC_PROBABILISTIC_CONFIG_FILE,
        output_variables=("Surface Air Temperature Change",),
        historical_emissions_path=CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE,
        harmonisation_year=HARMONISATION_YEAR,
        n_processes=multiprocessing.cpu_count(),
    )

    scm_results = scm_runner(infilled)

    exp_temperature = load_timeseries_csv(
        # TODO: check if this differs from the 'non whole pipeline' result
        CMIP7_SCENARIOMIP_OUT_DIR / "whole_pipeline" / f"{model}_{scenario}_GSAT.csv",
        lower_column_names=True,
        index_columns=[
            "climate_model",
            "model",
            "region",
            "run_id",
            "scenario",
            "unit",
            "variable",
        ],
        out_columns_type=int,
        out_columns_name="time",
    )

    assert_frame_equal(
        mi_loc(
            scm_results,
            exp_temperature.index.droplevel(
                exp_temperature.index.names.difference(["variable", "run_id"])
            ),
        ),
        exp_temperature,
        rtol=1e-5,
    )

    post_processor = create_cmip7_scenariomip_postprocessor(
        progress=False,
        n_processes=None,
    )
    post_processed = post_processor(scm_results)

    exp_quantiles = load_timeseries_csv(
        (
            CMIP7_SCENARIOMIP_OUT_DIR
            / "whole_pipeline"
            / f"assessed-warming-timeseries-quantiles_{model}.csv"
        ),
        lower_column_names=True,
        index_columns=[
            "climate_model",
            "model",
            "region",
            "scenario",
            "unit",
            "variable",
            "quantile",
        ],
        out_columns_type=int,
        out_columns_name="time",
    )

    exp_quantiles = update_index_levels_func(
        exp_quantiles, {"quantile": partial(np.round, decimals=4)}
    )
    processed_quantiles = update_index_levels_func(
        post_processed.timeseries_quantile, {"quantile": partial(np.round, decimals=4)}
    )
    assert_frame_equal(
        processed_quantiles.loc[:, exp_quantiles.columns], exp_quantiles, rtol=1e-5
    )

    exp_categories = pd.read_csv(
        CMIP7_SCENARIOMIP_OUT_DIR / "whole_pipeline" / f"categories_{model}.csv",
        index_col=["climate_model", "model", "scenario"],
    )
    exp_categories.columns.name = "metric"

    assert_frame_equal(
        post_processed.metadata_categories.unstack("metric"), exp_categories
    )
