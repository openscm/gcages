"""
Test infilling compared for CMIP7 ScenarioMIP
"""

from __future__ import annotations

import importlib
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas_openscm.index_manipulation import (
    set_index_levels_func,
    update_index_levels_func,
    update_levels_from_other,
)
from pandas_openscm.io import load_timeseries_csv

from gcages.cmip7_scenariomip.harmonisation import (
    create_cmip7_scenariomip_global_harmoniser,
)
from gcages.cmip7_scenariomip.infilling import (
    CMIP7ScenarioMIPInfiller,
)
from gcages.cmip7_scenariomip.post_processing import CMIP7ScenarioMIPPostProcessor
from gcages.cmip7_scenariomip.pre_processing import CMIP7ScenarioMIPPreProcessor
from gcages.cmip7_scenariomip.pre_processing.reaggregation import ReaggregatorBasic
from gcages.cmip7_scenariomip.pre_processing.reaggregation.basic import (
    get_required_timeseries_index,
)
from gcages.cmip7_scenariomip.scm_running import (
    CMIP7ScenarioMIPSCMRunner,
)
from gcages.completeness import get_missing_levels
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
def test_whole_pipeline(model, scenario, monkeypatch):  # noqa: PLR0915
    """Test a few scenarios, not all to save compute time"""
    # LOADING SCENARIO
    file = CMIP7_SCENARIOMIP_OUT_DIR / f"{model}_{scenario}_raw-scenario.csv"
    input_df = load_timeseries_csv(
        file,
        index_columns=["model", "scenario", "variable", "region", "unit"],
        out_columns_type=int,
        out_columns_name="year",
    )

    # In case the new data needs a bit of make-up
    input_df = input_df.loc[:, 2015:2100:1].dropna(how="all", axis="columns")
    input_df = input_df.T.interpolate(method="index").T

    model_regions = [
        r
        for r in input_df.index.get_level_values("region").unique()
        if r.startswith(model)
    ]

    reaggregator = ReaggregatorBasic(model_regions=model_regions)

    input_df = missing_reporting_zero_hack(reaggregator, input_df, model_regions)

    if importlib.util.find_spec("openscm_units") is None:
        # Loosen the tolerance given what we know about the units
        reaggregator.internal_consistency_tolerances["Emissions|CO2"]["atol"] = 1.0

    pre_processor = CMIP7ScenarioMIPPreProcessor(
        reaggregator=reaggregator,
        n_processes=None,  # run serially
        progress=False,
        run_checks=True,
    )
    pre_processed = pre_processor(input_df)

    # TODO should we move this ?
    # Hard override the global workflow emissions for CO2 AFOLU
    # to use globally reported numbers,
    # even if they're not consistent with region-sector reporting.
    pre_processed.global_workflow_emissions = pix.concat(
        [
            pre_processed.global_workflow_emissions.loc[
                ~pix.isin(variable="Emissions|CO2|Biosphere")
            ],
            input_df.loc[
                pix.isin(variable="Emissions|CO2|AFOLU", region="World")
            ].pix.assign(variable="Emissions|CO2|Biosphere"),
        ]
    )

    pre_processed.global_workflow_emissions_raw_names = pix.concat(
        [
            pre_processed.global_workflow_emissions_raw_names.loc[
                ~pix.isin(variable="Emissions|CO2|AFOLU")
            ],
            input_df.loc[pix.isin(variable="Emissions|CO2|AFOLU", region="World")],
        ]
    )

    # for attr in [
    #     "assumed_zero_emissions",
    #     "global_workflow_emissions",
    #     "global_workflow_emissions_raw_names",
    #     "gridding_workflow_emissions",
    # ]:
    #     # Interestingly, this won't fail if there are extra, unexpected columns
    #     # in the regression data against which we are comparing.
    #     dataframe_regression.check(
    #         getattr(pre_processed, attr).sort_index(),
    #                   basename=f"{input_file.stem}_{attr}"
    #     )

    # HARMONISATION
    pre_processed = pre_processed.global_workflow_emissions[
        pix.ismatch(
            region="World",
        )
    ]
    if pre_processed.empty:
        raise AssertionError

    # Harmonise
    # Only works if aneris installed
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
    harmonised_all = get_cmip7_scenariomip_harmonised_emissions(
        model=model,
        scenario=scenario,
        processed_cmip7_scenariomip_output_data_dir=CMIP7_SCENARIOMIP_OUT_DIR
        / "whole_pipeline",
    )

    exp = harmonised_all.loc[pix.ismatch(workflow="global")].reset_index(
        "workflow", drop=True
    )
    if exp.empty:
        raise AssertionError

    # Convert names to gcages naming before comparing
    exp = update_index_levels_func(
        exp,
        {
            "variable": lambda x: convert_variable_name(
                x,
                from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                to_convention=SupportedNamingConventions.GCAGES,
            )
        },
        copy=False,
    )
    exp.columns.name = "year"

    assert_frame_equal(harmonised, exp)

    ## INFILLING
    # Load infilled results
    exp = get_cmip7_scenariomip_complete_emissions(
        model=model,
        scenario=scenario,
        processed_cmip7_scenariomip_output_data_dir=CMIP7_SCENARIOMIP_OUT_DIR
        / "whole_pipeline",
    )
    exp = update_index_levels_func(
        exp,
        {
            "variable": lambda x: convert_variable_name(
                x,
                from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                to_convention=SupportedNamingConventions.GCAGES,
            )
        },
        copy=False,
    )
    exp.columns.name = "year"

    infiller = CMIP7ScenarioMIPInfiller.from_cmip7_scenariomip_config(
        cmip7_scenariomip_infilling_leader_emissions_file=PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR
        / "infilling_db_cmip7_scenariomip.csv",
        cmip7_ghg_inversions_file=PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR
        / "cmip7_ghg_inversions.csv",
        cmip7_scenariomip_global_historical_emissions_file=PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR
        / "history_cmip7_scenariomip.csv",
        ur=None,
    )
    infilled = infiller(harmonised)

    assert_frame_equal(infilled, exp)

    # MAGICC and post_processing
    # Select scenario and drop aggregated/cumulative rows
    infilled = infilled.loc[
        pix.ismatch(scenario=scenario)
        & ~pix.ismatch(variable=["**Kyoto**", "Cumulative**", "**CO2", "**GHG**"])
    ]

    # Loading expected results
    file = CMIP7_SCENARIOMIP_OUT_DIR / "whole_pipeline" / f"{model}_{scenario}_GSAT.csv"
    exp_temperature = load_timeseries_csv(
        file,
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

    scm_runner = CMIP7ScenarioMIPSCMRunner.from_cmip7_scenariomip_config(
        magicc_exe_path=guess_magicc_exe(CMIP7_SCENARIOMIP_MAGICC_EXECUTABLES_DIR),
        magicc_prob_distribution_path=CMIP7_SCENARIOMIP_MAGICC_PROBABILISTIC_CONFIG_FILE,
        output_variables=("Surface Air Temperature Change",),
        historical_emissions_path=CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE,
        harmonisation_year=HARMONISATION_YEAR,
        n_processes=multiprocessing.cpu_count(),
    )

    scm_results = scm_runner(infilled)

    assert_frame_equal(
        scm_results[
            scm_results.index.get_level_values("variable").str.contains(
                "Surface Air Temperature Change"
            )
        ].iloc[:10],
        exp_temperature,
        rtol=1e-6,
    )

    post_processor = CMIP7ScenarioMIPPostProcessor.from_cmip7_scenariomip_config()
    post_processed = post_processor(scm_results)

    # Loading and assessing quantiles timeseries results
    file = (
        CMIP7_SCENARIOMIP_OUT_DIR
        / "whole_pipeline"
        / f"assessed-warming-timeseries-quantiles_{model}.csv"
    )
    exp_quantiles = load_timeseries_csv(
        file,
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
    exp_quantiles.index = exp_quantiles.index.set_levels(
        exp_quantiles.index.levels[exp_quantiles.index.names.index("quantile")].round(
            4
        ),
        level="quantile",
    )
    processed_quantiles = post_processed.timeseries_quantile.iloc[:, 250:]
    processed_quantiles.index = processed_quantiles.index.set_levels(
        exp_quantiles.index.levels[exp_quantiles.index.names.index("quantile")].round(
            4
        ),
        level="quantile",
    )

    assert_frame_equal(
        processed_quantiles,
        exp_quantiles,
        rtol=1e-8,
    )

    # Loading and categories
    file = CMIP7_SCENARIOMIP_OUT_DIR / "whole_pipeline" / f"categories_{model}.csv"
    exp_categories = pd.read_csv(file)

    assert post_processed.metadata_categories.values[0] == exp_categories["value.1"][2]
    assert post_processed.metadata_categories.values[1] == exp_categories["value"][2]
