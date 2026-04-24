"""
Tests of the `gcages.cmip7_scenariomip.scm_running`
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gcages.assertions import MissingDataForTimesError
from gcages.cmip7_scenariomip.scm_running import (
    CMIP7ScenarioMIPSCMRunner,
    get_complete_scenarios_for_magicc,
)
from gcages.testing import guess_magicc_exe

PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR = (
    Path(__file__).parents[2]
    / "regression/cmip7-scenariomip/cmip7-scenariomip-workflow-inputs"
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

# Only works if openscm-runner installed
pytest.importorskip("openscm_runner.adapters")


@pytest.mark.skip_ci_default
@pytest.mark.slow
def test_get_complete_scenarios_for_magicc_adds_history_and_keeps_scenarios():
    scenario = pd.DataFrame(
        {
            2015: [10.0, 10.0],
            2016: [12.0, 14.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("M1", "S1", "World", "Emissions|CO2", "MtCO2/yr"),
                ("M1", "S1", "World", "Emissions|CH4", "MtCH4/yr"),
            ],
            names=["model", "scenario", "region", "variable", "unit"],
        ),
    )

    history = pd.DataFrame(
        {
            2013: [6.0, 6.0],
            2014: [8.0, 8.0],
            2015: [10.0, 10.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("M1", "S1", "World", "Emissions|CO2", "MtCO2/yr"),
                ("M1", "S1", "World", "Emissions|CH4", "MtCH4/yr"),
            ],
            names=["model", "scenario", "region", "variable", "unit"],
        ),
    )

    out = get_complete_scenarios_for_magicc(scenario, history, magicc_start_year=2014)

    # columns: 2014, 2015, 2016
    assert list(out.columns) == [2014, 2015, 2016]
    assert out.loc[("M1", "S1", "World", "Emissions|CO2", "MtCO2/yr"), 2014] == 8.0
    assert out.loc[("M1", "S1", "World", "Emissions|CH4", "MtCH4/yr"), 2014] == 8.0
    assert out.loc[("M1", "S1", "World", "Emissions|CO2", "MtCO2/yr"), 2015] == 10.0
    assert out.loc[("M1", "S1", "World", "Emissions|CH4", "MtCH4/yr"), 2016] == 14.0

    scm_runner_cfg = CMIP7ScenarioMIPSCMRunner.from_cmip7_scenariomip_config(
        magicc_exe_path=guess_magicc_exe(CMIP7_SCENARIOMIP_MAGICC_EXECUTABLES_DIR),
        magicc_prob_distribution_path=CMIP7_SCENARIOMIP_MAGICC_PROBABILISTIC_CONFIG_FILE,
        output_variables=("Surface Air Temperature Change",),
        run_checks=False,
    )
    scm_runner = CMIP7ScenarioMIPSCMRunner(
        climate_models_cfgs=scm_runner_cfg.climate_models_cfgs,
        output_variables=("Surface Air Temperature Change",),
        historical_emissions=history.reset_index(
            level=[lvl for lvl in ["model", "scenario"] if lvl in history.index.names],
            drop=True,
        ),
        harmonisation_year=2015,
        run_checks=True,
        res_column_type=int,
    )

    error_message = re.escape(
        '"in_emissions is missing data for the following times: [2100]. '
        "Available times: Index([2015, 2016], dtype='int64')"
    )
    with pytest.raises(MissingDataForTimesError, match=error_message):
        scm_runner.__call__(scenario)


def test_get_complete_scenarios_for_magicc_interpolates_missing_years():
    scenarios = pd.DataFrame(
        {
            2015: [10.0, 10.0],
            2016: [np.nan, np.nan],
            2017: [12.0, 14.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("M1", "S1", "CO2", "MtCO2/yr"),
                ("M1", "S1", "CH4", "MtCH4/yr"),
            ],
            names=["model", "scenario", "variable", "unit"],
        ),
    )

    history = pd.DataFrame(
        {
            2012: [6.0, 6.0],
            2013: [np.nan, np.nan],
            2014: [8.0, 8.0],
            2015: [10.0, 10.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("M1", "S1", "CO2", "MtCO2/yr"),
                ("M1", "S1", "CH4", "MtCH4/yr"),
            ],
            names=["model", "scenario", "variable", "unit"],
        ),
    )
    out = get_complete_scenarios_for_magicc(scenarios, history, magicc_start_year=2015)

    assert list(out.columns) == [2015, 2016, 2017]
    assert out.loc[("M1", "S1", "CO2", "MtCO2/yr"), 2016] == pytest.approx(11.0)
    assert out.loc[("M1", "S1", "CH4", "MtCH4/yr"), 2016] == 12.0


@pytest.mark.skip_ci_default
@pytest.mark.magicc_v760a3
@pytest.mark.parametrize(
    "scenario, history_path,run_checks, harmonisation_year,error_message",
    [
        (
            pd.DataFrame(
                {2020 + i: [12.0 + i / 10, 14.0 + i / 10] for i in range(10)},
                index=pd.MultiIndex.from_tuples(
                    [
                        ("M1", "S1", "CO2", "MtCO2/yr"),
                        ("M1", "S1", "CH4", "MtCH4/yr"),
                    ],
                    names=["model", "scenario", "variable", "unit"],
                ),
            ),
            None,
            True,
            2023,
            "`self.historical_emissions` must be set to check the infilling",
        ),
        (
            pd.DataFrame(
                {2020 + i: [12.0 + i / 10, 14.0 + i / 10] for i in range(10)},
                index=pd.MultiIndex.from_tuples(
                    [
                        ("M1", "S1", "CO2", "MtCO2/yr"),
                        ("M1", "S1", "CH4", "MtCH4/yr"),
                    ],
                    names=["model", "scenario", "variable", "unit"],
                ),
            ),
            CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE,
            True,
            None,
            "`self.harmonisation_year` must be set to check the infilling",
        ),
        (
            pd.DataFrame(
                {2020 + i: [12.0 + i / 10, 14.0 + i / 10] for i in range(100)},
                index=pd.MultiIndex.from_tuples(
                    [
                        ("M1", "S1", "Emissions|CO2", "MtCO2/yr"),
                        ("M1", "S1", "Emissions|CH4", "MtCH4/yr"),
                    ],
                    names=["model", "scenario", "variable", "unit"],
                ),
            ),
            None,
            False,
            2023,
            "Emissions starting year must be set to `2015`",
        ),
    ],
)
def test_cmip7_scenariomip_scmrunner(  # noqa: PLR0913
    scenario, history_path, run_checks, harmonisation_year, error_message, monkeypatch
):
    monkeypatch.delenv("MAGICC_EXECUTABLE_7", raising=False)
    scm_runner = CMIP7ScenarioMIPSCMRunner.from_cmip7_scenariomip_config(
        magicc_exe_path=guess_magicc_exe(CMIP7_SCENARIOMIP_MAGICC_EXECUTABLES_DIR),
        magicc_prob_distribution_path=CMIP7_SCENARIOMIP_MAGICC_PROBABILISTIC_CONFIG_FILE,
        output_variables=("Surface Air Temperature Change",),
        historical_emissions_path=history_path,
        harmonisation_year=harmonisation_year,
        run_checks=run_checks,
    )

    with pytest.raises(AssertionError, match=error_message):
        scm_runner(scenario)
