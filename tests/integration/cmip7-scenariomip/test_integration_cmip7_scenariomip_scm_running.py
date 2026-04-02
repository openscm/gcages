from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gcages.cmip7_scenariomip.scm_running import (
    CMIP7ScenarioMIPSCMRunner,
    check_cmip7_scenariomip_magicc7_version,
    get_complete_scenarios_for_magicc,
    load_magicc_cfgs,
)

PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR = (
    Path(__file__).parents[2]
    / "regression/cmip7-scenariomip/cmip7-scenariomip-workflow-inputs"
)
CMIP7_SCENARIOMIP_HISTORICAL_GLOBAL_EMISSIONS_FILE = (
    PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR / "history_cmip7_scenariomip.csv"
)

MAGIC_EXE = PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR / "magicc-v7.6.0a3/bin/magicc"
MAGICC_CMIP7_PROBABILISTIC_CONFIG_FILE = (
    PROCESSED_CMIP7_SCENARIOMIP_INPUT_DIR
    / "magicc-v7.6.0a3/configs/magicc-ar7-fast-track-drawnset-v0-3-0.json"
)

# Only works if pandas_indexing installed
pytest.importorskip("pandas_indexing")
# Only works if openscm-runner installed
pytest.importorskip("openscm_runner.adapters")


def test_load_magicc_cfgs_sets_common_and_physical_cfgs(tmp_path: Path):
    prob = tmp_path / "prob.json"
    prob.write_text(
        """{
          "configurations": [
            {
              "paraset_id": "cfg-1",
              "nml_allcfgs": {"SCENARIO": "foo", "STARTYEAR": 1750}
            }
          ]
        }""",
        encoding="utf-8",  # Optional, good practice
    )

    out = load_magicc_cfgs(
        prob, output_variables=("Surface Air Temperature Change",), startyear=1750
    )

    assert list(out) == ["MAGICC7"]
    assert len(out["MAGICC7"]) == 1
    cfg = out["MAGICC7"][0]
    assert cfg["run_id"] == "cfg-1"
    assert cfg["scenario"] == "foo"
    assert cfg["startyear"] == 1750
    assert cfg["out_ascii_binary"] == "BINARY"
    assert cfg["out_binary_format"] == 2
    assert cfg["out_dynamic_vars"]


def test_get_complete_scenarios_for_magicc_adds_history_and_keeps_scenarios():
    scenarios = pd.DataFrame(
        {
            2015: [10.0, 10.0],
            2016: [12.0, 14.0],
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
            2013: [6.0, 6.0],
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

    out = get_complete_scenarios_for_magicc(scenarios, history, magicc_start_year=2014)

    # columns: 2014, 2015, 2016
    assert list(out.columns) == [2014, 2015, 2016]
    assert out.loc[("M1", "S1", "CO2", "MtCO2/yr"), 2014] == 8.0
    assert out.loc[("M1", "S1", "CH4", "MtCH4/yr"), 2014] == 8.0
    assert out.loc[("M1", "S1", "CO2", "MtCO2/yr"), 2015] == 10.0
    assert out.loc[("M1", "S1", "CH4", "MtCH4/yr"), 2016] == 14.0


def test_call_success_non_magicc(monkeypatch):
    # Setup a runner for a generic model to skip MAGICC-specific interpolation
    runner = CMIP7ScenarioMIPSCMRunner(
        climate_models_cfgs={"GenericModel": [{"run_id": "test"}]},
        output_variables=("Surface Air Temperature Change",),
        run_checks=False,
    )

    mock_out = pd.DataFrame(
        [[1.0]],
        index=pd.MultiIndex.from_tuples(
            [
                ("M1", "S1", "Emissions|CO2", "MtCO2/yr"),
            ],
            names=["model", "scenario", "variable", "unit"],
        ),
        columns=[2020],
    )
    monkeypatch.setattr(
        "gcages.cmip7_scenariomip.scm_running.run_scms", lambda **kwargs: mock_out
    )

    in_emissions = mock_out.copy()  # Simplest valid input
    result = runner(in_emissions)

    assert not result.empty
    assert result.columns.dtype == int


def test_call_with_database(monkeypatch):
    from unittest.mock import MagicMock

    mock_db = MagicMock()
    mock_data = pd.DataFrame(
        [[1.0]],
        index=pd.MultiIndex.from_tuples(
            [
                ("M1", "S1", "Emissions|CO2", "MtCO2/yr"),
            ],
            names=["model", "scenario", "variable", "unit"],
        ),
        columns=[2020],
    )
    mock_db.load.return_value = mock_data

    runner = CMIP7ScenarioMIPSCMRunner(
        climate_models_cfgs={"Model": []},
        output_variables=(),
        db=mock_db,
        run_checks=False,
    )

    # Mock run_scms to return None (simulating output saved only to DB)
    monkeypatch.setattr(
        "gcages.cmip7_scenariomip.scm_running.run_scms", lambda **kwargs: None
    )

    result = runner(mock_data)
    mock_db.load.assert_called_once()
    assert result.equals(mock_data)


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
def test_CMIP7ScenarioMIPSCMRunner(  # noqa: PLR0913
    scenario, history_path, run_checks, harmonisation_year, error_message, monkeypatch
):
    monkeypatch.delenv("MAGICC_EXECUTABLE_7", raising=False)
    scm_runner = CMIP7ScenarioMIPSCMRunner.from_cmip7_scenariomip_config(
        magicc_exe_path=MAGIC_EXE,
        magicc_prob_distribution_path=MAGICC_CMIP7_PROBABILISTIC_CONFIG_FILE,
        output_variables=("Surface Air Temperature Change",),
        historical_emissions_path=history_path,
        harmonisation_year=harmonisation_year,
        run_checks=run_checks,
    )

    with pytest.raises(AssertionError, match=error_message):
        scm_runner.__call__(scenario)


def test_check_cmip7_scenariomip_magicc7_version(monkeypatch):
    import openscm_runner.adapters

    monkeypatch.setenv("MAGICC_EXECUTABLE_7", str(MAGIC_EXE))
    monkeypatch.setattr(
        openscm_runner.adapters.MAGICC7, "get_version", lambda: "v7.6.0a3"
    )
    check_cmip7_scenariomip_magicc7_version()

    # Force a version that isn't v7.6.0a3
    monkeypatch.setattr(
        openscm_runner.adapters.MAGICC7, "get_version", lambda: "v8.0.0"
    )

    with pytest.raises(AssertionError, match="v8.0.0"):
        check_cmip7_scenariomip_magicc7_version()
