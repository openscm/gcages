"""
Tests of `gcages.cmip7_scenariomip.scm_running.get_complete_scenarios_for_magicc
"""

import pandas as pd
import pytest

from gcages.cmip7_scenariomip.scm_running import get_complete_scenarios_for_magicc


@pytest.mark.parametrize(
    "scenario,status",
    [
        pytest.param(
            pd.DataFrame(
                [
                    [1.0, 2.0],
                    [3.0, 2.0],
                    [1.0, 2.0],
                    [3.0, 2.0],
                ],
                columns=[2015, 2100],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("model_1", "scenario_1", "World", "Emissions|BC", "Mt BC/yr"),
                        (
                            "model_1",
                            "scenario_1",
                            "World",
                            "Emissions|CO2",
                            "Mt CO2/yr",
                        ),
                        ("model_1", "scenario_1", "World", "Emissions|CO", "Mt CO/yr"),
                        ("model_1", "scenario_1", "World", "Emissions|BC", "Mt BC/yr"),
                    ],
                    names=["model", "scenario", "region", "variable", "unit"],
                ),
            ),
            "Fail",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    [1.0, 2.0],
                    [3.0, 2.0],
                    [1.0, 2.0],
                ],
                columns=[2015, 2100],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("model_1", "scenario_1", "World", "Emissions|BC", "Mt BC/yr"),
                        (
                            "model_1",
                            "scenario_1",
                            "World",
                            "Emissions|CO2",
                            "Mt CO2/yr",
                        ),
                        ("model_1", "scenario_1", "World", "Emissions|CO", "Mt CO/yr"),
                    ],
                    names=["model", "scenario", "region", "variable", "unit"],
                ),
            ),
            "Pass",
        ),
    ],
)
def test_assert_get_complete_scenarios_for_magicc(scenario, status):

    history = scenario[[2015]].copy()
    history[2012] = history[2015] * 0.8
    history[2010] = history[2015] * 0.7
    history = history.sort_index(axis=1)
    history = history[~history.index.duplicated(keep="first")].reset_index(
        ["model", "scenario"], drop=True
    )

    if status == "Fail":
        # Fail: two rows share the SAME index (Emissions|BC appears twice with
        # different values).
        with pytest.raises(
            ValueError,
            match="'scenarios' has duplicate index: model, scenario, variable",
        ):
            get_complete_scenarios_for_magicc(scenario, history, 2015)
    elif status == "Pass":
        scenario_magicc = get_complete_scenarios_for_magicc(scenario, history, 2012)
        assert not scenario_magicc.isnull().any().any()
