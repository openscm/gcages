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
            "Fail1",
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
            "Fail2",
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

    if status == "Fail1":
        # Fail1: two rows share the SAME index (Emissions|BC appears twice with
        # different values). The duplicate makes the scenario's row index non-unique,
        # so the concat(axis=1) inside get_complete_scenarios_for_magicc can't reindex
        # it so pandas raises InvalidIndexError. We assert that failure is raised.
        with pytest.raises(pd.errors.InvalidIndexError, match="uniquely valued Index"):
            get_complete_scenarios_for_magicc(scenario, history, 2015)
    elif status == "Fail2":
        # Fail2: all indices are unique, but Emissions|CO and Emissions|BC have
        # the same values (1.0, 2.0). The function builds its history-lookup key with
        # drop_duplicates(), which compares values and not the index, so CO is
        # collapsed into BC and never gets its historical prefix looked up.
        # We get a Nan in the mix.
        scenario_magicc = get_complete_scenarios_for_magicc(scenario, history, 2012)
        assert scenario_magicc.isnull().any().any()
