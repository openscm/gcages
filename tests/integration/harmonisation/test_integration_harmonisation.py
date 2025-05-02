"""
Tests of `gcages.harmonisation`
"""

import pandas as pd
import pytest

from gcages.harmonisation import AnerisHarmoniser, assert_harmonised


@pytest.mark.parametrize("harmonisation_year", (2020.0, 2030.0))
def test_basic(harmonisation_year):
    # Make sure things run without exploding
    historical_emissions = pd.DataFrame(
        [
            [1.0, 1.2, 1.5],
            [2.0, 1.2, 0.5],
        ],
        columns=[2010.0, 2020.0, 2030.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("v1", "r1", "GtCO2 / yr"),
                ("v2", "r1", "MtCH4 / yr"),
            ],
            names=["variable", "region", "unit"],
        ),
    )

    harmoniser = AnerisHarmoniser(
        historical_emissions=historical_emissions,
        harmonisation_year=harmonisation_year,
        progress=False,
        n_processes=None,
    )
    scenario_emissions = pd.DataFrame(
        [
            [1.3, 1.7, 1.0, 1.0],
            [1.0, 0.6, 0.5, 0.3],
        ],
        columns=[2020.0, 2030.0, 2050.0, 2100.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("s1", "m1", "v1", "r1", "GtCO2 / yr"),
                ("s1", "m1", "v2", "r1", "MtCH4 / yr"),
            ],
            names=["scenario", "model", "variable", "region", "unit"],
        ),
    )

    harmonised = harmoniser(scenario_emissions)

    assert_harmonised(
        harmonised,
        history=historical_emissions,
        harmonisation_time=harmonisation_year,
    )

    # Only data from the harmonisation year onwards is returned
    assert harmonised.columns[0] == harmonisation_year


# Tests to write:
# - check of harmonisation year and overrides in basic set up
# - check of scenario group levels (two scenarios,
#   one harmonised using reduce_ratio_2060 and one with reduce_ratio_2080)
# - error if historical is missing a timeseries that's needed
# - no error if historical has extra timeseries beyond what's needed
# - error if historical or scenarios is missing harmonisation year
# - validation of aneris overrides
# - validation of historical emissions
