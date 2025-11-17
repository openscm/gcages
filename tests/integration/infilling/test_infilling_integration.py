"""
Integration tests of `gcages.infilling`
"""

import pandas as pd
import pytest

# Unit tests to write:
# - just basic infilling
# - error if try to infill something that isn't there
# - error if one scenario is missing required lead gas
# - works for multiple scenarios with different missing variables
# - config passed correctly (infill with silicone in test,
#   make sure same result comes through via config)
# - regional infilling
# - unit handling


def test_regional_infilling_silicone(setup_pandas_accessors):
    """
    Full integration test of infilling

    This tries to test infilling including as many edge cases as possible.
    For complete testing of all the paths and error handling, see the unit tests.

    Edge cases we try to cover:

    - unit conversion
    - different source timeseries for infilling different target timeseries
    - infilling at the regional level
    """
    openscm_units = pytest.importorskip("openscm_units")

    scen_1 = pd.DataFrame(
        [
            [1.0, 2.0, 3.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 2.0],
            [100.0, 220.0, 300.0],
            [85.0, 200.0, 270.0],
            [15.0, 20.0, 30.0],
            [10.0, 20.0, 32.0],
        ],
        columns=[2020, 2040, 2050],
        index=pd.MultiIndex.from_tuples(
            [
                ("s1", "CO2", "World", "GtC / yr"),
                ("s1", "CO2", "R1", "GtC / yr"),
                ("s1", "CO2", "R2", "GtC / yr"),
                ("s1", "CH4", "World", "MtCH4 / yr"),
                ("s1", "CH4", "R1", "MtCH4 / yr"),
                ("s1", "CH4", "R2", "MtCH4 / yr"),
                ("s1", "Sulfur", "World", "MtS / yr"),
            ],
            names=["scenario", "species", "region", "unit"],
        ),
    )

    scen_2 = pd.DataFrame(
        [
            [1.0, 1.2, 1.5],
            [1.0, 1.0, 1.0],
            [0.0, 0.2, 0.5],
            [100.0, 200.0, 350.0],
            [85.0, 200.0, 270.0],
            [15.0, 0.0, 80.0],
            [10.0, 23.0, 35.0],
        ],
        columns=[2020, 2040, 2050],
        index=pd.MultiIndex.from_tuples(
            [
                ("s2", "CO2", "World", "GtC / yr"),
                ("s2", "CO2", "R1", "GtC / yr"),
                ("s2", "CO2", "R2", "GtC / yr"),
                ("s2", "CH4", "World", "MtCH4 / yr"),
                ("s2", "CH4", "R1", "MtCH4 / yr"),
                ("s2", "CH4", "R2", "MtCH4 / yr"),
                ("s2", "Sulfur", "World", "MtS / yr"),
            ],
            names=["scenario", "species", "region", "unit"],
        ),
    ).openscm.convert_unit(
        {"MtCH4 / yr": "GtCH4 / yr", "MtS / yr": "MtSO2 / yr"},
        ur=openscm_units.unit_registry,
    )

    scen_3 = pd.DataFrame(
        [
            [1.0, 0.2, -1.5],
            [1.0, 0.5, -1.0],
            [0.0, -0.3, -0.5],
            [10.0, 10.0, 5.0],
        ],
        columns=[2020, 2040, 2050],
        index=pd.MultiIndex.from_tuples(
            [
                ("s3", "CO2", "World", "GtC / yr"),
                ("s3", "CO2", "R1", "GtC / yr"),
                ("s3", "CO2", "R2", "GtC / yr"),
                ("s3", "Sulfur", "World", "MtS / yr"),
            ],
            names=["scenario", "species", "region", "unit"],
        ),
    ).openscm.convert_unit(
        pd.Series(
            ["MtCO2 / yr"],
            index=pd.MultiIndex.from_tuples(
                [("s3", "CO2", "R2")], names=["scenario", "species", "region"]
            ),
        ),
        ur=openscm_units.unit_registry,
    )

    db = pd.concat([scen_1, scen_2, scen_3])

    def get_db():
        return db

    assert False, "Up to here"
    infill(scen_input)

    # Check that expected timeseries are all in output

    # Check output values against doing it vs. siliecone by hand
    # Put all the difficult edge cases in here, except errors
    # Have specific tests for the rest
