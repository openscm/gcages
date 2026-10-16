"""
Test suite for OSCAR SCM
"""

from pathlib import Path

import pandas as pd
import pytest
from pandas_openscm.index_manipulation import update_index_levels_func

from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.scm_running.oscar import OSCARSCMRunner
from gcages.testing import get_ar6_infilled_emissions
from gcages.units_helpers import strip_pint_incompatible_characters_from_unit_string

CONFIG_DIR = Path(__file__).parents[0] / "configs"
PROCESSED_AR6_DB_DIR = (
    CONFIG_DIR.parents[2] / "regression" / "ar6" / "ar6-output-processed"
)


def strip_off_ar6_infilled_prefix_and_convert_to_gcages_and_fix_units(
    indf: pd.DataFrame,
) -> pd.DataFrame:
    indf = update_index_levels_func(
        indf,
        {
            "variable": lambda x: convert_variable_name(
                x.replace("AR6 climate diagnostics|Infilled|", ""),
                from_convention=SupportedNamingConventions.AR6_WG3,
                to_convention=SupportedNamingConventions.GCAGES,
            ),
            "unit": lambda x: strip_pint_incompatible_characters_from_unit_string(
                x
            ).replace("HFC245ca", "HFC245fa"),
        },
        copy=False,
    )

    return indf


def test_run_oscar(dataframe_regression):
    pytest.importorskip("oscar")

    complete = get_ar6_infilled_emissions(
        model="GCAM_5.3",
        scenario="NGFS2_Current_Policies",
        processed_ar6_output_data_dir=PROCESSED_AR6_DB_DIR,
    )
    # Select scenario and drop aggregated/cumulative rows
    is_aggregate = complete.index.get_level_values("variable").str.endswith(
        ("CO2", "F-Gases", "HFC", "PFC")
    ) | complete.index.get_level_values("variable").str.contains("Kyoto", regex=False)

    complete = complete[~is_aggregate]
    complete = strip_off_ar6_infilled_prefix_and_convert_to_gcages_and_fix_units(
        complete
    )

    runner = OSCARSCMRunner(
        output_variables=("Surface Air Temperature Change",),
    )
    res = runner(complete)

    # RESULTS seem to fluctuate sensibly between runs
    dataframe_regression.check(res, default_tolerance=dict(rtol=1e-2))
