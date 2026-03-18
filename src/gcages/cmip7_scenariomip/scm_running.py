"""
SCM-running configuration and related things for the updated workflow

This sets our defaults.
Individual notebooks can then override them as needed.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
from pandas_openscm.indexing import multi_index_lookup

from gcages.exceptions import MissingOptionalDependencyError
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.scm_running import (
    convert_openscm_runner_output_names_to_magicc_output_names,
)

SCM_OUTPUT_VARIABLES_DEFAULT: tuple[str, ...] = (
    # GSAT
    "Surface Air Temperature Change",
    # # GMST
    # "Surface Air Ocean Blended Temperature Change",
    # ERFs
    "Effective Radiative Forcing",
    "Effective Radiative Forcing|Anthropogenic",
    "Effective Radiative Forcing|Aerosols",
    "Effective Radiative Forcing|Aerosols|Direct Effect",
    "Effective Radiative Forcing|Aerosols|Direct Effect|BC",
    "Effective Radiative Forcing|Aerosols|Direct Effect|OC",
    "Effective Radiative Forcing|Aerosols|Direct Effect|SOx",
    "Effective Radiative Forcing|Aerosols|Indirect Effect",
    "Effective Radiative Forcing|Greenhouse Gases",
    "Effective Radiative Forcing|CO2",
    "Effective Radiative Forcing|CH4",
    "Effective Radiative Forcing|N2O",
    "Effective Radiative Forcing|F-Gases",
    "Effective Radiative Forcing|Montreal Protocol Halogen Gases",
    "Effective Radiative Forcing|Ozone",
    "Effective Radiative Forcing|Aviation|Cirrus",
    "Effective Radiative Forcing|Aviation|Contrail",
    "Effective Radiative Forcing|Aviation|H2O",
    "Effective Radiative Forcing|Black Carbon on Snow",
    # 'Effective Radiative Forcing|CH4 Oxidation Stratospheric',
    "CH4OXSTRATH2O_ERF",
    "Effective Radiative Forcing|Land-use Change",
    # "Effective Radiative Forcing|CFC11",
    # "Effective Radiative Forcing|CFC12",
    # "Effective Radiative Forcing|HCFC22",
    # "Effective Radiative Forcing|HFC125",
    # "Effective Radiative Forcing|HFC134a",
    # "Effective Radiative Forcing|HFC143a",
    # "Effective Radiative Forcing|HFC227ea",
    # "Effective Radiative Forcing|HFC23",
    # "Effective Radiative Forcing|HFC245fa",
    # "Effective Radiative Forcing|HFC32",
    # "Effective Radiative Forcing|HFC4310mee",
    # "Effective Radiative Forcing|CF4",
    # "Effective Radiative Forcing|C6F14",
    # "Effective Radiative Forcing|C2F6",
    # "Effective Radiative Forcing|SF6",
    # # Heat uptake
    # "Heat Uptake",
    # "Heat Uptake|Ocean",
    # Atmospheric concentrations
    "Atmospheric Concentrations|CO2",
    "Atmospheric Concentrations|CH4",
    "Atmospheric Concentrations|N2O",
    # # Carbon cycle
    # "Net Atmosphere to Land Flux|CO2",
    # "Net Atmosphere to Ocean Flux|CO2",
    # # permafrost
    # "Net Land to Atmosphere Flux|CO2|Earth System Feedbacks|Permafrost",
    # "Net Land to Atmosphere Flux|CH4|Earth System Feedbacks|Permafrost",
)
"""
Default variables to get from SCMs
"""


COMPLETE_EMISSIONS_INPUT_VARIABLES_GCAGES = [
    "Emissions|CO2|Biosphere",
    "Emissions|CO2|Fossil",
    "Emissions|BC",
    "Emissions|CH4",
    "Emissions|CO",
    "Emissions|N2O",
    "Emissions|NH3",
    "Emissions|NMVOC",
    "Emissions|NOx",
    "Emissions|OC",
    "Emissions|SOx",
    "Emissions|C2F6",
    "Emissions|C6F14",
    "Emissions|CF4",
    "Emissions|SF6",
    "Emissions|HFC125",
    "Emissions|HFC134a",
    "Emissions|HFC143a",
    "Emissions|HFC227ea",
    "Emissions|HFC23",
    "Emissions|HFC245fa",
    "Emissions|HFC32",
    "Emissions|HFC4310mee",
    "Emissions|CCl4",
    "Emissions|CFC11",
    "Emissions|CFC113",
    "Emissions|CFC114",
    "Emissions|CFC115",
    "Emissions|CFC12",
    "Emissions|CH3CCl3",
    "Emissions|HCFC141b",
    "Emissions|HCFC142b",
    "Emissions|HCFC22",
    "Emissions|Halon1202",
    "Emissions|Halon1211",
    "Emissions|Halon1301",
    "Emissions|Halon2402",
    "Emissions|C3F8",
    "Emissions|C4F10",
    "Emissions|C5F12",
    "Emissions|C7F16",
    "Emissions|C8F18",
    "Emissions|cC4F8",
    "Emissions|SO2F2",
    "Emissions|HFC236fa",
    "Emissions|HFC152a",
    "Emissions|HFC365mfc",
    "Emissions|CH2Cl2",
    "Emissions|CHCl3",
    "Emissions|CH3Br",
    "Emissions|CH3Cl",
    "Emissions|NF3",
]
"""
Complete set of input emissions using gcages' naming
"""

to_reporting_names = partial(
    convert_variable_name,
    from_convention=SupportedNamingConventions.GCAGES,
    to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
)

complete_index_gcages_names = pd.MultiIndex.from_product(
    [COMPLETE_EMISSIONS_INPUT_VARIABLES_GCAGES, ["World"]],
    names=["variable", "region"],
)
"""
Complete index using gcages' names
"""

complete_index_reporting_names = pd.MultiIndex.from_product(
    [
        [to_reporting_names(v) for v in COMPLETE_EMISSIONS_INPUT_VARIABLES_GCAGES],
        ["World"],
    ],
    names=["variable", "region"],
)
"""
Complete index using reporting names
"""


def load_magicc_cfgs(
    prob_distribution_path: Path,
    output_variables: tuple[str, ...] = SCM_OUTPUT_VARIABLES_DEFAULT,
    startyear: int = 1750,
) -> dict[str, list[dict[str, Any]]]:
    """
    Load MAGICC's configuration

    Parameters
    ----------
    prob_distribution_path
        Path to the file containing the probabilistic distribution

    output_variables
        Output variables

    startyear
        Starting year of the runs

    Returns
    -------
    :
        Config that can be used to run MAGICC
    """
    with open(prob_distribution_path) as fh:
        cfgs_raw = json.load(fh)

    cfgs_physical = [
        {
            "run_id": c["paraset_id"],
            **{k.lower(): v for k, v in c["nml_allcfgs"].items()},
        }
        for c in cfgs_raw["configurations"]
    ]

    common_cfg = {
        "startyear": startyear,
        # Note: endyear handled in gcages, which I don't love but is fine for now
        "out_dynamic_vars": convert_openscm_runner_output_names_to_magicc_output_names(
            output_variables
        ),
        "out_ascii_binary": "BINARY",
        "out_binary_format": 2,
    }

    run_config = [{**common_cfg, **physical_cfg} for physical_cfg in cfgs_physical]
    climate_models_cfgs = {"MAGICC7": run_config}

    return climate_models_cfgs


def get_complete_scenarios_for_magicc(
    scenarios: pd.DataFrame,
    history: pd.MultiIndex,
    magicc_start_year: int = 2015,
) -> pd.DataFrame:
    """
    Get complete scenarios for MAGICC

    Parameters
    ----------
    scenarios
        Scenarios

    history
        History

    magicc_start_year
        MAGICC's internal year in which it switches to scenario data

    Returns
    -------
    :
        Complete scenario to use with MAGICC
    """
    try:
        from pandas_indexing.core import concat as pix_concat
    except ImportError as exc:
        msg = "get_complete_scenarios_for_magicc"
        raise MissingOptionalDependencyError(
            msg, requirement="pandas_indexing"
        ) from exc

    scenarios_start_year = scenarios.columns.min()

    history_to_add = (
        multi_index_lookup(
            history,
            scenarios.reset_index(["model", "scenario"], drop=True)
            .drop_duplicates()
            .index,
        )
        .reset_index(["model", "scenario"], drop=True)
        .align(scenarios)[0]
        .loc[:, magicc_start_year : scenarios_start_year - 1]
    )

    complete_magicc = pix_concat(
        [
            history_to_add.reorder_levels(scenarios.index.names),
            scenarios,
        ],
        axis="columns",
    )
    # Also interpolate for MAGICC
    complete_magicc = complete_magicc.T.interpolate(method="index").T

    return complete_magicc
