"""
General tools for running FaIR
"""

from __future__ import annotations

import json
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from attrs import define, field
from pandas_openscm.db import OpenSCMDB
from pandas_openscm.index_manipulation import update_index_levels_func

from gcages.assertions import (
    assert_data_is_all_numeric,
    assert_has_index_levels,
    assert_index_is_multiindex,
)
from gcages.completeness import assert_all_groups_are_complete
from gcages.exceptions import MissingOptionalDependencyError
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.scm_running import run_scms
from gcages.units_helpers import assert_has_no_pint_incompatible_characters

FAIR_END_YEAR_MAX = 2110


def load_fair_probabilistic_config(
    config_file_slim: Path,
    config_file_common: Path,
    scenario_end_year: int,
    num_cfgs: int | None = None,
) -> list[dict[str, Any]]:
    """
    Load FaIR configuration from a probabilistic config file

    Parameters
    ----------
    config_file_slim
        Config file to load (slim)

    config_file_common
        Config file to load (common)

    scenario_end_year
        Scenario end year

        The natural forcing timeseries in `config_file_common`
        run from 1750 to 2110.
        They are cut down to match the emissions being run,
        so this must equal the last year of the emissions
        passed to the runner,
        otherwise FaIR receives arrays of the wrong length.

    num_cfgs
        Number of configs

    Returns
    -------
    :
        FaIR configurations to use when running FaIR
    """
    with open(config_file_slim) as fh:
        cfgs_raw = json.load(fh)

    with open(config_file_common) as fh:
        cfgs_common = json.load(fh)

    # Cut natural down to the correct length
    # TODO: Ask Zeb if the >= is correct
    if scenario_end_year >= FAIR_END_YEAR_MAX:
        msg = f"Scenario end year set to {scenario_end_year} must be < 2110"
        raise NotImplementedError(msg)

    last_natural_forcers_index = -(FAIR_END_YEAR_MAX - scenario_end_year)
    cfgs_common["natural"] = cfgs_common["natural"][:last_natural_forcers_index]
    cfgs_common["default_volcanic"] = cfgs_common["default_volcanic"][
        :last_natural_forcers_index
    ]

    e_pi = [0] * 40
    for idx in range(5, 12):
        e_pi[idx] = cfgs_common["E_pi"][idx - 5]

    cfgs = []
    for i, c in enumerate(cfgs_raw[:num_cfgs]):
        scale = [1] * 45
        c_pi = [0] * 31
        c_pi[1] = cfgs_common["C_pi"][0]
        c_pi[2] = cfgs_common["C_pi"][1]
        c_pi[3] = cfgs_common["C_pi"][2]
        c_pi[20] = cfgs_common["C_pi"][3]
        c_pi[25] = cfgs_common["C_pi"][4]
        c_pi[29] = cfgs_common["C_pi"][5]
        c_pi[30] = cfgs_common["C_pi"][6]
        scale[1] = c["scale"][0]
        scale[2] = c["scale"][1]
        for idx in range(3, 31):
            scale[idx] = c["scale"][2]
        scale[15] = scale[15] * cfgs_common["cfc11_adj"]
        scale[16] = scale[16] * cfgs_common["cfc12_adj"]
        scale[33] = c["scale"][3]
        scale[34] = c["scale"][4]
        scale[41] = c["scale"][5]
        scale[42] = c["scale"][6]
        scale[43] = c["scale"][7]
        c_pi[0] = c["C_pi_CO2"]
        f_solar = np.zeros(361)
        f_solar[:270] = (
            np.linspace(0, c["trend_solar"], 270)
            + np.array(cfgs_common["default_solar"])[:270] * c["scale"][8]
        )
        f_solar[270:351] = (
            c["trend_solar"]
            + np.array(cfgs_common["default_solar"])[270:351] * c["scale"][8]
        )
        f_solar[351:361] = cfgs_common["default_solar"][351:]
        # Cut down to the correct length
        f_solar = f_solar[:last_natural_forcers_index]

        this_cfg = {
            "run_id": i,
            "F2x": c["F2x"],
            "r0": c["r0"],
            "rt": c["rt"],
            "rc": c["rc"],
            "lambda_global": c["lambda_global"],
            "ocean_heat_capacity": c["ocean_heat_capacity"],
            "ocean_heat_exchange": c["ocean_heat_exchange"],
            "deep_ocean_efficacy": c["deep_ocean_efficacy"],
            "b_aero": [
                c["b_aero"][0],
                0.0,
                0.0,
                0.0,
                c["b_aero"][1],
                c["b_aero"][2],
                c["b_aero"][3],
            ],
            "ghan_params": c["ghan_params"],
            "scale": scale,
            "F_solar": f_solar.tolist(),
            "F_volcanic": cfgs_common["default_volcanic"],
            "C_pi": c_pi,
            "b_tro3": c["b_tro3"],
            "ozone_feedback": c["ozone_feedback"],
            "E_pi": e_pi,
            "ghg_forcing": cfgs_common["ghg_forcing"],
            "aCO2land": cfgs_common["aCO2land"],
            "stwv_from_ch4": cfgs_common["stwv_from_ch4"],
            "F_ref_BC": cfgs_common["F_ref_BC"],
            "E_ref_BC": cfgs_common["E_ref_BC"],
            "tropO3_forcing": cfgs_common["tropO3_forcing"],
            "natural": cfgs_common["natural"],
        }
        cfgs.append(this_cfg)

    return cfgs


FAIR_VERSION_REQUIRED = "1.6.2.1"
FAIR_SOURCE_URL = "https://github.com/OMS-NetZero/FAIR.git"
FAIR_SOURCE_REV = "f87269b7f968b8786d2446e3a4eed85b7e719191"


def check_fair_version() -> None:
    """
    Check that the installed FaIR is the build our configuration requires
    """
    try:
        import fair  # noqa: PLC0415
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "check_fair_version", requirement="fair"
        ) from exc

    if fair.__version__ != FAIR_VERSION_REQUIRED:
        msg = (
            f"Expected fair v{FAIR_VERSION_REQUIRED}, found v{fair.__version__}. "
            f"v{FAIR_VERSION_REQUIRED} is not on PyPI: it is fair v1.6.2 plus "
            "NumPy-compatibility fixes, installable with\n"
            f"    pip install 'fair @ git+{FAIR_SOURCE_URL}@{FAIR_SOURCE_REV}'\n"
            "Note that `pip install openscm-runner[fair]` pins fair<2 and will "
            "install a PyPI release (e.g. v1.6.4) that does NOT contain these fixes."
        )
        raise AssertionError(msg)


FAIR_OUTPUT_VARIABLES_DEFAULT = (
    "Atmospheric Concentrations|CH4",
    "Atmospheric Concentrations|CO2",
    "Atmospheric Concentrations|N2O",
    "Effective Radiative Forcing",
    "Effective Radiative Forcing|Aerosols",
    "Effective Radiative Forcing|Aerosols|Direct Effect",
    "Effective Radiative Forcing|Aerosols|Direct Effect|BC",
    "Effective Radiative Forcing|Aerosols|Direct Effect|OC",
    "Effective Radiative Forcing|Aerosols|Direct Effect|SOx",
    "Effective Radiative Forcing|Aerosols|Indirect Effect",
    "Effective Radiative Forcing|Anthropogenic",
    "Effective Radiative Forcing|C2F6",
    "Effective Radiative Forcing|C6F14",
    "Effective Radiative Forcing|CF4",
    "Effective Radiative Forcing|CFC11",
    "Effective Radiative Forcing|CFC12",
    "Effective Radiative Forcing|CH4",
    "Effective Radiative Forcing|CO2",
    "Effective Radiative Forcing|F-Gases",
    "Effective Radiative Forcing|Greenhouse Gases",
    "Effective Radiative Forcing|HCFC22",
    "Effective Radiative Forcing|HFC125",
    "Effective Radiative Forcing|HFC134a",
    "Effective Radiative Forcing|HFC143a",
    "Effective Radiative Forcing|HFC227ea",
    "Effective Radiative Forcing|HFC23",
    "Effective Radiative Forcing|HFC245fa",
    "Effective Radiative Forcing|HFC32",
    "Effective Radiative Forcing|HFC4310mee",
    "Effective Radiative Forcing|Montreal Protocol Halogen Gases",
    "Effective Radiative Forcing|N2O",
    "Effective Radiative Forcing|Ozone",
    "Effective Radiative Forcing|SF6",
    "Heat Uptake",
    "Surface Air Ocean Blended Temperature Change",
    "Surface Air Temperature Change",
)


@define
class FairSCMRunner:
    """
    FaIR Simple climate model runner

    This is a standalone SCM runner.
    """

    climate_models_cfgs: dict[str, list[dict[str, Any]]] = field(
        repr=lambda x: ", ".join(
            (
                f"{climate_model}: {len(cfgs)} configurations"
                for climate_model, cfgs in x.items()
            )
        )
    )
    """
    Climate models to run and the configuration to use with them
    """

    output_variables: tuple[str, ...] = FAIR_OUTPUT_VARIABLES_DEFAULT
    """
    Variables to include in the output
    """

    batch_size_scenarios: int | None = None
    """
    The number of scenarios to run at a time

    Smaller batch sizes use less memory, but take longer overall
    (all else being equal).

    If not supplied, all scenarios are run simultaneously.
    """

    db: OpenSCMDB | None = None
    """
    Database in which to store the output of the runs

    If not supplied, output of the runs is not stored.
    """

    res_column_type: type = int
    """
    Type to cast the result's column type to
    """

    verbose: bool = True
    """
    Should verbose messages be printed?

    This is a temporary hack while we think about how to handle logging
    """

    run_checks: bool = True
    """
    If `True`, run checks on both input and output data

    If you are sure about your workflow,
    you can disable the checks to speed things up
    (but we don't recommend this unless you really
    are confident about what you're doing).
    """

    progress: bool = True
    """
    Should progress bars be shown for each operation?
    """

    n_processes: int | None = multiprocessing.cpu_count()
    """
    Number of processes to use for parallel processing.

    Set to `None` to process in serial.
    """

    def __call__(
        self, in_emissions: pd.DataFrame, force_rerun: bool = False
    ) -> pd.DataFrame:
        """
        Run the simple climate model

        Parameters
        ----------
        in_emissions
            Emissions to run

        force_rerun
            Force scenarios to re-run (i.e. disable caching).

        Returns
        -------
        :
            Raw results from the simple climate model
        """
        if self.run_checks:
            assert_index_is_multiindex(in_emissions)
            assert_has_index_levels(
                in_emissions, ["variable", "unit", "model", "scenario"]
            )
            assert_has_no_pint_incompatible_characters(
                in_emissions.index.get_level_values("unit").unique()
            )
            assert_data_is_all_numeric(in_emissions)

        openscm_runner_emissions = update_index_levels_func(
            in_emissions,
            {
                "variable": partial(
                    convert_variable_name,
                    from_convention=SupportedNamingConventions.GCAGES,
                    to_convention=SupportedNamingConventions.OPENSCM_RUNNER,
                )
            },
        )

        scm_results_maybe = run_scms(
            openscm_runner_emissions,
            climate_models_cfgs=self.climate_models_cfgs,
            output_variables=self.output_variables,
            scenario_group_levels=["model", "scenario"],
            n_processes=self.n_processes if self.n_processes is not None else 1,
            db=self.db,
            verbose=self.verbose,
            batch_size_scenarios=self.batch_size_scenarios,
            force_rerun=force_rerun,
        )

        if self.db is not None:
            # Results aren't kept in memory during running, so have to load them now.
            # User can use `run_scms` directly if they want to process differently.
            out_maybe = self.db.load()
            if out_maybe is None:
                raise TypeError(out_maybe)

            out: pd.DataFrame = out_maybe

        else:
            if scm_results_maybe is None:
                raise TypeError(scm_results_maybe)

            out = scm_results_maybe

        out.columns = out.columns.astype(self.res_column_type)

        if self.run_checks:
            # All scenarios have output
            pd.testing.assert_index_equal(
                out.index.droplevel(
                    out.index.names.difference(["model", "scenario"])  # type: ignore # pandas-stubs out of date
                ).drop_duplicates(),
                in_emissions.index.droplevel(
                    in_emissions.index.names.difference(["model", "scenario"])  # type: ignore # pandas-stubs out of date
                ).drop_duplicates(),
                check_order=False,
            )
            # Expected output is provided
            assert_all_groups_are_complete(
                out,
                complete_index=pd.MultiIndex.from_arrays(
                    [list(self.output_variables)], names=["variable"]
                ),
            )

        return out

    @classmethod
    def load_configs(  # noqa: PLR0913
        cls,
        config_file_slim: Path,
        config_file_common: Path,
        scenario_end_year: int,
        num_cfgs: int | None = None,
        output_variables: tuple[str, ...] = FAIR_OUTPUT_VARIABLES_DEFAULT,
        batch_size_scenarios: int | None = None,
        db: OpenSCMDB | None = None,
        res_column_type: type = int,
        verbose: bool = True,
        run_checks: bool = True,
        progress: bool = True,
        n_processes: int | None = multiprocessing.cpu_count(),
    ) -> FairSCMRunner:
        """
        Initialise from the config files.

        Parameters
        ----------
        config_file_slim
            Path to the "slim" probabilistic config file.
            This holds the per-ensemble-member parameters.

        config_file_common
            Path to the "common" probabilistic config file.

            This holds the parameters shared by all ensemble members,
            including the natural forcing timeseries.

        scenario_end_year
            Last year of the scenarios that will be run.

        num_cfgs
            Number of ensemble members to use.

        output_variables
            Variables to include in the output

        batch_size_scenarios
            The number of scenarios to run at a time

        db
            Database to use for storing results.

        res_column_type
            Type of column to store in the output.

        verbose
            Should verbose messages be printed?

        run_checks
            Should checks of the input and output data be performed?

        progress
            Should progress bars be shown for each operation?

        n_processes
            Number of processes to use for parallel processing.

        Returns
        -------
            FaIR configs
        """
        # Check if it's v1.6.2
        check_fair_version()

        cfgs = load_fair_probabilistic_config(
            config_file_slim, config_file_common, scenario_end_year, num_cfgs
        )
        return cls(
            climate_models_cfgs={"FAIR": cfgs},
            output_variables=output_variables,
            batch_size_scenarios=batch_size_scenarios,
            db=db,
            res_column_type=res_column_type,
            verbose=verbose,
            run_checks=run_checks,
            progress=progress,
            n_processes=n_processes,
        )
