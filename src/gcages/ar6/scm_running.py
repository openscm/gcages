"""
Simple climate model (SCM) running part of the AR6 workflow
"""

from __future__ import annotations

import json
import multiprocessing
import os
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
    assert_has_data_for_times,
    assert_has_index_levels,
    assert_index_is_multiindex,
)
from gcages.completeness import assert_all_groups_are_complete
from gcages.exceptions import MissingOptionalDependencyError
from gcages.harmonisation import assert_harmonised
from gcages.hashing import get_file_hash
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.scm_running import (
    convert_openscm_runner_output_names_to_magicc_output_names,
    run_scms,
)
from gcages.units_helpers import assert_has_no_pint_incompatible_characters

DEFAULT_OUTPUT_VARIABLES: tuple[str, ...] = (
    # GSAT
    "Surface Air Temperature Change",
    # GMST
    "Surface Air Ocean Blended Temperature Change",
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
    # Heat uptake
    "Heat Uptake",
    # "Heat Uptake|Ocean",
    # Atmospheric concentrations
    "Atmospheric Concentrations|CO2",
    "Atmospheric Concentrations|CH4",
    "Atmospheric Concentrations|N2O",
    # Carbon cycle
    "Net Atmosphere to Land Flux|CO2",
    "Net Atmosphere to Ocean Flux|CO2",
    # Permafrost
    "Net Land to Atmosphere Flux|CO2|Earth System Feedbacks|Permafrost",
    "Net Land to Atmosphere Flux|CH4|Earth System Feedbacks|Permafrost",
)
"""
Default output variables

Note that it can be a bit of work
to get these variables to actually appear in the output,
depending on which simple climate model you're using.
"""


def check_ar6_magicc7_version() -> None:
    """
    Check that the MAGICC7 version is what was used in AR6
    """
    try:
        import openscm_runner.adapters
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "check_ar6_magicc7_version", requirement="openscm_runner"
        ) from exc

    if openscm_runner.adapters.MAGICC7.get_version() != "v7.5.3":  # type: ignore
        raise AssertionError(openscm_runner.adapters.MAGICC7.get_version())  # type: ignore


def load_ar6_magicc_probabilistic_config(filepath: Path) -> list[dict[str, Any]]:
    """
    Load the probabilistic config used with MAGICC in AR6

    Parameters
    ----------
    filepath
        Filepath from which to load the probabilistic configuration

    Returns
    -------
    :
        Probabilistic configuration used with MAGICC in AR6

    Raises
    ------
    AssertionError
        `filepath` points to a file that does not have the expected hash
    """
    fp_hash = get_file_hash(filepath, algorithm="sha256")
    fp_hash_exp = "f4481549e2309e3f32de9095bcfc1adc531b8ce985201690fd889d49def8a02f"
    if fp_hash != fp_hash_exp:
        msg = (
            f"The sha256 hash of {filepath} is {fp_hash}. "
            f"This does not match what we expect ({fp_hash_exp=})."
        )
        raise AssertionError(msg)

    with open(filepath) as fh:
        cfgs_raw = json.load(fh)

    cfgs = [
        {
            "run_id": c["paraset_id"],
            **{k.lower(): v for k, v in c["nml_allcfgs"].items()},
        }
        for c in cfgs_raw["configurations"]
    ]

    return cfgs


@define
class AR6SCMRunner:
    """
    Simple climate model runner that follows the same logic as was used in AR6

    If you want exactly the same behaviour as in AR6,
    initialise using [`from_ar6_config`][(c)]
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

    output_variables: tuple[str, ...]
    """
    Variables to include in the output
    """

    force_interpolate_to_yearly: bool = True
    """
    Should we interpolate scenarios we run to yearly steps before running the SCMs.
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

    historical_emissions: pd.DataFrame | None = None
    """
    Historical emissions used for harmonisation

    Only required if `run_checks` is `True` to check
    that the data to run is harmonised.
    """

    harmonisation_year: int | None = None
    """
    Year in which the data was harmonised

    Only required if `run_checks` is `True` to check
    that the data to run is harmonised.
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

            if self.historical_emissions is None:
                msg = "`self.historical_emissions` must be set to check the infilling"
                raise AssertionError(msg)

            if self.harmonisation_year is None:
                msg = "`self.harmonisation_year` must be set to check the infilling"
                raise AssertionError(msg)

            assert_has_data_for_times(
                in_emissions,
                name="in_emissions",
                times=[self.harmonisation_year, 2100],
                allow_nan=False,
            )

            assert_harmonised(
                in_emissions,
                history=self.historical_emissions,
                harmonisation_time=self.harmonisation_year,
                rounding=5,  # level of data storage in historical data often
            )
            assert_all_groups_are_complete(
                # The combo of the input and infilled should be complete
                in_emissions,
                complete_index=self.historical_emissions.index.droplevel("unit"),
            )

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
        if self.force_interpolate_to_yearly:
            # TODO: put interpolate to annual steps in pandas-openscm
            # Interpolate to ensure no nans.
            for y in range(
                openscm_runner_emissions.columns.min(),
                openscm_runner_emissions.columns.max() + 1,
            ):
                if y not in openscm_runner_emissions:
                    openscm_runner_emissions[y] = np.nan

            openscm_runner_emissions = (
                openscm_runner_emissions.sort_index(axis="columns")
                .T.interpolate("index")
                .T
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
            pd.testing.assert_index_equal(  # type: ignore # pandas-stubs out of date
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
    def from_ar6_config(  # noqa: PLR0913
        cls,
        magicc_exe_path: Path,
        magicc_prob_distribution_path: Path,
        output_variables: tuple[str, ...] = DEFAULT_OUTPUT_VARIABLES,
        batch_size_scenarios: int | None = None,
        db: OpenSCMDB | None = None,
        historical_emissions: pd.DataFrame | None = None,
        harmonisation_year: int | None = None,
        verbose: bool = True,
        run_checks: bool = True,
        progress: bool = True,
        n_processes: int | None = multiprocessing.cpu_count(),
    ) -> AR6SCMRunner:
        """
        Initialise from the config used in AR6

        Parameters
        ----------
        magicc_exe_path
            Path to the MAGICC executable to use.

            This should be a MAGICC v7.5.3 executable.

        magicc_prob_distribution_path
            Path to the MAGICC probabilistic distribution.

            This should be the AR6 probabilistic distribution.

        output_variables
            Variables to include in the output

        batch_size_scenarios
            The number of scenarios to run at a time

        db
            Database to use for storing results.

            If not supplied, raw outputs are not stored.

        historical_emissions
            Historical emissions used for harmonisation

            Only required if `run_checks` is `True` to check
            that the data is harmonised before running the SCMs.

        harmonisation_year
            Year in which the data was harmonised

            Only required if `run_checks` is `True` to check
            that the data is harmonised before running the SCMs.

        verbose
            Should verbose messages be printed?

            This is a temporary hack while we think about how to handle logging

        run_checks
            Should checks of the input and output data be performed?

            If this is turned off, things are faster,
            but error messages are much less clear if things go wrong.

        progress
            Should progress bars be shown for each operation?

        n_processes
            Number of processes to use for parallel processing.

            Set to `None` to process in serial.

        Returns
        -------
        :
            Initialised SCM runner
        """
        os.environ["MAGICC_EXECUTABLE_7"] = str(magicc_exe_path)
        check_ar6_magicc7_version()

        magicc_ar6_prob_cfg = load_ar6_magicc_probabilistic_config(
            magicc_prob_distribution_path
        )

        startyear = 1750
        common_cfg = {
            "startyear": startyear,
            "out_dynamic_vars": convert_openscm_runner_output_names_to_magicc_output_names(  # noqa: E501
                output_variables
            ),
            "out_ascii_binary": "BINARY",
            "out_binary_format": 2,
        }

        run_config = [{**common_cfg, **base_cfg} for base_cfg in magicc_ar6_prob_cfg]
        magicc_full_distribution_n_config = 600
        if len(run_config) != magicc_full_distribution_n_config:
            raise AssertionError(len(run_config))

        return cls(
            climate_models_cfgs={"MAGICC7": run_config},
            output_variables=output_variables,
            batch_size_scenarios=batch_size_scenarios,
            db=db,
            historical_emissions=historical_emissions,
            harmonisation_year=harmonisation_year,
            verbose=verbose,
            run_checks=run_checks,
            n_processes=n_processes,
            force_interpolate_to_yearly=True,  # MAGICC safer with annual input
            res_column_type=int,  # annual output by default
        )
