"""
SCM-running configuration and related things for the updated workflow

This sets our defaults.
Individual notebooks can then override them as needed.

TODO: reduce duplication with AR6 SCM runner
"""

from __future__ import annotations

import multiprocessing
from functools import partial
from pathlib import Path
from typing import Any, cast

import pandas as pd
from attrs import define, field
from pandas_openscm.db import OpenSCMDB
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.indexing import multi_index_lookup

from gcages.assertions import (
    assert_data_is_all_numeric,
    assert_has_data_for_times,
    assert_has_index_levels,
    assert_index_is_multiindex,
)
from gcages.cmip7_scenariomip.harmonisation import (
    load_cmip7_scenariomip_historical_emissions,
)
from gcages.completeness import assert_all_groups_are_complete
from gcages.harmonisation import assert_harmonised
from gcages.hashing import get_file_hash
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.scm_running import (
    DEFAULT_OUTPUT_VARIABLES,
    convert_openscm_runner_output_names_to_magicc_output_names,
    run_scms,
)
from gcages.scm_running.magicc import (
    check_magicc7_version,
    combine_probabilistic_and_common_cfg,
    load_magicc_probabilistic_config,
)
from gcages.units_helpers import assert_has_no_pint_incompatible_characters


def check_cmip7_scenariomip_magicc7_version(magicc_exe_path: Path) -> None:
    """
    Check that the MAGICC7 version is what was used in CMIP7 ScenarioMIP

    Parameters
    ----------
    magicc_exe_path
        Path to the MAGICC executable to use

    Raises
    ------
    AssertionError
        The MAGICC version is not what we expect

    MissingOptionalDependencyError
        [openscm-runner](https://github.com/openscm/openscm-runner) is not installed
    """
    check_magicc7_version(magicc_exe_path, expected_version="v7.6.0a3")


def load_cmip7_scenariomip_magicc_probabilistic_config(
    filepath: Path,
) -> list[dict[str, Any]]:
    """
    Load the probabilistic config used with MAGICC in CMIP7 ScenarioMIP

    Parameters
    ----------
    filepath
        Filepath from which to load the probabilistic configuration

    Returns
    -------
    :
        Probabilistic configuration used with MAGICC in CMIP7 ScenarioMIP

    Raises
    ------
    AssertionError
        `filepath` points to a file that does not have the expected hash
    """
    fp_hash = get_file_hash(filepath, algorithm="sha256")
    fp_hash_exp = "b386c89ddb3996a21b93658cb4a36efa68f6bed6ea979017c0eadcdc65aa6e72"
    if fp_hash != fp_hash_exp:
        msg = (
            f"The sha256 hash of {filepath} is {fp_hash}. "
            f"This does not match what we expect ({fp_hash_exp=})."
        )
        raise AssertionError(msg)

    cfgs = load_magicc_probabilistic_config(filepath)

    # Common config that affect MAGICC behaviour
    common_cfg = {"startyear": 1750}

    run_config = combine_probabilistic_and_common_cfg(cfgs, common_cfg=common_cfg)

    return run_config


def get_complete_scenarios_for_magicc(
    scenarios: pd.DataFrame,
    history: pd.DataFrame,
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
    scenarios_start_year = scenarios.columns.min()

    scenario_index = cast(
        pd.MultiIndex,
        scenarios.reset_index(["model", "scenario"], drop=True).drop_duplicates().index,
    )

    history_to_add = (
        multi_index_lookup(
            history,
            scenario_index,
        )
        .align(scenarios)[0]
        .loc[:, magicc_start_year : scenarios_start_year - 1]
    )

    complete_magicc = pd.concat(
        [
            history_to_add.reorder_levels(scenarios.index.names),
            scenarios,
        ],
        axis=1,
    )
    # Also interpolate for MAGICC
    complete_magicc = complete_magicc.T.interpolate(method="index").T
    return complete_magicc


@define
class CMIP7ScenarioMIPSCMRunner:
    """
    Simple climate model runner

    It follows the same logic as was used in CMIP7 SCENARIOMIP

    If you want exactly the same behaviour as in CMIP7 SCENARIOMIP
    initialise using [`from_cmip7_scenariomip_config`][(c)]
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

    Set to `None` to process serially.
    """

    magicc_exe_path: Path | None = None
    """
    Path to the MAGICC executable to use

    Only required if we're running MAGICC
    """

    def __call__(  # noqa: PLR0912
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

        if "MAGICC7" in self.climate_models_cfgs:
            if self.historical_emissions is None:
                # No history provided: assume emissions are already complete
                complete_emissions = in_emissions
                complete_emissions.columns = complete_emissions.columns.astype(int)
                # Validate MAGICC requirement
                magicc_start_year = 2015
                if complete_emissions.columns.min() != magicc_start_year:
                    msg = (
                        "Emissions starting year must be set to `2015` "
                        "when running MAGICC7 without providing `historical_emissions`"
                    )
                    raise AssertionError(msg)
            else:
                # History provided merge with scenarios
                complete_emissions = get_complete_scenarios_for_magicc(
                    scenarios=in_emissions,
                    history=self.historical_emissions,
                )
                complete_emissions.columns = complete_emissions.columns.astype(int)
        else:
            # Not running MAGICC, use emissions as-is
            complete_emissions = in_emissions

        # Start function to split out
        # `run_scms_gcages` ?
        openscm_runner_emissions = update_index_levels_func(
            complete_emissions,
            {
                "variable": partial(
                    convert_variable_name,
                    from_convention=SupportedNamingConventions.GCAGES,
                    to_convention=SupportedNamingConventions.OPENSCM_RUNNER,
                )
            },
        )

        scm_results_maybe = run_scms(
            scenarios=openscm_runner_emissions,
            climate_models_cfgs=self.climate_models_cfgs,
            output_variables=self.output_variables,
            scenario_group_levels=["model", "scenario"],
            # TODO: fix value in run_scms
            n_processes=self.n_processes if self.n_processes is not None else 1,
            db=self.db,
            verbose=self.verbose,
            batch_size_scenarios=self.batch_size_scenarios,
            force_rerun=force_rerun,
            magicc_exe_path=self.magicc_exe_path,
        )

        if self.db is not None:
            # Results aren't kept in memory during running, so have to load them now.
            # User can use `run_scms` directly if they want to process differently.
            # TODO: only load the scenarios we ran
            out_maybe = self.db.load()
            if out_maybe is None:
                raise TypeError(out_maybe)

            out: pd.DataFrame = out_maybe

        else:
            if scm_results_maybe is None:
                raise TypeError(scm_results_maybe)

            out = scm_results_maybe

        out.columns = out.columns.astype(self.res_column_type)
        # End function to split out

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
    def from_cmip7_scenariomip_config(  # noqa: PLR0913
        cls,
        magicc_exe_path: Path,
        magicc_prob_distribution_path: Path,
        output_variables: tuple[str, ...] = DEFAULT_OUTPUT_VARIABLES,
        batch_size_scenarios: int | None = None,
        db: OpenSCMDB | None = None,
        # TODO: switch to `historical_emissions`
        # and add helper for loading historical_emissions for MAGICC
        historical_emissions_path: Path | None = None,
        harmonisation_year: int = 2023,
        verbose: bool = True,
        run_checks: bool = True,
        progress: bool = True,
        n_processes: int | None = multiprocessing.cpu_count(),
    ) -> CMIP7ScenarioMIPSCMRunner:
        """
        Initialise from the config used in CMIP7 ScenarioMIP

        Parameters
        ----------
        magicc_exe_path
            Path to the MAGICC executable to use.

            This should be a MAGICC v7.6.0a3 executable.

        magicc_prob_distribution_path
            Path to the MAGICC probabilistic distribution.

            This should be the CMIP7 ScenarioMIP probabilistic distribution.

        output_variables
            Variables to include in the output

        batch_size_scenarios
            The number of scenarios to run at a time

        db
            Database to use for storing results.

            If not supplied, raw outputs are not stored.

        historical_emissions_path
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

            Set to `None` to process serially.

        Returns
        -------
        :
            Initialised SCM runner
        """
        check_cmip7_scenariomip_magicc7_version(magicc_exe_path)

        if historical_emissions_path is not None:
            # Load history
            historical_emissions = load_cmip7_scenariomip_historical_emissions(
                filepath=historical_emissions_path,
                check_hash=True,
            )
            historical_emissions = update_index_levels_func(
                historical_emissions,
                {
                    "variable": lambda x: convert_variable_name(
                        x,
                        from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                        to_convention=SupportedNamingConventions.GCAGES,
                    )
                },
                copy=False,
            )

            historical_emissions = historical_emissions.reset_index(
                level=[
                    lvl
                    for lvl in ["model", "scenario"]
                    if lvl in historical_emissions.index.names
                ],
                drop=True,
            )

        else:
            historical_emissions = None

        magicc_prob_cfg = load_cmip7_scenariomip_magicc_probabilistic_config(
            magicc_prob_distribution_path,
        )

        common_cfg = {
            "out_dynamic_vars": convert_openscm_runner_output_names_to_magicc_output_names(  # noqa: E501
                output_variables
            ),
            "out_ascii_binary": "BINARY",
            "out_binary_format": 2,
        }

        run_config = combine_probabilistic_and_common_cfg(
            magicc_prob_cfg, common_cfg=common_cfg
        )

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
            res_column_type=int,  # annual output by default
            magicc_exe_path=magicc_exe_path,
        )
