"""
SCM-running configuration and related things for the updated workflow

This sets our defaults.
Individual notebooks can then override them as needed.
"""

from __future__ import annotations

import json
import multiprocessing
import os
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
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.scm_running import (
    convert_openscm_runner_output_names_to_magicc_output_names,
    run_scms,
)
from gcages.units_helpers import assert_has_no_pint_incompatible_characters

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
        .reset_index(["model", "scenario"], drop=True)
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
class CMIP7_SCENARIOMIP_SCMRunner:
    """
    Simple climate model runner

    It follows the same logic as was used in CMIP7_SCENARIOMIP

    If you want exactly the same behaviour as in CMIP7_SCENARIOMIP
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

    # force_interpolate_to_yearly: bool = True
    # """
    # Should we interpolate scenarios we run to yearly steps before running the SCMs.
    # """

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
        # TODO MZ: not sure that's the best solution
        in_emissions.columns.name = "year"
        in_emissions = in_emissions.dropna(axis=1, how="all")

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
                history=self.historical_emissions.reset_index(
                    level=[
                        lvl
                        for lvl in ["model", "scenario"]
                        if lvl in self.historical_emissions.index.names
                    ],
                    drop=True,
                ),
                harmonisation_time=self.harmonisation_year,
                rounding=5,  # level of data storage in historical data often
            )
            assert_all_groups_are_complete(
                # The combo of the input and infilled should be complete
                in_emissions,
                complete_index=self.historical_emissions.index.droplevel(
                    ["model", "scenario", "unit"]
                ),
            )
        if self.historical_emissions is None:
            complete_emissions = in_emissions
        else:
            complete_emissions = get_complete_scenarios_for_magicc(
                scenarios=in_emissions,
                history=self.historical_emissions,
            )

        complete_emissions.columns = complete_emissions.columns.astype(int)

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
        # TODO delete?
        # if self.force_interpolate_to_yearly:
        #     # TODO: put interpolate to annual steps in pandas-openscm
        #     # Interpolate to ensure no nans.
        #     for y in range(
        #         openscm_runner_emissions.columns.min(),
        #         openscm_runner_emissions.columns.max() + 1,
        #     ):
        #         if y not in openscm_runner_emissions:
        #             openscm_runner_emissions[y] = np.nan
        #
        #     openscm_runner_emissions = (
        #         openscm_runner_emissions.sort_index(axis="columns")
        #         .T.interpolate("index")
        #         .T
        #     )
        scm_results_maybe = run_scms(
            scenarios=openscm_runner_emissions,
            climate_models_cfgs=self.climate_models_cfgs,
            output_variables=self.output_variables,
            scenario_group_levels=["model", "scenario"],
            n_processes=self.n_processes if self.n_processes is not None else 1,
            db=self.db,
            verbose=self.verbose,
            batch_size_scenarios=self.batch_size_scenarios,
            force_rerun=True,
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
    def from_cmip7_scenariomip_config(  # noqa: PLR0913
        cls,
        magicc_exe_path: Path,
        magicc_prob_distribution_path: Path,
        output_variables: tuple[str, ...] = SCM_OUTPUT_VARIABLES_DEFAULT,
        batch_size_scenarios: int | None = None,
        db: OpenSCMDB | None = None,
        historical_emissions_path: Path | None = None,
        harmonisation_year: int = 2023,
        verbose: bool = True,
        run_checks: bool = True,
        progress: bool = True,
        n_processes: int | None = multiprocessing.cpu_count(),
    ) -> CMIP7_SCENARIOMIP_SCMRunner:
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

            Set to `None` to process in serial.

        Returns
        -------
        :
            Initialised SCM runner
        """
        os.environ["MAGICC_EXECUTABLE_7"] = str(magicc_exe_path)
        # TODO?
        # check_cmip7_scenariomip_magicc7_version()

        # TODO Is it appropriate that we want a Path here and
        # a df in the class?
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
        else:
            historical_emissions = None

        magicc_prob_cfg = load_magicc_cfgs(
            prob_distribution_path=magicc_prob_distribution_path,
            output_variables=output_variables,
            startyear=1750,
        )

        return cls(
            climate_models_cfgs=magicc_prob_cfg,
            output_variables=output_variables,
            batch_size_scenarios=batch_size_scenarios,
            db=db,
            historical_emissions=historical_emissions,
            harmonisation_year=harmonisation_year,
            verbose=verbose,
            run_checks=run_checks,
            n_processes=n_processes,
            # force_interpolate_to_yearly=True,  # MAGICC safer with annual input
            res_column_type=int,  # annual output by default
        )
