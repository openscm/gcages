"""
Simple climate model (SCM) running part of the AR6 workflow
"""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from attrs import define, field
from pandas_openscm.db import EmptyDBError, OpenSCMDB
from pandas_openscm.indexing import multi_index_lookup
from pandas_openscm.parallelisation import ParallelOpConfig

from gcages.convention_mapping import (
    convert_openscm_runner_output_names_to_magicc_output_names,
    transform_iamc_to_openscm_runner_variable,
)
from gcages.exceptions import MissingOptionalDependencyError
from gcages.hashing import get_file_hash
from gcages.units_helpers import strip_pint_incompatible_characters_from_units

AR6_OUTPUT_VARIABLES: tuple[str, ...] = (
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
    "Effective Radiative Forcing|CFC11",
    "Effective Radiative Forcing|CFC12",
    "Effective Radiative Forcing|HCFC22",
    "Effective Radiative Forcing|Ozone",
    "Effective Radiative Forcing|HFC125",
    "Effective Radiative Forcing|HFC134a",
    "Effective Radiative Forcing|HFC143a",
    "Effective Radiative Forcing|HFC227ea",
    "Effective Radiative Forcing|HFC23",
    "Effective Radiative Forcing|HFC245fa",
    "Effective Radiative Forcing|HFC32",
    "Effective Radiative Forcing|HFC4310mee",
    "Effective Radiative Forcing|CF4",
    "Effective Radiative Forcing|C6F14",
    "Effective Radiative Forcing|C2F6",
    "Effective Radiative Forcing|SF6",
    # Heat uptake
    "Heat Uptake",
    # "Heat Uptake|Ocean",
    # Atmospheric concentrations
    "Atmospheric Concentrations|CO2",
    "Atmospheric Concentrations|CH4",
    "Atmospheric Concentrations|N2O",
    # carbon cycle
    "Net Atmosphere to Land Flux|CO2",
    "Net Atmosphere to Ocean Flux|CO2",
    # permafrost
    "Net Land to Atmosphere Flux|CO2|Earth System Feedbacks|Permafrost",
    "Net Land to Atmosphere Flux|CH4|Earth System Feedbacks|Permafrost",
)
"""
Output variables captured in AR6

Note that it can be a bit of work
to get these variables to actually appear in the output,
depending on which model you're using.
"""


def run_scm(  # noqa: PLR0913
    scenarios: pd.DataFrame,
    climate_models_cfgs: dict[str, list[dict[str, Any]]],
    output_variables: tuple[str, ...],
    n_processes: int,
    db: OpenSCMDB | None = None,
    verbose: bool = True,
    progress: bool = True,
    batch_size_scenarios: int | None = None,
    force_interpolate_to_yearly: bool = True,
    force_rerun: bool = False,
) -> pd.DataFrame:
    try:
        import openscm_runner.run
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "run_scm", requirement="openscm_runner"
        ) from exc

    try:
        import scmdata
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "run_scm", requirement="openscm_runner"
        ) from exc

    if force_interpolate_to_yearly:
        # Interpolate to ensure no nans.
        for y in range(scenarios.columns.min(), scenarios.columns.max() + 1):
            if y not in scenarios:
                scenarios[y] = np.nan

        scenarios = scenarios.sort_index(axis="columns").T.interpolate("index").T

    mod_scens_to_run = scenarios.pix.unique(["model", "scenario"])
    climate_models_cfgs_iter = climate_models_cfgs.items()
    if progress:
        pconfig = ParallelOpConfig.from_user_facing(
            progress=progress,
            progress_results_kwargs=dict(desc="Climate models"),
            max_workers=None,
        )
        climate_models_cfgs_iter = pconfig.progress_results(
            climate_models_cfgs_iter, desc="Climate models"
        )

    for climate_model, cfg in climate_models_cfgs_iter:
        if climate_model == "MAGICC7":
            # Avoid MAGICC's last year jump
            magicc_extra_years = 3
            cfg_use = [
                {**c, "endyear": scenarios.columns.max() + magicc_extra_years}
                for c in cfg
            ]
            os.environ["MAGICC_WORKER_NUMBER"] = str(n_processes)

        else:
            cfg_use = cfg

        if batch_size_scenarios is None:
            scenario_batches = [scenarios]
        else:
            scenario_batches = []
            for i in range(0, mod_scens_to_run.size, batch_size_scenarios):
                start = i
                stop = min(i + batch_size_scenarios, mod_scens_to_run.shape[0])

                scenario_batches.append(
                    multi_index_lookup(scenarios, mod_scens_to_run[start:stop])
                )

        if progress:
            pconfig = ParallelOpConfig.from_user_facing(
                progress=progress,
                progress_results_kwargs=dict(desc="Scenario batches"),
                max_workers=None,
            )
            scenario_batches = pconfig.progress_results(
                scenario_batches, desc="Scenario batch"
            )

        if db is None:
            res_l = []

        for scenario_batch in scenario_batches:
            if force_rerun or db is None:
                run_all = True

            else:
                try:
                    existing_metadata = db.load_metadata()
                    run_all = False

                except EmptyDBError:
                    run_all = True

            if run_all:
                scenario_batch_to_run = scenario_batch

            else:
                db_mod_scen = existing_metadata.pix.unique(["model", "scenario"])

                already_run = multi_index_lookup(scenario_batch, db_mod_scen)
                scenario_batch_to_run = scenario_batch.loc[
                    scenario_batch.index.difference(already_run.index)
                ]

                already_run_mod_scen = already_run.pix.unique(["model", "scenario"])
                if not already_run_mod_scen.empty and verbose:
                    # There are nicer ways to do this than verbose,
                    # but thinking through logging is a problem for another day
                    # (making loguru a required dependency might be the answer,
                    # I don't know if it has any other dependencies).
                    print(
                        "Not re-running already run scenarios:\n"
                        f"{already_run_mod_scen.to_frame(index=False)}"
                    )

                if scenario_batch_to_run.empty:
                    continue

            if climate_model == "MAGICC7":
                # Avoid MAGICC's last year jump
                scenario_batch_to_run = scenario_batch_to_run.copy()
                last_year = scenario_batch_to_run.columns.max()
                scenario_batch_to_run[last_year + magicc_extra_years] = (
                    scenario_batch_to_run[last_year]
                )
                scenario_batch_to_run = (
                    scenario_batch_to_run.sort_index(axis="columns")
                    .T.interpolate("index")
                    .T
                )

            batch_res = openscm_runner.run.run(
                scenarios=scmdata.ScmRun(scenario_batch_to_run, copy_data=True),
                climate_models_cfgs={climate_model: cfg_use},
                output_variables=output_variables,
            ).timeseries(time_axis="year")

            if climate_model == "MAGICC7":
                # Chop off the extra years
                batch_res = batch_res.iloc[:, :-magicc_extra_years]
                # Chop out regional results
                batch_res = batch_res.loc[
                    batch_res.index.get_level_values("region") == "World"
                ]

            if db is not None:
                db.save(batch_res)
            else:
                res_l.append(batch_res)

    if db is not None:
        res = db.load(mod_scens_to_run)
    else:
        res = pd.concat(res_l)

    return res


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

    if openscm_runner.adapters.MAGICC7.get_version() != "v7.5.3":
        raise AssertionError(openscm_runner.adapters.MAGICC7.get_version())


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
            raise NotImplementedError

        # TODO:
        #   - enable optional checks for:
        #       - only known variable names are in in_emissions
        #       - only data with a useable time axis is in there
        #       - metadata is appropriate/usable

        # Strip off any prefixes that might be there
        to_run = in_emissions.pix.assign(
            variable=in_emissions.index.get_level_values("variable").map(
                lambda x: x.replace("AR6 climate diagnostics|Infilled|", "")
            )
        ).sort_index(axis="columns")

        # Transform variable names
        to_run = to_run.pix.assign(
            variable=to_run.index.get_level_values("variable")
            .map(transform_iamc_to_openscm_runner_variable)
            .values
        )

        # Deal with the HFC245 bug
        to_run = to_run.pix.assign(
            variable=to_run.index.get_level_values("variable")
            .str.replace("HFC245ca", "HFC245fa")
            .values,
            unit=to_run.index.get_level_values("unit")
            .str.replace("HFC245ca", "HFC245fa")
            .values,
        )

        to_run = strip_pint_incompatible_characters_from_units(to_run)

        scm_results = run_scm(
            to_run,
            climate_models_cfgs=self.climate_models_cfgs,
            output_variables=self.output_variables,
            n_processes=self.n_processes,
            db=self.db,
            batch_size_scenarios=self.batch_size_scenarios,
            force_interpolate_to_yearly=self.force_interpolate_to_yearly,
            force_rerun=force_rerun,
        )

        # Apply AR6 naming scheme
        out: pd.DataFrame = scm_results.pix.format(
            variable="AR6 climate diagnostics|{variable}"
        )
        out = out.pix.assign(
            variable=out.index.get_level_values("variable")
            .str.replace(
                "Surface Air Temperature Change", "Raw Surface Temperature (GSAT)"
            )
            .values,
        )
        out.columns = out.columns.astype(self.res_column_type)

        # TODO:
        #   - enable optional checks for:
        #       - input and output scenarios are the same

        return out

    @classmethod
    def from_ar6_config(  # noqa: PLR0913
        cls,
        magicc_exe_path: Path,
        magicc_prob_distribution_path: Path,
        output_variables: tuple[str, ...] = AR6_OUTPUT_VARIABLES,
        db: OpenSCMDB | None = None,
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

        db
            Database to use for storing results.

            If not supplied, raw outputs are not stored.

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
            db=db,
            run_checks=run_checks,
            n_processes=n_processes,
            force_interpolate_to_yearly=True,  # MAGICC safer with annual input
            res_column_type=int,  # annual output by default
        )
