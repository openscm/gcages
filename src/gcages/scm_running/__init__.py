"""
General simple climate model (SCM) running tools
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any, Optional, cast

import pandas as pd
from pandas_openscm.db import EmptyDBError, OpenSCMDB
from pandas_openscm.indexing import multi_index_lookup, multi_index_match
from pandas_openscm.parallelisation import ParallelOpConfig

from gcages.exceptions import MissingOptionalDependencyError


def convert_openscm_runner_output_names_to_magicc_output_names(
    openscm_runner_names: Iterable[str],
) -> tuple[str, ...]:
    """
    Get output names for the call to MAGICC

    Parameters
    ----------
    openscm_runner_names
        OpenSCM-Runner output names

    Returns
    -------
    :
        MAGICC output names
    """
    # TODO: move this to OpenSCM-Runner or fix up pymagicc
    # (not doing now because of the headache of upgrading those packages)
    try:
        import pymagicc.definitions  # type: ignore
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "convert_openscm_runner_output_names_to_magicc_output_names",
            requirement="pymagicc",
        ) from exc

    res_l = []
    for openscm_runner_variable in openscm_runner_names:
        if openscm_runner_variable == "Surface Air Temperature Change":
            # A fun inconsistency
            res_l.append("SURFACE_TEMP")
        elif openscm_runner_variable == "Effective Radiative Forcing|HFC4310mee":
            # Another fun inconsistency
            magicc_var = pymagicc.definitions.convert_magicc7_to_openscm_variables(
                "Effective Radiative Forcing|HFC4310",
                inverse=True,
            )
            res_l.append(magicc_var)
        else:
            magicc_var = pymagicc.definitions.convert_magicc7_to_openscm_variables(
                openscm_runner_variable,
                inverse=True,
            )
            res_l.append(magicc_var)

    return tuple(res_l)


def batch_df(  # noqa: D103
    df: pd.DataFrame, batch_index: pd.MultiIndex, batch_size: int | None
) -> list[pd.DataFrame]:
    # TOOD: move this to pandas-openscm
    if batch_size is None:
        batches = [df]

    else:
        batches = []
        for i in range(0, batch_index.size, batch_size):
            start = i
            stop = min(i + batch_size, batch_index.shape[0])

            batches.append(multi_index_lookup(df, batch_index[start:stop]))

    return batches


def run_batch(
    batch: pd.DataFrame,
    climate_models_cfgs: dict[str, list[dict[str, Any]]],
    output_variables: tuple[str, ...],
) -> pd.DataFrame:
    """
    Run a batch of scenarios

    Parameters
    ----------
    batch
        Batch to run

    climate_models_cfgs
        Climate model to run and its configuration

        Passed to [openscm_runner.run.run]

    output_variables
        Output variables to retrieve from the climate model

        Passed to [openscm_runner.run.run]

    Returns
    -------
    :
        Results of running the batch
    """
    # TODO: move to openscm-runner
    # (not there at the moment because of maintenance issues)

    try:
        import openscm_runner.run
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "run_batch", requirement="openscm_runner"
        ) from exc

    try:
        import scmdata
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "run_batch", requirement="scmdata"
        ) from exc

    batch_res = openscm_runner.run.run(  # type: ignore
        scenarios=scmdata.ScmRun(batch, copy_data=True),
        climate_models_cfgs=climate_models_cfgs,
        output_variables=output_variables,
    ).timeseries(time_axis="year")

    return cast(pd.DataFrame, batch_res)


def get_scenarios_to_run_after_checking_cache(  # noqa: PLR0913
    scenarios: pd.DataFrame,
    climate_model: str,
    db: OpenSCMDB,
    scenario_group_levels: list[str],
    climate_model_level: str,
    verbose: bool,
) -> pd.DataFrame | None:
    """
    Get the scenarios to run after checking the database cache

    Parameters
    ----------
    scenarios
        Full set of scenarios

    climate_model
        Climate model we are going to run with

    db
        Database in which results are being stored

    scenario_group_levels
        Index levels which define scenario groups

        Typically something like ["model", "scenario"]

    climate_model_level
        Climate model level in the database's metadata

        This level should store information
        about the climate model used to run the scenario.

    verbose
        If we skip running scenarios, should we print a message showing which ones

    Returns
    -------
    :
        Scenarios to run

        If all scenarios have already been run for `climate_model`,
        we return `None`
    """
    # TODO: move to openscm-runner
    # (not there at the moment because of maintenance issues)
    try:
        existing_metadata = db.load_metadata()
    except EmptyDBError:
        # Empty DB, we know we need to run everything
        return scenarios

    check_levels = [*scenario_group_levels, climate_model_level]
    db_already_run = existing_metadata.droplevel(
        existing_metadata.names.difference(check_levels)  # type: ignore # pandas-stubs out of date
    ).unique()

    # TODO: set_new_single_value_levels into pandas-openscm's index_manipulation
    new_values = [climate_model]
    new_names = [climate_model_level]
    if not isinstance(scenarios.index, pd.MultiIndex):
        raise TypeError(scenarios.index)

    batch_output_exp_index = pd.MultiIndex(
        codes=[
            *scenarios.index.codes,
            *([[0] * scenarios.index.shape[0]] * len(new_values)),
        ],
        levels=[*scenarios.index.levels, *[pd.Index([value]) for value in new_values]],
        names=[*scenarios.index.names, *new_names],
    )

    already_run_idx = multi_index_match(batch_output_exp_index, db_already_run)
    batch_to_run = scenarios.loc[~already_run_idx, :]
    already_run = scenarios.loc[already_run_idx, :]
    already_run_disp = already_run.index.droplevel(
        already_run.index.names.difference(check_levels)  # type: ignore # pandas-stubs out of date
    ).unique()
    if not already_run_disp.empty and verbose:
        # There are nicer ways to do this than verbose,
        # but thinking through logging is a problem for another day
        # (making loguru a required dependency might be the answer,
        # I don't know if it has any other dependencies).
        print(
            "Not re-running already run scenarios:\n"
            f"{already_run_disp.to_frame(index=False)}"
        )

    if batch_to_run.empty:
        return None

    return batch_to_run


def run_scms(  # noqa: PLR0912, PLR0913
    scenarios: pd.DataFrame,
    climate_models_cfgs: dict[str, list[dict[str, Any]]],
    output_variables: tuple[str, ...],
    scenario_group_levels: list[str],
    n_processes: int,
    db: OpenSCMDB | None = None,
    db_climate_model_level: str = "climate_model",
    verbose: bool = True,
    progress: bool = True,
    batch_size_scenarios: int | None = None,
    force_rerun: bool = False,
) -> pd.DataFrame | None:
    """
    Run simple climate models (SCMs)

    Parameters
    ----------
    scenarios
        Scenarios to run

    climate_models_cfgs
        Climate model to run and its configuration

        Passed to [openscm_runner.run.run]

    output_variables
        Output variables to retrieve from the climate model

        Passed to [openscm_runner.run.run]

    scenario_group_levels
        Index levels which define scenario groups

        Typically something like ["model", "scenario"]

    n_processes
        Number of parallel processes to use while running

    db
        Database in which to save the results

        If not provided, results are not saved along the way

    db_climate_model_level
        Climate model level in the database's metadata

        This level should store information
        about the climate model used to run the scenario.

    verbose
        If we skip running scenarios because they have already been run,
        should we print a message showing which ones?

    progress
        Should progress bar(s) be displayed?

    batch_size_scenarios
        How many scenarios should be run in a single batch?

        Running more scenarios at once is faster,
        but it runs the risk of running out of memory.

    force_rerun
        Should we force the scenarios to be re-run, even if they are already in `db`

    Returns
    -------
    :
        Results of running the SCM

        If `db` is provided, returns `None`
        (so you as the user can decided whether all the output
        should be in memory at once)
    """
    # TODO: move to openscm-runner
    # (not there at the moment because of maintenance issues)
    try:
        import openscm_runner.adapters
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "run_scms", requirement="openscm_runner"
        ) from exc

    scens_to_run = scenarios.index.droplevel(
        scenarios.index.names.difference(scenario_group_levels)  # type: ignore # pandas-stubs out of date
    ).unique()
    climate_models_cfgs_iter = climate_models_cfgs.items()
    if progress:
        pconfig = ParallelOpConfig.from_user_facing(
            progress=progress,
            progress_results_kwargs=dict(desc="Climate models"),
            max_workers=None,  # This loop always goes in serial
        )
        climate_models_cfgs_iter = pconfig.progress_results(  # type: ignore # something weird happening here
            climate_models_cfgs_iter, desc="Climate models"
        )

    for climate_model, cfg in climate_models_cfgs_iter:
        cfg_use = cfg
        if force_rerun or db is None:
            scenarios_use: Optional[pd.DataFrame] = scenarios

        else:
            if climate_model == "MAGICC7":
                # Urgh
                climate_model_check = (
                    f"MAGICC{openscm_runner.adapters.MAGICC7.get_version()}"  # type: ignore
                )
            else:
                climate_model_check = climate_model

            scenarios_use = get_scenarios_to_run_after_checking_cache(
                scenarios,
                climate_model=climate_model_check,
                db=db,
                scenario_group_levels=scenario_group_levels,
                climate_model_level=db_climate_model_level,
                verbose=verbose,
            )
            if scenarios_use is None:
                # Already all run for this climate model
                continue

        if climate_model == "MAGICC7":
            # Avoid MAGICC's last year jump
            magicc_extra_years = 3
            cfg_use = [
                {**c, "endyear": scenarios.columns.max() + magicc_extra_years}
                for c in cfg_use
            ]
            os.environ["MAGICC_WORKER_NUMBER"] = str(n_processes)

            if scenarios_use is None:
                raise TypeError(scenarios_use)
            scenarios_use = scenarios_use.copy()
            last_year = scenarios_use.columns.max()
            scenarios_use[last_year + magicc_extra_years] = scenarios_use[last_year]
            scenarios_use = (
                scenarios_use.sort_index(axis="columns").T.interpolate("index").T
            )

        if scenarios_use is None:
            raise TypeError(scenarios_use)

        scenario_batches = batch_df(
            scenarios_use,
            batch_index=scens_to_run,
            batch_size=batch_size_scenarios,
        )

        if progress:
            pconfig = ParallelOpConfig.from_user_facing(
                progress=progress,
                progress_results_kwargs=dict(desc="Scenario batches"),
                max_workers=None,
            )
            scenario_batches = pconfig.progress_results(  # type: ignore # not sure what is happening here
                scenario_batches, desc="Scenario batch"
            )

        if db is None:
            res_l = []

        for scenario_batch in scenario_batches:
            batch_res = run_batch(
                batch=scenario_batch,
                climate_models_cfgs={climate_model: cfg_use},
                output_variables=output_variables,
            )

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
        # Assume the user doesn't want all the data in memory at once
        # (I really don't like this pattern, but can't fix it now)
        return None

    res = pd.concat(res_l)

    return res
