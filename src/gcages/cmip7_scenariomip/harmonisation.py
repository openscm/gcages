"""
Harmonisation helpers for the CMIP7 ScenarioMIP workflow
"""

from __future__ import annotations

import multiprocessing
import platform
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.io import load_timeseries_csv

from gcages.harmonisation.aneris import AnerisHarmoniser
from gcages.hashing import get_file_hash
from gcages.renaming import SupportedNamingConventions, convert_variable_name


def load_cmip7_scenariomip_country_historical_emissions(
    filepath: Path,
    check_hash: bool = False,
) -> pd.DataFrame:
    """
    Load historical emissions for CMIP7 ScenarioMIP harmonisation.

    Parameters
    ----------
    filepath
        Path from which to load the file

    check_hash
        Check file hash

    Returns
    -------
    :
        Historical emissions

    Raises
    ------
    AssertionError
        `filepath` does not have the expected hash.

        We expect to be reading the file from
        https://zenodo.org/records/17845154/files/country-history_202511261223_202511040855_202512032146_202512021030_7e32405ade790677a6022ff498395bff00d9792d.csv?download=1
    """
    if check_hash:
        fp_hash = get_file_hash(filepath, algorithm="md5")
        if platform.system() in "Windows":
            fp_hash_exp = "ec92f75325a4d2e112b393e1379e818a"
        else:
            fp_hash_exp = "ec92f75325a4d2e112b393e1379e818a"

        if fp_hash != fp_hash_exp:
            msg = (
                f"The md5 hash of {filepath} is {fp_hash}. "
                f"This does not match what we expect {fp_hash_exp=}."
            )
            raise AssertionError(msg)

    res = load_timeseries_csv(
        filepath,
        lower_column_names=True,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_columns_type=int,
    )

    res.columns.name = "year"

    return res


def load_cmip7_scenariomip_global_historical_emissions(
    filepath: Path,
    check_hash: bool = False,
) -> pd.DataFrame:
    """
    Load historical emissions for CMIP7 ScenarioMIP harmonisation.

    Parameters
    ----------
    filepath
        Path from which to load the file

    check_hash
        Check file hash

    Returns
    -------
    :
        Historical emissions

    Raises
    ------
    AssertionError
        `filepath` does not have the expected hash.

        We expect to be reading the file from
        https://zenodo.org/records/17845154/files/global-workflow-history_202511261223_202511040855_202512032146_202512021030_7e32405ade790677a6022ff498395bff00d9792d_202511040855_202512071232_202511040855_202511040855_0002_0002.csv?download=1
    """
    if check_hash:
        fp_hash = get_file_hash(filepath, algorithm="md5")
        if platform.system() in "Windows":
            fp_hash_exp = "4aeb5e372df52b7beb54eedf5936d162"
        else:
            fp_hash_exp = "19482df604f1dc746fb354ef66ef9047"

        if fp_hash != fp_hash_exp:
            msg = (
                f"The md5 hash of {filepath} is {fp_hash}. "
                f"This does not match what we expect {fp_hash_exp=}."
            )
            raise AssertionError(msg)

    res = load_timeseries_csv(
        filepath,
        lower_column_names=True,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_columns_type=int,
    )
    res.columns.name = "year"

    return res


def load_aneris_overrides_file(filepath: Path) -> pd.Series[str]:
    """
    Load aneris overrides for CMIP7 ScenarioMIP harmonisation.

    Parameters
    ----------
    filepath
        Path from which to load the file

    Returns
    -------
    :
        Aneris overrides
    """
    raw = pd.read_csv(filepath)

    if "method" not in raw.columns:
        msg = "'method' column is required in the overrides CSV"
        raise KeyError(msg)

    index_cols = raw.columns.difference(["method"])
    res = raw.set_index(list(index_cols))["method"].astype(str)

    return res


@dataclass
class _HarmonisationConfig:
    """Configuration object for CMIP7 ScenarioMIP harmonisation routines."""

    historical_emissions: pd.DataFrame
    """CMIP7 ScenarioMIP historical emissions."""
    aneris_overrides: pd.Series
    """Aneris overrides for the global workflow."""
    rename_variables: bool = False
    """On global level variables might need some renaming."""


def create_cmip7_scenariomip_global_harmoniser(
    cmip7_scenariomip_global_historical_emissions_file: Path,
    aneris_global_overrides_file: Path,
    run_checks: bool = True,
    progress: bool = True,
    n_processes: int | None = multiprocessing.cpu_count(),
) -> AnerisHarmoniser:
    """
    Create an Aneris harmoniser configured for CMIP7 ScenarioMIP global emissions.

    This function loads the global historical emissions dataset, converts variable
    names to the GCAGES convention, applies harmonisation overrides, and returns an
    ``AnerisHarmoniser`` instance that reflects the ScenarioMIP global workflow.

    Parameters
    ----------
    cmip7_scenariomip_global_historical_emissions_file
        Path to the CMIP7 ScenarioMIP global historical emissions dataset

    aneris_global_overrides_file
        Path to the Aneris overrides file specifying harmonisation methods per variable

    run_checks
        Whether to perform internal validation checks on inputs and outputs

    progress
        Whether to show progress indicators during harmonisation

    n_processes
        Number of parallel processes to use. Defaults to all available CPU cores.

    Returns
    -------
    :
        A fully configured harmoniser ready for ScenarioMIP global-level harmonisation.

    Notes
    -----
    This function primarily prepares inputs and delegates construction to
    ``_create_cmip7_scenariomip_harmoniser``. It ensures variables are translated
    from the CMIP7 ScenarioMIP to the GCAGES naming convention.
    """
    historical = load_cmip7_scenariomip_global_historical_emissions(
        filepath=cmip7_scenariomip_global_historical_emissions_file,
        check_hash=True,
    )

    aneris_overrides = load_aneris_overrides_file(aneris_global_overrides_file)

    # Type juggling for mypy: from series to dataframe back to series
    overrides_df = aneris_overrides.to_frame(name="method")

    overrides_df = update_index_levels_func(
        overrides_df,
        {
            "variable": lambda x: convert_variable_name(
                x,
                from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                to_convention=SupportedNamingConventions.GCAGES,
            )
        },
        copy=False,
    )

    aneris_overrides = overrides_df["method"]

    config = _HarmonisationConfig(
        historical_emissions=historical,
        aneris_overrides=aneris_overrides,
        rename_variables=True,
    )

    return _create_cmip7_scenariomip_harmoniser(
        config,
        run_checks,
        progress,
        n_processes,
    )


def create_cmip7_scenariomip_country_harmoniser(
    cmip7_scenariomip_country_historical_emissions_file: Path,
    aneris_country_overrides_file: Path,
    run_checks: bool = True,
    progress: bool = True,
    n_processes: int | None = multiprocessing.cpu_count(),
) -> AnerisHarmoniser:
    """
    Create an Aneris harmoniser for CMIP7 ScenarioMIP country-level emissions.

    This function loads a regionally resolved CMIP7 ScenarioMIP historical emissions
    dataset and corresponding Aneris overrides, builds a harmonisation configuration,
    and returns an ``AnerisHarmoniser`` for country-level.

    Parameters
    ----------
    cmip7_scenariomip_country_historical_emissions_file
        Path to the CMIP7 ScenarioMIP country-level historical emissions dataset

    aneris_country_overrides_file
        Path to the Aneris overrides file specifying harmonisation methods

    run_checks
        Whether to perform internal validation checks on inputs and outputs

    progress
        Whether to show progress indicators during harmonisation

    n_processes
        Number of parallel processes to use. Defaults to all available CPU cores.

    Returns
    -------
    :
        A fully configured harmoniser ready for ScenarioMIP country-level harmonisation.

    Notes
    -----
    In contrast to ``create_cmip7_scenariomip_global_harmoniser``, this function
    does not rename variable names as the country-level data already follow
    expected conventions.
    """
    historical_emissions = load_cmip7_scenariomip_country_historical_emissions(
        filepath=cmip7_scenariomip_country_historical_emissions_file,
        check_hash=False,  # TODO change this
    )

    aneris_overrides = load_aneris_overrides_file(aneris_country_overrides_file)

    config = _HarmonisationConfig(
        historical_emissions=historical_emissions,
        aneris_overrides=aneris_overrides,
    )

    return _create_cmip7_scenariomip_harmoniser(
        config,
        run_checks,
        progress,
        n_processes,
    )


def _create_cmip7_scenariomip_harmoniser(
    config: _HarmonisationConfig,
    run_checks: bool = True,
    progress: bool = True,
    n_processes: int | None = multiprocessing.cpu_count(),
) -> AnerisHarmoniser:
    """
    Create a CMIP7 ScenarioMIP harmoniser.

    This internal function, optionally renames variables
    to the GCAGES convention, and constructs the final ``AnerisHarmoniser`` instance.

    Parameters
    ----------
    config
        Configuration object containing input data, overrides, and behaviour flags.

    run_checks
        Whether to perform validation checks during harmonisation

    progress
        Whether to show progress indicators during execution

    n_processes
        Number of parallel worker processes

    Returns
    -------
    :
        Harmoniser instance configured for CMIP7 ScenarioMIP's workflow.
    """
    historical_emissions = config.historical_emissions

    # Drop model + scenario
    historical_emissions = historical_emissions.reset_index(
        historical_emissions.index.names.difference(["variable", "region", "unit"]),
        drop=True,
    )
    # If renaming is needed
    if config.rename_variables:
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

    return AnerisHarmoniser(
        historical_emissions=historical_emissions,
        harmonisation_year=2023,
        aneris_overrides=config.aneris_overrides,
        run_checks=run_checks,
        progress=progress,
        n_processes=n_processes,
    )
