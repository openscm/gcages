"""
Harmonisation helpers for the CMIP7 ScenarioMIP workflow
"""

from __future__ import annotations

import multiprocessing
import platform
from pathlib import Path

import pandas as pd
from pandas_openscm.index_manipulation import update_index_levels_func
from pandas_openscm.io import load_timeseries_csv

from gcages.harmonisation.aneris import AnerisHarmoniser
from gcages.hashing import get_file_hash
from gcages.renaming import SupportedNamingConventions, convert_variable_name


def load_cmip7_scenariomip_historical_emissions(
    filepath: Path, check_hash: bool = False
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
    res = raw.set_index(list(raw.columns.difference(["method"])))["method"]

    return res


def create_cmip7_scenariomip_global_harmoniser(
    cmip7_scenariomip_global_historical_emissions_file: Path,
    aneris_global_overrides_file: Path,
    run_checks: bool = True,
    progress: bool = True,
    n_processes: int | None = multiprocessing.cpu_count(),
) -> AnerisHarmoniser:
    """
    Create an a harmoniser configured for CMIP7 ScenarioMIP's global workflow.

    Parameters
    ----------
    cmip7_scenariomip_global_historical_emissions_file
        File containing CMIP7 ScenarioMIP historical emissions.

    aneris_global_overrides_file
        File containing aneris overrides for the global workflow.

    run_checks
        Should checks of input and output data be performed?

    progress
        Should progress bars be shown?

    n_processes
        Number of processes to use for parallel processing.

    Returns
    -------
    :
        Harmoniser that will behave in line with CMIP7 ScenarioMIP's global workflow
    """
    historical_emissions = load_cmip7_scenariomip_historical_emissions(
        filepath=cmip7_scenariomip_global_historical_emissions_file,
        check_hash=True,
    )

    # Drop out the model and scenario levels
    historical_emissions = historical_emissions.reset_index(
        historical_emissions.index.names.difference(["variable", "region", "unit"]),  # type: ignore # pandas-stubs out of date
        drop=True,
    )

    # Use gcages naming to match pre-processed outputs.
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

    aneris_overrides = load_aneris_overrides_file(aneris_global_overrides_file)
    # Type juggling for mypy: from series to dataframe back to series
    aneris_overrides_df = aneris_overrides.to_frame(name="method")
    updated_df = update_index_levels_func(
        aneris_overrides_df,
        {
            "variable": lambda x: convert_variable_name(
                x,
                from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                to_convention=SupportedNamingConventions.GCAGES,
            )
        },
        copy=False,
    )
    aneris_overrides = updated_df["method"]

    return AnerisHarmoniser(
        historical_emissions=historical_emissions,
        # Hard-coded as this was what was used.
        # If people want a different year, we can change the interface
        # but that requires thinking about historical emissions too
        # so we deliberately hard-code here.
        harmonisation_year=2023,
        aneris_overrides=aneris_overrides,
        run_checks=run_checks,
        progress=progress,
        n_processes=n_processes,
    )
