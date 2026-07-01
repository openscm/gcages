"""
Harmonisation helpers for the CMIP7 ScenarioMIP workflow
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pandas as pd
from pandas_openscm.io import load_timeseries_csv

from gcages.cmip7_scenariomip.harmonisation import load_aneris_overrides_file
from gcages.harmonisation.aneris import AnerisHarmoniser
from gcages.renaming import SupportedNamingConventions, rename_variables


def load_historical_emissions(
    historical_emissions_file: Path,
) -> pd.DataFrame:
    """
    Load historical emissions for CMIP7 ScenarioMIP harmonisation.

    Parameters
    ----------
    historical_emissions_file
        Path from which to load the file

    Returns
    -------
    :
        Historical emissions in GCAGES format
    """
    historical_emissions = load_timeseries_csv(
        historical_emissions_file,
        lower_column_names=True,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_columns_type=int,
    )

    # Drop out the model and scenario levels
    historical_emissions = historical_emissions.reset_index(
        historical_emissions.index.names.difference(["variable", "region", "unit"]),  # type: ignore # pandas-stubs out of date
        drop=True,
    )

    # Use gcages naming to match pre-processed outputs.
    historical_emissions = rename_variables(
        historical_emissions,
        from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        to_convention=SupportedNamingConventions.GCAGES,
    )

    return historical_emissions


def create_scijune2026_global_harmoniser(  # noqa: PLR0913
    historical_emissions_file: Path,
    aneris_overrides_file: Path,
    harmonisation_year: int,
    run_checks: bool = True,
    progress: bool = True,
    n_processes: int | None = multiprocessing.cpu_count(),
) -> AnerisHarmoniser:
    """
    Create an Aneris harmoniser configured for SCI June 2026 harmoniser.

    Parameters
    ----------
    historical_emissions_file
        File containing CMIP7 ScenarioMIP historical emissions.

    aneris_overrides_file
        File containing aneris overrides for the global workflow.

    harmonisation_year
        Year in which to harmonise

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
    historical_emissions = load_historical_emissions(
        historical_emissions_file,
    )

    aneris_overrides = load_aneris_overrides_file(aneris_overrides_file)
    # TODO: remove this as it isn't needed for pandas-openscm 0.8.1
    aneris_overrides_df = aneris_overrides.to_frame(name="method")

    updated_df = rename_variables(
        aneris_overrides_df,
        from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        to_convention=SupportedNamingConventions.GCAGES,
    )
    aneris_overrides = updated_df["method"]

    return AnerisHarmoniser(
        historical_emissions=historical_emissions,
        harmonisation_year=harmonisation_year,
        aneris_overrides=aneris_overrides,
        run_checks=run_checks,
        progress=progress,
        n_processes=n_processes,
    )
