"""
Harmonisation helpers for the CMIP7 ScenarioMIP workflow
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pandas as pd
from attrs import define
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


@define
class SCIHarmoniser:
    """
    Harmoniser that follows the same logic as was used in SCI

    """

    historical_emissions: pd.DataFrame
    """
    Historical emissions to use for harmonisation
    """

    aneris_overrides: pd.Series[str] | None
    """
    Overrides to supply to `aneris.convenience.harmonise_all`

    For source code and docs,
    see e.g. [https://github.com/iiasa/aneris/blob/v0.4.2/src/aneris/convenience.py]().
    """

    harmonisation_year: int = 2023
    """
    Year in which to harmonise
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

    n_processes: int = multiprocessing.cpu_count()
    """
    Number of processes to use for parallel processing.

    Set to 1 to process in serial.
    """

    def __call__(self, in_emissions: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonise

        Parameters
        ----------
        in_emissions
            Emissions to harmonise

        Returns
        -------
        :
            Harmonised emissions
        """
        harmoniser = AnerisHarmoniser(
            historical_emissions=self.historical_emissions,
            aneris_overrides=self.aneris_overrides,
            harmonisation_year=self.harmonisation_year,
            run_checks=self.run_checks,
            progress=self.progress,
            n_processes=self.n_processes,
        )

        harmonised = harmoniser(in_emissions)

        return harmonised

    # TODO: add overrides in line with AR6 without external file?
    @classmethod
    def from_files(  # noqa: PLR0913
        cls,
        historical_emissions_file: Path,
        aneris_overrides_file: Path,
        harmonisation_year: int = 2023,
        run_checks: bool = True,
        progress: bool = True,
        n_processes: int | None = multiprocessing.cpu_count(),
    ) -> SCIHarmoniser:
        """
        Create an Aneris harmoniser configured.

        Parameters
        ----------
        historical_emissions_file
            File containing historical emissions.

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
            Initialised harmoniser
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

        return SCIHarmoniser(
            historical_emissions=historical_emissions,
            aneris_overrides=aneris_overrides,
            harmonisation_year=harmonisation_year,
            run_checks=run_checks,
            progress=progress,
            n_processes=n_processes,
        )
