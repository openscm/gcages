"""
Infilling configuration and related things for the CMIP7 ScenarioMIP workflow
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from attrs import define
from pandas_openscm.io import load_timeseries_csv

from gcages.cmip7_scenariomip.infilling import CMIP7ScenarioMIPInfiller
from gcages.exceptions import MissingOptionalDependencyError
from gcages.harmonisation.common import assert_harmonised
from gcages.renaming import SupportedNamingConventions, rename_variables
from gcages.sci_june_2026.harmonisation import load_historical_emissions

if TYPE_CHECKING:
    from pint import UnitRegistry


@define
class SCIJune2026Infiller:
    """
    Infiller that follows the same logic as was used in SCI

    """

    infilling_db: pd.DataFrame
    """
    Infilling leaders data base for each variable.
    """

    ghg_inversions: pd.DataFrame
    """
    Green house gasses inversion data frame.
    """

    historical_emissions: pd.DataFrame
    """
    Historical emissions used for harmonisation
    """
    harmonisation_year: int = 2023
    """
    Year in which the data was harmonised
    """

    pre_industrial_year: int = 1750
    """
    Pre-Industrial year
    """
    run_checks: bool = True
    """
    If `True`, run checks on both input and output data

    If you are sure about your workflow,
    you can disable the checks to speed things up
    (but we don't recommend this unless you really
    are confident about what you're doing).
    """

    ur: UnitRegistry | None = None
    """
    UnitRegistry
    """

    def __call__(self, in_emissions: pd.DataFrame) -> pd.DataFrame:
        """
        Create an a infilled df for SCI simple climate model run.

        Parameters
        ----------
        in_emissions
            Emissions to infill

        Returns
        -------
        :
            Infilled emissions DataFrame
        """
        if self.ur is None:
            try:
                import openscm_units

                self.ur = openscm_units.unit_registry
            except ImportError as exc:
                raise MissingOptionalDependencyError(
                    "openscm_units",
                    requirement="openscm_units",
                ) from exc

        if self.run_checks:
            assert_harmonised(
                self.infilling_db,
                history=self.historical_emissions.reset_index(
                    level=[
                        lvl
                        for lvl in ["model", "scenario"]
                        if lvl in self.historical_emissions.index.names
                    ],
                    drop=True,
                ),
                harmonisation_time=self.harmonisation_year,
                history_unit_level="unit",
                ur=self.ur,
            )

        # Notes: currently this uses RMSClosest under the hood.
        # That's probably not a bad decision, and avoids the OC-BC decoupling from AR6,
        # so definitely good for this SCI paper.
        # Can be investigated more in future in other applications.
        infiller = CMIP7ScenarioMIPInfiller(
            infilling_db=self.infilling_db,
            historical_emissions=self.historical_emissions,
            cmip7_ghg_inversions=self.ghg_inversions,
            harmonisation_year=self.harmonisation_year,
            pre_industrial_year=self.pre_industrial_year,
            run_checks=True,
            # TODO: add something like this back in
            # progress=True,
            ur=self.ur,
        )

        complete = infiller(in_emissions)

        return complete

    @classmethod
    def from_files(  # noqa: PLR0913
        cls,
        infilling_leader_emissions_file: Path,
        ghg_inversions_file: Path,
        historical_emissions_file: Path,
        pi_year: int = 1750,
        harmonisation_year: int = 2023,
        ur: UnitRegistry | None = None,
        run_checks: bool = True,
    ) -> SCIJune2026Infiller:
        """
        Initialise from the config used in AR6

        Parameters
        ----------
        infilling_leader_emissions_file
            File containing the infilling leaders database

            This is for all emissions except GHGs.

        ghg_inversions_file
            File containing the infilling database for GHGs inversions

        historical_emissions_file
            File containing the historical emissions used for harmonisation

        pi_year
            Pre-Industrial Year

        harmonisation_year
            Harmonisation Year

        run_checks
            Should checks of the input and output data be performed?

            If this is turned off, things are faster,
            but error messages are much less clear if things go wrong.

        Returns
        -------
        :
            Initialised CMIP7ScenarioMIPInfiller
        """
        if ur is None:
            try:
                import openscm_units

                ur = openscm_units.unit_registry
            except ImportError as exc:
                raise MissingOptionalDependencyError(
                    "openscm_units",
                    requirement="openscm_units",
                ) from exc

        infilling_db = load_timeseries_csv(
            infilling_leader_emissions_file,
            lower_column_names=True,
            index_columns=["model", "scenario", "region", "variable", "unit"],
            out_columns_type=int,
            # out_columns_name="year",
        )

        # CMIP7 GHG inversions
        ghg_inversions = load_timeseries_csv(
            ghg_inversions_file,
            lower_column_names=True,
            index_columns=["model", "scenario", "region", "variable", "unit"],
            out_columns_type=int,
            # out_columns_name="year",
        )
        # History
        historical_emissions = load_historical_emissions(
            historical_emissions_file=historical_emissions_file
        )

        # Use gcages naming convention.
        # infilling_db = update_index_levels_func(
        #     infilling_db,
        #     {
        #         "variable": lambda x: convert_variable_name(
        #             x,
        #             from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        #             to_convention=SupportedNamingConventions.GCAGES,
        #         )
        #     },
        #     copy=False,
        # )
        infilling_db = rename_variables(
            infilling_db,
            from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
            to_convention=SupportedNamingConventions.GCAGES,
        )

        ghg_inversions = rename_variables(
            ghg_inversions,
            from_convention=SupportedNamingConventions.OPENSCM_RUNNER,
            to_convention=SupportedNamingConventions.GCAGES,
        )

        # historical_emissions = rename_variables(
        #     historical_emissions,
        #     from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        #     to_convention=SupportedNamingConventions.GCAGES,
        # )

        return cls(
            infilling_db=infilling_db,
            historical_emissions=historical_emissions,
            ghg_inversions=ghg_inversions,
            harmonisation_year=harmonisation_year,
            pre_industrial_year=pi_year,
            run_checks=run_checks,
            ur=ur,
        )
