"""
Infilling configuration and related things for the SCIJune2026 workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pandas_openscm.io import load_timeseries_csv

from gcages.cmip7_scenariomip.infilling import CMIP7ScenarioMIPInfiller
from gcages.exceptions import MissingOptionalDependencyError
from gcages.harmonisation.common import assert_harmonised
from gcages.renaming import SupportedNamingConventions, rename_variables
from gcages.sci_june_2026.harmonisation import (
    load_historical_emissions,
)

if TYPE_CHECKING:
    from pint import UnitRegistry


def create_scijune2026_infiller(  # noqa: PLR0913
    infilling_leader_emissions_file: Path,
    ghg_inversions_file: Path,
    historical_emissions_file: Path,
    harmonisation_year: int = 2023,
    pre_industrial_year: int = 1750,
    run_checks: bool = True,
    ur: UnitRegistry | None = None,
) -> CMIP7ScenarioMIPInfiller:
    """
    Infiller that follows the same logic as was used in SCI

    Parameters
    ----------
    infilling_leader_emissions_file
        Infilling leaders database file for infilling leader emissions.

    ghg_inversions_file
        Green house gasses inversion dataframe file.

    historical_emissions_file
        Historical emissions file used for harmonisation

    harmonisation_year
        Year in which to harmonise

    pre_industrial_year
        Pre-Industrial year

    run_checks
        If `True`, run checks on both input and output data

        If you are sure about your workflow,
        you can disable the checks to speed things up
        (but we don't recommend this unless you really
        are confident about what you're doing).

    ur
        UnitRegistry

    Returns
    -------
    :
    """
    if ur is None:
        try:
            import openscm_units  # noqa: PLC0415

            ur = openscm_units.unit_registry
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "openscm_units",
                requirement="openscm_units",
            ) from exc

    infilling_leader_emissions = load_timeseries_csv(
        infilling_leader_emissions_file,
        lower_column_names=True,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_columns_type=int,
    )
    # CMIP7 GHG inversions
    ghg_inversions = load_timeseries_csv(
        ghg_inversions_file,
        lower_column_names=True,
        index_columns=["model", "scenario", "region", "variable", "unit"],
        out_columns_type=int,
    )
    # History
    historical_emissions = load_historical_emissions(
        historical_emissions_file=historical_emissions_file,
    )
    # Use gcages naming convention.
    infilling_leader_emissions = rename_variables(
        infilling_leader_emissions,
        from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
        to_convention=SupportedNamingConventions.GCAGES,
    )
    ghg_inversions = rename_variables(
        ghg_inversions,
        from_convention=SupportedNamingConventions.OPENSCM_RUNNER,
        to_convention=SupportedNamingConventions.GCAGES,
    )

    if run_checks:
        assert_harmonised(
            infilling_leader_emissions,
            history=historical_emissions.reset_index(
                level=[
                    lvl
                    for lvl in ["model", "scenario"]
                    if lvl in historical_emissions.index.names
                ],
                drop=True,
            ),
            harmonisation_time=harmonisation_year,
            history_unit_level="unit",
            ur=ur,
        )

    # Notes: currently this uses RMSClosest under the hood.
    # That's probably not a bad decision, and avoids the OC-BC decoupling from AR6.
    # Can be investigated more in future in other applications.
    return CMIP7ScenarioMIPInfiller(
        infilling_db=infilling_leader_emissions,
        historical_emissions=historical_emissions,
        cmip7_ghg_inversions=ghg_inversions,
        harmonisation_year=harmonisation_year,
        pre_industrial_year=pre_industrial_year,
        run_checks=run_checks,
        # TODO: add something like this back in
        # progress=progress,
        ur=ur,
    )
