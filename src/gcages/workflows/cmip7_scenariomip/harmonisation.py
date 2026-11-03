"""
Harmonisation for the CMIP7 ScenarioMIP workflow

In CMIP7 ScenarioMIP, emissions were harmonised at two levels:

- region-sector: the emissions used for gridding,
  harmonised at the level of each model region and sector
- global-only: the emissions used for the global workflow,
  harmonised at the global level

The emissions that are infilled (and then used with simple climate models)
are the region-sector harmonised emissions summed to the global level,
except for CO2 AFOLU and any variables that aren't in the region-sector emissions,
for which the global-only harmonised emissions are used.
"""

from __future__ import annotations

import multiprocessing

import numpy as np
import pandas as pd
from attrs import define
from pandas_openscm.indexing import multi_index_match

from gcages.cmip7_scenariomip.gridding_emissions import to_global_workflow_emissions
from gcages.harmonisation import AnerisHarmoniser
from gcages.pandas_indexing_helpers import import_pandas_indexing
from gcages.workflows.cmip7_scenariomip.inputs import (
    convert_cmip7_scenariomip_names_to_gcages,
)

INDEX_LEVELS = ["model", "scenario", "region", "variable", "unit"]
"""Index levels of the data we harmonise"""

CO2_FOSSIL_NAME = "Emissions|CO2|Energy and Industrial Processes"
"""Name of fossil CO2 emissions in the global-only emissions"""

CO2_AFOLU_NAME = "Emissions|CO2|AFOLU"
"""Name of CO2 AFOLU emissions in the global-only emissions"""


@define
class CMIP7ScenarioMIPWorkflowHarmonisedResult:
    """
    Result of harmonisation in the CMIP7 ScenarioMIP workflow
    """

    region_sector: pd.DataFrame
    """
    Results of region-sector harmonisation

    These use CMIP7 ScenarioMIP's naming convention.
    """

    global_only: pd.DataFrame
    """
    Results of global-only harmonisation

    These use CMIP7 ScenarioMIP's naming convention.
    """

    for_infilling: pd.DataFrame
    """
    Emissions that are passed to the infilling step

    These use gcages' naming convention.
    """


def interpolate_to_annual(
    df: pd.DataFrame, start_year: int, end_year: int = 2100
) -> pd.DataFrame:
    """
    Interpolate to annual values between `start_year` and `end_year`

    Parameters
    ----------
    df
        Data to interpolate

    start_year
        First year for which to have annual values

    end_year
        Last year for which to have annual values

    Returns
    -------
    :
        Interpolated data
    """
    res = df.copy()
    for y in range(start_year, end_year + 1):
        if y not in res:
            res[y] = np.nan

    res = res.sort_index(axis="columns")
    res = res.T.interpolate(method="index").T

    return res


def assert_all_timeseries_have_methods(
    df: pd.DataFrame, methods: pd.Series[str], name: str
) -> None:
    """
    Assert that a harmonisation method is defined for all timeseries

    Parameters
    ----------
    df
        Data to check

    Methods
    -------
        Harmonisation methods

    name
        Name of the data to use in error messages

    Raises
    ------
    ValueError
        A method is not defined for some timeseries in `df`
    """
    df_methods_index = df.index.droplevel(
        df.index.names.difference(methods.index.names)  # type: ignore # pandas-stubs confused
    ).reorder_levels(methods.index.names)
    missing = df_methods_index[~df_methods_index.isin(methods.index)]
    if not missing.empty:
        msg = (
            f"No harmonisation method for these timeseries in the {name} emissions. "
            "We only have the methods used for the scenarios "
            "that were run in CMIP7 ScenarioMIP. "
            f"Missing:\n{missing.to_frame(index=False)}"
        )
        raise ValueError(msg)


def set_hist_zero_to_zero(
    harmonised: pd.DataFrame, methods: pd.Series[str]
) -> pd.DataFrame:
    """
    Set timeseries harmonised with the 'hist_zero' method to zero

    Parameters
    ----------
    harmonised
        Harmonised data

    Methods
    -------
        Harmonisation methods

    Returns
    -------
    :
        Harmonised data with timeseries harmonised with 'hist_zero' set to zero
    """
    hist_zero = methods[methods == "hist_zero"].index
    if (
        hist_zero.get_level_values("variable")
        .unique()
        .str.startswith("Carbon Removal")
        .any()
    ):
        msg = "'hist_zero' should not be used for any CDR variables"
        raise AssertionError(msg)

    res = harmonised.copy()
    res.loc[multi_index_match(res.index, hist_zero), :] = 0.0  # type: ignore # pandas-stubs confused

    return res


def combine_for_infilling(
    region_sector: pd.DataFrame, global_only: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine the harmonised emissions into the emissions to infill

    We use the region-sector emissions summed to the global level where we can,
    except for CO2 AFOLU.
    We use the global-only emissions otherwise.

    Parameters
    ----------
    region_sector
        Results of region-sector harmonisation

    global_only
        Results of global-only harmonisation

    Returns
    -------
    :
        Emissions to infill (using CMIP7 ScenarioMIP's naming convention)
    """
    pix = import_pandas_indexing("combine_for_infilling")

    from_region_sector = to_global_workflow_emissions(
        region_sector,
        global_workflow_co2_fossil_sector="Energy and Industrial Processes",
        global_workflow_co2_biosphere_sector="AFOLU",
    )
    from_region_sector = from_region_sector.loc[~pix.isin(variable=CO2_AFOLU_NAME)]
    # Include CDR in fossil CO2 emissions
    from_region_sector = pix.aggregatelevel(
        from_region_sector,
        variable={CO2_FOSSIL_NAME: [CO2_FOSSIL_NAME, "Carbon Removal|CO2"]},
    ).sort_index()

    from_global_only = global_only.loc[
        ~pix.isin(variable=pix.uniquelevel(from_region_sector, "variable"))
    ]

    res: pd.DataFrame = pix.concat([from_region_sector, from_global_only])
    res.columns.name = "year"

    return res


@define
class CMIP7ScenarioMIPWorkflowHarmoniser:
    """
    Harmoniser for the CMIP7 ScenarioMIP workflow

    The harmonisation is done with
    [AnerisHarmoniser][gcages.harmonisation.AnerisHarmoniser],
    using the given harmonisation methods.
    """

    historical_emissions: pd.DataFrame
    """
    Historical emissions to use for harmonisation

    These should use CMIP7 ScenarioMIP's naming convention
    and include the history at both the region-sector and global level.
    """

    region_sector_methods: pd.Series[str]
    """
    Harmonisation methods to use for the region-sector harmonisation

    The method must be defined for every timeseries
    (so that we know which timeseries were harmonised with 'hist_zero').
    """

    global_methods: pd.Series[str]
    """
    Harmonisation methods to use for the global-only harmonisation

    The method must be defined for every timeseries.
    """

    harmonisation_year: int
    """Year in which to harmonise"""

    run_checks: bool = True
    """
    If `True`, run checks on both input and output data

    For the region-sector harmonisation, we only check
    that a harmonisation method is defined for every timeseries,
    because some timeseries are deliberately not harmonised
    (e.g. those that use the methods 'model_zero' and 'hist_zero').
    """

    progress: bool = True
    """Should progress bars be shown?"""

    n_processes: int | None = multiprocessing.cpu_count()
    """
    Number of processes to use for parallel processing.

    Set to `None` to process in serial.
    """

    def harmonise(
        self, emissions: pd.DataFrame, methods: pd.Series[str], run_checks: bool
    ) -> pd.DataFrame:
        """
        Harmonise using the given methods

        Parameters
        ----------
        emissions
            Emissions to harmonise

        Methods
        -------
            Harmonisation methods to use

        run_checks
            Should [AnerisHarmoniser][gcages.harmonisation.AnerisHarmoniser]
            run its checks?

        Returns
        -------
        :
            Harmonised emissions
        """
        # Only pass the history for the timeseries in `emissions`
        # (aneris raises an error if there is history
        # for any other combinations of variable and region).
        history = self.historical_emissions.loc[
            multi_index_match(
                self.historical_emissions.index,  # type: ignore # pandas-stubs confused
                emissions.index.droplevel(
                    emissions.index.names.difference(["variable", "region"])  # type: ignore # pandas-stubs confused
                ).unique(),
            )
        ]

        harmoniser = AnerisHarmoniser(
            historical_emissions=history,
            harmonisation_year=self.harmonisation_year,
            aneris_overrides=methods,
            run_checks=run_checks,
            progress=self.progress,
            n_processes=self.n_processes,
        )

        res = harmoniser(emissions)
        res.columns.name = "year"

        return res

    def __call__(
        self,
        region_sector_emissions: pd.DataFrame,
        global_emissions: pd.DataFrame,
    ) -> CMIP7ScenarioMIPWorkflowHarmonisedResult:
        """
        Harmonise

        Parameters
        ----------
        region_sector_emissions
            Emissions to harmonise at the region-sector level

            These should use CMIP7 ScenarioMIP's naming convention.

        global_emissions
            Emissions to harmonise at the global level

            These should use CMIP7 ScenarioMIP's naming convention.

        Returns
        -------
        :
            Harmonised emissions
        """
        # Interpolate to ensure there are no NaNs in the future
        region_sector_in = interpolate_to_annual(
            region_sector_emissions.reorder_levels(INDEX_LEVELS),
            start_year=self.harmonisation_year,
        )
        global_in = interpolate_to_annual(
            global_emissions.reorder_levels(INDEX_LEVELS),
            start_year=self.harmonisation_year,
        )

        if self.run_checks:
            assert_all_timeseries_have_methods(
                region_sector_in, self.region_sector_methods, name="region-sector"
            )
            assert_all_timeseries_have_methods(
                global_in, self.global_methods, name="global"
            )

        region_sector = set_hist_zero_to_zero(
            self.harmonise(
                region_sector_in, methods=self.region_sector_methods, run_checks=False
            ),
            methods=self.region_sector_methods,
        )

        global_only = self.harmonise(
            global_in, methods=self.global_methods, run_checks=self.run_checks
        )

        for_infilling = convert_cmip7_scenariomip_names_to_gcages(
            combine_for_infilling(region_sector=region_sector, global_only=global_only)
        )

        return CMIP7ScenarioMIPWorkflowHarmonisedResult(
            region_sector=region_sector,
            global_only=global_only,
            for_infilling=for_infilling,
        )
