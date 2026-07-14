"""
Pre-processing part of the SCI workflow
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Mapping
from functools import partial

import pandas as pd
from attrs import define
from numpy import nan

from gcages.ar6.pre_processing import reclassify_variables, run_parallel_pre_processing
from gcages.assertions import (
    assert_data_is_all_numeric,
    assert_has_index_levels,
    assert_index_is_multiindex,
)
from gcages.exceptions import MissingOptionalDependencyError
from gcages.renaming import (
    SupportedNamingConventions,
    rename_variables,
)
from gcages.units_helpers import strip_pint_incompatible_characters_from_units


@define
class SCIJune2026PreProcessor:
    """
    Pre-processor that follows the same logic as was used in SCI

    """

    emissions_out: tuple[str, ...]
    """
    Names of emissions that can be included in the result of pre-processing

    Not all these emissions need to be there,
    but any names which are not in this list will be removed as part of pre-processing.
    """

    reclassifications: Mapping[str, tuple[str, ...]] | None = None
    """
    Variables that should be reclassified as being part of another variable

    Form:

    ```python
    {
        variable_to_add_to: (variable_to_rename_1, variable_to_rename_2),
        ...
    }
    ```

    For example
    ```python
    {
        "Emissions|CO2|Energy and Industrial Processes": (
            "Emissions|CO2|Other",
            "Emissions|CO2|Waste",
        )
    }
    ```
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

    def __call__(self, in_emissions: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-process

        Parameters
        ----------
        in_emissions
            Emissions to pre-process

        Returns
        -------
        :
            Pre-processed emissions
        """
        try:
            from pandas_indexing.selectors import isin, ismatch  # noqa: PLC0415
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "SCIPreProcessor.__call__", requirement="pandas_indexing"
            ) from exc

        if self.run_checks:
            assert_index_is_multiindex(in_emissions)
            assert_data_is_all_numeric(in_emissions)
            assert_has_index_levels(in_emissions, ["variable", "unit"])

        rp = partial(
            run_parallel_pre_processing,
            progress=self.progress,
            n_processes=self.n_processes,
        )

        # Negative values checking
        co2_locator = ismatch(variable="**CO2**")
        non_co2 = in_emissions.loc[~co2_locator]
        keep_condition = (non_co2 >= 0.0) | non_co2.isnull()
        rows_to_drop = ~keep_condition.all(axis=1)

        if rows_to_drop.any():
            negatives = (
                non_co2.loc[rows_to_drop]
                .reset_index()[["model", "scenario", "region", "variable"]]
                .to_string(index=False)
            )
            msg = f"Below threshold values found in: \n {negatives}"
            raise ValueError(msg)

        if self.reclassifications is not None:
            # TODO: switch to using something like the ScenarioMIP reaggregator.
            # The reason we want to use this is that we want to check reporting
            # consistency and re-aggregation together,
            # so we can give sensible error messages.
            in_emissions = rp(  # type: ignore
                in_emissions,
                func_to_call=reclassify_variables,
                progress_bar_desc="For each model-scenario, reclassifying variables",
                reclassifications=self.reclassifications,
            )

        res: pd.DataFrame = in_emissions.loc[isin(variable=self.emissions_out)]

        # Interpolate to annual steps
        for y in range(2010, 2100 + 1):
            if y not in res:
                res.loc[:, y] = nan

        res = res.T.interpolate(method="index").T.sort_index(axis="columns")

        # Convert to gcages naming conventions
        res = rename_variables(
            res,
            from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
            to_convention=SupportedNamingConventions.GCAGES,
        )
        res = strip_pint_incompatible_characters_from_units(
            res, units_index_level="unit"
        )

        return res

    @classmethod
    def from_standard_config(
        cls,
        run_checks: bool = True,
        progress: bool = True,
        n_processes: int | None = multiprocessing.cpu_count(),
    ) -> SCIJune2026PreProcessor:
        """
        Initialise from config that was used in AR6

        Parameters
        ----------
        run_checks
            Should checks of the input and output data be performed?

            If this is turned off, things are faster,
            but error messages are much less clear if things go wrong.

        progress
            Should a progress bar be shown for each operation?

        n_processes
            Number of processes to use for parallel processing.

            Set to `None` to process in serial.

        Returns
        -------
        :
            Initialised Pre-processor
        """
        # In the CMIP7_SCENARIOMIP naming convention,
        # i.e. what we expect to receive as input
        emissions_out = (
            "Emissions|BC",
            "Emissions|C2F6",
            "Emissions|C6F14",
            "Emissions|CF4",
            "Emissions|CO",
            # "Emissions|CO2", # Not used
            "Emissions|CO2|AFOLU",
            "Emissions|CO2|Energy and Industrial Processes",
            "Emissions|CH4",
            # "Emissions|F-Gases",  # Not used
            # "Emissions|HFC",  # Not used
            "Emissions|HFC|HFC125",
            "Emissions|HFC|HFC134a",
            "Emissions|HFC|HFC143a",
            "Emissions|HFC|HFC227ea",
            "Emissions|HFC|HFC23",
            "Emissions|HFC|HFC245fa",
            "Emissions|HFC|HFC32",
            "Emissions|HFC|HFC43-10",
            # "Emissions|Kyoto Gases",  # Not used
            "Emissions|N2O",
            "Emissions|NH3",
            "Emissions|NOx",
            "Emissions|OC",
            # "Emissions|PFC",  # Not used
            "Emissions|SF6",
            "Emissions|Sulfur",
            "Emissions|VOC",
        )
        reclassifications = {
            "Emissions|CO2|Energy and Industrial Processes": (
                "Emissions|CO2|Other",
                "Emissions|CO2|Waste",
                "Emissions|CO2|Other Capture and Removal",
                "Emissions|CO2|Product Use",
            )
        }

        return cls(
            emissions_out=emissions_out,
            reclassifications=reclassifications,
            run_checks=run_checks,
            n_processes=n_processes,
            progress=progress,
        )
