"""
Post-processing of simple climate model output for the CMIP7 ScenarioMIP workflow
"""

from __future__ import annotations

import pandas as pd
from attrs import define, field
from pandas_openscm.grouping import (
    fix_index_name_after_groupby_quantile,
    groupby_except,
)

from gcages.cmip7_scenariomip.post_processing import CMIP7ScenarioMIPPostProcessor
from gcages.post_processing import PostProcessingResult

GSAT_VARIABLE = "Surface Air Temperature Change"
"""Name of the GSAT variable in the simple climate model output"""

ERF_VARIABLES = (
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
    "Effective Radiative Forcing|Ozone",
    "Effective Radiative Forcing|Tropospheric Ozone",
    "Effective Radiative Forcing|Stratospheric Ozone",
    "Effective Radiative Forcing|Solar",
    "Effective Radiative Forcing|Volcanic",
)
"""Effective radiative forcing variables reported in CMIP7 ScenarioMIP"""

SCM_OUTPUT_VARIABLES = (GSAT_VARIABLE, *ERF_VARIABLES)
"""Variables to get from the simple climate model"""


@define
class CMIP7ScenarioMIPWorkflowPostProcessor:
    """
    Post-processor for the CMIP7 ScenarioMIP workflow

    This uses
    [CMIP7ScenarioMIPPostProcessor][gcages.cmip7_scenariomip.post_processing.CMIP7ScenarioMIPPostProcessor]
    to post-process GSAT
    (assessed warming, peak warming, warming in 2100, exceedance probabilities etc.)
    and adds the quantiles of the effective radiative forcing variables
    to the timeseries quantiles.
    """

    gsat_post_processor: CMIP7ScenarioMIPPostProcessor = field(
        factory=CMIP7ScenarioMIPPostProcessor.from_cmip7_scenariomip_config
    )
    """Post-processor to use for GSAT"""

    erf_variables: tuple[str, ...] = ERF_VARIABLES
    """Effective radiative forcing variables of which to calculate quantiles"""

    last_year: int | None = 2100
    """
    Last year to consider in the post-processing

    In CMIP7 ScenarioMIP, the simple climate model was run to 2500
    (with the extended emissions)
    but the reported metrics (e.g. peak warming, exceedance probabilities)
    only consider the years up to 2100.
    If `None`, all years are considered.
    """

    def __call__(self, scm_results: pd.DataFrame) -> PostProcessingResult:
        """
        Post-process

        Parameters
        ----------
        scm_results
            Raw results from the simple climate model

        Returns
        -------
        :
            Post-processed results
        """
        if self.last_year is not None:
            scm_results = scm_results.loc[:, : self.last_year]

        gsat_res = self.gsat_post_processor(scm_results)

        erf = scm_results.loc[
            scm_results.index.get_level_values("variable").isin(self.erf_variables)
        ]
        erf_quantiles = fix_index_name_after_groupby_quantile(
            groupby_except(erf, "run_id").quantile(
                list(self.gsat_post_processor.percentiles_to_calculate)  # type: ignore # pandas-stubs confused
            ),
            new_name="quantile",
        )

        return PostProcessingResult(
            timeseries_run_id=gsat_res.timeseries_run_id,
            timeseries_quantile=pd.concat(
                [
                    gsat_res.timeseries_quantile,
                    erf_quantiles.reorder_levels(
                        gsat_res.timeseries_quantile.index.names
                    ),
                ]
            ),
            timeseries_exceedance_probabilities=gsat_res.timeseries_exceedance_probabilities,
            metadata_run_id=gsat_res.metadata_run_id,
            metadata_quantile=gsat_res.metadata_quantile,
            metadata_exceedance_probabilities=gsat_res.metadata_exceedance_probabilities,
            metadata_categories=gsat_res.metadata_categories,
        )
