"""
Post-processing in line the CMIP7 ScenarioMIP workflow
"""

from __future__ import annotations

import multiprocessing

import numpy as np

from gcages.ar6.post_processing import (
    AR6PostProcessor,
)


def create_cmip7_scenariomip_postprocessor(
    run_checks: bool = True,
    progress: bool = True,
    n_processes: int | None = multiprocessing.cpu_count(),
) -> AR6PostProcessor:
    """
    Create a post-processor configured for CMIP7 ScenarioMIP

    Parameters
    ----------
    run_checks
        Should checks of the input and output data be performed?

        If this is turned off, things are faster,
        but error messages are much less clear if things go wrong.

    progress
        Should progress bars be shown for each operation?

    n_processes
        Number of processes to use for parallel processing.

        Set to `None` to process serially.


    Returns
    -------
    :
        Initialised post-processor
    """
    res = AR6PostProcessor(
        gsat_assessment_median=0.85,
        gsat_assessment_time_period=tuple(range(1995, 2014 + 1)),
        gsat_assessment_pre_industrial_period=tuple(range(1850, 1900 + 1)),
        quantiles_of_interest=(
            0.05,
            0.10,
            1.0 / 6.0,
            0.33,
            0.5,
            0.67,
            5.0 / 6.0,
            0.90,
            0.95,
        ),
        exceedance_thresholds_of_interest=np.arange(1.0, 4.01, 0.5),
        raw_gsat_variable_in="Surface Air Temperature Change",
        assessed_gsat_variable="Surface Temperature (GSAT)",
        run_checks=run_checks,
        progress=progress,
        n_processes=n_processes,
    )

    return res
