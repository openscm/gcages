"""
Pre-processing part of the workflow
"""

from __future__ import annotations

import multiprocessing

import pandas as pd
from attrs import define


@define
class CMIP7ScenarioMIPPreProcessor:
    """
    Pre-processor for CMIP7's ScenarioMIP
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
