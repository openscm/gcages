"""
Runner of the CMIP7 ScenarioMIP workflow
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import pandas as pd
from attrs import define

from gcages.cmip7_scenariomip.infilling import CMIP7ScenarioMIPInfiller
from gcages.cmip7_scenariomip.pre_processing import CMIP7ScenarioMIPPreProcessingResult
from gcages.cmip7_scenariomip.scm_running import (
    CMIP7ScenarioMIPSCMRunner,
    check_cmip7_scenariomip_magicc7_version,
    load_magicc_cfgs,
)
from gcages.post_processing import PostProcessingResult
from gcages.workflows.cmip7_scenariomip.extension import CMIP7ScenarioMIPExtender
from gcages.workflows.cmip7_scenariomip.harmonisation import (
    CMIP7ScenarioMIPWorkflowHarmonisedResult,
    CMIP7ScenarioMIPWorkflowHarmoniser,
)
from gcages.workflows.cmip7_scenariomip.inputs import (
    GLOBAL_HISTORY_FILE,
    INFILLING_DB_FILE,
    REGION_SECTOR_HISTORY_FILE,
    Downloader,
    PoochDownloader,
    convert_cmip7_scenariomip_names_to_gcages,
    load_cmip7_ghg_inversions,
    load_harmonisation_methods,
    load_parquet,
)
from gcages.workflows.cmip7_scenariomip.post_processing import (
    SCM_OUTPUT_VARIABLES,
    CMIP7ScenarioMIPWorkflowPostProcessor,
)
from gcages.workflows.cmip7_scenariomip.pre_processing import (
    CMIP7ScenarioMIPWorkflowPreProcessor,
)

HARMONISATION_YEAR = 2023
"""Year in which emissions were harmonised in CMIP7 ScenarioMIP"""

PRE_INDUSTRIAL_YEAR = 1750
"""Pre-industrial year used in CMIP7 ScenarioMIP"""


@define
class CMIP7ScenarioMIPWorkflowResult:
    """
    Result of running the CMIP7 ScenarioMIP workflow
    """

    raw: pd.DataFrame
    """Raw input"""

    pre_processed: CMIP7ScenarioMIPPreProcessingResult
    """Pre-processed data"""

    harmonised: CMIP7ScenarioMIPWorkflowHarmonisedResult
    """Harmonised data"""

    infilled: pd.DataFrame
    """Infilled data"""

    extended: pd.DataFrame | None = None
    """
    Infilled data, extended beyond 2100

    `None` if the runner doesn't extend the data.
    """

    scm_results: pd.DataFrame | None = None
    """
    Raw results of running the simple climate model

    `None` if the runner doesn't run a simple climate model.
    """

    post_processed: PostProcessingResult | None = None
    """
    Post-processed results of running the simple climate model

    `None` if the runner doesn't run a simple climate model.
    """


@define
class CMIP7ScenarioMIPWorkflowRunner:
    """
    Runner of the entire CMIP7 ScenarioMIP workflow

    If you want exactly the same behaviour as in CMIP7 ScenarioMIP,
    initialise using [`from_cmip7_scenariomip_config`][(c)].
    """

    pre_processor: CMIP7ScenarioMIPWorkflowPreProcessor
    """Pre-processor"""

    harmoniser: CMIP7ScenarioMIPWorkflowHarmoniser
    """Harmoniser"""

    infiller: CMIP7ScenarioMIPInfiller
    """Infiller"""

    extender: CMIP7ScenarioMIPExtender | None = None
    """
    Extender of the infilled data beyond 2100

    If `None`, the data is not extended.
    """

    scm_runner: CMIP7ScenarioMIPSCMRunner | None = None
    """
    Simple climate model runner

    It is run with the extended data if `extender` is supplied,
    otherwise with the infilled data.
    If `None`, no simple climate model is run.
    """

    post_processor: CMIP7ScenarioMIPWorkflowPostProcessor | None = None
    """
    Post-processor of the simple climate model results

    Only used if `scm_runner` is supplied.
    """

    def __call__(self, in_emissions: pd.DataFrame) -> CMIP7ScenarioMIPWorkflowResult:
        """
        Run the workflow

        Parameters
        ----------
        in_emissions
            Raw emissions

        Returns
        -------
        :
            Results of each step of the workflow
        """
        pre_processed = self.pre_processor(in_emissions)

        harmonised = self.harmoniser(
            region_sector_emissions=pre_processed.gridding_workflow_emissions,
            global_emissions=pre_processed.global_workflow_emissions_raw_names,
        )

        infilled = self.infiller(harmonised.for_infilling)

        extended = (
            self.extender(infilled, harmonised) if self.extender is not None else None
        )

        scm_results = None
        post_processed = None
        if self.scm_runner is not None:
            scm_results = self.scm_runner(
                extended if extended is not None else infilled
            )

            if self.post_processor is not None:
                post_processed = self.post_processor(scm_results)

        return CMIP7ScenarioMIPWorkflowResult(
            raw=in_emissions,
            pre_processed=pre_processed,
            harmonised=harmonised,
            infilled=infilled,
            extended=extended,
            scm_results=scm_results,
            post_processed=post_processed,
        )

    @classmethod
    def from_cmip7_scenariomip_config(  # noqa: PLR0913
        cls,
        downloader: Downloader | None = None,
        magicc_exe_path: Path | None = None,
        magicc_prob_distribution_path: Path | None = None,
        run_checks: bool = True,
        progress: bool = True,
        n_processes: int | None = multiprocessing.cpu_count(),
    ) -> CMIP7ScenarioMIPWorkflowRunner:
        """
        Initialise with the configuration used in CMIP7 ScenarioMIP

        This reproduces v0.2 of the harmonised and infilled emissions
        and climate outcomes for the CMIP7 ScenarioMIP marker scenarios
        (https://zenodo.org/records/19825038).
        The harmonisation methods and extension configuration
        are the ones that were used for these scenarios,
        so only these scenarios are supported.

        Parameters
        ----------
        downloader
            Downloader to use to get input data from Zenodo

            If not supplied, we use [PoochDownloader][gcages.workflows.cmip7_scenariomip.inputs.PoochDownloader]
            with its default cache directory.

        magicc_exe_path
            Path to the MAGICC executable to use

            This should be a MAGICC v7.6.0a3 executable.
            If not supplied, MAGICC is not run
            (and the workflow stops after extending the emissions).

        magicc_prob_distribution_path
            Path to the MAGICC probabilistic distribution to use

            This should be the CMIP7 ScenarioMIP probabilistic distribution
            (`magicc-ar7-fast-track-drawnset-v0-3-0.json`).
            Must be supplied if `magicc_exe_path` is supplied.

        run_checks
            Should checks of the input and output data be performed?

            If this is turned off, things are faster,
            but error messages are much less clear if things go wrong.

        progress
            Should progress bars be shown?

        n_processes
            Number of processes to use for parallel processing
            during harmonisation and when running MAGICC

            (Pre-processing is always done in serial,
            see [CMIP7ScenarioMIPWorkflowPreProcessor][gcages.workflows.cmip7_scenariomip.CMIP7ScenarioMIPWorkflowPreProcessor].)

            Set to `None` to process in serial.

        Returns
        -------
        :
            Initialised runner

        Raises
        ------
        ValueError
            Only one of `magicc_exe_path` and `magicc_prob_distribution_path`
            is supplied
        """  # noqa: E501
        if (magicc_exe_path is None) != (magicc_prob_distribution_path is None):
            msg = (
                "Either supply both `magicc_exe_path` "
                "and `magicc_prob_distribution_path` or neither. "
                f"Received {magicc_exe_path=} and {magicc_prob_distribution_path=}"
            )
            raise ValueError(msg)

        if downloader is None:
            downloader = PoochDownloader()

        global_history = load_parquet(GLOBAL_HISTORY_FILE, downloader=downloader)
        region_sector_history = load_parquet(
            REGION_SECTOR_HISTORY_FILE, downloader=downloader
        )

        history_for_harmonisation = pd.concat([region_sector_history, global_history])
        # aneris explodes if any history year is NaN,
        # even ones we don't use
        history_for_harmonisation = history_for_harmonisation.dropna(axis="columns")
        history_for_harmonisation = history_for_harmonisation.reset_index(
            ["model", "scenario"], drop=True
        )

        harmoniser = CMIP7ScenarioMIPWorkflowHarmoniser(
            historical_emissions=history_for_harmonisation,
            region_sector_methods=load_harmonisation_methods(
                "harmonisation-methods_region-sector.csv"
            ),
            global_methods=load_harmonisation_methods(
                "harmonisation-methods_global.csv"
            ),
            harmonisation_year=HARMONISATION_YEAR,
            run_checks=run_checks,
            progress=progress,
            n_processes=n_processes,
        )

        global_history_gcages = convert_cmip7_scenariomip_names_to_gcages(
            global_history
        )
        infiller = CMIP7ScenarioMIPInfiller(
            infilling_db=convert_cmip7_scenariomip_names_to_gcages(
                load_parquet(INFILLING_DB_FILE, downloader=downloader)
            ),
            cmip7_ghg_inversions=load_cmip7_ghg_inversions(),
            historical_emissions=global_history_gcages,
            harmonisation_year=HARMONISATION_YEAR,
            pre_industrial_year=PRE_INDUSTRIAL_YEAR,
            run_checks=run_checks,
        )

        extender = CMIP7ScenarioMIPExtender(historical_emissions=global_history)

        if magicc_exe_path is None or magicc_prob_distribution_path is None:
            scm_runner = None
            post_processor = None

        else:
            os.environ["MAGICC_EXECUTABLE_7"] = str(magicc_exe_path)
            check_cmip7_scenariomip_magicc7_version()

            scm_runner = CMIP7ScenarioMIPSCMRunner(
                climate_models_cfgs=load_magicc_cfgs(
                    prob_distribution_path=magicc_prob_distribution_path,
                    output_variables=SCM_OUTPUT_VARIABLES,
                    startyear=1750,
                ),
                output_variables=SCM_OUTPUT_VARIABLES,
                historical_emissions=global_history_gcages.reset_index(
                    ["model", "scenario"], drop=True
                ),
                harmonisation_year=HARMONISATION_YEAR,
                verbose=False,
                run_checks=run_checks,
                progress=progress,
                n_processes=n_processes,
            )
            post_processor = CMIP7ScenarioMIPWorkflowPostProcessor()

        return cls(
            pre_processor=CMIP7ScenarioMIPWorkflowPreProcessor(progress=progress),
            harmoniser=harmoniser,
            infiller=infiller,
            extender=extender,
            scm_runner=scm_runner,
            post_processor=post_processor,
        )
