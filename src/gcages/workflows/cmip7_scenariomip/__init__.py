"""
The CMIP7 ScenarioMIP workflow

This reproduces the workflow used to create the harmonised and infilled emissions
for the CMIP7 ScenarioMIP marker scenarios (https://zenodo.org/records/19825038),
which was run with https://github.com/iiasa/emissions_harmonization_historical
(commit 376987dbe3b66fcd23537519a16771e21cba6d1e).
"""

from __future__ import annotations

from gcages.workflows.cmip7_scenariomip.extension import (
    CMIP7ScenarioMIPExtender,
)
from gcages.workflows.cmip7_scenariomip.harmonisation import (
    CMIP7ScenarioMIPWorkflowHarmonisedResult,
    CMIP7ScenarioMIPWorkflowHarmoniser,
)
from gcages.workflows.cmip7_scenariomip.inputs import (
    Downloader,
    PoochDownloader,
    ZenodoFile,
)
from gcages.workflows.cmip7_scenariomip.markers import (
    CMIP7_SCENARIOMIP_MARKERS,
    CMIP7ScenarioMIPMarker,
    PublishedClimateAssessment,
    load_published_climate_assessment,
    load_published_emissions,
)
from gcages.workflows.cmip7_scenariomip.post_processing import (
    CMIP7ScenarioMIPWorkflowPostProcessor,
)
from gcages.workflows.cmip7_scenariomip.pre_processing import (
    CMIP7ScenarioMIPWorkflowPreProcessor,
)
from gcages.workflows.cmip7_scenariomip.runner import (
    CMIP7ScenarioMIPWorkflowResult,
    CMIP7ScenarioMIPWorkflowRunner,
)

__all__ = [
    "CMIP7_SCENARIOMIP_MARKERS",
    "CMIP7ScenarioMIPExtender",
    "CMIP7ScenarioMIPMarker",
    "CMIP7ScenarioMIPWorkflowHarmonisedResult",
    "CMIP7ScenarioMIPWorkflowHarmoniser",
    "CMIP7ScenarioMIPWorkflowPostProcessor",
    "CMIP7ScenarioMIPWorkflowPreProcessor",
    "CMIP7ScenarioMIPWorkflowResult",
    "CMIP7ScenarioMIPWorkflowRunner",
    "Downloader",
    "PoochDownloader",
    "PublishedClimateAssessment",
    "ZenodoFile",
    "load_published_climate_assessment",
    "load_published_emissions",
]
