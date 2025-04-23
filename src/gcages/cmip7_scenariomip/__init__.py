"""
CMIP7 ScenarioMIP components
"""

from __future__ import annotations

from gcages.cmip7_scenariomip.pre_processing.pre_processor import (
    CMIP7ScenarioMIPPreProcessingResult,
    CMIP7ScenarioMIPPreProcessor,
)

InternalConsistencyError = ValueError

__all__ = [
    "CMIP7ScenarioMIPPreProcessingResult",
    "CMIP7ScenarioMIPPreProcessor",
    "InternalConsistencyError",
]
