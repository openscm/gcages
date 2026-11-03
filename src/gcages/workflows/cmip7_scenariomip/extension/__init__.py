"""
Extension of emissions beyond 2100, as done in CMIP7 ScenarioMIP

This is a port of the extension logic in
https://github.com/iiasa/emissions_harmonization_historical
(commit 376987dbe3b66fcd23537519a16771e21cba6d1e),
which is based on FLEX (https://github.com/benmsanderson/FLEX, MIT licence).
"""

from __future__ import annotations

from gcages.workflows.cmip7_scenariomip.extension.extender import (
    CMIP7_SCENARIOMIP_EXTENSION_CONFIGS,
    CMIP7ScenarioMIPExtender,
    CMIP7ScenarioMIPExtensionConfig,
)

__all__ = [
    "CMIP7_SCENARIOMIP_EXTENSION_CONFIGS",
    "CMIP7ScenarioMIPExtender",
    "CMIP7ScenarioMIPExtensionConfig",
]
