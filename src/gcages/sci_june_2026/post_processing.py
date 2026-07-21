"""
SCIJune2026 post-processing.

SCIJune2026 uses the standard CMIP7ScenarioMIPPostProcessor for post-processing.
No SCI-specific post-processing implementation is provided here because it
would duplicate the existing CMIP7ScenarioMIPPostProcessor logic.
"""

from gcages.cmip7_scenariomip.post_processing import CMIP7ScenarioMIPPostProcessor

SCIJune2026PostProcessor = CMIP7ScenarioMIPPostProcessor
