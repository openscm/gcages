"""
SCM-running configuration and related things for the updated workflow

SCIJune2026 uses the standard CMIP7ScenarioMIPSCMRunner.
No SCI-specific post-processing implementation is provided here because it
would duplicate the existing CMIP7ScenarioMIPSCMRunner logic.
"""

from gcages.cmip7_scenariomip.scm_running import CMIP7ScenarioMIPSCMRunner

SCIJune2026SCMRunner = CMIP7ScenarioMIPSCMRunner
