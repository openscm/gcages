"""
CMIP7 ScenarioMIP components
"""

from __future__ import annotations

from gcages.cmip7_scenariomip.harmonisation import (
    create_cmip7_scenariomip_country_harmoniser,
    create_cmip7_scenariomip_global_harmoniser,
    load_aneris_overrides_file,
    load_cmip7_scenariomip_country_historical_emissions,
    # load_cmip7_scenariomip_global_historical_emissions,
)

# from gcages.cmip7_scenariomip.harmonisation_country import (
#     create_cmip7_scenariomip_country_harmoniser,
#     load_cmip7_scenariomip_country_historical_emissions,
# )
from gcages.cmip7_scenariomip.infilling import (
    create_cmip7_scenariomip_infilled_df,
)
from gcages.cmip7_scenariomip.pre_processing import (
    CMIP7ScenarioMIPPreProcessingResult,
    CMIP7ScenarioMIPPreProcessor,
    ReaggregatorBasic,
    ReaggregatorLike,
)
from gcages.cmip7_scenariomip.scm_running_aux import (
    CMIP7_SCENARIOMIP_SCMRunner,
    get_complete_scenarios_for_magicc,
    load_magicc_cfgs,
)

__all__ = [
    "CMIP7ScenarioMIPPreProcessingResult",
    "CMIP7ScenarioMIPPreProcessor",
    "CMIP7_SCENARIOMIP_SCMRunner",
    "ReaggregatorBasic",
    "ReaggregatorLike",
    "create_cmip7_scenariomip_country_harmoniser",
    "create_cmip7_scenariomip_global_harmoniser",
    "create_cmip7_scenariomip_infilled_df",
    "get_complete_scenarios_for_magicc",
    "load_aneris_overrides_file",
    "load_cmip7_scenariomip_country_historical_emissions",
    "load_cmip7_scenariomip_historical_emissions",
    "load_magicc_cfgs",
]
