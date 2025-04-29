"""
Pre-processing part of the workflow

This is extremely fiddly
because of the way the data is reported,
which is frankly, a mess
because of how it blends data
that is a regional-sum with data that has regional detail
and how the variable name is a blend of different bits of information
(species, sectoral information etc.)
with no easy way to decode what is what using a machine
(you have to hardcode lots of edge cases
e.g. Emissions|CO2|Energy is "Emissions", then the species then the sector
but Emissions|HFC|HFC245 is "Emissions" then the "HFC" string then the species,
i.e. completely different information is provided after each "|").

This module implements the logic for this processing.
The complexity comes in the re-aggregation
([gcages.cmip7_scenariomip.pre_processing.reaggregation][]),
which has to handle converting from whatever is reported
(and a huge amount of different possibilities have to be supported)
to the sectors used for gridding.
From there, the workflow can be standardised
(as is done in
[pre_processor.do_pre_processing][(m).]).
"""

from __future__ import annotations

from gcages.cmip7_scenariomip.pre_processing.pre_processor import (
    CMIP7ScenarioMIPPreProcessingResult,
    CMIP7ScenarioMIPPreProcessor,
    ReaggregatorLike,
)
from gcages.cmip7_scenariomip.pre_processing.reaggregation import (
    ReaggregatorBasic,
    ToCompleteResult,
)

__all__ = [
    "CMIP7ScenarioMIPPreProcessingResult",
    "CMIP7ScenarioMIPPreProcessor",
    "ReaggregatorBasic",
    "ReaggregatorLike",
    "ToCompleteResult",
]
