"""
Pre-processing part of the workflow

This is extremely fiddly
because of the way the data is reported
(some is reported at the regional level,
other bits only at the global level,
the pipe means different things for different variables
(for most variables, after the pipe you get sectors,
for HFCs you get the actual gas)
and we need to be able to move between all these conventions).

This module implements the logic for this processing.
There are a number of definitions in [constants][(m).constants].
It is likely possible to change these.
However, it would be extremely difficult to test
that the constants can be altered
and the whole module stays consistent.
As a result, we have written it like this to make clearer
that this entire module is more or less coupled,
If you alter any of the constants,
we don't guarantee correct behaviour.

The underlying logic is this:

- we're doing region-sector harmonisation
- hence we need regions and sectors lined up very specifically with CEDS
- we also need to keep information that will be used only at the global level
  (e.g. CFCs, HFCs)
- we process with sane tables that are easy to sum,
  but have to report with the same format we got in
"""

from __future__ import annotations

from gcages.cmip7_scenariomip.pre_processing.completeness import (
    get_required_model_region_index_input,
    get_required_world_index_input,
)
from gcages.cmip7_scenariomip.pre_processing.constants import (
    REQUIRED_MODEL_REGION_VARIABLES_INPUT,
    REQUIRED_WORLD_VARIABLES_INPUT,
)

__all__ = [
    "REQUIRED_MODEL_REGION_VARIABLES_INPUT",
    "REQUIRED_WORLD_VARIABLES_INPUT",
    "get_required_model_region_index_input",
    "get_required_world_index_input",
]
