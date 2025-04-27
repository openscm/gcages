"""
Reaggregation of timeseries from raw reporting to sectors needed for gridding

The idea here is that we receive raw data following some variable specification
Based on this, we reaggregate to the variables needed for gridding
(see [gcages.cmip7_scenariomip.gridding_emissions][]).
In order to do the reaggregation sensibly,
two things must be true:

1. all the timeseries we require must be there
1. the data must be internally consistent
    - including consideration of any optional timeseries

Reaggregation is a data problem
i.e. the hard part is making sure
that the data we receive matches our data model.
As a result, the code is highly coupled with the data we expect
(writing general solutions is hard).
This is why we have written the code
that supports each data model in a standalone module,
rather than trying to write a general solution
(which was extremely difficult when we tried to do it that way from the start,
we think because it creates couplings
which are incredibly difficult to reason through).
"""

from __future__ import annotations

from gcages.cmip7_scenariomip.pre_processing.reaggregation.basic import (
    ReaggregatorBasic,
)
from gcages.cmip7_scenariomip.pre_processing.reaggregation.common import (
    ToCompleteResult,
)

__all__ = ["ReaggregatorBasic", "ToCompleteResult"]
