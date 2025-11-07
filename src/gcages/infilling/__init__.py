"""
General infilling tools

Infillers can infer timeseries of one variable
based on timeseries of another variable
and a database of information from which the relationship
between the two variables can be derived.

The infilling is generally split into two parts:

1. configuration
2. infilling

We make this split so that the potentially expensive step,
deriving the relationship between variables,
can essentially be cached, avoiding unnecessary computation
while also avoiding object creation at runtime
(i.e. making the infiller object depend on the exact emissions
that need to be infilled).
"""

from __future__ import annotations

from gcages.infilling.common import Infiller
from gcages.infilling.silicone import (
    SiliconeBasedInfillingConfig,
    create_silicone_based_infiller,
)

__all__ = ["Infiller", "SiliconeBasedInfillingConfig", "create_silicone_based_infiller"]
