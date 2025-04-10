"""
Type hints that are used throughout
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from typing_extensions import TypeAlias

NUMERIC_DATA: TypeAlias = Union[float, int, np.floating, np.integer]
"""
Type alias for a value that can be used in the data of a [TimeseriesDataFrame][(m).]
"""

TIME_POINT: TypeAlias = Union[float, int]
"""
Type alias for a value that can be used in the columns of a [TimeseriesDataFrame][(m).]
"""

TimeseriesDataFrame: TypeAlias = pd.DataFrame
"""
Type alias for the [pandas.DataFrame][pd.DataFrame] shape we use throughout

For typing purposes, this is just a direct alias of [pandas.DataFrame][pd.DataFrame].
However, the point of defining this
is to provide greater clarity of the kind of data we expect.

We expect a collection of timeseries.
These timeseries are defined by the columns, which we expect to be timepoints.
We expect that the index contains metadata about each timeseries.
As a result, the data itself should be numerical only (no strings, no lists, no dicts).

An example of this kind of data is given below.
Note, in line with the description above:

1. Data is in the body of the [pandas.DataFrame][pd.DataFrame]
1. The columns define the time axis
1. All metadata is contained in the index

```python
                        2015  2100
scenario variable unit
sa       va       W      1.0   2.1
         vb       W      3.0   2.0
```
"""
