"""
Tests of `gcages.harmonisation.assert_harmonised`
"""

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest

from gcages.harmonisation import NotHarmonisedError, assert_harmonised
from gcages.harmonisation.common import align_history_to_data_at_time


def get_df(index):
    return pd.DataFrame(
        np.zeros((index.shape[0], 3)),
        columns=range(3),
        index=index,
    )


@pytest.mark.parametrize(
    "df, history, harmonisation_time, exp",
    (
        pytest.param(
            pd.DataFrame(
                [
                    [1.0, 2.0],
                    [3.0, 2.0],
                ],
                columns=[2015, 2100],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("sa", "va", "W"),
                        ("sa", "vb", "W"),
                    ],
                    names=["scenario", "variable", "unit"],
                ),
            ),
            pd.DataFrame(
                [
                    [1.1, 1.0],
                    [2.2, 3.0],
                ],
                columns=[2010, 2015],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("va", "W"),
                        ("vb", "W"),
                    ],
                    names=["variable", "unit"],
                ),
            ),
            2015,
            does_not_raise(),
            id="harmonised",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    [1.0, 2.1],
                    [3.0, 2.0],
                    [1.0, 2.2],
                    [3.0, 2.3],
                ],
                columns=[2015, 2100],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("sa", "va", "W"),
                        ("sa", "vb", "W"),
                        ("sb", "va", "W"),
                        ("sb", "vb", "W"),
                    ],
                    names=["scenario", "variable", "unit"],
                ),
            ),
            pd.DataFrame(
                [
                    [1.1, 1.0],
                    [2.2, 3.0],
                ],
                columns=[2010, 2015],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("va", "W"),
                        ("vb", "W"),
                    ],
                    names=["variable", "unit"],
                ),
            ),
            2015,
            does_not_raise(),
            id="harmonised-multiple-scenarios",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    [1.0, 2.1],
                    [3.0, 2.0],
                    [1.0, 2.2],
                ],
                columns=[2015, 2100],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("sa", "va", "W"),
                        ("sa", "vb", "W"),
                        ("sb", "va", "W"),
                    ],
                    names=["scenario", "variable", "unit"],
                ),
            ),
            pd.DataFrame(
                [
                    [1.1, 1.0],
                    [2.2, 3.0],
                ],
                columns=[2010, 2015],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("va", "W"),
                        ("vb", "W"),
                    ],
                    names=["variable", "unit"],
                ),
            ),
            2015,
            does_not_raise(),
            id="harmonised-multiple-scenarios-different-variables",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    [1.0, 2.0],
                    [3.0, 2.0],
                ],
                columns=[2015, 2100],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("sa", "va", "W"),
                        ("sa", "vb", "W"),
                    ],
                    names=["scenario", "variable", "unit"],
                ),
            ),
            pd.DataFrame(
                [
                    [1.1, 1.0],
                    [2.2, 3.0],
                    [-1.0, -2.2],
                ],
                columns=[2010, 2015],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("va", "W"),
                        ("vb", "W"),
                        ("vc", "W"),
                    ],
                    names=["variable", "unit"],
                ),
            ),
            2015,
            does_not_raise(),
            id="harmonised-extra-history-timeseries",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    [1.0, 2.0],
                    [3.0, 2.0],
                    [1.0, 2.1],
                    [3.0, 2.3],
                ],
                columns=[2015, 2100],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("sa", "ma", "va", "W"),
                        ("sa", "ma", "vb", "W"),
                        ("sa", "mb", "va", "W"),
                        ("sa", "mb", "vb", "W"),
                    ],
                    names=["scenario", "model", "variable", "unit"],
                ),
            ),
            pd.DataFrame(
                [
                    [1.1, 1.0],
                    [2.2, 3.0],
                    [-1.0, -2.2],
                ],
                columns=[2010, 2015],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("va", "W"),
                        ("vb", "W"),
                        ("vc", "W"),
                    ],
                    names=["variable", "unit"],
                ),
            ),
            2015,
            does_not_raise(),
            id="harmonised-model-scenario-extra-history-timeseries",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    [1.0, 2.1],
                    [3.0, 2.0],
                ],
                columns=[2015, 2100],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("sa", "va", "W"),
                        ("sa", "vb", "W"),
                    ],
                    names=["scenario", "variable", "unit"],
                ),
            ),
            pd.DataFrame(
                [
                    [1.1, 1.1],
                    [2.2, 3.1],
                ],
                columns=[2010, 2015],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("va", "W"),
                        ("vb", "W"),
                    ],
                    names=["variable", "unit"],
                ),
            ),
            2015,
            pytest.raises(
                NotHarmonisedError,
                match=re.escape("The DataFrame is not harmonised in 2015. comparison="),
            ),
            id="unharmonised-single",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    [1.0, 2.1],
                    [3.0, 2.0],
                    [-1.0, -2.1],
                ],
                columns=[2015, 2100],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("sa", "va", "W"),
                        ("sa", "vb", "W"),
                        ("sa", "vc", "W"),
                    ],
                    names=["scenario", "variable", "unit"],
                ),
            ),
            pd.DataFrame(
                [
                    [1.1, 1.1],
                    [2.2, 3.0],
                    [-1.2, -1.8],
                ],
                columns=[2010, 2015],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("va", "W"),
                        ("vb", "W"),
                        ("vc", "W"),
                    ],
                    names=["variable", "unit"],
                ),
            ),
            2015,
            pytest.raises(
                NotHarmonisedError,
                match=re.escape("The DataFrame is not harmonised in 2015. comparison="),
            ),
            id="unharmonised-single-scenario-multiple-variables",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    [1.1, 2.1],
                    [3.0, 2.0],
                    [1.0, 2.2],
                    [3.1, 2.3],
                ],
                columns=[2015, 2100],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("sa", "va", "W"),
                        ("sa", "vb", "W"),
                        ("sb", "va", "W"),
                        ("sb", "vb", "W"),
                    ],
                    names=["scenario", "variable", "unit"],
                ),
            ),
            pd.DataFrame(
                [
                    [1.1, 1.0],
                    [2.2, 3.0],
                ],
                columns=[2010, 2015],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("va", "W"),
                        ("vb", "W"),
                    ],
                    names=["variable", "unit"],
                ),
            ),
            2015,
            pytest.raises(
                NotHarmonisedError,
                match=re.escape("The DataFrame is not harmonised in 2015. comparison="),
            ),
            id="unharmonised-multiple-scenarios",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    [1.0, 2.0],
                    [3.0, 2.0],
                    [1.0, 2.1],
                    [3.0, 2.3],
                ],
                columns=[2015.0, 2100.0],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("sa", "ma", "va", "W"),
                        ("sa", "ma", "vb", "W"),
                        ("sa", "mb", "va", "W"),
                        ("sa", "mb", "vb", "W"),
                    ],
                    names=["scenario", "model", "variable", "unit"],
                ),
            ),
            pd.DataFrame(
                [
                    [1.1, 1.0],
                    [2.2, 3.0],
                    [-1.0, -2.2],
                ],
                columns=[2014.5, 2015.0],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("va", "W"),
                        ("vb", "W"),
                        ("vc", "W"),
                    ],
                    names=["variable", "unit"],
                ),
            ),
            2015.0,
            does_not_raise(),
            id="harmonised-model-scenario-extra-history-timeseries-float-times",
        ),
    ),
)
def test_assert_harmonised(df, history, harmonisation_time, exp):
    with exp:
        assert_harmonised(df, history=history, harmonisation_time=harmonisation_time)


def test_align_history_to_data_in_year_same_index_error():
    df = pd.DataFrame(
        [
            [1.0, 2.1],
            [3.0, 2.0],
            [1.0, 2.2],
            [3.0, 2.3],
        ],
        columns=[2015, 2100],
        index=pd.MultiIndex.from_tuples(
            [
                ("sa", "va", "W"),
                ("sa", "vb", "W"),
                ("sb", "va", "W"),
                ("sb", "vb", "W"),
            ],
            names=["scenario", "variable", "unit"],
        ),
    )
    history = pd.DataFrame(
        [
            [1.1, 1.0],
            [2.2, 3.0],
        ],
        columns=[2010, 2015],
        index=pd.MultiIndex.from_tuples(
            [
                ("historical", "va", "W"),
                ("historical", "vb", "W"),
            ],
            names=["scenario", "variable", "unit"],
        ),
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "history did not align properly with df. "
            "history and df have the same index levels "
            "(['scenario', 'variable', 'unit'])"
        ),
    ):
        align_history_to_data_at_time(df, history=history, time=2015)


def test_align_history_to_data_in_year_different_units_error():
    df = pd.DataFrame(
        [
            [1.0, 2.1],
            [3.0, 2.0],
            [1.0, 2.2],
            [3.0, 2.3],
        ],
        columns=[2015, 2100],
        index=pd.MultiIndex.from_tuples(
            [
                ("sa", "va", "W / m^2"),
                ("sa", "vb", "t / yr"),
                ("sb", "va", "W / m^2"),
                ("sb", "vb", "t/yr"),
            ],
            names=["scenario", "variable", "unit"],
        ),
    )
    history = pd.DataFrame(
        [
            [1.1, 1.0],
            [2.2, 3.0],
        ],
        columns=[2010, 2015],
        index=pd.MultiIndex.from_tuples(
            [
                ("va", "W/m^2"),
                ("vb", "t/yr"),
            ],
            names=["variable", "unit"],
        ),
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "history did not align properly with df. "
            "The following units only appear in `df`, "
            "which might be why the data isn't aligned: ['W / m^2', 't / yr']."
        ),
    ):
        align_history_to_data_at_time(df, history=history, time=2015)
