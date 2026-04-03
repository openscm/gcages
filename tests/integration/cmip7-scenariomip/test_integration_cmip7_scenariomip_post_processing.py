import numpy as np
import pandas as pd
import pytest

from gcages.cmip7_scenariomip.post_processing import CMIP7ScenarioMIPPostProcessor

pix = pytest.importorskip("pandas_indexing")


def create_dummy_scm_results(years=range(1850, 2101), rand_weight=1):
    """Creates a dummy DataFrame mimicking SCM output for multiple runs/scenarios."""
    index = pd.MultiIndex.from_product(
        [
            ["MAGICC7"],
            ["ModelA"],
            ["Scenario1"],
            ["World"],
            ["run_1", "run_2"],  # run_id
            ["Surface Air Temperature Change"],
            ["K"],
        ],
        names=[
            "climate_model",
            "model",
            "scenario",
            "region",
            "run_id",
            "variable",
            "unit",
        ],
    )

    data = []
    for i in range(len(index)):
        end_val = 1.0 + (i * 0.5)
        base = np.linspace(0, end_val, len(years))
        rng = np.random.default_rng()
        noise = rng.normal(0, 0.1, len(years)) * rand_weight
        data.append(base + noise)

    return pd.DataFrame(data, index=index, columns=years)


def test_post_processor_initialization():
    """Tests the factory method from_cmip7_scenariomip_config."""
    processor = CMIP7ScenarioMIPPostProcessor.from_cmip7_scenariomip_config()

    assert processor.gsat_variable_name == "Surface Air Temperature Change"
    assert processor.gsat_assessment_median == 0.85
    assert 1850 in processor.gsat_assessment_pre_industrial_period
    assert 2014 in processor.gsat_assessment_time_period


@pytest.mark.parametrize("missing_col", ["model", "scenario", "run_id"])
def test_post_processor_validation_errors(missing_col):
    """Tests that _check_in_df catches missing metadata."""
    processor = CMIP7ScenarioMIPPostProcessor.from_cmip7_scenariomip_config()
    df = create_dummy_scm_results()

    # Drop a required level
    df_invalid = df.reset_index(missing_col, drop=True)

    with pytest.raises(
        ValueError, match=f"missing required metadata levels:.*{missing_col}"
    ):
        processor(df_invalid)


def test_post_processor_missing_years():
    """Tests that validation fails if required assessment years are missing."""
    processor = CMIP7ScenarioMIPPostProcessor.from_cmip7_scenariomip_config()
    # Create data only from 2020-2100 (missing pre-industrial period)
    df = create_dummy_scm_results(years=range(2020, 2101))

    with pytest.raises(
        ValueError, match="Input data is missing years required for assessment"
    ):
        processor(df)


def test_post_processor_synthetic_input():
    """Tests that validation fails if required assessment years are missing."""
    processor = CMIP7ScenarioMIPPostProcessor.from_cmip7_scenariomip_config()
    df = create_dummy_scm_results(years=range(1850, 2101), rand_weight=0)

    post_processed = processor(df)

    assert (
        post_processed.metadata_exceedance_probabilities.unstack("threshold").values
        == [100, 100, 0]
    ).all()
    quantiles = (
        post_processed.metadata_quantile.loc[pix.isin(quantile=[0.05, 0.5, 0.95])]
        .unstack(["quantile", "metric"])
        .round(2)
        .values
    )
    assert (
        quantiles[~np.isnan(quantiles)]
        == np.array([1.12, 1.33, 1.53, 1.12, 1.33, 1.53, 2100, 2100, 2100])
    ).all()
    assert (
        post_processed.metadata_run_id.values.round(2)
        == np.array([1.10, 1.55, 1.10, 1.55, 2100, 2100])
    ).all()
