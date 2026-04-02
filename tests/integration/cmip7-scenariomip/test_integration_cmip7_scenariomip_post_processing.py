import numpy as np
import pandas as pd
import pytest

from gcages.cmip7_scenariomip.post_processing import CMIP7ScenarioMIPPostProcessor


def create_dummy_scm_results(years=range(1850, 2101)):
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
    for _ in range(len(index)):
        base = np.linspace(0, 3.0, len(years))
        rng = np.random.default_rng()
        noise = rng.normal(0, 0.1, len(years))
        data.append(base + noise)

    return pd.DataFrame(data, index=index, columns=years)


def test_post_processor_initialization():
    """Tests the factory method from_cmip7_scenariomip_config."""
    processor = CMIP7ScenarioMIPPostProcessor.from_cmip7_scenariomip_config()

    assert processor.gsat_variable_name == "Surface Air Temperature Change"
    assert processor.gsat_assessment_median == 0.85
    assert 1850 in processor.gsat_assessment_pre_industrial_period
    assert 2014 in processor.gsat_assessment_time_period


def test_post_processor_call_integration(monkeypatch):
    """
    Integration test for the __call__ method.
    """
    # Setup
    processor = CMIP7ScenarioMIPPostProcessor.from_cmip7_scenariomip_config()
    scm_results = create_dummy_scm_results()

    monkeypatch.setattr(
        "gcages.cmip7_scenariomip.post_processing.get_temperatures_in_line_with_assessment",
        lambda in_df,
        **kwargs: in_df,  # Return as-is for simplicity in testing structure
    )
    monkeypatch.setattr(
        "gcages.cmip7_scenariomip.post_processing.get_exceedance_probabilities_over_time",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "gcages.cmip7_scenariomip.post_processing.get_exceedance_probabilities",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "gcages.cmip7_scenariomip.post_processing.categorise_scenarios",
        lambda **kwargs: pd.Series(["C1", "C1"], name="category"),
    )

    # Execution
    result = processor(scm_results)

    # Assertions on PostProcessingResult object
    assert hasattr(result, "timeseries_quantile")
    assert hasattr(result, "metadata_categories")

    # Check that quantiles were calculated (index should now contain 'quantile')
    assert "quantile" in result.timeseries_quantile.index.names

    # Check that variable name was updated to the assessment name
    assert (
        result.timeseries_run_id.index.get_level_values("variable")
        == processor.gsat_in_line_with_assessment_variable_name
    ).all()


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
