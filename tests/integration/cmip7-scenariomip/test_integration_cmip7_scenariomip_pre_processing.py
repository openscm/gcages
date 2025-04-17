"""
Integration tests of our pre-processing for CMIP7 ScenarioMIP
"""

import pytest


@pytest.fixture(scope="session")
def example_raw_input():
    # breakpoint()
    pass


@pytest.fixture(scope="session")
def processed_output(example_raw_input):
    pre_processor = CMIP7ScenarioMIPPreProcessor.from_ar6_config(
        n_processes=None,  # run serially
    )

    processed = pre_processor(example_raw_input)

    yield processed


def test_transport_shuffling(example_raw_input, processed_output):
    """
    Test the moving of the transport data

    Domestic and international aviation should be added to make |Aircraft.
    Transportation should have domestic aviation
    subtracted to make |Transportation Sector.
    """
    assert False, "Implement"


def test_industrial_sector_aggregation(example_raw_input, processed_output):
    exp_sector = "Industrial Sector"
    exp_contributing_sectors = [
        "Energy|Supply",
        "Energy|Demand|Industry",
        "Energy|Demand|Other Sector",
        "Industrial Processes",
        "Other",
    ]
    assert False, "Implement"


def test_output_sectors(processed_output):
    exp_output_sectors = [
        "Energy Sector",
        "International Shipping",
        "Residential Commercial Other"
        "Solvents Production and Application"
        "Agriculture"
        "Agricultural Waste Burning",
        "Forest Burning",
        "Grassland Burning",
        "Peat Burning",
    ]
    assert False, "Implement"


def test_output_internal_consistency(processed_output):
    assert False, "check that sum over sectors makes sense"
    assert False, "check that sum over regions makes sense"
    assert False, "check that sum over sectors and regions makes sense"
    assert False, "check that sum over regions and sectors makes sense"


def test_input_not_internally_consistent_error(example_raw_input):
    # break the internal consistency at some level
    # make sure that a sensible error message is raised
    # (don't bother trying to guess what has gone wrong, too hard basket)
    assert False, "implement"


# Underlying logic:
# - we're doing region-sector harmonisation
# - hence we need regions and sectors lined up very specifically with CEDS
# - hence this kind of pre-processing
#   (if you want different pre-processing, use something more like AR6)
