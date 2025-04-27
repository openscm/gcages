import pandas as pd
import pytest
from attrs import define

from gcages.cmip7_scenariomip import (
    CMIP7ScenarioMIPPreProcessingResult,
    CMIP7ScenarioMIPPreProcessor,
)
from gcages.cmip7_scenariomip.pre_processing.pre_processor import ReaggregatorLike
from gcages.cmip7_scenariomip.pre_processing.reaggregation import ReaggregatorBasic
from gcages.cmip7_scenariomip.pre_processing.reaggregation.basic import (
    get_example_input,
)


def pytest_generate_tests(metafunc):
    if "example_input_output" in metafunc.fixturenames:
        metafunc.parametrize(
            "example_input_output",
            [
                "basic",
                # "domestic_aviation_global_only"
            ],
            indirect=True,
        )


@define
class ExampleInputOutput:
    input: pd.DataFrame
    model_regions: tuple[str, ...]
    output: CMIP7ScenarioMIPPreProcessingResult
    reaggregator: ReaggregatorLike


@pytest.fixture(scope="session")
def example_input_output(request):
    """
    Get example input and output

    Only need basic tests here,
    the tests of all the variations between minimum and complete reporting
    should go in the tests of the different reaggregation options.
    """
    if request.param == "basic":
        model_regions_raw = ("China", "Pacific OECD")
        model = "model_am"
        model_regions = [f"{model}|{r}" for r in model_regions_raw]
        reaggregator = ReaggregatorBasic(model_regions=model_regions)
        pre_processor = CMIP7ScenarioMIPPreProcessor(
            reaggregator=reaggregator,
            n_processes=None,  # run serially
            progress=False,
        )

        input = get_example_input(model_regions=model_regions, model=model)

    else:
        raise NotImplementedError(request.param)

    processed = pre_processor(input)

    return ExampleInputOutput(
        input=input,
        model_regions=model_regions,
        output=processed,
        reaggregator=reaggregator,
    )
