import pandas as pd
import pytest
from attrs import define

from gcages.cmip7_scenariomip import CMIP7ScenarioMIPPreProcessor
from gcages.testing import get_cmip7_scenariomip_like_input


def pytest_generate_tests(metafunc):
    if "example_input_output" in metafunc.fixturenames:
        metafunc.parametrize(
            "example_input_output",
            [
                "all_reported",
                # "min_reported",
            ],
            indirect=True,
        )


@define
class ExampleInputOutput:
    input: pd.DataFrame
    output: pd.DataFrame


@pytest.fixture(scope="session")
def example_input_output(request):
    pre_processor = CMIP7ScenarioMIPPreProcessor(
        n_processes=None,  # run serially
        progress=False,
    )

    if request.param == "all_reported":
        example_input = get_cmip7_scenariomip_like_input()

    # TODO: add over reporting e.g. with extra regions or variables
    # that we don't use
    # TODO: add more combos of reporting in between based on IAM submissions
    # elif request.param == "min_reported":
    #     # TODO: pass the drop outs in so that the totals are correct
    #     example_input = get_cmip7_scenariomip_like_input()
    #     # Drop out everything that isn't compulsory
    #     # see https://docs.google.com/spreadsheets/d/1_j09LpJYsBip37RfVxpSCb2czar1P1d8ncc27Csubug/edit?gid=660607711#gid=660607711
    #     example_input = example_input.loc[
    #         ~(
    #             (
    #                 pix.ismatch(
    #                     variable=[
    #                         f"**|{sectors}"
    #                         for sectors in [
    #                             "Emissions|*|Energy|Demand|Transportation|Domestic Aviation",
    #                             "Emissions|*|Energy|Demand|Bunkers|International Aviation",
    #                             "Emissions|*|Energy|Demand|Bunkers|International Shipping",
    #                         ]
    #                     ],
    #                 )
    #                 & pix.isin(region="World")
    #             )
    #             | pix.ismatch(
    #                 variable=[
    #                     f"**CO2|{sectors}"
    #                     for sectors in [
    #                         "AFOLU|Land|Fires|Forest Burning",
    #                         "AFOLU|Land|Fires|Grassland Burning",
    #                     ]
    #                 ]
    #             )
    #             | pix.ismatch(
    #                 variable=[
    #                     f"**CH4|{sectors}"
    #                     for sectors in [
    #                         "Energy|Demand|Transportation|Domestic Aviation",
    #                         "Energy|Demand|Bunkers|International Aviation",
    #                     ]
    #                 ]
    #             )
    #             | pix.ismatch(
    #                 variable=[
    #                     f"**{species}|{sectors}"
    #                     for species in ["BC", "CO", "OC", "Sulfur"]
    #                     for sectors in [
    #                         "AFOLU|Land|Harvested Wood Products",
    #                         "AFOLU|Land|Land Use and Land-Use Change",
    #                         "AFOLU|Land|Other",
    #                         "AFOLU|Land|Wetlands",
    #                     ]
    #                 ]
    #             )
    #             | pix.ismatch(variable=["**AFOLU|Land|Fires|Peat Burning"])
    #             | pix.ismatch(
    #                 variable=[
    #                     f"**{species}|Product Use"
    #                     for species in ["BC", "CH4", "CO", "NOx", "OC", "Sulfur"]
    #                 ]
    #             )
    #         )
    #     ]

    else:
        raise NotImplementedError(request.param)

    processed = pre_processor(example_input)

    return ExampleInputOutput(input=example_input_output)
