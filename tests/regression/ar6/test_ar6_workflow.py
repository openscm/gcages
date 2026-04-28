"""
Full regression tests of the AR6 workflow
"""

import multiprocessing

import pytest

# Mark the entire module.
# If you want to run workflow tests that aren't skipped in CI by default,
# put them somewhere else.
pytest.mark.skip_ci_default

pytest.mark.magicc_v753

AR6_WORKFLOW_INPUTS_DIR = Path(__file__).parents[0] / "ar6-workflow-inputs"


def test_ar6_workflow():
    # all scenarios
    # in parallel
    # (tests of serial vs parallel behaviour should be in tests of specific stages,
    # to keep things fast)
    res = run_ar6_workflow(
        infilling_db_path=AR6_WORKFLOW_INPUTS_DIR / "infilling_db_ar6.csv",
        infilling_db_cfcs_path=AR6_WORKFLOW_INPUTS_DIR / "infilling_db_ar6_cfcs.csv",
        magicc_exe_path=guess_magicc_exe(AR6_WORKFLOW_INPUTS_DIR / "magicc-v7.5.3/bin"),
        magicc_ar6_probabilistic_config_path=(
            AR6_WORKFLOW_INPUTS_DIR
            / "magicc-ar6-0fd0f62-f023edb-drawnset"
            / "0fd0f62-derived-metrics-id-f023edb-drawnset.json"
        ),
        config=AR6WorkflowConfig(
            # Don't use all processes for testing
            n_processes_scms=multiprocessing.cpu_count() - 2,
            # Otherwise, leave the defaults
            # progress_scms,
            # n_processes_pre_processing,
            # progress_pre_processing,
            # n_processes_harmonisation,
            # progress_harmonisation,
            # n_processes_infilling,
            # progress_infilling,
            # n_processes_post_processing,
            # progress_post_processing,
            # run_checks,
        ),
        # output_variables
    )

    # check pre-processing results (not possible?)
    # check harmonisation results
    # check infilling results
    # check SCM results (not possible?)
    # check post-processing results
