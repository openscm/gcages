# Regression testing strategy

We provide workflows that have been used in or replicate key scientific processes.
To check these, we reproduce the full set of results.
However, this reproduction is relatively computationally expensive.
As a result, we have to have a slightly more complicated testing strategy
that allows us to check reproduction while also offering testing paths
that facilitate fast iteration during development.

The strategy is as follows.
For each workflow we support, we put our tests in `tests/regression/[workflow_name]`.
The workflow tests then go in `tests/regression/[workflow_name]/test_[workflow_name]_workflow.py`.
These test functions that support running the complete workflows from end-to-end in full.
As a result, these tests are relatively expensive.
To avoid making our CI very slow, these tests are marked with `pytest.mark.skip_ci_default`
and these tests are skipped in the CI, except for commits to the main branch.

We then include tests of individual stages in the workflow
(e.g. harmonisation, infilling)
in `tests/regression/[workflow_name]/test_[workflow_name]_[stage].py`
These tests are deliberately made fast and not marked with `pytest.mark.skip_ci_default`
so that they are run in the CI on every pull request
(and give us at least basic coverage of all the different workflow stages).
If this is possible while still running the stage as it was done in the workflow,
then we simply compare against the same inputs and outputs as the full workflow (for the given stage).
However, if this is not possible, we simplify and test against computationally faster
workflows that capture the key elements as much as possible.
For example, the AR6 workflow harmonisation is very fast to run,
so we simply run harmonisation in full when testing the AR6 harmonisation.
However, the AR6 workflow simple climate model running is slow,
so we only run the simple climate model with a fraction of its full probabilistic ensemble
when testing the simple climate model running step
(and leave the workflow tests to cover the case where the simple climate model is run in full).
