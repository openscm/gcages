# Rationale behind our testing

Status: this document outlines where we want to get to.
Neither the code nor the tests follows this document exactly yet.

Our general testing strategy is like other repositories:
unit tests, integration tests, regression tests etc. as they make sense.
These go under `tests/unit`, `tests/integration`, `tests/regression` etc. as needed.
In general, we want gcages to be as thin as possible.
As a result, most specific unit, integration or regression tests
should be in the upstream tool's repository rather than gcages'.

## Reproducing key scientific workflows

However, a key part of gcages is its ability to support or reproduce
the climate assessment done in key scientific applications.
For each workflow we support, we have the ability to test it in full.
However, it is not feasible/sensible/desirable
to re-run these workflows in full every time we run the tests locally or in CI.
Instead, we take a more pragmatic approach.

By default, we don't run any of the workflow reproduction tests.
When opting in with flags specific to a given workflow,
a given number of scenarios from the workflow will be run in full (default is 2).
[TODO flag to pick the number]
These scenarios are picked randomly from the full set of available scenarios.
If you wish to run some specific or otherwise custom set,
update the testing code and then re-run.

The rationale for this is that the workflow tests are slow,
each taking minutes on a standard laptop.
You don't want these to run when doing a 'standard' `pytest tests` run.

We run a given number of scenarios from each workflow in CI
to check that our key workflows still work.
These tests should be marked with a specific mark
e.g. `pytest.mark.workflow_reproduction`.
We only run these on merges to main to save CI time.
This runs the risk that we merge breaking changes to main.
We think this risk is acceptable for the benefits it brings,
and the consequences of it are mitigated because we find out about the failure immediately,
so can quickly release a fix once the failure is identified (almost always before any release to e.g. PyPI).
Each of these workflows should also have a manual dispatch trigger
so that they can be run for a specific branch before merging
if there is a need/desire to check specific workflows' reproduction tests in advance of merging to main.

If we see that the tests are flaky i.e. only fail for some scenarios,
this shows that something about a given scenario is causing a failure.
This is a sign that one step in the workflow is not sufficiently well tested by other more targeted tests.
The remedy is to add these more targeted tests, rather than expanding the suite of reproduction tests
(as this causes a permanent increase in the test suite's runtime, which is exactly what we are trying to avoid).

Running a custom, rather than randomly chosen set, is a rare thing to do.
As a result, we are ok with requiring users to alter the test code to make this happen,
rather than paying the price of maintaining a set of flags which are hardly used and not tested.
If we find that we need specific custom sets often, we can reconsider
but at this stage we expect that this need will be a sign of issues in other test coverage,
rather than a strong argument to run more reproduction tests.

The workflow specific tests live under `tests/workflows/<workflow-name>`.
Reproduction tests therefore live under `tests/workflows/<workflow-name>/reproduction`.
Other workflow specific tests e.g. workflow specific unit, integration or regression tests
can live in sub-folders of `tests/workflows/<workflow-name>`.
