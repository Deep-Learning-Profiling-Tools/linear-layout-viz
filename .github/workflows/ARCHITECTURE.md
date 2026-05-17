The `.github/workflows/` directory contains automation for the LL-viz wrapper repository.

`ci.yml` validates the `tensor-viz/` submodule from a clean checkout. It installs the Python package, installs Node dependencies, installs Chromium for Playwright, then runs typecheck, tests, and build. This catches both TypeScript/Python unit failures and browser startup failures in the demo app.

`deploy-pages.yml` builds the static viewer demo from the submodule and publishes `tensor-viz/packages/viewer-demo/dist` to GitHub Pages. It does not run the full test suite because deployment should consume a commit that CI has already validated.

Workflow changes should preserve the submodule checkout step. Without it, CI can pass repository setup while testing none of the viewer code.
