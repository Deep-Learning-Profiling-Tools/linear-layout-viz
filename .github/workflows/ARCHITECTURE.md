The `.github/workflows/` directory contains automation for the LL-viz app repository.

`ci.yml` validates the root LL-viz frontend from a clean checkout. Until the new `@tensor-viz/*` npm packages are published, it checks out the tensor-viz extraction branch into a temporary directory, packs `@tensor-viz/viewer-core` and `@tensor-viz/viewer-demo`, installs those tarballs through npm, installs Chromium for Playwright, then runs typecheck, unit tests, browser e2e tests, and build. This catches linear-layout parser/model failures, package-boundary import failures, and browser startup regressions before release.

`deploy-pages.yml` builds the root Vite app with the same temporary tensor-viz tarball install path and publishes `dist/` to GitHub Pages. It does not run the full test suite because deployment should consume a commit that CI has already validated.

Workflow changes should keep CI pointed at the root package. Tensor-viz is consumed through npm package artifacts, so bringing back submodule checkout would hide package-boundary failures that this repository is meant to catch. Once `@tensor-viz/viewer-core` and `@tensor-viz/viewer-demo` are published, replace the temporary checkout/pack steps with a plain `npm install`.
