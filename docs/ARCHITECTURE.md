The root `docs/` directory holds project-level documentation assets for the hosted manual and visual examples.

`manual-site/` is the static manual site output. `manual-vids/` contains short interaction clips referenced by the manual. `sample-svgs/` contains exported examples that make the visualizer's output inspectable without running the app.

The root `MANUAL.md` is the source of truth for user-facing viewer behavior in this repository. When behavior changes, update `MANUAL.md` first, then refresh or curate the assets that illustrate that behavior. Do not hand-edit generated manual output when the source document or sync script should own the change.

For package-level tensor-viz API documentation, use the upstream `tensor-viz`
repository. For LL-viz demo assets, keep source media in this docs tree and
finish with `npm run build` from the repository root.
