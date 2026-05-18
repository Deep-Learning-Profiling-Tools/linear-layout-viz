This repository is the LL-viz app built on the reusable `tensor-viz` viewer packages.

The root owns the Triton linear-layout demo inputs and project documentation. `demo_linear_layout.py` defines the sample layouts that users can run locally. `linear_layout_viz.py` converts Triton `LinearLayout` objects into tensor-viz session data, including hardware/logical tensors, colors, markers, and linear-layout metadata.

The browser app lives at the root. `src/main.ts` imports `startDemoApp(...)` from `@tensor-viz/viewer-demo` and passes the local linear-layout extension factory. That package boundary is intentional: tensor-viz owns generic viewer rendering and shell behavior, while LL-viz owns GPU layout presets, compose-layout parsing, linear-layout widgets, and fallback tabs.

The TypeScript extension lives in `src/extensions/linear-layout/`. `extension.ts` is the only file the tensor-viz shell sees; model files parse and evaluate layouts, preset files describe instruction families declaratively, and widget files render the sidebar controls.

The root CI installs npm dependencies, runs typecheck, runs unit tests, runs Playwright against the real app, and builds `dist/` for Pages. Python helpers depend on the published `tensor-viz` Python package rather than a checked-out submodule.
