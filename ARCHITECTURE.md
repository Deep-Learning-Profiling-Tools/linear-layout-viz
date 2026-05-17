This repository is the LL-viz wrapper around the reusable `tensor-viz` viewer.

The root owns the Triton linear-layout demo inputs and project documentation. `demo_linear_layout.py` defines the sample layouts that users can run locally. `linear_layout_viz.py` converts Triton `LinearLayout` objects into tensor-viz session data, including hardware/logical tensors, colors, markers, and linear-layout metadata.

The `tensor-viz/` submodule owns the viewer implementation. Its TypeScript packages render the UI, its Python package serves local sessions, and its linear-layout extension owns presets, widgets, parsing, and browser behavior. Root code should call into that package instead of duplicating viewer logic here.

The root CI checks the submodule because that is where the app builds and tests. When the submodule pointer changes, CI installs Python and Node dependencies inside `tensor-viz/`, installs Chromium for Playwright, runs typecheck, runs tests, and builds packaged frontend assets.

Keep new root code narrowly focused on LL-viz examples or project-level docs. Runtime behavior, widget behavior, rendering, and preset data should normally be changed inside `tensor-viz/` with architecture notes next to that subsystem.
