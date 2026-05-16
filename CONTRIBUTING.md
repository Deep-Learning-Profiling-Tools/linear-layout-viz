# Contributing

**NOTE: This project was purely built with coding agents, and any contributions will be purely reviewed by coding agents. As for now, only PRs that add/modify layout presets will be considered as the rest of the codebase is subject to change.**

## Setup

This repository wraps the `tensor-viz/` submodule, where the viewer code lives.

```bash
git submodule update --init --recursive
cd tensor-viz
npm install
python -m venv .venv
source .venv/bin/activate
pip install -e .
npm run build
```

To check that linear-layout-viz is installed correctly, run this in the project root directory after building tensor-viz above:

```bash
python demo_linear_layout.py
```

## Testing Changes

Run the full test suite from `tensor-viz/`:

```bash
npm run test
```

Before submitting any code change, also run:

```bash
npm run build
```

The build is required even if tests pass because it refreshes the built demo
assets served by the Python package.

Focused checks are useful while iterating:

```bash
npm run test --workspace @tensor-viz/viewer-core
npm run test --workspace @tensor-viz/viewer-demo
PYTHONPATH=python/src python -m unittest discover -s python/tests -p 'test_*.py'
```

## File Structure

- `README.md`: project overview, public website links, and top-level setup.
- `MANUAL.md`: user-facing viewer interaction guide.
- `linear_layout_viz.py`, `demo_linear_layout.py`: root-level linear layout demos.
- `assets/`: source media used by the README and project pages.
- `docs/manual-site/`, `docs/manual-vids/`, `docs/sample-svgs/`: generated or
  curated documentation assets.
- `tensor-viz/`: submodule containing the Python package, TypeScript packages,
  demo app, tests, docs, and build tooling.

Inside `tensor-viz/`:

- `packages/viewer-core/src/`: reusable viewer engine, layout math, session
  model, rendering, and core tests.
- `packages/viewer-demo/src/`: the full linear-layout demo app and UI.
- `packages/viewer-demo/src/linear-layout-presets/`: instruction-family preset
  definitions.
- `packages/viewer-demo/src/widgets/`: sidebar widgets split by user workflow.
- `python/src/tensor_viz/`: Python package and local server.
- `python/tests/`: Python documentation and API tests.
- `docs/`: Sphinx and TypeDoc documentation sources.
- `tools/`: synchronization scripts for examples and built demo assets.

## Architecture Guides

To add/modify features within the current system architecture, architecture docs live next to the code they describe:

- [Linear layout demo app](./tensor-viz/packages/viewer-demo/src/ARCHITECTURE.md):
  parsing, composition, runtime metadata, and generated Python.
- [Linear layout presets](./tensor-viz/packages/viewer-demo/src/linear-layout-presets/ARCHITECTURE.md):
  adding NVIDIA, AMD, or other preset families.
- [Linear layout widgets](./tensor-viz/packages/viewer-demo/src/widgets/ARCHITECTURE.md):
  sidebar UI ownership and shared widget code.
- [Viewer core](./tensor-viz/packages/viewer-core/src/ARCHITECTURE.md):
  reusable layout, view, session, and rendering behavior.
- [Python package](./tensor-viz/python/src/tensor_viz/ARCHITECTURE.md):
  Python API, session normalization, and local serving.
- [Root documentation assets](./docs/ARCHITECTURE.md) and
  [tensor-viz package docs](./tensor-viz/docs/ARCHITECTURE.md): manual pages,
  Sphinx docs, generated references, and demo asset synchronization.

## Documentation Standards

For a given design change (e.g. adding/modifying a subsystem), update the relevant
`ARCHITECTURE.md` file(s) explaining the purpose of the subsystem, folder file
structure, what each file does, and instructions for workflows using your subsystem.
Keep user-facing behavior in `MANUAL.md` or `tensor-viz/docs/`, and keep this file
limited to setup, checks, review expectations, and pointers.

When a change makes existing documentation misleading, update the docs in the
same change. Prefer one source of truth with links over copying the same
procedure into several files.
