# Contributing

**NOTE: This project was built with coding agents, and contributions are reviewed
with coding-agent assistance. For now, preset additions and fixes are the
expected external contribution path.**

## Setup

Install the root frontend dependencies and the published Python viewer package:

```bash
npm install
npx playwright install chromium
python -m venv .venv
source .venv/bin/activate
pip install tensor-viz
npm run build
```

If the matching `@tensor-viz/viewer-core` and `@tensor-viz/viewer-demo` npm
packages have not been published yet, build local tarballs from the tensor-viz
extraction branch and install them with `npm install --no-save <tarball paths>`.
The CI workflow does this automatically until those packages are available from
the registry.

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

To check the Python demo, run:

```bash
python demo_linear_layout.py
```

## Testing Changes

Run the full frontend suite before submitting a change:

```bash
npm run typecheck
npm run test
npm run build
```

Focused checks are useful while iterating:

```bash
npm run test:unit
npm run test:e2e
npm run sync:linear-layout-examples
```

`npm run sync:linear-layout-examples` rewrites the generated baked-example block
inside `src/extensions/linear-layout/linear-layout.ts` from
`demo_linear_layout.py`. Run it after changing the Python demo layouts.

## File Structure

- `src/main.ts`: root Vite entrypoint that starts the tensor-viz shell with the
  LL-viz extension factory.
- `src/extensions/linear-layout/`: parser/model code, preset definitions,
  sidebar widgets, viewer synchronization, and unit tests.
- `linear_layout_viz.py`, `demo_linear_layout.py`: Python helpers and examples
  that produce tensor-viz sessions for Triton `LinearLayout` objects.
- `e2e/`: Playwright smoke tests for the real browser app.
- `assets/`, `docs/`, `MANUAL.md`: website media, manual assets, and
  user-facing docs.

Tensor-viz itself is no longer vendored here. Generic viewer behavior belongs in
the `Deep-Learning-Profiling-Tools/tensor-viz` repository and should be consumed
through the published `@tensor-viz/*` npm packages.

## Architecture Guides

Architecture docs live next to the code they describe:

- [Root LL-viz app](./ARCHITECTURE.md)
- [Linear layout extension](./src/extensions/linear-layout/ARCHITECTURE.md)
- [Linear layout presets](./src/extensions/linear-layout/presets/ARCHITECTURE.md)
- [Linear layout widgets](./src/extensions/linear-layout/widgets/ARCHITECTURE.md)
- [Root documentation assets](./docs/ARCHITECTURE.md)
- [Browser e2e tests](./e2e/ARCHITECTURE.md)

When a change modifies a subsystem, update the relevant `ARCHITECTURE.md` in the
same change. Keep user-facing behavior in `MANUAL.md`; keep this file focused on
setup, checks, and review expectations.
