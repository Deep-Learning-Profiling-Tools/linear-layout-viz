# linear-layout-viz

Minimal wrapper repo for experiments that build on the local [`tensor-viz`](./tensor-viz) project.

## Structure

- `tensor-viz/`: git submodule pointing at `Deep-Learning-Profiling-Tools/tensor-viz`

## Setup

```bash
git submodule update --init --recursive
cd tensor-viz
npm install
python -m venv .venv
source .venv/bin/activate
pip install -e .
npm run build
```

## GitHub Pages

The repo now includes [`.github/workflows/deploy-pages.yml`](./.github/workflows/deploy-pages.yml), which builds the static Vite demo from `tensor-viz/packages/viewer-demo` and publishes it to GitHub Pages.

Important limitation: GitHub Pages is static-only. The public site can render browser-side linear layouts from the new sidebar and still open local `.npy` files, but it cannot run the Python `tensor_viz.viz(...)` server or serve `/api/session.json`.

### One-time GitHub setup

1. Push this repo, including the submodule pointer you want Pages to build.
2. In GitHub, open `Settings -> Pages`.
3. Under `Build and deployment`, set `Source` to `GitHub Actions`.
4. Push to `main`, or run the `Deploy GitHub Pages` workflow manually from the Actions tab.

### Local preview of the same static site

```bash
cd tensor-viz
npm install
npm run build --workspace @tensor-viz/viewer-demo
```

The built site is written to `tensor-viz/packages/viewer-demo/dist`.

### Linear Layout Sidebar Schema

The public site's layout editor accepts JSON shaped like Triton's `LinearLayout.from_bases(...)` call:

```json
{
  "name": "Blocked Layout",
  "bases": [
    ["warp", [[0, 8], [0, 16]]],
    ["thread", [[4, 0], [8, 0], [0, 1], [0, 2], [0, 4]]],
    ["register", [[1, 0], [2, 0]]]
  ],
  "out_dims": ["x", "y"]
}
```

`out_dims` can also be written as `[name, size]` pairs when you want to pin the output shape explicitly.

## Notes

- the submodule points at `https://github.com/Deep-Learning-Profiling-Tools/tensor-viz`
- the wrapper repo builds whatever committed `tensor-viz` revision the submodule pointer references
- if you want Pages to include newer `tensor-viz` changes, commit them in `tensor-viz` first and then update the submodule pointer here
