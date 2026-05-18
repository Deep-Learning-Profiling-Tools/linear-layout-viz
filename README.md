# linear-layout-viz

![Cover](./assets/cover.png)

_Linear layout in Figure 1a of the linear layout paper visualized with the visualizer._

A visualizer for Triton linear layouts.
- [Website](https://deep-learning-profiling-tools.github.io/linear-layout-viz/)
- Paper: [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using F_2](https://arxiv.org/pdf/2505.23819)

See [MANUAL.md](./MANUAL.md) for the viewer interaction guide.
See [docs/sample-svgs/](./docs/sample-svgs/README.md) for example exported SVGs.

## Structure

- `src/extensions/linear-layout/`: LL-viz extension, parser/model code, presets,
  widgets, and tests.
- `linear_layout_viz.py`, `demo_linear_layout.py`: Python helpers that generate
  tensor-viz session data for Triton `LinearLayout` objects.
- `assets/`, `docs/`, `MANUAL.md`: website media and user-facing guides.

## Setup

```bash
npm install
python -m venv .venv
source .venv/bin/activate
pip install tensor-viz
npm run build
```

### One-time GitHub setup

1. Push this repo.
2. In GitHub, open `Settings -> Pages`.
3. Under `Build and deployment`, set `Source` to `GitHub Actions`.
4. Push to `main`, or run the `Deploy GitHub Pages` workflow manually from the Actions tab.

### Local preview of the same static site

```bash
npm install
npm run build
```

The built site is written to `dist/`.

## Usage

- For day-to-day viewer usage, see [MANUAL.md](./MANUAL.md).
- The manual covers selection, inspector, matrix view, tabs, slicing, HSL coloring, cell text, display toggles, and saving SVG output.
- For example exports, see [docs/sample-svgs/](./docs/sample-svgs/README.md).
