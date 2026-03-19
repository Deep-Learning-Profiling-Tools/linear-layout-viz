# linear-layout-viz

Minimal wrapper repo for experiments that build on the local [`tensor-viz`](./tensor-viz) project.

## Structure

- `tensor-viz/`: git submodule pointing at the sibling `../tensor-viz` checkout

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

## Notes

- the submodule currently uses a local relative URL: `../tensor-viz`
- the submodule points at the committed `HEAD` of `tensor-viz`, not its uncommitted working tree
- if you want this wrapper to follow newer `tensor-viz` changes, commit them in `tensor-viz` and then update the submodule pointer here
