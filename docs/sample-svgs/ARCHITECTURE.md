The `docs/sample-svgs/` directory contains exported SVG examples from the viewer.

These files are visual regression aids and documentation examples. They let contributors inspect expected output without starting the app. The local `README.md` explains what each sample represents.

When layout rendering, color mapping, or SVG export changes intentionally, refresh the affected SVGs and note the behavior change in the commit. Do not hand-edit SVG internals as a substitute for fixing renderer output.
