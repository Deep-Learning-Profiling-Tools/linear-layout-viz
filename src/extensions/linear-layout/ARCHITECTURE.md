The `extensions/linear-layout/` directory is the LL-viz feature package inside the tensor-viz demo shell.

The extension exists so linear-layout work can grow without making the generic viewer app know about GPU instruction families, compose-layout syntax, propagated labels, or multi-input hover behavior. `extension.ts` is the only file the shell imports. It registers sidebar widgets, the Propagate Outputs toolbar control, tab/session lifecycle hooks, baked fallback tabs, hover popup behavior, tensor-view axis labels, multi-input sliders, inspector coordinate rows, and selection synchronization.

`linear-layout.ts` is the center of the model. It owns compose-layout parsing, operation evaluation, preset normalization, matrix previews, generated Python, and metadata embedded into viewer tabs. If a change affects layout syntax, composition semantics, output labels, matrix blocks, or the session metadata emitted for a rendered layout, start there.

`linear-layout-state.ts` bridges saved viewer tabs and browser storage back into live sidebar state. It should stay focused on cloning, validation, tab synchronization, and persistence. `linear-layout-viewer-sync.ts` bridges in the other direction: it translates current runtime metadata into viewer labels, colors, selection, hover popups, and multi-input display state.

The two subdirectories keep contributor workflows local:

- `presets/` contains instruction-family preset data and selector metadata.
- `widgets/` contains sidebar UI split by workflow.

When changing parsing or composition, update `linear-layout.test.ts` with the smallest test that captures the behavior. Good tests here build a `ComposeRuntime`, inspect emitted metadata or generated Python, then assert the mapping behavior that would break in the UI.
