# Manual

**NOTE:** This manual is better viewed within the visualizer - there's a button called "Manual" in the bottom-left corner that has video previews of all the actions.


This viewer shows paired hardware and logical layouts and lets you inspect how cells map between them.

## Selection

- `Left click + drag`: create a new selection.
- `Shift + left click + drag`: add cells to the current selection.
- `Ctrl + left click + drag`: remove cells from the current selection.
- `Click` on empty space: clear the current selection.

Selections are paired across hardware and logical layouts. Selecting cells in one view highlights the mapped cells in the other.

Videos:

- [Left click + drag](./docs/manual-vids/0-lclick-drag.mp4)
- [Shift + left click + drag](./docs/manual-vids/1-shift-lclick-drag.mp4)
- [Ctrl + left click + drag](./docs/manual-vids/2-ctrl-lclick-drag.mp4)
- [Click empty space to deselect](./docs/manual-vids/3-lclick-out-bounds.mp4)

## Inspector

The `Inspector` widget shows information for the cell under the cursor:

- hovered tensor
- hardware coordinate
- logical coordinate
- hardware coordinate in binary
- logical coordinate in binary
- tensor shape
- rank

For linear-layout tabs, `Hardware Coord` is the `T/W/R` coordinate and `Logical Coord` is the `A/B/...` coordinate.

Video:

- [Inspector](./docs/manual-vids/4-inspector.mp4)

## Show Matrix

`Show Matrix` displays the current rendered layout matrix.

- It is tab-specific.
- It updates when the layout is rendered.
- Row and column labels are color-coded to match the dimension-line colors.

Video:

- [Show Matrix](./docs/manual-vids/5-show-matrix.mp4)

## Tabs

- Click a tab to switch layouts.
- `Add New Tab` duplicates the current tab as a new editable tab.
- Click the `x` on a tab to close it.
- Each tab stores its own layout, slicing state, HSL settings, and cell-text settings.

Video:

- [Tabs](./docs/manual-vids/6-tabs.mp4)

## Slicing

Use the `Tensor View` widget to change the active tensor view and slice hidden axes.

- Edit the `View String` to permute or hide axes.
- Use the slice sliders to move through hidden-axis indices.

For paired linear-layout tabs, slicing one tensor filters the corresponding visible cells in the other tensor as well.

Video:

- [Slicing](./docs/manual-vids/7-slicing.mp4)

## HSL Coloring

Use the `HSL Mapping` widget to control the cell colors.

- Drag the `H`, `S`, and `L` chips to swap which axis drives each channel.
- Edit the start/end numeric fields for each channel range.
- Click `Recolor Layout` to apply the updated coloring.

The mapping is per tab.

Video:

- [HSL coloring](./docs/manual-vids/8-hsl-coloring.mp4)

## Cell Text

Use the `Cell Text` widget to overlay hardware ids on cells.

- `Warp Id`
- `Thread Id`
- `Register Id`

Text appears on both the hardware and logical layouts. Logical cells show the mapped hardware ids for the corresponding element.

Video:

- [Cell text](./docs/manual-vids/9-cell-text.mp4)

## Display Toggles

From the `Display` menu:

- `Toggle Block Gaps`: show or hide spacing between higher-level layout blocks.
- `Toggle Dimension Lines`: show or hide tensor outlines, dimension guides, and axis labels.

These can also be changed from the sidebar widgets where available.

Video:

- [Display toggles](./docs/manual-vids/10-display-toggles.mp4)

## Save

Use `File -> Save as SVG` to export the current 2D view.

- Shortcut: `Ctrl+S`
- The SVG includes tensor names, dimension lines, axis labels, cell colors, and supported overlays visible in the current 2D view.

Video:

- [Save as SVG](./docs/manual-vids/11-save.mp4)

## Notes

- GitHub Pages serves the static frontend only.
- The local Python demo can additionally serve `/api/session.json` and tensor payload files.
