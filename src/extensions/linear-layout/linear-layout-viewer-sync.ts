import {
    coordFromKey,
    coordKey,
    type LoadedBundleDocument,
    type SelectionCoords,
    type TensorViewer,
} from '@tensor-viz/viewer-core';
import {
    applyLinearLayoutDisplay,
    coordsForRootIndexes,
    displayedRootIndexForCoord,
    linearLayoutDisplayModel,
    linearLayoutSelectionMapForMeta,
    rootIndexesForCoords,
} from './linear-layout-multi-input.js';
import {
    isLinearLayoutTab,
    snapshotTensorViews,
    type InspectorCoordEntry,
    type LinearLayoutCellTextState,
    type LinearLayoutSelectionMap,
    type LinearLayoutTensorViewsState,
    type LinearLayoutUiContext,
} from './linear-layout-state.js';
import { rootColorsForLayoutState } from './linear-layout.js';

/**
 * One colored line in the linear-layout hover popup for a viewer cell that is
 * shared by multiple source labels.
 *
 * `color` is the CSS color used for the popup marker, and `text` is the label
 * shown to identify the contributing input or propagated output.
 *
 * @example
 * const entry: LinearLayoutHoverPopupEntry = {
 *   color: '#38bdf8',
 *   text: 'A[warp=0, lane=3]',
 * };
 *
 * console.assert(entry.text.startsWith('A['));
 */
export type LinearLayoutHoverPopupEntry = {
    color: string;
    text: string;
};

/**
 * Collects tensor-view slider and slice state that should survive regenerating
 * or saving a linear-layout tab.
 *
 * Inactive tabs use the last tab-local snapshot saved in extension state. The
 * active tab overlays that saved snapshot with the live viewer snapshot so
 * unsaved slice changes are preserved before new specs are applied.
 *
 * @param ctx - Linear-layout UI context that provides the active tab id, the per-tab `linearLayoutTensorViewsStates` cache, and the viewer snapshot.
 * @param tabId - Tab id to preserve, or `null` to return an empty preservation state; defaults to the current active tab id.
 * @returns A tensor-view state object suitable for passing into linear-layout tab regeneration or serialization.
 * @noThrows The function only reads maps, compares tab ids, spreads plain state objects, and snapshots the current viewer view state.
 * @example
 * const preserved = preservedLinearLayoutTensorViews(ctx, ctx.getActiveTabId());
 *
 * // The result can be supplied when applying regenerated tensor specs so the
 * // current tab keeps its visible slices instead of resetting to defaults.
 * applyLinearLayoutSpecsToTab(ctx, nextSpecs, ctx.viewer.getSnapshot(), undefined, preserved);
 */
export function preservedLinearLayoutTensorViews(
    ctx: LinearLayoutUiContext,
    tabId: string | null = ctx.getActiveTabId(),
): LinearLayoutTensorViewsState {
    const stored = tabId ? ctx.state.linearLayoutTensorViewsStates.get(tabId) ?? {} : {};
    // inactive tabs only have their last saved snapshots; the active tab also
    // needs the live viewer snapshot so applying specs preserves unsaved slices.
    if (!tabId || ctx.getActiveTabId() !== tabId) return { ...stored };
    return { ...stored, ...snapshotTensorViews(ctx.viewer.getSnapshot()) };
}

/**
 * Returns the cached selection mapping for a linear-layout tab, or derives it
 * from the tab manifest metadata and stores it for later hover, inspector, and
 * multi-input synchronization.
 *
 * Tabs without usable linear-layout metadata return `null`, allowing callers to
 * fall back to ordinary viewer selection behavior.
 *
 * @param ctx - Linear-layout UI context whose state contains the per-tab `linearLayoutSelectionMaps` cache.
 * @param tab - Loaded linear-layout bundle document whose id keys the cache and whose manifest metadata describes root inputs, outputs, and tensor coordinate mappings.
 * @returns The selection map used to translate viewer selections between source and output tensors, or `null` when the tab metadata cannot produce one.
 * @noThrows The function performs a cache lookup and delegates metadata interpretation to `linearLayoutSelectionMapForMeta`; missing metadata is represented by `null` rather than an exception.
 * @example
 * const map = linearLayoutSelectionMapForTab(ctx, linearLayoutTab);
 *
 * if (map) {
 *   console.assert(ctx.state.linearLayoutSelectionMaps.get(linearLayoutTab.id) === map);
 *   console.assert(map.orderedTensorIds.length > 0);
 * }
 */
export function linearLayoutSelectionMapForTab(ctx: LinearLayoutUiContext, tab: LoadedBundleDocument): LinearLayoutSelectionMap | null {
    const cached = ctx.state.linearLayoutSelectionMaps.get(tab.id);
    if (cached) return cached;
    // selection maps are pure metadata derived from the manifest, so cache them
    // per tab and invalidate when a tab is regenerated.
    const map = linearLayoutSelectionMapForMeta(tab);
    if (!map) return null;
    ctx.state.linearLayoutSelectionMaps.set(tab.id, map);
    return map;
}

/**
 * Builds the inspector coordinate rows shown while hovering a tensor cell in a linear-layout tab.
 *
 * When no linear-layout mapping is available, the inspector falls back to the hovered tensor's own
 * coordinate, shape, and axis labels. When a mapping is available, the hovered tensor coordinate is
 * translated to its shared root index and one row is returned for each tensor in layout order.
 *
 * @param _ctx - Linear-layout UI context passed by the extension hook; currently unused by this row builder.
 * @param hover - Live tensor hover record from the viewer, including the hovered tensor id/name and tensor coordinate, or a falsy value when no cell is hovered.
 * @param hoveredStatus - Viewer status for the hovered tensor, used for fallback axis labels and shape when no layout map is present.
 * @param linearLayout - Selection map for the active linear-layout tab, or null when the active tab has no usable mapping metadata.
 * @returns Inspector rows for the hover: an empty array for no hover or an unmapped coordinate, one fallback row without a layout map, or one root-aligned row per layout tensor.
 * @noThrows Missing hover, missing layout metadata, and unmapped root coordinates are handled with empty or fallback row arrays instead of throwing.
 * @example
 * const rows = inspectorCoordEntries(ctx, {
 *   tensorId: 'q',
 *   tensorName: 'Q',
 *   tensorCoord: [0, 1],
 * }, { axisLabels: ['row', 'col'], shape: [2, 2] }, null);
 *
 * expect(rows).toEqual([{
 *   title: 'Q',
 *   labels: ['row', 'col'],
 *   shape: [2, 2],
 *   coord: [0, 1],
 *   hovered: true,
 * }]);
 */
export function inspectorCoordEntries(
    _ctx: LinearLayoutUiContext,
    hover: ReturnType<TensorViewer['getHover']>,
    hoveredStatus: ReturnType<TensorViewer['getTensorStatus']> | null,
    linearLayout: LinearLayoutSelectionMap | null,
): InspectorCoordEntry[] {
    if (!hover) return [];
    if (!linearLayout) {
        return [{
            title: hover.tensorName,
            labels: hoveredStatus?.axisLabels.slice() ?? [],
            shape: hoveredStatus?.shape.slice() ?? [],
            coord: hover.tensorCoord,
            hovered: true,
        }];
    }
    const display = linearLayoutDisplayModel(_ctx, linearLayout);
    const rootIndex = displayedRootIndexForCoord(display, linearLayout, hover.tensorId, hover.tensorCoord);
    if (rootIndex === null) return [];
    return linearLayout.orderedTensorIds.map((tensorId) => {
        const entry = linearLayout.tensors.get(tensorId)!;
        return {
            title: entry.meta.title,
            labels: entry.meta.axisLabels,
            shape: entry.meta.shape,
            coord: coordFromKey(entry.rootToTensorKeys[rootIndex] ?? ''),
            hovered: tensorId === hover.tensorId,
        };
    });
}

/**
 * Applies the linear-layout cell-text overlay to viewer tensors for the active tab.
 *
 * If there is no active linear-layout tab, labels are cleared from every tensor in the inspector
 * model. If the active tab has a mapping but the selected cell-text options produce no labels, the
 * labels are cleared only from tensors in that mapping.
 *
 * @param ctx - Linear-layout UI context that provides the active tab, saved cell-text state, layout metadata, and viewer label mutator.
 * @returns Nothing; callers observe the effect through `viewer.setTensorCellLabels` calls that install or clear per-cell labels.
 * @noThrows The function treats missing active tabs, missing layout mappings, and disabled label options as label-clearing cases rather than exceptional states.
 * @example
 * const calls: Array<[string, string[] | null]> = [];
 * const ctx = makeLinearLayoutContext({
 *   activeTab: null,
 *   inspectorTensors: [{ id: 'A' }, { id: 'B' }],
 *   setTensorCellLabels: (id, labels) => calls.push([id, labels]),
 * });
 *
 * applyLinearLayoutCellText(ctx);
 *
 * expect(calls).toEqual([
 *   ['A', null],
 *   ['B', null],
 * ]);
 */
export function applyLinearLayoutCellText(ctx: LinearLayoutUiContext): void {
    const tab = activeLinearLayoutTab(ctx);
    if (!tab) {
        ctx.viewer.getInspectorModel().tensors.forEach((tensor) => ctx.viewer.setTensorCellLabels(tensor.id, null));
        return;
    }
    const labels = linearLayoutCellLabelsForTab(ctx, tab, ctx.state.linearLayoutCellTextState);
    if (!labels) {
        const mapping = linearLayoutSelectionMapForTab(ctx, tab);
        mapping?.orderedTensorIds.forEach((tensorId) => ctx.viewer.setTensorCellLabels(tensorId, null));
        return;
    }
    labels.forEach(({ tensorId, labels: tensorLabels }) => {
        ctx.viewer.setTensorCellLabels(tensorId, tensorLabels);
    });
}

/**
 * Creates the hover-popup rows that reveal every propagated root hidden inside a many-to-one layout cell.
 *
 * The popup is suppressed when there is no hover, no layout map, an injective layout, output-propagation
 * mode, an unknown tensor id, or a hovered coordinate that is not present in the mapped tensor.
 *
 * @param ctx - Linear-layout UI context containing propagation mode and cell-text preferences used to format popup labels and colors.
 * @param hover - Live viewer hover record with the tensor id and tensor coordinate under the pointer, or a falsy value when the pointer is not over a tensor cell.
 * @param linearLayout - Selection map for the active linear-layout tab, including tensor coordinate indexes, root indexes, labels, shapes, and color inputs.
 * @returns Popup entries for the roots contained in the hovered cell; each entry contains a CSS `rgb(...)` color and either configured cell text or axis-label coordinate text.
 * @noThrows Unsupported popup situations are represented by an empty entry array, and missing root colors fall back to black.
 * @example
 * const entries = linearLayoutHoverPopupEntries(ctx, { tensorId: 'input', tensorCoord: [0] }, manyToOneLayout);
 *
 * expect(entries).toEqual([
 *   { color: 'rgb(255 0 0)', text: 'X:0' },
 *   { color: 'rgb(0 0 255)', text: 'X:1' },
 * ]);
 *
 * expect(linearLayoutHoverPopupEntries(ctx, null, manyToOneLayout)).toEqual([]);
 */
export function linearLayoutHoverPopupEntries(
    ctx: LinearLayoutUiContext,
    hover: ReturnType<TensorViewer['getHover']>,
    linearLayout: LinearLayoutSelectionMap | null,
): LinearLayoutHoverPopupEntry[] {
    if (!hover || !linearLayout) return [];
    // injective layouts already have one root per cell.  The popup is only
    // needed when a many-to-one cell hides extra roots in input-propagation mode.
    if (linearLayout.injective || ctx.state.linearLayoutState.propagateOutputs) return [];
    const tensor = linearLayout.tensors.get(hover.tensorId);
    if (!tensor) return [];
    const flat = tensor.coordKeyToFlatIndex.get(coordKey(hover.tensorCoord));
    if (flat === undefined) return [];
    const rootColors = rootColorsForLayoutState(
        ctx.state.linearLayoutState.propagateOutputs ? linearLayout.finalOutputLabels : linearLayout.rootInputLabels,
        ctx.state.linearLayoutState.propagateOutputs ? linearLayout.finalOutputShape : linearLayout.rootInputShape,
        ctx.state.linearLayoutState,
    );
    return (tensor.cellRootIndexes[flat] ?? []).map((rootIndex) => {
        const coord = propagatedCoordForRoot(linearLayout, rootIndex, ctx.state.linearLayoutState.propagateOutputs);
        const color: [number, number, number] = (
            rootColors[propagatedIndexForRoot(linearLayout, rootIndex, ctx.state.linearLayoutState.propagateOutputs)] ?? [0, 0, 0]
        );
        return {
            color: `rgb(${color.map((value) => Math.round(value * 255)).join(' ')})`,
            text: linearLayoutCellTextForCoord(
                coord,
                ctx.state.linearLayoutState.propagateOutputs ? linearLayout.finalOutputLabels : linearLayout.rootInputLabels,
                ctx.state.linearLayoutCellTextState,
            ) || (
                ctx.state.linearLayoutState.propagateOutputs ? linearLayout.finalOutputLabels : linearLayout.rootInputLabels
            ).map((label, axis) => `${label}:${coord[axis] ?? 0}`).join('\n'),
        };
    });
}

/**
 * Reapplies viewer display filtering and cell-label overlays for the active linear-layout tab.
 *
 * This is used after tensor-view or multi-input state changes so the canvas, inspector labels, and
 * layout-derived visibility stay aligned with the active tab's selection map.
 *
 * @param ctx - Linear-layout UI context that exposes the active tab, layout metadata lookup, viewer display controls, and cell-text state.
 * @returns Nothing; callers observe updated viewer display state and tensor cell labels when the active tab has a valid linear-layout mapping.
 * @noThrows Non-linear-layout tabs, missing active tabs, and tabs without mapping metadata are no-op synchronization cases.
 * @example
 * const ctx = makeLinearLayoutContext({ activeTab: imageTab });
 *
 * syncLinearLayoutViewFilters(ctx);
 *
 * expect(ctx.viewer.setTensorCellLabels).not.toHaveBeenCalled();
 * expect(ctx.viewer.setTensorViewFilter).not.toHaveBeenCalled();
 */
export function syncLinearLayoutViewFilters(ctx: LinearLayoutUiContext): void {
    const tab = ctx.getActiveTab();
    if (!tab || !isLinearLayoutTab(tab)) return;
    const mapping = linearLayoutSelectionMapForTab(ctx, tab);
    if (!mapping) return;
    applyLinearLayoutDisplay(ctx);
    applyLinearLayoutCellText(ctx);
}

/**
 * Mirrors a committed viewer selection through the active linear-layout mapping.
 *
 * When the active tab contains linear-layout metadata, the selected source tensor
 * coordinates are translated to the corresponding layout coordinates and written
 * back to the viewer. The syncing flag prevents that viewer update from being
 * processed again as a fresh selection event.
 *
 * @param ctx - Linear-layout UI context that provides the active tab, mapping metadata, viewer instance, and recursion guard state.
 * @param selection - Tensor-id keyed coordinates reported by the viewer selection event before linear-layout remapping.
 * @returns Nothing; callers observe the effect through `ctx.viewer.setSelectedCoords` when a non-empty mapped selection differs from the original selection.
 * @noThrows The function only reads the active tab/mapping and returns early when synchronization is already in progress, no linear-layout tab is active, no mapping exists, or the mapped selection is empty/unchanged.
 * @example
 * const sourceSelection = new Map([["input", [[0, 1]]]]);
 * const mappedSelection = new Map([["layout", [[3, 4]]]]);
 * const calls: SelectionCoords[] = [];
 * const ctx = makeLinearLayoutCtx({
 *   activeTab: linearLayoutTab,
 *   mappedSelection,
 *   viewer: { setSelectedCoords: (next) => calls.push(next) },
 * });
 *
 * syncLinearLayoutSelection(ctx, sourceSelection);
 *
 * console.assert(calls.length === 1);
 * console.assert(calls[0].get("layout")?.[0]?.join(",") === "3,4");
 * console.assert(ctx.state.syncingLinearLayoutSelection === false);
 */
export function syncLinearLayoutSelection(ctx: LinearLayoutUiContext, selection: SelectionCoords): void {
    if (ctx.state.syncingLinearLayoutSelection) return;
    const tab = ctx.getActiveTab();
    if (!tab || !isLinearLayoutTab(tab)) return;
    const mapping = linearLayoutSelectionMapForTab(ctx, tab);
    if (!mapping) return;
    const nextSelection = mappedSelectionFromSource(ctx, selection, mapping);
    if (nextSelection.size === 0 || selectionsMatch(selection, nextSelection)) return;
    // viewer selection updates emit again; this guard prevents linear-layout
    // remapping from recursively feeding its own output back into the viewer.
    ctx.state.syncingLinearLayoutSelection = true;
    ctx.viewer.setSelectedCoords(nextSelection);
    ctx.state.syncingLinearLayoutSelection = false;
}

/**
 * Updates the viewer's preview selection, translating it through the active linear-layout tab when possible.
 *
 * Preview selections come from transient hover or drag feedback. Non-linear-layout
 * tabs keep those coordinates unchanged, while linear-layout tabs preview the
 * mapped coordinates so highlighted cells match the composed layout view.
 *
 * @param ctx - Linear-layout UI context that supplies the active tab, optional layout mapping metadata, and viewer preview-selection API.
 * @param selection - Tensor-id keyed coordinates from the viewer's current preview selection interaction.
 * @returns Nothing; callers observe the result through `ctx.viewer.setPreviewSelectedCoords` receiving either the original selection or its mapped layout coordinates.
 * @noThrows The function has no validation branch and no explicit throw path; if the active tab is missing, not a linear-layout tab, or lacks mapping metadata, it simply forwards the original preview selection.
 * @example
 * const preview = new Map([["input", [[1, 0]]]]);
 * const mappedPreview = new Map([["layout", [[5, 2]]]]);
 * let lastPreview: SelectionCoords | undefined;
 * const ctx = makeLinearLayoutCtx({
 *   activeTab: linearLayoutTab,
 *   mappedSelection: mappedPreview,
 *   viewer: { setPreviewSelectedCoords: (next) => { lastPreview = next; } },
 * });
 *
 * syncLinearLayoutSelectionPreview(ctx, preview);
 *
 * console.assert(lastPreview?.get("layout")?.[0]?.join(",") === "5,2");
 */
export function syncLinearLayoutSelectionPreview(ctx: LinearLayoutUiContext, selection: SelectionCoords): void {
    const tab = ctx.getActiveTab();
    if (!tab || !isLinearLayoutTab(tab)) {
        ctx.viewer.setPreviewSelectedCoords(selection);
        return;
    }
    const mapping = linearLayoutSelectionMapForTab(ctx, tab);
    ctx.viewer.setPreviewSelectedCoords(mapping ? mappedSelectionFromSource(ctx, selection, mapping) : selection);
}

/**
 * Returns the active loaded tab only when it belongs to the linear-layout extension.
 *
 * Helpers in this module use this guard before reading compose-layout metadata or
 * writing layout-specific viewer state, so ordinary tensor tabs are treated as no
 * active linear-layout session.
 *
 * @param ctx - Linear-layout UI context whose `getActiveTab()` result is inspected with `isLinearLayoutTab`.
 * @returns The active `LoadedBundleDocument` when it is a linear-layout tab; otherwise `null` for missing tabs or non-linear-layout documents.
 * @noThrows The helper only calls `ctx.getActiveTab()` and applies the linear-layout tab type guard before returning the tab or `null`.
 * @example
 * const ctx = { getActiveTab: () => tensorOnlyTab } as LinearLayoutUiContext;
 *
 * console.assert(activeLinearLayoutTab(ctx) === null);
 *
 * const layoutCtx = { getActiveTab: () => linearLayoutTab } as LinearLayoutUiContext;
 * console.assert(activeLinearLayoutTab(layoutCtx) === linearLayoutTab);
 */
function activeLinearLayoutTab(ctx: LinearLayoutUiContext): LoadedBundleDocument | null {
    const tab = ctx.getActiveTab();
    return tab && isLinearLayoutTab(tab) ? tab : null;
}

/**
 * Compares two tensor selection maps by tensor id and coordinate membership.
 *
 * Coordinate order is ignored: two selections match when they contain the same
 * tensor ids and each tensor has the same set of coordinate arrays after they are
 * converted to coordinate keys.
 *
 * @param left - Viewer selection map keyed by tensor id, with each value containing selected coordinate arrays for that tensor.
 * @param right - Selection map to compare against `left`, typically coordinates produced by linear-layout remapping.
 * @returns `true` when both maps contain identical tensor ids and identical coordinate sets for every tensor; `false` when a tensor is missing, counts differ, or any coordinate differs.
 * @noThrows The comparison is limited to `Map` lookups, array iteration, and coordinate-key construction; it does not mutate the inputs or call viewer APIs.
 * @example
 * const left = new Map([["tensor-a", [[0, 1], [2, 3]]]]);
 * const sameDifferentOrder = new Map([["tensor-a", [[2, 3], [0, 1]]]]);
 * const different = new Map([["tensor-a", [[0, 1], [9, 9]]]]);
 *
 * console.assert(selectionsMatch(left, sameDifferentOrder) === true);
 * console.assert(selectionsMatch(left, different) === false);
 */
function selectionsMatch(left: SelectionCoords, right: Map<string, number[][]>): boolean {
    if (left.size !== right.size) return false;
    for (const [tensorId, coords] of right) {
        const leftCoords = left.get(tensorId);
        if (!leftCoords) return false;
        const leftKeys = new Set(leftCoords.map((coord) => coordKey(coord)));
        const rightKeys = new Set(coords.map((coord) => coordKey(coord)));
        if (leftKeys.size !== rightKeys.size) return false;
        for (const key of rightKeys) {
            if (!leftKeys.has(key)) return false;
        }
    }
    return true;
}

/**
 * Chooses the tensor whose selected coordinates should drive linear-layout selection propagation.
 *
 * @param ctx - Linear-layout UI context whose viewer state may contain the currently active tensor id.
 * @param selection - Map from tensor id to selected tensor coordinates; tensors with no coordinate entries are ignored.
 * @param mapping - Linear-layout selection map whose ordered tensor ids define the fallback priority when no active selected tensor exists.
 * @returns The only selected tensor id, the active selected tensor id when multiple tensors are selected, the first selected tensor in mapping order as a fallback, or null when no mapped tensor has selected coordinates.
 * @noThrows Reads viewer state and collection contents only; absent selections and inactive tensors are represented by null or ordered fallbacks instead of exceptions.
 * @example
 * const selection = new Map<string, number[][]>([
 *   ['input', [[0, 1]]],
 *   ['output', [[0, 3]]],
 * ]);
 * const mapping = { orderedTensorIds: ['input', 'output'] } as LinearLayoutSelectionMap;
 * const ctx = { viewer: { getState: () => ({ activeTensorId: 'output' }) } } as LinearLayoutUiContext;
 *
 * selectionSourceTensorId(ctx, selection, mapping);
 * // => 'output'
 *
 * selectionSourceTensorId(ctx, new Map(), mapping);
 * // => null
 */
function selectionSourceTensorId(ctx: LinearLayoutUiContext, selection: SelectionCoords, mapping: LinearLayoutSelectionMap): string | null {
    const nonEmpty = mapping.orderedTensorIds.filter((tensorId) => (selection.get(tensorId)?.length ?? 0) > 0);
    if (nonEmpty.length === 0) return null;
    if (nonEmpty.length === 1) return nonEmpty[0]!;
    const activeId = ctx.viewer.getState().activeTensorId;
    return activeId && nonEmpty.includes(activeId) ? activeId : nonEmpty[0]!;
}

/**
 * Propagates a selection on one linear-layout tensor to matching coordinates on every mapped tensor.
 *
 * @param ctx - Linear-layout UI context used to read the active tensor and display slicing state that can hide some root indexes.
 * @param selection - Current viewer selection, keyed by tensor id, from which one non-empty tensor is chosen as the propagation source.
 * @param mapping - Linear-layout mapping that relates tensor coordinates to shared root indexes across the tab's tensors.
 * @returns A new selection map containing coordinates for tensors that share the source root indexes in the current display, or an empty map when no source tensor or source coordinates are selected.
 * @noThrows Missing source selections return an empty map, and the mapping/display helpers are used only to translate available coordinate arrays.
 * @example
 * const selection = new Map<string, number[][]>([['input', [[0, 1]]]]);
 * const nextSelection = mappedSelectionFromSource(ctx, selection, mapping);
 *
 * nextSelection.get('input');
 * // => [[0, 1]]
 * nextSelection.get('output');
 * // => coordinates whose root indexes match input[0, 1]
 *
 * mappedSelectionFromSource(ctx, new Map(), mapping).size;
 * // => 0
 */
function mappedSelectionFromSource(
    ctx: LinearLayoutUiContext,
    selection: SelectionCoords,
    mapping: LinearLayoutSelectionMap,
): SelectionCoords {
    const sourceTensorId = selectionSourceTensorId(ctx, selection, mapping);
    if (!sourceTensorId) return new Map();
    const sourceCoords = selection.get(sourceTensorId) ?? [];
    if (sourceCoords.length === 0) return new Map();
    const display = linearLayoutDisplayModel(ctx, mapping);
    const rootIndexes = rootIndexesForCoords(mapping, sourceTensorId, sourceCoords);
    const nextSelection = new Map<string, number[][]>();
    mapping.orderedTensorIds.forEach((tensorId) => {
        const coords = coordsForRootIndexes(mapping, tensorId, rootIndexes, display.sliceRootIndexes);
        if (coords.length) nextSelection.set(tensorId, coords);
    });
    return nextSelection;
}

/**
 * Formats the enabled linear-layout axis labels for a single tensor cell coordinate.
 *
 * @param coord - Tensor coordinate whose axis values are paired with label names by array index.
 * @param labels - Ordered root-input or final-output label names for the displayed coordinate axes.
 * @param state - Cell-label visibility flags; only labels with a truthy flag are included in the text.
 * @returns Newline-separated `label:value` lines for enabled labels that have a matching coordinate axis, or an empty string when no requested label can be shown.
 * @noThrows The formatter only indexes arrays and reads boolean label flags; disabled labels and labels beyond the coordinate length are skipped.
 * @example
 * linearLayoutCellTextForCoord([2, 5, 7], ['m', 'n', 'k'], { m: true, n: false, k: true });
 * // => 'm:2\nk:7'
 *
 * linearLayoutCellTextForCoord([2], ['m', 'n'], { n: true });
 * // => ''
 */
function linearLayoutCellTextForCoord(coord: number[], labels: string[], state: LinearLayoutCellTextState): string {
    return labels
        .flatMap((label, axis) => (state[label] && axis < coord.length ? [`${label}:${coord[axis] ?? 0}`] : []))
        .join('\n');
}

/**
 * Builds per-tensor cell label overlays for a loaded linear-layout tab.
 *
 * @param ctx - Linear-layout UI context containing the tab mapping state, propagate-outputs setting, and current display slicing model.
 * @param tab - Loaded bundle document whose linear-layout metadata is converted into tensor/root coordinate mappings.
 * @param state - Cell-label visibility flags selected by the user for root input labels or propagated output labels.
 * @returns One entry per mapped tensor with label text for displayable cells, or null when the tab has no linear-layout mapping or none of the relevant labels are enabled.
 * @noThrows Missing mappings and disabled label sets are treated as unavailable overlay data and return null; individual cells without a displayed root index are skipped.
 * @example
 * const labels = linearLayoutCellLabelsForTab(ctx, tab, { m: true, n: false });
 *
 * labels?.[0];
 * // => { tensorId: 'input', labels: [{ coord: [0, 0], text: 'm:0' }, ...] }
 *
 * linearLayoutCellLabelsForTab(ctx, tabWithoutLinearLayoutMetadata, { m: true });
 * // => null
 */
function linearLayoutCellLabelsForTab(
    ctx: LinearLayoutUiContext,
    tab: LoadedBundleDocument,
    state: LinearLayoutCellTextState,
): Array<{ tensorId: string; labels: Array<{ coord: number[]; text: string }> }> | null {
    const mapping = linearLayoutSelectionMapForTab(ctx, tab);
    const labels = ctx.state.linearLayoutState.propagateOutputs ? mapping?.finalOutputLabels : mapping?.rootInputLabels;
    if (!mapping || !labels?.some((label) => state[label])) return null;
    const display = linearLayoutDisplayModel(ctx, mapping);
    return mapping.orderedTensorIds.map((tensorId) => {
        const tensor = mapping.tensors.get(tensorId)!;
        const labels = tensor.cellRootIndexes.map((roots, flat) => {
            const rootIndex = display.displayedRootIndexByTensor.get(tensorId)?.[flat] ?? roots[0] ?? null;
            if (rootIndex === null) return null;
            return {
                coord: coordFromKey(tensor.rootToTensorKeys[rootIndex] ?? ''),
                text: linearLayoutCellTextForCoord(
                    propagatedCoordForRoot(mapping, rootIndex, ctx.state.linearLayoutState.propagateOutputs),
                    ctx.state.linearLayoutState.propagateOutputs ? mapping.finalOutputLabels : mapping.rootInputLabels,
                    state,
                ),
            };
        }).filter((entry): entry is { coord: number[]; text: string } => entry !== null);
        return { tensorId, labels };
    });
}

/**
 * Resolves a root cell index to the coordinate that viewer overlays should display.
 *
 * When Propagate Outputs is enabled, the coordinate comes from the root-to-final-output
 * mapping; otherwise it comes from the original root input coordinate. Hover popups,
 * ghost labels, and inspector rows use this coordinate with the matching input/output
 * axis labels.
 *
 * @param mapping - Linear-layout selection map containing `rootKeys` and `rootToFinalKeys` encoded as coordinate keys.
 * @param rootIndex - Index of the root element currently referenced by a tensor cell or ghost layer.
 * @param propagateOutputs - Whether to read the final-output coordinate (`true`) or the root-input coordinate (`false`).
 * @returns Decoded coordinate components for the selected root element; a missing key decodes as the empty coordinate.
 * @noThrows Missing mapping entries are normalized to an empty key before decoding, so absent propagation data does not create a throw path here.
 * @example
 * const mapping = {
 *   rootKeys: ['0,1'],
 *   rootToFinalKeys: ['2,3'],
 * } as LinearLayoutSelectionMap;
 *
 * propagatedCoordForRoot(mapping, 0, false); // [0, 1]
 * propagatedCoordForRoot(mapping, 0, true); // [2, 3]
 */
function propagatedCoordForRoot(mapping: LinearLayoutSelectionMap, rootIndex: number, propagateOutputs: boolean): number[] {
    const key = propagateOutputs ? mapping.rootToFinalKeys[rootIndex] : mapping.rootKeys[rootIndex];
    return coordFromKey(key ?? '');
}

/**
 * Flattens the displayed coordinate for a root cell into the color-table index used by viewer synchronization.
 *
 * The helper uses the same Propagate Outputs choice as `propagatedCoordForRoot`, then folds the coordinate
 * through either the root input shape or the final output shape in row-major order. Callers use the result to
 * pick the stable color for hover entries and ghost layers.
 *
 * @param mapping - Linear-layout selection map containing encoded root/final keys and their corresponding shapes.
 * @param rootIndex - Index of the root element whose displayed coordinate should be converted to a flat index.
 * @param propagateOutputs - Whether to flatten within `finalOutputShape` (`true`) or `rootInputShape` (`false`).
 * @returns Row-major flat index into the root-input or final-output color array for the selected root element.
 * @noThrows The function only reads mapping arrays and reduces numeric coordinates; callers provide the matching shape arrays created with the selection map.
 * @example
 * const mapping = {
 *   rootKeys: ['1,2'],
 *   rootToFinalKeys: ['0,3'],
 *   rootInputShape: [4, 5],
 *   finalOutputShape: [2, 8],
 * } as LinearLayoutSelectionMap;
 *
 * propagatedIndexForRoot(mapping, 0, false); // 7  (1 * 5 + 2)
 * propagatedIndexForRoot(mapping, 0, true); // 3   (0 * 8 + 3)
 */
function propagatedIndexForRoot(mapping: LinearLayoutSelectionMap, rootIndex: number, propagateOutputs: boolean): number {
    const coord = propagatedCoordForRoot(mapping, rootIndex, propagateOutputs);
    const shape = propagateOutputs ? mapping.finalOutputShape : mapping.rootInputShape;
    return coord.reduce((index, value, axis) => (index * shape[axis]!) + value, 0);
}
