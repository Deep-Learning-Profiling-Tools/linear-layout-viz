import {
    coordFromKey,
    coordKey,
    parseTensorView,
    serializeTensorViewEditor,
    visibleTensorCoords,
    type LoadedBundleDocument,
} from '@tensor-viz/viewer-core';
import { rootColorsForLayoutState } from './linear-layout.js';
import { composeLayoutMetaForTab, type LinearLayoutSelectionMap, type LinearLayoutUiContext } from './linear-layout-state.js';

/**
 * Display-time index map for a linear-layout selection, linking each tensor's visible coordinates to the root input cells currently shown in the viewer and to any ghosted duplicate roots.
 *
 * The multi-input hover and slider code uses this model to decide which root indexes are visible in the active slice, which root index each tensor coordinate represents, and which duplicate cells should be drawn as ghost overlays.
 *
 * @example
 * const display: LinearLayoutDisplayModel = {
 *   rootIndexes: new Set([0, 1, 2]),
 *   sliceRootIndexes: new Set([0, 2]),
 *   displayedRootIndexByTensor: new Map([
 *     ['accumulator', [0, null, 2]],
 *   ]),
 *   visibleCoordsByTensor: new Map([
 *     ['accumulator', [[0, 0], [0, 1], [0, 2]]],
 *   ]),
 *   ghostRootIndexesByTensor: new Map([
 *     ['accumulator', [{ coord: [0, 2], rootIndex: 1, layer: 0 }]],
 *   ]),
 * };
 *
 * expect(display.displayedRootIndexByTensor.get('accumulator')?.[2]).toBe(2);
 * expect(display.ghostRootIndexesByTensor.get('accumulator')?.[0].coord).toEqual([0, 2]);
 */
export type LinearLayoutDisplayModel = {
    rootIndexes: Set<number>;
    sliceRootIndexes: Set<number> | null;
    displayedRootIndexByTensor: Map<string, Array<number | null>>;
    visibleCoordsByTensor: Map<string, number[][]>;
    ghostRootIndexesByTensor: Map<string, Array<{ coord: number[]; rootIndex: number; layer: number }>>;
};

/**
 * Describes the optional slider shown when the focused linear-layout tensor cell
 * represents multiple root inputs. `null` means the UI should hide the slider,
 * either because no tensor is focused, output propagation is active, no mapping
 * is available, or every visible cell maps to at most one root input.
 *
 * @example
 * const visibleSlider: LinearLayoutMultiInputModel = {
 *     focusedTensorId: 'compose-step-2',
 *     value: 0,
 *     size: 4,
 * };
 *
 * const hiddenSlider: LinearLayoutMultiInputModel = null;
 */
export type LinearLayoutMultiInputModel = {
    focusedTensorId: string;
    value: number;
    size: number;
} | null;

/**
 * Builds the coordinate lookup tables used to keep linear-layout tensor hovers,
 * selections, colors, and ghost layers aligned with the root input space and the
 * final propagated output space for a loaded tab.
 *
 * @param tab - Loaded bundle document whose manifest tensor ids and embedded compose-layout metadata are inspected.
 * @returns A selection map containing root-input labels, final-output labels, tensor coordinate indexes, and loaded tensor ids, or `null` when the tab has no compose-layout metadata or the metadata contains no tensors.
 * @noThrows Missing or empty compose-layout metadata is treated as an unsupported tab and reported with `null`; the function only derives in-memory arrays and maps from the loaded document.
 * @example
 * const mapping = linearLayoutSelectionMapForMeta(tab);
 * if (mapping) {
 *     expect(mapping.orderedTensorIds).toContain('compose-step-1');
 *     expect(mapping.rootKeyToIndex.get('0')).toBe(0);
 * } else {
 *     expect(mapping).toBeNull();
 * }
 */
export function linearLayoutSelectionMapForMeta(
    tab: LoadedBundleDocument,
): LinearLayoutSelectionMap | null {
    const meta = composeLayoutMetaForTab(tab);
    if (!meta || meta.tensors.length === 0) return null;
    const loadedTensorIds = new Set(tab.manifest.tensors.map((tensor) => tensor.id));
    const finalOutputShape = meta.finalOutputBitCounts.map((bits) => bits === 0 ? 1 : 2 ** bits);
    const rootInputShape = meta.rootInputBitCounts.map((bits) => bits === 0 ? 1 : 2 ** bits);
    const rootKeys = meta.tensors[0]!.rootToTensor.map((coord) => coordKey(coord));
    const rootToFinalKeys = meta.tensors[0]!.tensorToFinal.map((coord) => coord ? coordKey(coord) : '');
    const tensors = new Map<string, LinearLayoutSelectionMap['tensors'] extends Map<string, infer T> ? T : never>();
    meta.tensors.forEach((tensorMeta) => {
        if (!loadedTensorIds.has(tensorMeta.id)) return;
        const rootToTensorKeys = tensorMeta.rootToTensor.map((coord) => coordKey(coord));
        const coordKeyToFlatIndex = new Map<string, number>();
        const cellRootIndexes = Array.from({ length: tensorMeta.shape.reduce((total, value) => total * value, 1) }, () => [] as number[]);
        // non-injective tensors can map many root inputs into one cell.  Keep
        // all roots by flat cell so hover, selection, and ghost layers agree.
        rootToTensorKeys.forEach((tensorKey, rootIndex) => {
            const flat = coordFromKey(tensorKey).reduce((index, value, axis) => (index * tensorMeta.shape[axis]!) + value, 0);
            coordKeyToFlatIndex.set(tensorKey, flat);
            cellRootIndexes[flat]!.push(rootIndex);
        });
        tensors.set(tensorMeta.id, { meta: tensorMeta, rootToTensorKeys, coordKeyToFlatIndex, cellRootIndexes });
    });
    return {
        injective: meta.injective,
        rootInputLabels: meta.rootInputLabels.slice(),
        rootInputShape,
        rootKeys: rootKeys.slice(),
        rootKeyToIndex: new Map(rootKeys.map((key, index) => [key, index])),
        finalOutputLabels: meta.finalOutputLabels.slice(),
        finalOutputShape,
        rootToFinalKeys,
        tensors,
        orderedTensorIds: meta.tensors.map((tensor) => tensor.id).filter((id) => tensors.has(id)),
    };
}

/**
 * Decides whether the focused tensor needs the multi-input slider and, when it
 * does, returns the slider range and selected root-input offset for that tensor.
 *
 * @param ctx - Linear-layout UI context that provides the active tensor id, the propagate-outputs flag, and saved per-tensor slider positions.
 * @param mapping - Selection map for the active linear-layout tab, or `null` when the tab cannot provide linear-layout coordinate metadata.
 * @returns Slider state for the focused non-injective tensor cell: `focusedTensorId`, `size` as the largest number of root inputs sharing one tensor cell, and `value` clamped into `[-1, size - 1]`; returns `null` when no slider should be rendered.
 * @noThrows Missing mapping, missing focus, propagated-output mode, unknown tensor ids, and one-to-one mappings are normal UI states and return `null` instead of raising an error.
 * @example
 * const model = linearLayoutMultiInputModel(ctx, mapping);
 * expect(model).toEqual({
 *     focusedTensorId: 'compose-step-2',
 *     size: 4,
 *     value: 0,
 * });
 *
 * ctx.state.linearLayoutState.propagateOutputs = true;
 * expect(linearLayoutMultiInputModel(ctx, mapping)).toBeNull();
 */
export function linearLayoutMultiInputModel(
    ctx: LinearLayoutUiContext,
    mapping: LinearLayoutSelectionMap | null,
): LinearLayoutMultiInputModel {
    const focusedTensorId = ctx.viewer.getState().activeTensorId;
    if (!mapping || !focusedTensorId) return null;
    if (ctx.state.linearLayoutState?.propagateOutputs) return null;
    const tensor = mapping.tensors.get(focusedTensorId);
    if (!tensor) return null;
    const size = Math.max(0, ...tensor.cellRootIndexes.map((roots) => roots.length));
    // the slider exists only for many-to-one cells; injective or currently
    // one-to-one views should not expose an extra control.
    if (size <= 1) return null;
    const storedValue = ctx.state.linearLayoutMultiInputState[focusedTensorId] ?? -1;
    const value = storedValue < 0 ? -1 : Math.min(size - 1, storedValue);
    return { focusedTensorId, value, size };
}

/**
 * Synchronizes the active linear-layout tab into the tensor viewer by writing
 * per-cell root indexes, RGB colors, visible coordinates, and ghost layers for
 * every loaded tensor in the layout mapping.
 *
 * @param ctx - Linear-layout UI context that supplies the active tab, layout state, cached selection maps, and viewer rendering methods such as `setTensorData`, `colorTensor`, `setTensorVisibleCoords`, and `setTensorGhostLayers`.
 * @returns Nothing; callers observe the update through the viewer tensors being recolored, sliced, and assigned ghost-layer annotations.
 * @noThrows If there is no active tab or the tab has no linear-layout selection map, the function returns before touching the viewer; otherwise it performs deterministic viewer API calls from existing metadata and UI state.
 * @example
 * applyLinearLayoutDisplay(ctx);
 *
 * expect(ctx.viewer.setTensorData).toHaveBeenCalledWith(
 *     'compose-step-1',
 *     expect.any(Float32Array),
 *     'float32',
 * );
 * expect(ctx.viewer.colorTensor).toHaveBeenCalledWith('compose-step-1', expect.any(Float32Array));
 * expect(ctx.viewer.setTensorVisibleCoords).toHaveBeenCalledWith('compose-step-1', expect.any(Array));
 */
export function applyLinearLayoutDisplay(ctx: LinearLayoutUiContext): void {
    const tab = ctx.getActiveTab();
    if (!tab) return;
    const mapping = linearLayoutSelectionMapForTab(ctx, tab);
    if (!mapping) return;
    const display = linearLayoutDisplayModel(ctx, mapping);
    const [colorLabels, colorShape] = ctx.state.linearLayoutState.propagateOutputs
        ? [mapping.finalOutputLabels, mapping.finalOutputShape]
        : [mapping.rootInputLabels, mapping.rootInputShape];
    const colors = rootColorsForLayoutState(
        colorLabels,
        colorShape,
        ctx.state.linearLayoutState,
    );
    mapping.orderedTensorIds.forEach((tensorId) => {
        const tensor = mapping.tensors.get(tensorId)!;
        const displayed = display.displayedRootIndexByTensor.get(tensorId) ?? [];
        const data = new Float32Array(tensor.meta.shape.reduce((total, value) => total * value, 1)).fill(-1);
        const rgb = new Float32Array(data.length * 3);
        displayed.forEach((rootIndex, flat) => {
            if (rootIndex === null) return;
            data[flat] = rootIndex;
            rgb.set(colors[propagatedIndexForRoot(mapping, rootIndex, ctx.state.linearLayoutState.propagateOutputs)]!, flat * 3);
        });
        // data, colors, visible coords, and ghost layers are updated together so
        // rendering cannot show stale hidden roots after slicing or slider edits.
        ctx.viewer.setTensorData(tensorId, data, 'float32');
        ctx.viewer.colorTensor(tensorId, rgb);
        ctx.viewer.setTensorVisibleCoords(tensorId, display.visibleCoordsByTensor.get(tensorId) ?? []);
        ctx.viewer.setTensorGhostLayers(tensorId, ctx.state.linearLayoutState.propagateOutputs ? null : display.ghostRootIndexesByTensor.get(tensorId)?.map((entry) => ({
            coord: entry.coord,
            color: colors[propagatedIndexForRoot(mapping, entry.rootIndex, ctx.state.linearLayoutState.propagateOutputs)]!
                .map((value) => Math.round(value * 255)) as [number, number, number],
            bias: [entry.layer * 0.18, -(entry.layer * 0.18)] as const,
            layer: entry.layer,
            text: linearLayoutGhostText(
                propagatedCoordForRoot(mapping, entry.rootIndex, ctx.state.linearLayoutState.propagateOutputs),
                ctx.state.linearLayoutState.propagateOutputs ? mapping.finalOutputLabels : mapping.rootInputLabels,
                ctx.state.linearLayoutCellTextState,
            ),
        })) ?? null);
    });
}

/**
 * Builds the linear-layout render model that decides which compose-root indexes are visible in each tensor view.
 *
 * The model intersects active tensor-view slice selections, optionally narrows the result to the root selected by the focused multi-input slider, and records the root displayed in each tensor cell. When a tensor cell maps to multiple roots, the first root is rendered as the main cell and the remaining roots are returned as ghost layers.
 *
 * @param ctx - Linear-layout UI context containing the viewer slice state, focused tensor-view slider state, and current extension settings.
 * @param mapping - Selection map produced from linear-layout metadata, including ordered tensor ids, root keys, tensor shapes, coordinate lookup tables, and per-cell root-index memberships.
 * @returns Display model used by render and selection synchronization code: the visible root-index set, the slice-only root-index set, displayed root indexes by tensor cell, visible coordinates by tensor, and ghost root layers for non-injective cells.
 * @noThrows The function only reads Maps, Sets, arrays, and context state; missing optional focus/slice state is treated as an absent filter rather than as an exceptional condition.
 * @example
 * const display = linearLayoutDisplayModel(ctx, mapping);
 *
 * expect(Array.from(display.rootIndexes)).toEqual([2]);
 * expect(display.displayedRootIndexByTensor.get('compose-root')).toEqual([null, null, 2, null]);
 */
export function linearLayoutDisplayModel(
    ctx: LinearLayoutUiContext,
    mapping: LinearLayoutSelectionMap,
): LinearLayoutDisplayModel {
    const sliceVisibleRootIndexes = sliceVisibleRootIndexesByTensor(ctx, mapping);
    const slicedRoots = intersectRootIndexes(sliceVisibleRootIndexes.values(), mapping.rootKeys.length);
    const multiInput = linearLayoutMultiInputModel(ctx, mapping);
    // visibility is the intersection of active tensor-view slices, then
    // optionally narrowed to one many-to-one member by the focused tensor slider.
    const focusedRoots = multiInput
        ? focusedRootIndexes(mapping, multiInput.focusedTensorId, multiInput.value, sliceVisibleRootIndexes)
        : null;
    const rootIndexes = focusedRoots ?? slicedRoots ?? new Set(Array.from({ length: mapping.rootKeys.length }, (_entry, index) => index));
    const displayedRootIndexByTensor = new Map<string, Array<number | null>>();
    const visibleCoordsByTensor = new Map<string, number[][]>();
    const ghostRootIndexesByTensor = new Map<string, Array<{ coord: number[]; rootIndex: number; layer: number }>>();
    mapping.orderedTensorIds.forEach((tensorId) => {
        const tensor = mapping.tensors.get(tensorId)!;
        const visibleRoots = tensor.cellRootIndexes.map((roots) => roots.filter((rootIndex) => rootIndexes.has(rootIndex)));
        const displayed = visibleRoots.map((roots) => roots[0] ?? null);
        displayedRootIndexByTensor.set(tensorId, displayed);
        visibleCoordsByTensor.set(tensorId, displayed.flatMap((rootIndex, flat) => (
            rootIndex === null ? [] : [unravelIndex(flat, tensor.meta.shape)]
        )));
        ghostRootIndexesByTensor.set(tensorId, visibleRoots.flatMap((roots, flat) => (
            // root zero is rendered as the main cell; additional roots become
            // offset ghost layers so non-injective cells remain inspectable.
            roots.slice(1).map((rootIndex, layer) => ({
                coord: unravelIndex(flat, tensor.meta.shape),
                rootIndex,
                layer: layer + 1,
            }))
        )));
    });
    return { rootIndexes, sliceRootIndexes: slicedRoots, displayedRootIndexByTensor, visibleCoordsByTensor, ghostRootIndexesByTensor };
}

/**
 * Converts selected tensor coordinates into the compose-root indexes represented by those cells.
 *
 * This is used when a user selects cells in one tensor view and the linear-layout extension needs the shared root indexes to project that selection into the other tensor views.
 *
 * @param mapping - Linear-layout selection map containing the tensor coordinate-to-flat-index table and the root indexes attached to each tensor cell.
 * @param tensorId - Identifier of the tensor whose coordinates were selected, such as `compose-step-1` or `compose-root`.
 * @param coords - Tensor coordinates from the viewer selection, with each entry matching the tensor rank, for example `[[0]]` for a one-dimensional cell.
 * @returns Set of root indexes attached to the requested coordinates; missing tensor ids, coordinates outside the tensor, and cells without roots contribute no entries.
 * @noThrows The tensor and coordinate lookups are guarded: an unknown tensor id or coordinate key returns an empty contribution instead of throwing.
 * @example
 * const selectedRoots = rootIndexesForCoords(mapping, 'compose-step-1', [[0]]);
 *
 * expect(Array.from(selectedRoots)).toEqual([0, 2]);
 * expect(Array.from(rootIndexesForCoords(mapping, 'missing-tensor', [[0]]))).toEqual([]);
 */
export function rootIndexesForCoords(
    mapping: LinearLayoutSelectionMap,
    tensorId: string,
    coords: number[][],
): Set<number> {
    const tensor = mapping.tensors.get(tensorId);
    if (!tensor) return new Set();
    return new Set(coords.flatMap((coord) => {
        const flat = tensor.coordKeyToFlatIndex.get(coordKey(coord));
        return flat === undefined ? [] : tensor.cellRootIndexes[flat] ?? [];
    }));
}

/**
 * Projects compose-root selections back into the coordinates of a tensor view.
 *
 * A tensor coordinate is returned when any root attached to that tensor cell is in `selectedRootIndexes`. When `visibleRootIndexes` is provided, the coordinate must also contain at least one root that survives the current slice filter.
 *
 * @param mapping - Linear-layout selection map containing tensor shapes and each cell's compose-root memberships.
 * @param tensorId - Identifier of the tensor view to project into, such as `compose-root` or an intermediate compose step.
 * @param selectedRootIndexes - Root indexes gathered from the source selection and propagated through the linear-layout graph.
 * @param visibleRootIndexes - Optional slice-filter root set from the display model; pass `null` to include matching coordinates even when they are outside the current slice filter.
 * @returns Tensor coordinates whose cells match the selected roots and, when supplied, the visible-root filter; returns an empty array for unknown tensors or an empty selected-root set.
 * @noThrows The function checks for a missing tensor and empty selection before reading cell data, and the optional visibility filter is handled as a normal `null` case.
 * @example
 * const selectedRoots = new Set([0, 2]);
 *
 * expect(coordsForRootIndexes(mapping, 'compose-root', selectedRoots, null)).toEqual([[0], [2]]);
 * expect(coordsForRootIndexes(mapping, 'compose-root', selectedRoots, new Set([2]))).toEqual([[2]]);
 */
export function coordsForRootIndexes(
    mapping: LinearLayoutSelectionMap,
    tensorId: string,
    selectedRootIndexes: Set<number>,
    visibleRootIndexes: Set<number> | null = null,
): number[][] {
    const tensor = mapping.tensors.get(tensorId);
    if (!tensor || selectedRootIndexes.size === 0) return [];
    return tensor.cellRootIndexes.flatMap((roots, flat) => {
        const matchesSelection = roots.some((rootIndex) => selectedRootIndexes.has(rootIndex));
        const matchesVisible = visibleRootIndexes === null || roots.some((rootIndex) => visibleRootIndexes.has(rootIndex));
        return matchesSelection && matchesVisible ? [unravelIndex(flat, tensor.meta.shape)] : [];
    });
}

/**
 * Reads the root index currently rendered for one tensor cell in a linear-layout display model.
 *
 * Hover popups and inspector rows use this helper to connect a visible tensor coordinate back to the compose-root entry that the display model chose for that cell.
 *
 * @param display - Display model returned by `linearLayoutDisplayModel`, including the per-tensor array of displayed root indexes.
 * @param mapping - Linear-layout selection map used to translate the tensor coordinate into the flat cell index used by the display arrays.
 * @param tensorId - Identifier of the tensor that owns the coordinate being inspected.
 * @param coord - Tensor coordinate to inspect, with one number per tensor axis.
 * @returns The displayed compose-root index for the cell, or `null` when the tensor id is unknown, the coordinate is outside the tensor, or the display model has no root for that cell.
 * @noThrows Unknown tensor ids, unknown coordinate keys, and missing display entries are all checked with guarded lookups and return `null`.
 * @example
 * const display = linearLayoutDisplayModel(ctx, mapping);
 *
 * expect(displayedRootIndexForCoord(display, mapping, 'compose-root', [2])).toBe(2);
 * expect(displayedRootIndexForCoord(display, mapping, 'compose-root', [99])).toBeNull();
 */
export function displayedRootIndexForCoord(
    display: LinearLayoutDisplayModel,
    mapping: LinearLayoutSelectionMap,
    tensorId: string,
    coord: number[],
): number | null {
    const tensor = mapping.tensors.get(tensorId);
    if (!tensor) return null;
    const flat = tensor.coordKeyToFlatIndex.get(coordKey(coord));
    if (flat === undefined) return null;
    return display.displayedRootIndexByTensor.get(tensorId)?.[flat] ?? null;
}

/**
 * Narrows the visible linear-layout roots to the member selected by the focused tensor's multi-input slider.
 *
 * Each tensor cell can map to several root indexes. This picks the `index`th root from each focused-tensor cell,
 * after applying any tensor-view slice filter for that tensor, so hover and display synchronization can show one
 * many-to-one member at a time.
 *
 * @param mapping - Linear-layout selection map whose `tensors` entry contains `focusedTensorId` and per-cell root-index lists.
 * @param focusedTensorId - Tensor id for the slider that currently controls the many-to-one focus.
 * @param index - Zero-based slider position to select from each focused tensor cell's root list; negative values disable focus.
 * @param sliceVisibleRootIndexes - Root indexes still visible for each tensor after tensor-view slicing.
 * @returns A set of selected root indexes, or `null` when the index is negative or the focused tensor is not present in the mapping.
 * @noThrows Missing focused tensors and disabled slider indexes are represented as `null`; missing slice filters mean all roots in the focused tensor remain eligible.
 * @example
 * const mapping = {
 *   tensors: new Map([
 *     ['rhs', { cellRootIndexes: [[0, 2], [1, 3]] }],
 *   ]),
 * } as LinearLayoutSelectionMap;
 * const visibleByTensor = new Map([['rhs', new Set([2, 3])]]);
 *
 * Array.from(focusedRootIndexes(mapping, 'rhs', 0, visibleByTensor)!).sort();
 * // => [2, 3]
 *
 * focusedRootIndexes(mapping, 'rhs', -1, visibleByTensor);
 * // => null
 */
function focusedRootIndexes(
    mapping: LinearLayoutSelectionMap,
    focusedTensorId: string,
    index: number,
    sliceVisibleRootIndexes: Map<string, Set<number>>,
): Set<number> | null {
    if (index < 0) return null;
    const tensor = mapping.tensors.get(focusedTensorId);
    if (!tensor) return null;
    const visibleRoots = sliceVisibleRootIndexes.get(focusedTensorId) ?? null;
    return new Set(tensor.cellRootIndexes.flatMap((roots) => {
        const filteredRoots = visibleRoots ? roots.filter((rootIndex) => visibleRoots.has(rootIndex)) : roots;
        const rootIndex = filteredRoots[index];
        return rootIndex === undefined ? [] : [rootIndex];
    }));
}

/**
 * Builds the per-tensor root visibility filters implied by each tensor's current tensor-view slice.
 *
 * The display model uses this map to hide linear-layout roots that are outside the active slices before applying
 * multi-input focus. Tensors whose view does not resolve to any visible roots are omitted.
 *
 * @param ctx - Linear-layout UI context whose viewer provides tensor status, tensor-view editor snapshots, and hidden indices.
 * @param mapping - Selection map that orders tensor ids and maps visible tensor coordinates back to linear-layout root indexes.
 * @returns A map from tensor id to the root indexes visible in that tensor's parsed slice; tensors with no visible roots are absent.
 * @noThrows Invalid tensor-view text is converted by `slicedTensorCoords` into `null`, which contributes an empty root set that is filtered out.
 * @example
 * const visibleByTensor = sliceVisibleRootIndexesByTensor(ctx, mapping);
 *
 * visibleByTensor.get('lhs');
 * // => Set containing the root indexes for coordinates still visible in the lhs tensor view
 *
 * visibleByTensor.has('tensor-with-invalid-view');
 * // => false
 */
function sliceVisibleRootIndexesByTensor(
    ctx: LinearLayoutUiContext,
    mapping: LinearLayoutSelectionMap,
): Map<string, Set<number>> {
    return new Map(mapping.orderedTensorIds.map((tensorId) => {
        const coords = slicedTensorCoords(ctx, tensorId);
        return [tensorId, coords ? rootIndexesForCoords(mapping, tensorId, coords) : new Set<number>()] as const;
    }).filter(([_tensorId, roots]) => roots.size > 0));
}

/**
 * Finds the linear-layout root indexes that remain visible in every active tensor-view slice.
 *
 * The result is the shared visibility mask used before optional focused-tensor narrowing. An empty iterable means no
 * tensor-view slice is constraining the display.
 *
 * @param sets - Visible root-index sets produced for each tensor that currently has an active slice filter.
 * @param rootCount - Total number of roots in the mapping; accepted for call-site symmetry with visibility calculations but not needed to compute the intersection.
 * @returns A set containing only root indexes present in every supplied set, or `null` when no sets were supplied.
 * @noThrows The function only iterates the supplied sets and allocates result sets; absence of slice filters is reported as `null` rather than an exception.
 * @example
 * const visibleRoots = intersectRootIndexes([
 *   new Set([0, 1, 3]),
 *   new Set([1, 3, 4]),
 * ], 5);
 *
 * Array.from(visibleRoots!).sort();
 * // => [1, 3]
 *
 * intersectRootIndexes([], 5);
 * // => null
 */
function intersectRootIndexes(sets: Iterable<Set<number>>, rootCount: number): Set<number> | null {
    let intersection: Set<number> | null = null;
    for (const set of sets) {
        intersection = intersection
            ? new Set(Array.from<number>(intersection).filter((rootIndex) => set.has(rootIndex)))
            : new Set(set);
    }
    if (intersection) return intersection;
    return null;
}

/**
 * Converts a tensor's current tensor-view editor state into the coordinates that remain visible after slicing.
 *
 * The linear-layout extension uses these coordinates to map viewer slices back to composed root indexes for
 * multi-input visibility and hover synchronization.
 *
 * @param ctx - Linear-layout UI context whose viewer can read the tensor status and current tensor-view snapshot.
 * @param tensorId - Id of the tensor whose shape, axis labels, hidden indices, and editor text should be parsed.
 * @returns Visible tensor coordinates as index arrays, or `null` when the tensor-view editor text cannot be parsed for the tensor shape.
 * @noThrows Tensor-view syntax errors are returned as `null` parse results, letting callers treat invalid editor state as no slice-derived visibility filter.
 * @example
 * const coords = slicedTensorCoords(ctx, 'lhs');
 *
 * coords;
 * // => [[0, 0], [0, 1]] for a 2-D tensor view that leaves the first row visible
 *
 * // If the lhs tensor-view editor contains invalid syntax:
 * slicedTensorCoords(ctx, 'lhs');
 * // => null
 */
function slicedTensorCoords(ctx: LinearLayoutUiContext, tensorId: string): number[][] | null {
    const status = ctx.viewer.getTensorStatus(tensorId);
    const snapshot = ctx.viewer.getTensorView(tensorId);
    const parsed = parseTensorView(
        status.shape.slice(),
        serializeTensorViewEditor(snapshot.editor),
        snapshot.hiddenIndices,
        status.axisLabels,
    );
    return !parsed.ok ? null : visibleTensorCoords(parsed.spec);
}

/**
 * Decodes a row-major flat tensor position into one coordinate per axis of the supplied shape.
 *
 * @param index - Zero-based flat position in storage order for a tensor with the supplied dimensions.
 * @param shape - Tensor dimensions ordered from outermost axis to innermost axis; an empty shape represents a scalar.
 * @returns Coordinate tuple for the flat position, with `coord.length === shape.length`; scalars return an empty tuple.
 * @noThrows Uses only array allocation and arithmetic on caller-supplied numbers; scalar shapes return before indexing.
 * @example
 * unravelIndex(5, [2, 3]);
 * // => [1, 2]
 *
 * unravelIndex(0, []);
 * // => []
 */
function unravelIndex(index: number, shape: number[]): number[] {
    if (shape.length === 0) return [];
    const coord = new Array(shape.length).fill(0);
    let remainder = index;
    for (let axis = shape.length - 1; axis >= 0; axis -= 1) {
        const size = shape[axis] ?? 1;
        coord[axis] = remainder % size;
        remainder = Math.floor(remainder / size);
    }
    return coord;
}

/**
 * Builds the hover ghost text that shows enabled linear-layout axis labels beside their coordinate values.
 *
 * @param coord - Coordinate tuple for the hovered root or propagated tensor cell.
 * @param labels - Axis labels in the same order as `coord`; labels past the coordinate length are ignored.
 * @param state - Map whose truthy entries mark which axis labels should be shown in the hover text.
 * @returns Newline-delimited `label:value` lines for enabled labels, or `null` when no enabled label has a coordinate.
 * @noThrows Only filters and formats the provided arrays and object; missing coordinate entries default to `0`.
 * @example
 * linearLayoutGhostText([3, 1, 0], ['m', 'n', 'k'], { m: true, n: false, k: true });
 * // => 'm:3\nk:0'
 *
 * linearLayoutGhostText([3, 1], ['m', 'n'], { m: false, n: false });
 * // => null
 */
function linearLayoutGhostText(coord: number[], labels: string[], state: Record<string, boolean>): string | null {
    const text = labels
        .flatMap((label, axis) => (state[label] && axis < coord.length ? [`${label}:${coord[axis] ?? 0}`] : []))
        .join('\n');
    return text || null;
}

/**
 * Resolves the coordinate displayed for a root input cell, either in root-input space or after propagation to final-output space.
 *
 * @param mapping - Linear-layout selection map containing `rootKeys` for input coordinates and `rootToFinalKeys` for propagated output coordinates.
 * @param rootIndex - Zero-based index of the root input cell whose selection or hover coordinate is being displayed.
 * @param propagateOutputs - Whether the Propagate Outputs control is enabled and final-output coordinates should be used.
 * @returns Coordinate decoded from the selected mapping key; missing keys resolve to the empty coordinate key.
 * @noThrows Reads mapping arrays with fallback to an empty key before decoding, so absent entries do not throw.
 * @example
 * const mapping = {
 *   rootKeys: ['0,0', '0,1'],
 *   rootToFinalKeys: ['1,0', '1,1'],
 * } as LinearLayoutSelectionMap;
 *
 * propagatedCoordForRoot(mapping, 1, false);
 * // => [0, 1]
 *
 * propagatedCoordForRoot(mapping, 1, true);
 * // => [1, 1]
 */
function propagatedCoordForRoot(mapping: LinearLayoutSelectionMap, rootIndex: number, propagateOutputs: boolean): number[] {
    const key = propagateOutputs ? mapping.rootToFinalKeys[rootIndex] : mapping.rootKeys[rootIndex];
    return coordFromKey(key ?? '');
}

/**
 * Converts a root cell's displayed coordinate into the row-major flat index used to look up propagated selection colors.
 *
 * @param mapping - Linear-layout selection map containing root/final coordinate keys and their corresponding input/output shapes.
 * @param rootIndex - Zero-based index of the root input cell whose color-buffer position is needed.
 * @param propagateOutputs - Whether to flatten the propagated final-output coordinate instead of the original root-input coordinate.
 * @returns Row-major flat index within `rootInputShape` when propagation is off, or within `finalOutputShape` when it is on.
 * @noThrows Delegates coordinate lookup to `propagatedCoordForRoot` and reduces over the returned coordinate; missing keys reduce to `0`.
 * @example
 * const mapping = {
 *   rootInputShape: [2, 3],
 *   finalOutputShape: [3, 2],
 *   rootKeys: ['0,0', '0,1'],
 *   rootToFinalKeys: ['1,0', '1,1'],
 * } as LinearLayoutSelectionMap;
 *
 * propagatedIndexForRoot(mapping, 1, false);
 * // => 1
 *
 * propagatedIndexForRoot(mapping, 1, true);
 * // => 3
 */
function propagatedIndexForRoot(mapping: LinearLayoutSelectionMap, rootIndex: number, propagateOutputs: boolean): number {
    const shape = propagateOutputs ? mapping.finalOutputShape : mapping.rootInputShape;
    return propagatedCoordForRoot(mapping, rootIndex, propagateOutputs)
        .reduce((index, value, axis) => (index * shape[axis]!) + value, 0);
}

/**
 * Retrieves the selection-synchronization map for a loaded linear-layout tab, reusing the per-tab cache before parsing the tab metadata.
 *
 * Hover popups, multi-input sliders, and inspector coordinate rows use this map to translate viewer selections back to propagated linear-layout labels.
 *
 * @param ctx - Linear-layout UI context whose state owns the `linearLayoutSelectionMaps` cache keyed by tab id.
 * @param tab - Loaded viewer tab that may contain compose-layout metadata embedded when the layout was rendered.
 * @returns The cached or newly parsed selection map for `tab`, or `null` when the tab metadata does not describe a linear-layout selection mapping.
 * @noThrows Cache lookup and metadata probing are guarded by null checks; tabs without usable mapping metadata are reported as `null` instead of raising an error.
 * @example
 * const first = linearLayoutSelectionMapForTab(ctx, linearLayoutTab);
 * if (first) {
 *   console.assert(ctx.state.linearLayoutSelectionMaps.get(linearLayoutTab.id) === first);
 *   console.assert(linearLayoutSelectionMapForTab(ctx, linearLayoutTab) === first);
 * }
 *
 * const missing = linearLayoutSelectionMapForTab(ctx, ordinaryTensorTab);
 * console.assert(missing === null);
 */
function linearLayoutSelectionMapForTab(ctx: LinearLayoutUiContext, tab: LoadedBundleDocument): LinearLayoutSelectionMap | null {
    const cached = ctx.state.linearLayoutSelectionMaps.get(tab.id);
    if (cached) return cached;
    const mapping = linearLayoutSelectionMapForMeta(tab);
    if (!mapping) return null;
    ctx.state.linearLayoutSelectionMaps.set(tab.id, mapping);
    return mapping;
}
