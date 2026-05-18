import type { LoadedBundleDocument, TensorViewSnapshot, ViewerSnapshot, TensorViewer } from '@tensor-viz/viewer-core';
import { escapeInfo } from '@tensor-viz/viewer-demo/extension-api';
import {
    autoColorLayoutState,
    buildComposeRuntime,
    cloneComposeLayoutState,
    composeLayoutStateFromLegacySpec,
    defaultComposeLayoutState,
    emptyComposeLayoutState,
    isComposeLayoutMeta,
    isComposeLayoutState,
    matchedComposeLayoutPresetSelection,
    type ComposeLayoutMeta,
    type ComposeLayoutState,
    type ComposeTensorMeta,
    type MatrixBlock,
} from './linear-layout.js';

/**
 * Describes one row added to the tensor inspector by the linear-layout extension for the hovered cell.
 * The row names the source/output coordinate being reported, the axis labels used to display it, the tensor shape,
 * the resolved coordinate when one is available, and whether that row corresponds to the currently hovered tensor.
 *
 * @example
 * const entry: InspectorCoordEntry = {
 *     title: 'Output coordinate',
 *     labels: ['m', 'n'],
 *     shape: [16, 8],
 *     coord: [3, 5],
 *     hovered: true,
 * };
 *
 * entry.labels.map((label, axis) => `${label}:${entry.coord?.[axis]}`).join(', ');
 * // 'm:3, n:5'
 */
export type InspectorCoordEntry = {
    title: string;
    labels: string[];
    shape: number[];
    coord: number[] | null;
    hovered: boolean;
};

/**
 * User-visible status message emitted by the linear-layout sidebar after a compose-layout action succeeds or fails.
 * The tone selects the banner styling, while text contains the message shown to the user.
 *
 * @example
 * const notice: LinearLayoutNotice = {
 *     tone: 'error',
 *     text: 'Unable to parse compose layout: expected a closing bracket.',
 * };
 *
 * notice.tone;
 * // 'error'
 */
export type LinearLayoutNotice = {
    tone: 'error' | 'success';
    text: string;
};

/**
 * Per-label visibility flags for coordinate text rendered inside linear-layout tensor cells.
 * Keys are propagated/root axis labels such as `batch`, `m`, or `n`; a true value means matching cell labels include
 * that axis coordinate, and a false value suppresses it.
 *
 * @example
 * const state: LinearLayoutCellTextState = {
 *     m: true,
 *     n: false,
 * };
 *
 * state.m;
 * // true
 */
export type LinearLayoutCellTextState = Record<string, boolean>;
/**
 * Selected input index for each multi-input tensor slider in a linear-layout tab.
 * Keys are tensor ids from the rendered viewer manifest, and values are the zero-based input slot currently displayed
 * for that tensor.
 *
 * @example
 * const state: LinearLayoutMultiInputState = {
 *     'mma.accumulator': 0,
 *     'mma.operand': 2,
 * };
 *
 * state['mma.operand'];
 * // 2
 */
export type LinearLayoutMultiInputState = Record<string, number>;
/**
 * Stores the tensor-view snapshots captured for each tensor in a linear-layout tab.
 *
 * Keys are tensor ids from the loaded bundle, and values are the viewer-core
 * `TensorViewSnapshot` objects that should be cloned into tab snapshots,
 * duplicated tabs, or restored sessions so per-tensor slicing and view settings
 * survive linear-layout tab lifecycle changes.
 *
 * @example
 * const tensorViews: LinearLayoutTensorViewsState = {
 *   input: inputTensorViewSnapshot,
 *   output: outputTensorViewSnapshot,
 * };
 *
 * const snapshotForPersistence = cloneLinearLayoutTensorViewsState(tensorViews);
 * console.assert(snapshotForPersistence.input !== tensorViews.input);
 */
export type LinearLayoutTensorViewsState = Record<string, TensorViewSnapshot>;
/**
 * Sidebar compose-layout form state associated with a linear-layout tab.
 *
 * This is the saved form model for the compose-layout expression, preset
 * choices, and related options that are copied between the live sidebar,
 * per-tab state maps, and viewer snapshots when a linear-layout session is
 * saved or restored.
 *
 * @example
 * const stateByTab = new Map<string, LinearLayoutFormState>();
 * stateByTab.set(tab.id, cloneLinearLayoutState(currentComposeLayoutState));
 *
 * const restored = stateByTab.get(tab.id);
 * console.assert(restored !== currentComposeLayoutState);
 */
export type LinearLayoutFormState = ComposeLayoutState;
/**
 * Display channel names supported by the linear-layout color/label mapping UI.
 *
 * `H`, `S`, and `L` are the only channels that widgets may assign to tensor
 * labels when validating channel mappings for linear-layout render controls.
 *
 * @example
 * const channel: LinearLayoutChannel = 'H';
 * const mapping: Record<LinearLayoutChannel, string> = {
 *   H: 'batch',
 *   S: 'row',
 *   L: 'col',
 * };
 *
 * console.assert(mapping[channel] === 'batch');
 */
export type LinearLayoutChannel = 'H' | 'S' | 'L';

/**
 * Coordinate lookup table used to synchronize selections across tensors in a
 * rendered linear-layout tab.
 *
 * The map records the root input coordinate space, the final output coordinate
 * space, and per-tensor coordinate keys so hover, inspector rows, multi-input
 * sliders, and propagated-output selection can translate between a displayed
 * tensor cell and the corresponding root/final coordinates.
 *
 * @example
 * const selectionMap: LinearLayoutSelectionMap = {
 *   injective: true,
 *   rootInputLabels: ['m', 'n'],
 *   rootInputShape: [2, 2],
 *   rootKeys: ['0,0', '0,1', '1,0', '1,1'],
 *   rootKeyToIndex: new Map([['0,0', 0], ['0,1', 1], ['1,0', 2], ['1,1', 3]]),
 *   finalOutputLabels: ['m', 'n'],
 *   finalOutputShape: [2, 2],
 *   rootToFinalKeys: ['0,0', '0,1', '1,0', '1,1'],
 *   tensors: new Map([[
 *     'accumulator',
 *     {
 *       meta: accumulatorMeta,
 *       rootToTensorKeys: ['0,0', '0,1', '1,0', '1,1'],
 *       coordKeyToFlatIndex: new Map([['0,0', 0], ['0,1', 1], ['1,0', 2], ['1,1', 3]]),
 *       cellRootIndexes: [[0], [1], [2], [3]],
 *     },
 *   ]]),
 *   orderedTensorIds: ['accumulator'],
 * };
 *
 * const rootIndex = selectionMap.rootKeyToIndex.get('1,0');
 * console.assert(rootIndex === 2);
 */
export type LinearLayoutSelectionMap = {
    injective: boolean;
    rootInputLabels: string[];
    rootInputShape: number[];
    rootKeys: string[];
    rootKeyToIndex: Map<string, number>;
    finalOutputLabels: string[];
    finalOutputShape: number[];
    rootToFinalKeys: string[];
    tensors: Map<string, {
        meta: ComposeTensorMeta;
        rootToTensorKeys: string[];
        coordKeyToFlatIndex: Map<string, number>;
        cellRootIndexes: number[][];
    }>;
    orderedTensorIds: string[];
};

/**
 * Holds the mutable state owned by the linear-layout extension while the demo sidebar is open.
 *
 * The object keeps the active compose-layout form, per-tab form snapshots, per-tab cell-text,
 * multi-input, tensor-view, and selection caches, plus UI-only flags such as the matrix preview,
 * notice banner, and selection-sync guard.
 *
 * @example
 * const state: LinearLayoutUiState = {
 *     linearLayoutState: defaultLinearLayoutState(),
 *     linearLayoutStates: new Map(),
 *     linearLayoutCellTextState: defaultLinearLayoutCellTextState(),
 *     linearLayoutCellTextStates: new Map(),
 *     linearLayoutMultiInputState: defaultLinearLayoutMultiInputState(),
 *     linearLayoutMultiInputStates: new Map(),
 *     linearLayoutTensorViewsStates: new Map(),
 *     linearLayoutSelectionMaps: new Map(),
 *     linearLayoutNotice: { tone: 'success', text: 'Layout applied.' },
 *     linearLayoutMatrixPreview: '[1 0]\n[0 1]',
 *     showLinearLayoutMatrix: true,
 *     syncingLinearLayoutSelection: false,
 * };
 *
 * state.linearLayoutStates.set('tab-a', state.linearLayoutState);
 * console.assert(state.linearLayoutStates.has('tab-a'));
 */
export type LinearLayoutUiState = {
    linearLayoutState: LinearLayoutFormState;
    linearLayoutStates: Map<string, LinearLayoutFormState>;
    linearLayoutCellTextState: LinearLayoutCellTextState;
    linearLayoutCellTextStates: Map<string, LinearLayoutCellTextState>;
    linearLayoutMultiInputState: LinearLayoutMultiInputState;
    linearLayoutMultiInputStates: Map<string, LinearLayoutMultiInputState>;
    linearLayoutTensorViewsStates: Map<string, LinearLayoutTensorViewsState>;
    linearLayoutSelectionMaps: Map<string, LinearLayoutSelectionMap>;
    linearLayoutNotice: LinearLayoutNotice | null;
    linearLayoutMatrixPreview: string;
    showLinearLayoutMatrix: boolean;
    syncingLinearLayoutSelection: boolean;
};

/**
 * Bundles the viewer instance, sidebar DOM nodes, tab/session callbacks, and shared
 * linear-layout state passed to linear-layout widget renderers and action handlers.
 *
 * Render helpers use this context to read the active tab, update extension widgets, persist
 * session tabs, and trigger a full re-render after compose-layout form changes.
 *
 * @example
 * const context: LinearLayoutUiContext = {
 *     viewer,
 *     viewport: document.createElement('div'),
 *     linearLayoutPresetWidget: document.createElement('section'),
 *     linearLayoutWidget: document.createElement('section'),
 *     linearLayoutVisibleTensorsWidget: document.createElement('section'),
 *     cellTextWidget: document.createElement('section'),
 *     linearLayoutColorWidget: document.createElement('section'),
 *     state,
 *     widgetTitle: (widgetId, info) => `${widgetId}: ${info}`,
 *     getActiveTab: () => tabs.find((tab) => tab.id === activeTabId),
 *     getActiveTabId: () => activeTabId,
 *     getSessionTabs: () => tabs,
 *     setSessionTabs: (nextTabs) => { tabs = nextTabs; },
 *     loadTab: async (id) => { activeTabId = id; },
 *     renderLinearLayoutEditorWidgets: () => { rendered = true; },
 * };
 *
 * context.setSessionTabs([]);
 * console.assert(context.getSessionTabs().length === 0);
 */
export type LinearLayoutUiContext = {
    viewer: TensorViewer;
    viewport: HTMLElement;
    linearLayoutPresetWidget: HTMLElement;
    linearLayoutWidget: HTMLElement;
    linearLayoutVisibleTensorsWidget: HTMLElement;
    cellTextWidget: HTMLElement;
    linearLayoutColorWidget: HTMLElement;
    state: LinearLayoutUiState;
    widgetTitle: (widgetId: string, info: string) => string;
    getActiveTab: () => LoadedBundleDocument | undefined;
    getActiveTabId: () => string | null;
    getSessionTabs: () => LoadedBundleDocument[];
    setSessionTabs: (tabs: LoadedBundleDocument[]) => void;
    loadTab: (id: string) => Promise<void>;
    renderLinearLayoutEditorWidgets: () => void;
};

const LINEAR_LAYOUT_CHANNELS: LinearLayoutChannel[] = ['H', 'S', 'L'];
const LINEAR_LAYOUT_STORAGE_KEY = 'tensor-viz-linear-layout-spec';

/**
 * Reads the saved compose-layout editor form from browser localStorage and normalizes it for
 * the live linear-layout sidebar.
 *
 * Current saved forms are validated and cloned before use. Older saved editor/spec objects are
 * migrated into the current form shape when recognizable. Missing, malformed, or unreadable
 * storage entries fall back to the built-in default form.
 *
 * @returns The form state used to initialize the linear-layout editor fields, either cloned from storage, migrated from a legacy save, or created from defaults.
 * @noThrows localStorage access and JSON parsing are wrapped in a catch block, so blocked storage, invalid JSON, and unexpected stored shapes return the default form instead of propagating an exception.
 * @example
 * window.localStorage.setItem(
 *     LINEAR_LAYOUT_STORAGE_KEY,
 *     JSON.stringify({ ...defaultLinearLayoutState(), operationText: 'out = input' }),
 * );
 *
 * const state = loadLinearLayoutState();
 *
 * console.assert(state.operationText === 'out = input');
 *
 * @example
 * window.localStorage.setItem(LINEAR_LAYOUT_STORAGE_KEY, '{not valid json');
 *
 * const state = loadLinearLayoutState();
 *
 * console.assert(state.operationText === defaultLinearLayoutState().operationText);
 */
export function loadLinearLayoutState(): LinearLayoutFormState {
    const fallback = defaultLinearLayoutState();
    try {
        const stored = window.localStorage.getItem(LINEAR_LAYOUT_STORAGE_KEY);
        if (!stored) return fallback;
        const parsed = JSON.parse(stored);
        if (isLinearLayoutState(parsed)) return cloneLinearLayoutState(parsed);
        if (parsed && typeof parsed === 'object' && ('specsText' in parsed || 'operationText' in parsed)) {
            return { ...fallback, ...(parsed as Partial<LinearLayoutFormState>) };
        }
        if (parsed && typeof parsed === 'object' && (parsed.basesText || parsed.bases)) {
            return legacyEditorState(parsed as Record<string, unknown>, fallback);
        }
        if (parsed && typeof parsed === 'object' && (parsed.input_dims || parsed.bases)) {
            return composeLayoutStateFromLegacySpec(parsed, 'Layout_1');
        }
    } catch {
        return fallback;
    }
    return fallback;
}

/**
 * Persists the current compose-layout editor form to browser localStorage for the next demo
 * session.
 *
 * The saved JSON is later consumed by {@link loadLinearLayoutState} to restore sidebar fields
 * such as specs text, operation text, layout name, and propagation options.
 *
 * @param state - Complete linear-layout form state to serialize under the linear-layout storage key.
 * @returns Nothing; callers observe the effect by reading the saved JSON from localStorage or by reloading the editor state later.
 * @noThrows localStorage writes and JSON serialization are wrapped in a catch block because private browsing, quota limits, or restricted browser settings can reject storage writes.
 * @example
 * const state = { ...defaultLinearLayoutState(), operationText: 'out = input' };
 *
 * storeLinearLayoutState(state);
 *
 * const saved = JSON.parse(window.localStorage.getItem(LINEAR_LAYOUT_STORAGE_KEY) ?? '{}');
 * console.assert(saved.operationText === 'out = input');
 */
export function storeLinearLayoutState(state: LinearLayoutFormState): void {
    try {
        window.localStorage.setItem(LINEAR_LAYOUT_STORAGE_KEY, JSON.stringify(state));
    } catch {
        // ignore storage failures in restricted browsers
    }
}

/**
 * Checks whether an unknown saved value has the compose-layout form-state shape used by the linear-layout sidebar.
 *
 * Use this before restoring `composeLayoutState` from a viewer manifest, tab snapshot, or localStorage so invalid data can fall back to a default state.
 *
 * @param value - Unknown value read from persisted viewer state, browser storage, or extension metadata.
 * @returns `true` when `value` can be treated as a `LinearLayoutFormState`; otherwise `false` so callers can ignore or migrate the value.
 * @noThrows The check delegates to the structural compose-layout state guard and does not parse layout text or touch browser APIs.
 * @example
 * const saved = defaultLinearLayoutState();
 *
 * if (isLinearLayoutState(saved)) {
 *   const restored = cloneLinearLayoutState(saved);
 *   console.assert(restored !== saved);
 * }
 *
 * console.assert(isLinearLayoutState({ specsText: 42 }) === false);
 */
export function isLinearLayoutState(value: unknown): value is LinearLayoutFormState {
    return isComposeLayoutState(value);
}

/**
 * Builds the normal fallback form state for the linear-layout sidebar.
 *
 * The extension uses this state when no tab-local compose-layout state is available, when localStorage is empty, or when metadata only supplies a subset of the layout fields.
 *
 * @returns A fresh `LinearLayoutFormState` populated with the default compose-layout text, input name, preset selection, visibility, mapping, and range fields expected by the sidebar.
 * @noThrows The default state is assembled from in-memory compose-layout defaults and does not read storage, parse user layout text, or inspect the active viewer tab.
 * @example
 * const state = defaultLinearLayoutState();
 *
 * console.assert(isLinearLayoutState(state));
 * console.assert(typeof state.specsText === 'string');
 * console.assert(typeof state.operationText === 'string');
 */
export function defaultLinearLayoutState(): LinearLayoutFormState {
    return defaultComposeLayoutState();
}

/**
 * Builds a blank compose-layout form state for starting a new linear-layout tab.
 *
 * New tab creation uses this before generating the tab document so the sidebar begins from an empty editable layout rather than a restored or preset-backed state.
 *
 * @returns A fresh `LinearLayoutFormState` with the empty compose-layout fields used to seed a new document and sidebar session.
 * @noThrows The empty state is created from static compose-layout defaults and does not read viewer state, storage, or user-authored layout text.
 * @example
 * const state = emptyLinearLayoutState();
 *
 * console.assert(isLinearLayoutState(state));
 * console.assert(typeof state.specsText === 'string');
 * console.assert(typeof state.operationText === 'string');
 */
export function emptyLinearLayoutState(): LinearLayoutFormState {
    return emptyComposeLayoutState();
}

/**
 * Copies a linear-layout sidebar form-state snapshot for tab-local storage or restoration.
 *
 * Use the clone when saving state into the extension's per-tab maps or when restoring persisted compose-layout metadata so later widget edits do not mutate the saved snapshot by reference.
 *
 * @param state - Valid linear-layout form state from the active sidebar, a tab-state map, or validated persisted `composeLayoutState` metadata.
 * @returns An independent `LinearLayoutFormState` containing the same compose-layout text, preset selection, visibility, mapping, and range data as `state`.
 * @noThrows Cloning only copies fields from an already-typed form-state object and does not validate, parse, or access external resources.
 * @example
 * const original = defaultLinearLayoutState();
 * const cloned = cloneLinearLayoutState(original);
 *
 * console.assert(cloned !== original);
 * console.assert(cloned.specsText === original.specsText);
 */
export function cloneLinearLayoutState(state: LinearLayoutFormState): LinearLayoutFormState {
    return cloneComposeLayoutState(state);
}

/**
 * Builds the initial cell-text visibility map for a linear-layout tab.
 *
 * Each label starts enabled so newly created or synchronized layouts show cell text for every root input or final output label discovered from the layout metadata.
 *
 * @param labels - Ordered linear-layout input or output labels from compose-layout metadata; duplicate labels collapse to one visibility entry.
 * @returns A label-to-boolean map with every supplied label set to `true`, ready to store as `linearLayoutCellTextState` for the tab.
 * @noThrows The function only maps the provided string array into a plain object and does not read viewer state, validate layout syntax, or perform I/O.
 * @example
 * const state = defaultLinearLayoutCellTextState(['A', 'B']);
 * expect(state).toEqual({ A: true, B: true });
 *
 * const emptyState = defaultLinearLayoutCellTextState();
 * expect(emptyState).toEqual({});
 */
export function defaultLinearLayoutCellTextState(labels: string[] = []): LinearLayoutCellTextState {
    return Object.fromEntries(labels.map((label) => [label, true]));
}

/**
 * Copies a linear-layout cell-text visibility map before it is cached for a tab or mutated by the sidebar.
 *
 * @param state - Plain object whose keys are linear-layout labels and whose values indicate whether that label's cell text is visible.
 * @returns A shallow copy containing the same label visibility flags, so callers can update the copy without changing the original map reference.
 * @noThrows The clone uses object spread on the supplied cell-text state and performs no parsing, storage access, or layout synchronization.
 * @example
 * const original = { A: true, B: false };
 * const clone = cloneLinearLayoutCellTextState(original);
 * clone.A = false;
 *
 * expect(clone).toEqual({ A: false, B: false });
 * expect(original).toEqual({ A: true, B: false });
 */
export function cloneLinearLayoutCellTextState(state: LinearLayoutCellTextState): LinearLayoutCellTextState {
    return { ...state };
}

/**
 * Creates the empty multi-input display state used when a tab has no saved slider or hover choices.
 *
 * @returns An empty `LinearLayoutMultiInputState` object that can be stored for a linear-layout tab and populated as multi-input controls are used.
 * @noThrows The function returns a new object literal and does not inspect viewer metadata, browser storage, or tab contents.
 * @example
 * const state = defaultLinearLayoutMultiInputState();
 * expect(state).toEqual({});
 */
export function defaultLinearLayoutMultiInputState(): LinearLayoutMultiInputState {
    return {};
}

/**
 * Copies the per-tab multi-input state used by linear-layout sliders and hover display controls.
 *
 * @param state - Plain multi-input state object currently associated with the active tab or loaded from saved viewer metadata.
 * @returns A shallow copy with the same multi-input entries, suitable for storing in the tab cache without sharing the top-level object reference.
 * @noThrows The clone uses object spread on the supplied multi-input state and does not evaluate layouts, read storage, or touch the DOM.
 * @example
 * const original = { operand: 1 } as LinearLayoutMultiInputState;
 * const clone = cloneLinearLayoutMultiInputState(original);
 * clone.operand = 2;
 *
 * expect(clone).toEqual({ operand: 2 });
 * expect(original).toEqual({ operand: 1 });
 */
export function cloneLinearLayoutMultiInputState(state: LinearLayoutMultiInputState): LinearLayoutMultiInputState {
    return { ...state };
}

/**
 * Checks whether an unknown persisted tab value can be used as the linear-layout multi-input slider state.
 *
 * @param value - Candidate value loaded from viewer metadata or browser tab state; valid entries are object properties whose values are integers greater than or equal to -1.
 * @returns `true` when every stored multi-input slot is an integer index or the `-1` sentinel, allowing TypeScript to treat `value` as `LinearLayoutMultiInputState`; otherwise `false`.
 * @noThrows The guard rejects null and non-object inputs before inspecting entries, and it only performs type checks and integer comparisons on the remaining property values.
 * @example
 * const restored = { lhs: 0, rhs: -1 };
 * isLinearLayoutMultiInputState(restored); // true
 *
 * const invalid = { lhs: 0.5, rhs: 'all' };
 * isLinearLayoutMultiInputState(invalid); // false
 */
export function isLinearLayoutMultiInputState(value: unknown): value is LinearLayoutMultiInputState {
    if (!value || typeof value !== 'object') return false;
    return Object.values(value as Record<string, unknown>).every((entry) => Number.isInteger(entry) && Number(entry) >= -1);
}

/**
 * Copies the per-tensor view state saved for a linear-layout tab without sharing mutable hidden-index arrays.
 *
 * @param state - Mapping from tensor id to the tensor view snapshot that stores the editor expression and hidden dimension indices for that tensor.
 * @returns A new tensor-id mapping with the same editor strings and freshly copied `hiddenIndices` arrays, suitable for duplicating tabs or embedding state in a saved viewer snapshot.
 * @noThrows For a valid `LinearLayoutTensorViewsState`, cloning only enumerates own entries and calls `slice()` on each existing `hiddenIndices` array.
 * @example
 * const original = {
 *   activations: { editor: '[:, 0:4]', hiddenIndices: [2] },
 * };
 * const cloned = cloneLinearLayoutTensorViewsState(original);
 *
 * cloned.activations.hiddenIndices.push(3);
 * original.activations.hiddenIndices; // [2]
 */
export function cloneLinearLayoutTensorViewsState(state: LinearLayoutTensorViewsState): LinearLayoutTensorViewsState {
    return Object.fromEntries(Object.entries(state).map(([tensorId, view]) => [
        tensorId,
        { editor: view.editor, hiddenIndices: view.hiddenIndices.slice() },
    ]));
}

/**
 * Checks whether an unknown persisted tab value is the linear-layout map of cell-text visibility flags.
 *
 * @param value - Candidate value loaded from viewer metadata or browser tab state; valid entries are object properties whose values are booleans keyed by linear-layout cell or input label.
 * @returns `true` when every stored label flag is boolean, allowing TypeScript to treat `value` as `LinearLayoutCellTextState`; otherwise `false`.
 * @noThrows The guard rejects null and non-object inputs before entry inspection, and it only checks the JavaScript type of each property value.
 * @example
 * const restored = { inputA: true, accumulator: false };
 * isLinearLayoutCellTextState(restored); // true
 *
 * const invalid = { inputA: 'visible' };
 * isLinearLayoutCellTextState(invalid); // false
 */
export function isLinearLayoutCellTextState(value: unknown): value is LinearLayoutCellTextState {
    if (!value || typeof value !== 'object') return false;
    return Object.values(value as Record<string, unknown>).every((entry) => typeof entry === 'boolean');
}

/**
 * Captures the current viewer tensor view expressions and hidden-axis selections for linear-layout tab state.
 *
 * @param snapshot - Viewer snapshot whose `tensors` array contains each tensor id and its current `view.editor` expression and `view.hiddenIndices` array.
 * @returns A `LinearLayoutTensorViewsState` keyed by tensor id, used to restore or persist the tensor views associated with a linear-layout tab.
 * @noThrows For a valid `ViewerSnapshot`, the function only maps the snapshot tensor array and copies each tensor's existing `hiddenIndices` array.
 * @example
 * const snapshot = {
 *   tensors: [
 *     { id: 'weights', view: { editor: '[0, :, :]', hiddenIndices: [0] } },
 *     { id: 'output', view: { editor: '[:, :]', hiddenIndices: [] } },
 *   ],
 * } as ViewerSnapshot;
 *
 * snapshotTensorViews(snapshot);
 * // {
 * //   weights: { editor: '[0, :, :]', hiddenIndices: [0] },
 * //   output: { editor: '[:, :]', hiddenIndices: [] },
 * // }
 */
export function snapshotTensorViews(snapshot: ViewerSnapshot): LinearLayoutTensorViewsState {
    return Object.fromEntries(snapshot.tensors.map((tensor) => [
        tensor.id,
        { editor: tensor.view.editor, hiddenIndices: tensor.view.hiddenIndices.slice() },
    ]));
}

/**
 * Reads the compose-layout metadata embedded in a loaded viewer tab and returns it only when it matches the linear-layout metadata schema.
 *
 * @param tab - Loaded bundle document whose `manifest.viewer.composeLayoutMeta` field may contain metadata emitted by the linear-layout renderer.
 * @returns The validated `ComposeLayoutMeta` used by the extension for labels, tensor visibility, selection mapping, and controls; `null` when the tab has no valid compose-layout metadata.
 * @noThrows The function only performs optional manifest property access and a type-guard check, so malformed or missing metadata is treated as `null` rather than thrown.
 * @example
 * const meta = composeLayoutMetaForTab(linearLayoutTab);
 * console.assert(meta?.inputName === 'accumulator');
 * console.assert(meta?.tensors[0]?.id === 'mma_output');
 *
 * const missing = composeLayoutMetaForTab(plainTensorTab);
 * console.assert(missing === null);
 */
export function composeLayoutMetaForTab(tab: LoadedBundleDocument): ComposeLayoutMeta | null {
    const candidate = (tab.manifest.viewer as { composeLayoutMeta?: unknown }).composeLayoutMeta;
    return isComposeLayoutMeta(candidate) ? candidate : null;
}

/**
 * Classifies a loaded tab as a linear-layout tab when its viewer manifest contains valid compose-layout metadata.
 *
 * @param tab - Loaded bundle document to test before enabling linear-layout widgets, toolbar controls, hover entries, or tensor-view contributions.
 * @returns `true` when `composeLayoutMetaForTab(tab)` finds valid metadata; otherwise `false` so callers can leave the generic tensor viewer behavior unchanged.
 * @noThrows The predicate delegates to metadata validation that returns `null` for absent or malformed compose-layout data instead of throwing.
 * @example
 * console.assert(isLinearLayoutTab(linearLayoutTab) === true);
 * console.assert(isLinearLayoutTab(plainTensorTab) === false);
 */
export function isLinearLayoutTab(tab: LoadedBundleDocument): boolean {
    return composeLayoutMetaForTab(tab) !== null;
}

/**
 * Restores the live linear-layout sidebar state for a tab from the per-tab cache, saved manifest state, or compose-layout metadata defaults.
 *
 * @param ctx - Linear-layout UI context containing the mutable sidebar state, per-tab `linearLayoutStates` cache, and matrix-preview refresh dependencies.
 * @param tab - Loaded bundle document whose `id` selects cached state and whose viewer metadata may provide saved `composeLayoutState` or compose-layout defaults.
 * @returns Nothing. For linear-layout tabs, updates `ctx.state.linearLayoutState`, stores a cloned copy under `tab.id`, and refreshes the matrix preview; for non-linear-layout tabs, leaves the current state unchanged.
 * @noThrows With a valid UI context and loaded tab, the function validates optional manifest state before cloning it and otherwise builds defaults from guarded metadata, so missing or malformed tab fields fall back without an expected exception.
 * @example
 * syncLinearLayoutState(ctx, linearLayoutTab);
 * console.assert(ctx.state.linearLayoutState.specsText === 'A: f32[16,16]');
 * console.assert(ctx.state.linearLayoutStates.has(linearLayoutTab.id));
 *
 * const before = ctx.state.linearLayoutState;
 * syncLinearLayoutState(ctx, plainTensorTab);
 * console.assert(ctx.state.linearLayoutState === before);
 */
export function syncLinearLayoutState(ctx: LinearLayoutUiContext, tab: LoadedBundleDocument): void {
    if (!isLinearLayoutTab(tab)) return;
    const stored = ctx.state.linearLayoutStates.get(tab.id);
    if (stored) {
        ctx.state.linearLayoutState = cloneLinearLayoutState(stored);
        refreshLinearLayoutMatrixPreview(ctx);
        return;
    }
    const candidate = (tab.manifest.viewer as { composeLayoutState?: unknown }).composeLayoutState;
    if (isComposeLayoutState(candidate)) {
        ctx.state.linearLayoutState = cloneLinearLayoutState(candidate);
        ctx.state.linearLayoutStates.set(tab.id, cloneLinearLayoutState(candidate));
        refreshLinearLayoutMatrixPreview(ctx);
        return;
    }
    const meta = composeLayoutMetaForTab(tab);
    const autoColor = meta ? autoColorLayoutState(meta.specsText, meta.operationText) : null;
    ctx.state.linearLayoutState = meta
        ? {
            ...defaultLinearLayoutState(),
            specsText: meta.specsText,
            operationText: meta.operationText,
            inputName: meta.inputName ?? defaultLinearLayoutState().inputName,
            presetSelection: matchedComposeLayoutPresetSelection({
                specsText: meta.specsText,
                operationText: meta.operationText,
                inputName: meta.inputName ?? defaultLinearLayoutState().inputName,
            }),
            visibleTensors: Object.fromEntries(meta.tensors.map((tensor) => [tensor.id, tensor.visible])),
            mapping: autoColor?.mapping ?? defaultLinearLayoutState().mapping,
            ranges: autoColor?.ranges ?? defaultLinearLayoutState().ranges,
        }
        : defaultLinearLayoutState();
    ctx.state.linearLayoutStates.set(tab.id, cloneLinearLayoutState(ctx.state.linearLayoutState));
    refreshLinearLayoutMatrixPreview(ctx);
}

/**
 * Synchronizes the sidebar cell-label display state for a tab, choosing saved cell-text settings or defaults derived from compose-layout input/output labels.
 *
 * @param ctx - Linear-layout UI context containing `linearLayoutCellTextState`, the per-tab `linearLayoutCellTextStates` cache, and the current `linearLayoutState.propagateOutputs` mode.
 * @param tab - Loaded bundle document whose `id` selects cached cell-text state and whose viewer metadata may contain saved `linearLayoutCellTextState` plus root and final output labels.
 * @returns Nothing. Updates `ctx.state.linearLayoutCellTextState` and caches a clone for linear-layout tabs, or resets cell-text state to the default when the tab is not a linear-layout document.
 * @noThrows With a valid UI context and loaded tab, the function guards optional manifest state before cloning and uses empty/default labels when metadata is absent, so invalid cell-text metadata is ignored rather than thrown.
 * @example
 * ctx.state.linearLayoutState.propagateOutputs = true;
 * syncLinearLayoutCellTextState(ctx, linearLayoutTab);
 * console.assert(ctx.state.linearLayoutCellTextState.labels[0] === 'C[0,0]');
 * console.assert(ctx.state.linearLayoutCellTextStates.has(linearLayoutTab.id));
 *
 * syncLinearLayoutCellTextState(ctx, plainTensorTab);
 * console.assert(ctx.state.linearLayoutCellTextState.labels.length === 0);
 */
export function syncLinearLayoutCellTextState(ctx: LinearLayoutUiContext, tab: LoadedBundleDocument): void {
    if (!isLinearLayoutTab(tab)) {
        ctx.state.linearLayoutCellTextState = defaultLinearLayoutCellTextState();
        return;
    }
    const stored = ctx.state.linearLayoutCellTextStates.get(tab.id);
    if (stored) {
        ctx.state.linearLayoutCellTextState = cloneLinearLayoutCellTextState(stored);
        return;
    }
    const candidate = (tab.manifest.viewer as { linearLayoutCellTextState?: unknown }).linearLayoutCellTextState;
    if (isLinearLayoutCellTextState(candidate)) {
        ctx.state.linearLayoutCellTextState = cloneLinearLayoutCellTextState(candidate);
        ctx.state.linearLayoutCellTextStates.set(tab.id, cloneLinearLayoutCellTextState(candidate));
        return;
    }
    const meta = composeLayoutMetaForTab(tab);
    const labels = ctx.state.linearLayoutState.propagateOutputs
        ? meta?.finalOutputLabels ?? []
        : meta?.rootInputLabels ?? [];
    ctx.state.linearLayoutCellTextState = defaultLinearLayoutCellTextState(labels);
    ctx.state.linearLayoutCellTextStates.set(tab.id, cloneLinearLayoutCellTextState(ctx.state.linearLayoutCellTextState));
}

/**
 * Restores the sidebar multi-input slider state for a selected linear-layout tab.
 *
 * Non-linear-layout tabs reset the multi-input controls to their defaults. Linear-layout tabs first reuse the
 * per-tab cached state, then fall back to validated `linearLayoutMultiInputState` metadata from the tab manifest,
 * and finally create and cache a default state for that tab id.
 *
 * @param ctx - Linear-layout UI context whose `state.linearLayoutMultiInputState` and per-tab
 * `linearLayoutMultiInputStates` cache are updated.
 * @param tab - Loaded viewer document whose `id`, tab type, and manifest metadata select the multi-input state.
 * @returns Nothing; callers read the refreshed multi-input controls from `ctx.state.linearLayoutMultiInputState`.
 * @noThrows The function only performs type-guard checks, map lookups, cloning, and default creation; invalid or
 * missing tab metadata is ignored and replaced with a default multi-input state.
 * @example
 * const tab = {
 *   id: 'mma-tab',
 *   manifest: { viewer: { linearLayoutMultiInputState: savedMultiInputState } },
 * } as LoadedBundleDocument;
 *
 * syncLinearLayoutMultiInputState(ctx, tab);
 *
 * expect(ctx.state.linearLayoutMultiInputState).toEqual(savedMultiInputState);
 * expect(ctx.state.linearLayoutMultiInputStates.get('mma-tab')).toEqual(savedMultiInputState);
 */
export function syncLinearLayoutMultiInputState(ctx: LinearLayoutUiContext, tab: LoadedBundleDocument): void {
    if (!isLinearLayoutTab(tab)) {
        ctx.state.linearLayoutMultiInputState = defaultLinearLayoutMultiInputState();
        return;
    }
    const stored = ctx.state.linearLayoutMultiInputStates.get(tab.id);
    if (stored) {
        ctx.state.linearLayoutMultiInputState = cloneLinearLayoutMultiInputState(stored);
        return;
    }
    const candidate = (tab.manifest.viewer as { linearLayoutMultiInputState?: unknown }).linearLayoutMultiInputState;
    if (isLinearLayoutMultiInputState(candidate)) {
        ctx.state.linearLayoutMultiInputState = cloneLinearLayoutMultiInputState(candidate);
        ctx.state.linearLayoutMultiInputStates.set(tab.id, cloneLinearLayoutMultiInputState(candidate));
        return;
    }
    ctx.state.linearLayoutMultiInputState = defaultLinearLayoutMultiInputState();
    ctx.state.linearLayoutMultiInputStates.set(tab.id, cloneLinearLayoutMultiInputState(ctx.state.linearLayoutMultiInputState));
}

/**
 * Rebuilds the compose-layout matrix preview shown in the linear-layout sidebar.
 *
 * The preview is generated from the supplied form state, or from the current sidebar form when no state is passed.
 * If the compose-layout text cannot be parsed or evaluated, the preview is cleared so the UI does not display stale
 * matrix blocks.
 *
 * @param ctx - Linear-layout UI context whose `state.linearLayoutMatrixPreview` string is replaced.
 * @param state - Compose-layout form state containing `specsText`, `operationText`, input name, mapping, and ranges
 * used to build the runtime matrix blocks. Defaults to `ctx.state.linearLayoutState`.
 * @returns Nothing; callers render the updated `ctx.state.linearLayoutMatrixPreview` value.
 * @noThrows Compose-runtime and matrix-preview failures are caught and converted to an empty preview string.
 * @example
 * ctx.state.linearLayoutMatrixPreview = 'stale preview';
 *
 * refreshLinearLayoutMatrixPreview(ctx, {
 *   ...ctx.state.linearLayoutState,
 *   specsText: 'not valid compose-layout syntax',
 * });
 *
 * expect(ctx.state.linearLayoutMatrixPreview).toBe('');
 */
export function refreshLinearLayoutMatrixPreview(ctx: LinearLayoutUiContext, state = ctx.state.linearLayoutState): void {
    try {
        ctx.state.linearLayoutMatrixPreview = matrixPreviewFromBlocks(buildComposeRuntime(state).matrixBlocks);
    } catch {
        ctx.state.linearLayoutMatrixPreview = '';
    }
}

/**
 * Converts an older saved linear-layout editor payload into the current compose-layout form state.
 *
 * Legacy `basesText` or `bases.thread`/`bases.warp`/`bases.register` values become `T`, `W`, and `R` basis rows in
 * `specsText`. Legacy channel mappings and ranges are normalized onto the current `H`, `S`, and `L` channel fields,
 * while the fallback input name supplies the migrated operation input.
 *
 * @param raw - Parsed legacy editor object that may contain `basesText`, `bases`, `mapping`, and `ranges` fields from
 * an older saved tab or browser-storage entry.
 * @param fallback - Current default form state used for fields not present in the legacy payload, including
 * `inputName`, initial mapping values, and initial channel ranges.
 * @returns A current `LinearLayoutFormState` with generated compose-layout `specsText`, operation `Layout_1`,
 * normalized mapping/ranges, empty visible tensor overrides, and a preset selection matched to the generated form.
 * @throws Error when a legacy bases string is not valid JSON, is not a JSON array, contains a non-array basis row, or
 * contains a non-number basis entry.
 * @example
 * const migrated = legacyEditorState(
 *   {
 *     bases: { thread: '[[1,0]]', warp: '[[0,1]]', register: '[]' },
 *     mapping: { H: 'thread' },
 *     ranges: { H: ['0', '16'] },
 *   },
 *   fallback
 * );
 *
 * expect(migrated.operationText).toBe('Layout_1');
 * expect(migrated.specsText).toContain('T: [[1,0]]');
 * expect(migrated.mapping.H).toBe('T');
 * expect(migrated.ranges.H).toEqual(['0', '16']);
 *
 * expect(() => legacyEditorState({ bases: { thread: '[not json]' } }, fallback)).toThrow(
 *   'T bases must be valid JSON.'
 * );
 */
function legacyEditorState(raw: Record<string, unknown>, fallback: LinearLayoutFormState): LinearLayoutFormState {
    const textByLabel = new Map<string, string>([['T', '[]'], ['W', '[]'], ['R', '[]']]);
    if (typeof raw.basesText === 'string') {
        raw.basesText.split('\n').forEach((line) => {
            const match = line.trim().match(/^([TWR])\s*:\s*(.+)$/i);
            if (match) textByLabel.set(match[1]!.toUpperCase(), match[2]!.trim());
        });
    }
    if (raw.bases && typeof raw.bases === 'object') {
        const bases = raw.bases as Record<string, unknown>;
        if (typeof bases.thread === 'string') textByLabel.set('T', bases.thread);
        if (typeof bases.warp === 'string') textByLabel.set('W', bases.warp);
        if (typeof bases.register === 'string') textByLabel.set('R', bases.register);
    }
    const rows = ['T', 'W', 'R'].map((label) => parseBasesField(label, textByLabel.get(label) ?? '[]'));
    const outputRank = Math.max(1, ...rows.flatMap((entry) => entry.map((basis) => basis.length)));
    const outputs = Array.from({ length: outputRank }, (_entry, axis) => String.fromCharCode(65 + axis));
    const mapping = { ...fallback.mapping };
    if (raw.mapping && typeof raw.mapping === 'object') {
        Object.entries(raw.mapping as Record<string, unknown>).forEach(([channel, axisName]) => {
            const normalized = String(channel).toUpperCase() as LinearLayoutChannel;
            if (!LINEAR_LAYOUT_CHANNELS.includes(normalized) || typeof axisName !== 'string') return;
            mapping[normalized] = axisName === 'thread' ? 'T' : axisName === 'warp' ? 'W' : axisName === 'register' ? 'R' : 'none';
        });
    }
    const ranges = {
        H: [...fallback.ranges.H],
        S: [...fallback.ranges.S],
        L: [...fallback.ranges.L],
    } as Record<LinearLayoutChannel, [string, string]>;
    if (raw.ranges && typeof raw.ranges === 'object') {
        Object.entries(raw.ranges as Record<string, unknown>).forEach(([channel, range]) => {
            const normalized = String(channel).toUpperCase() as LinearLayoutChannel;
            if (!LINEAR_LAYOUT_CHANNELS.includes(normalized) || !Array.isArray(range) || range.length !== 2) return;
            ranges[normalized] = [String(range[0]), String(range[1])];
        });
    }
    const specsText = [
        `Layout_1: [T,W,R] -> [${outputs.join(',')}]`,
        ...['T', 'W', 'R'].map((label, axis) => `${label}: ${JSON.stringify(rows[axis])}`),
    ].join('\n');
    const operationText = 'Layout_1';
    return {
        specsText,
        operationText,
        inputName: fallback.inputName,
        presetSelection: matchedComposeLayoutPresetSelection({
            specsText,
            operationText,
            inputName: fallback.inputName,
        }),
        visibleTensors: {},
        propagateOutputs: false,
        mapping,
        ranges,
    };
}

/**
 * Parses one legacy linear-layout bases field into numeric basis vectors.
 *
 * Empty or whitespace-only text means the legacy editor did not provide bases for that axis and is treated as an
 * empty vector list. Non-empty text must be JSON shaped as an array of arrays of numbers.
 *
 * @param label - Axis label used in validation messages, such as `T`, `W`, or `R`.
 * @param value - JSON text from a legacy bases field; expected shape is `number[][]` such as `[[1,0],[0,1]]`.
 * @returns The parsed basis vectors as `number[][]`; returns `[]` for blank input.
 * @throws Error when `value` is malformed JSON, the parsed value is not an array, any basis row is not an array, or
 * any basis entry is missing, non-numeric, or `NaN`.
 * @example
 * expect(parseBasesField('T', '[[1,0],[0,1]]')).toEqual([
 *   [1, 0],
 *   [0, 1],
 * ]);
 * expect(parseBasesField('R', '   ')).toEqual([]);
 * expect(() => parseBasesField('W', '[[1,"x"]]')).toThrow('W basis 1[2] must be a number.');
 */
function parseBasesField(label: string, value: string): number[][] {
    if (!value.trim()) return [];
    let parsed: unknown;
    try {
        parsed = JSON.parse(value);
    } catch {
        throw new Error(`${label} bases must be valid JSON.`);
    }
    if (!Array.isArray(parsed)) throw new Error(`${label} bases must be a JSON array.`);
    return parsed.map((basis, index) => {
        if (!Array.isArray(basis)) throw new Error(`${label} basis ${index + 1} must be an array.`);
        return basis.map((entry, axis) => {
            if (typeof entry !== 'number' || Number.isNaN(entry)) {
                throw new Error(`${label} basis ${index + 1}[${axis + 1}] must be a number.`);
            }
            return entry;
        });
    });
}

/**
 * Renders compose-runtime matrix blocks as the escaped HTML snippet shown in the
 * linear-layout sidebar preview.
 *
 * Each block becomes a titled matrix with axis-colored row and column labels and
 * `matrix-one`/`matrix-zero` cells for selected and empty entries.
 *
 * @param blocks - Matrix preview blocks emitted by the compose runtime, including a title, row and column labels with axis indexes, and a rectangular `values` grid where `1` marks an active mapping.
 * @returns HTML markup for the matrix preview container; callers can assign it to the sidebar preview state for rendering.
 * @noThrows The formatter only maps over the supplied block data, pads strings, and escapes labels; parse/evaluation failures happen before blocks are passed in.
 * @example
 * const html = matrixPreviewFromBlocks([
 *   {
 *     title: 'MMA A -> Accumulator',
 *     rows: [{ label: 'm0', axis: 0 }],
 *     columns: [{ label: 'n0', axis: 1 }],
 *     values: [[1]],
 *   },
 * ]);
 *
 * console.assert(html.includes('<div class="matrix-block-title">MMA A -&gt; Accumulator</div>'));
 * console.assert(html.includes('<span class="matrix-one">1</span>'));
 */
function matrixPreviewFromBlocks(blocks: MatrixBlock[]): string {
    return blocks.map((block) => {
        const labelWidth = Math.max(1, ...block.rows.map((row) => row.label.length));
        const columnWidths = block.columns.map((column) => Math.max(1, column.label.length));
        const header = block.columns.length === 0
            ? '<span class="matrix-zero">0</span>'
            : `${' '.repeat(labelWidth)} | ${block.columns.map((column, index) => (
                `<span class="matrix-label matrix-axis-${column.axis % 3}">${escapeInfo(column.label.padStart(columnWidths[index] ?? 1))}</span>`
            )).join(' ')}`;
        const rows = block.rows.length === 0
            ? []
            : block.rows.map((row, rowIndex) => (
                `<span class="matrix-label matrix-axis-${row.axis % 3}">${escapeInfo(row.label.padStart(labelWidth))}</span> | ${block.columns.map((_column, columnIndex) => {
                    const value = block.values[rowIndex]?.[columnIndex] === 1 ? '1' : '0';
                    const klass = value === '1' ? 'matrix-one' : 'matrix-zero';
                    return `<span class="${klass}">${value.padStart(columnWidths[columnIndex] ?? 1)}</span>`;
                }).join(' ')}`
            ));
        return [
            '<div class="matrix-block">',
            `<div class="matrix-block-title">${escapeInfo(block.title)}</div>`,
            '<div class="matrix-block-body">',
            header,
            ...rows,
            '</div>',
            '</div>',
        ].join('\n');
    }).join('\n');
}
