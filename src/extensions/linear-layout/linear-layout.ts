import {
    createBundleManifest,
    product,
    unravelIndex,
    type LoadedBundleDocument,
    type TensorViewSnapshot,
    type ViewerSnapshot,
} from '@tensor-viz/viewer-core';
import {
    cloneComposeLayoutPresetSelection,
    emptyComposeLayoutPresetSelection,
    isComposeLayoutPresetSelection,
    matchedComposeLayoutPresetSelection,
    type ComposeLayoutPresetSelection,
} from './linear-layout-preset-model.js';
import {
    formatSpecsText,
    parseLayoutSpecs,
    type NamedLayoutSpec,
} from './linear-layout-parser.js';
export {
    cloneComposeLayoutPresetSelection,
    composeLayoutPresetCatalog,
    composeLayoutPresetFields,
    composeLayoutPresetForSelection,
    composeLayoutPresetOptions,
    composeLayoutPresets,
    emptyComposeLayoutPresetSelection,
    isComposeLayoutPresetSelection,
    matchedComposeLayoutPresetSelection,
    normalizeComposeLayoutPresetSelection,
} from './linear-layout-preset-model.js';
export type {
    ComposeLayoutPreset,
    ComposeLayoutPresetField,
    ComposeLayoutPresetOptions,
    ComposeLayoutPresetSelection,
} from './linear-layout-preset-model.js';
export type { NamedLayoutSpec } from './linear-layout-parser.js';

/**
 * HSL color channel that the compose-layout UI can bind to an axis-derived value.
 *
 * `H`, `S`, and `L` correspond to hue, saturation, and lightness controls in the
 * generated matrix preview and tensor cell coloring.
 *
 * @example
 * const channel: ComposeChannel = 'H';
 * const ranges: Record<ComposeChannel, [string, string]> = {
 *   H: ['0', '360'],
 *   S: ['45', '90'],
 *   L: ['35', '70'],
 * };
 * ranges[channel]; // ['0', '360']
 */
export type ComposeChannel = 'H' | 'S' | 'L';
/**
 * Axis label selected for a compose color channel, or `none` when that channel is not driven by an axis.
 *
 * The sidebar stores one value for each HSL channel so presets and sessions can restore
 * which tensor axis controls hue, saturation, and lightness.
 *
 * @example
 * const hueAxis: ComposeMappingValue = 'row';
 * const disabledLightness: ComposeMappingValue = 'none';
 *
 * const mapping: Record<ComposeChannel, ComposeMappingValue> = {
 *   H: hueAxis,
 *   S: 'col',
 *   L: disabledLightness,
 * };
 * mapping.L; // 'none'
 */
export type ComposeMappingValue = string | 'none';

/**
 * Sidebar-owned compose-layout editor state that is parsed into a rendered linear-layout tab.
 *
 * Stores the raw spec and operation text, selected preset, tensor visibility flags, channel mappings,
 * output propagation toggle, and editable channel ranges before `buildComposeRuntime` converts them into
 * viewer metadata, matrix previews, and generated Python.
 *
 * @example
 * const state: ComposeLayoutState = {
 *     specsText: 'A: [m, n]\nB: [n, k]',
 *     operationText: 'A @ B',
 *     inputName: 'A',
 *     presetSelection: { familyId: 'custom', presetId: 'custom' },
 *     visibleTensors: { A: true, B: true, output: true },
 *     propagateOutputs: true,
 *     mapping: { x: 'm', y: 'n', z: 'k' },
 *     ranges: { x: ['0', '16'], y: ['0', '16'], z: ['0', '16'] },
 * };
 *
 * state.visibleTensors.A; // true
 */
export type ComposeLayoutState = {
    specsText: string;
    operationText: string;
    inputName: string;
    presetSelection: ComposeLayoutPresetSelection;
    visibleTensors: Record<string, boolean>;
    propagateOutputs: boolean;
    mapping: Record<ComposeChannel, ComposeMappingValue>;
    ranges: Record<ComposeChannel, [string, string]>;
};

/**
 * Row or column descriptor in a compose-layout matrix preview.
 *
 * The label is shown in the preview table, while `axis` keeps the numeric source-axis index used by
 * the evaluated layout metadata.
 *
 * @example
 * const rowAxis: MatrixAxis = { label: 'input m', axis: 0 };
 *
 * rowAxis.label; // 'input m'
 * rowAxis.axis; // 0
 */
export type MatrixAxis = {
    label: string;
    axis: number;
};

/**
 * One titled matrix preview emitted for a parsed compose-layout tensor or mapping step.
 *
 * Rows and columns describe the displayed axes, and `values[row][column]` contains the numeric mapping
 * entry shown at that row and column intersection.
 *
 * @example
 * const block: MatrixBlock = {
 *     title: 'A root to output',
 *     rows: [{ label: 'm', axis: 0 }, { label: 'n', axis: 1 }],
 *     columns: [{ label: 'x', axis: 0 }, { label: 'y', axis: 1 }],
 *     values: [
 *         [1, 0],
 *         [0, 1],
 *     ],
 * };
 *
 * block.values[1]![0]; // 0
 */
export type MatrixBlock = {
    title: string;
    rows: MatrixAxis[];
    columns: MatrixAxis[];
    values: number[][];
};

/**
 * Metadata embedded in a rendered linear-layout tab for one tensor in the compose chain.
 *
 * It gives the viewer the tensor title, expression text, root-or-step role, axis labels, shape,
 * coordinate transforms from the root tensor and into the final output, plus whether the tensor should
 * initially be displayed.
 *
 * @example
 * const meta: ComposeTensorMeta = {
 *     id: 'A',
 *     title: 'Input A',
 *     exprText: 'A: [m, n]',
 *     kind: 'root',
 *     axisLabels: ['m', 'n'],
 *     shape: [16, 32],
 *     rootToTensor: [
 *         [1, 0],
 *         [0, 1],
 *     ],
 *     tensorToFinal: [[1, 0], [0, 1]],
 *     visible: true,
 * };
 *
 * meta.shape; // [16, 32]
 * meta.tensorToFinal[0]; // [1, 0]
 */
export type ComposeTensorMeta = {
    id: string;
    title: string;
    exprText: string;
    kind: 'root' | 'step';
    axisLabels: string[];
    shape: number[];
    rootToTensor: number[][];
    tensorToFinal: Array<number[] | null>;
    visible: boolean;
};

/**
 * Serialized compose-layout metadata embedded in viewer snapshots so restored tabs can rebuild labels,
 * visible tensor records, and root-to-output mapping without reparsing generic viewer state.
 *
 * @example
 * const meta: ComposeLayoutMeta = {
 *     version: 3,
 *     specsText: 'row(a:2) -> r:2',
 *     operationText: 'row',
 *     inputName: 'a',
 *     injective: true,
 *     rootInputLabels: ['a'],
 *     rootInputBitCounts: [2],
 *     finalOutputLabels: ['r'],
 *     finalOutputBitCounts: [2],
 *     tensors: [],
 * };
 *
 * console.assert(meta.version === 3);
 * console.assert(meta.rootInputLabels[0] === 'a');
 */
export type ComposeLayoutMeta = {
    version: 3;
    specsText: string;
    operationText: string;
    inputName?: string;
    injective: boolean;
    rootInputLabels: string[];
    rootInputBitCounts: number[];
    finalOutputLabels: string[];
    finalOutputBitCounts: number[];
    tensors: ComposeTensorMeta[];
};

/**
 * Fully evaluated compose-layout model used by sidebar widgets, tensor rendering, generated Python,
 * and the metadata saved into the viewer tab.
 *
 * @example
 * const runtime: ComposeRuntime = {
 *     specs: [],
 *     inputLabels: ['lane'],
 *     inputBitCounts: [2],
 *     inputShape: [4],
 *     finalOutputLabels: ['row'],
 *     finalOutputBitCounts: [2],
 *     finalOutputShape: [4],
 *     injective: true,
 *     tensors: [],
 *     matrixBlocks: [],
 *     pythonCode: 'row = Layout.row_major([4])',
 *     meta: {
 *         version: 3,
 *         specsText: 'row(lane:2) -> row:2',
 *         operationText: 'row',
 *         inputName: 'lane',
 *         injective: true,
 *         rootInputLabels: ['lane'],
 *         rootInputBitCounts: [2],
 *         finalOutputLabels: ['row'],
 *         finalOutputBitCounts: [2],
 *         tensors: [],
 *     },
 * };
 *
 * console.assert(runtime.finalOutputShape[0] === 4);
 * console.assert(runtime.meta.operationText === 'row');
 */
export type ComposeRuntime = {
    specs: NamedLayoutSpec[];
    inputLabels: string[];
    inputBitCounts: number[];
    inputShape: number[];
    finalOutputLabels: string[];
    finalOutputBitCounts: number[];
    finalOutputShape: number[];
    injective: boolean;
    tensors: ComposeTensorMeta[];
    matrixBlocks: MatrixBlock[];
    pythonCode: string;
    meta: ComposeLayoutMeta;
};

/**
 * Parsed compose-layout operation AST: a layout name, its inverse, a product of independent layouts,
 * or an application that composes an outer layout with an inner layout.
 *
 * @example
 * const expr: LayoutExpr = {
 *     kind: 'apply',
 *     outer: { kind: 'name', name: 'swizzle' },
 *     inner: { kind: 'inverse', expr: { kind: 'name', name: 'rowMajor' } },
 * };
 *
 * console.assert(expr.kind === 'apply');
 * console.assert(expr.outer.kind === 'name' && expr.outer.name === 'swizzle');
 */
type LayoutExpr =
    | { kind: 'name'; name: string }
    | { kind: 'inverse'; expr: LayoutExpr }
    | { kind: 'product'; left: LayoutExpr; right: LayoutExpr }
    | { kind: 'apply'; outer: LayoutExpr; inner: LayoutExpr };

/**
 * Result of evaluating one layout expression, including propagated axis labels, bit counts,
 * dense transform matrix, and the Python variable or expression that represents it.
 *
 * @example
 * const evaluated: EvaluatedLayout = {
 *     kind: 'name',
 *     exprText: 'rowMajor',
 *     codeRef: 'rowMajor',
 *     inputs: ['lane'],
 *     outputs: ['row'],
 *     inputBitCounts: [2],
 *     outputBitCounts: [2],
 *     matrix: [
 *         [1, 0],
 *         [0, 1],
 *     ],
 * };
 *
 * console.assert(evaluated.outputs.join(',') === 'row');
 * console.assert(evaluated.matrix[0][0] === 1);
 */
type EvaluatedLayout = {
    kind: LayoutExpr['kind'];
    exprText: string;
    codeRef: string;
    inputs: string[];
    outputs: string[];
    inputBitCounts: number[];
    outputBitCounts: number[];
    matrix: number[][];
    spec?: NamedLayoutSpec;
};

/**
 * Connects one generated Python statement to the evaluated layout that the
 * linear-layout sidebar should preview beside that line.
 *
 * @example
 * const codeLine: CodeLine = {
 *     title: 'compose H then S',
 *     layout: evaluatedLayout,
 *     line: "HS = compose_layout(H, S)",
 * };
 * console.assert(codeLine.line.includes('compose_layout'));
 * console.assert(codeLine.layout === evaluatedLayout);
 */
type CodeLine = {
    title: string;
    layout: EvaluatedLayout;
    line: string;
};

/**
 * Catalog entry for a baked linear-layout demo tab before the entry is cloned
 * and converted into a viewer document.
 *
 * @example
 * const example: ExampleState = {
 *     title: 'Blocked matmul',
 *     state: {
 *         specsText: 'H: (M, K) -> (M/block, K)',
 *         operationText: 'H',
 *         inputName: 'x',
 *         presetSelection: null,
 *         visibleTensors: {},
 *         propagateOutputs: false,
 *         mapping: { H: 'none', S: 'none', L: 'none' },
 *         ranges: defaultColorRanges(),
 *     },
 * };
 * console.assert(example.title === 'Blocked matmul');
 * console.assert(example.state.inputName === 'x');
 */
type ExampleState = {
    title: string;
    state: ComposeLayoutState;
};

const DEFAULT_INPUT_NAME = 'Input Space';

const DEFAULT_COLOR_RANGES = {
    H: ['0', '0.8'],
    S: ['1', '0.2'],
    L: ['1', '0.2'],
} as const;

const AUTO_COLOR_CHANNELS: ComposeChannel[] = ['H', 'L', 'S'];

const LEGACY_AXIS_ALIASES = {
    warp: 'W',
    thread: 'T',
    register: 'R',
} as const;

const DEFAULT_EMPTY_SPEC_TEXT = [
    'Layout_1: [T,W,R] -> [A,B]',
    'T: [[0,1],[0,2],[0,4],[0,8],[0,16]]',
    'W: []',
    'R: [[1,0],[2,0]]',
].join('\n');

// these examples are synced into docs by tools/sync-linear-layout-examples.py;
// avoid editing generated docs directly when changing their source text here.
// sync-linear-layout-examples:start
const BLOCKED_LAYOUT_TEXT = [
    'Blocked_Layout: [T,W,R] -> [Y,X]',
    'T: [[4,0],[8,0],[0,1],[0,2],[0,4]]',
    'W: [[0,8],[0,16]]',
    'R: [[1,0],[2,0]]',
].join('\n');
const MMA_A_LAYOUT__M16N8K16_TEXT = [
    'MMA_A_Layout__m16n8k16: [T,W,R] -> [Y,X]',
    'T: [[0,2],[0,4],[1,0],[2,0],[4,0]]',
    'W: []',
    'R: [[0,1],[8,0],[0,8]]',
].join('\n');
const MMA_B_LAYOUT__M16N8K16_TEXT = [
    'MMA_B_Layout__m16n8k16: [T,W,R] -> [Y,X]',
    'T: [[2,0],[4,0],[0,1],[0,2],[0,4]]',
    'W: []',
    'R: [[1,0],[8,0]]',
].join('\n');
const MMA_C_LAYOUT__M16N8K16_TEXT = [
    'MMA_C_Layout__m16n8k16: [T,W,R] -> [Y,X]',
    'T: [[0,2],[0,4],[1,0],[2,0],[4,0]]',
    'W: []',
    'R: [[0,1],[8,0]]',
].join('\n');
const SHARED_MEMORY_128B_SWIZZLE_TEXT = [
    'Shared_Memory_128B_Swizzle: [Y,X] -> [Y,X]',
    'Y: [[1,1],[2,2],[4,4]]',
    'X: [[0,1],[0,2],[0,4]]',
].join('\n');
const SLICED_LAYOUT_TEXT = [
    'Sliced_Layout: [S,R] -> [Y,X]',
    'S: [[1,0],[2,0]]',
    'R: [[0,0],[0,0],[0,0]]',
].join('\n');

const BAKED_EXAMPLE_DEFINITIONS = [
    { title: 'Blocked Layout', specsText: BLOCKED_LAYOUT_TEXT, operationText: 'Blocked_Layout', inputName: 'Hardware Layout' },
    { title: 'MMA A Layout (m16n8k16)', specsText: MMA_A_LAYOUT__M16N8K16_TEXT, operationText: 'MMA_A_Layout__m16n8k16', inputName: 'Hardware Layout' },
    { title: 'MMA B Layout (m16n8k16)', specsText: MMA_B_LAYOUT__M16N8K16_TEXT, operationText: 'MMA_B_Layout__m16n8k16', inputName: 'Hardware Layout' },
    { title: 'MMA C Layout (m16n8k16)', specsText: MMA_C_LAYOUT__M16N8K16_TEXT, operationText: 'MMA_C_Layout__m16n8k16', inputName: 'Hardware Layout' },
    { title: 'Shared Memory 128B Swizzle', specsText: SHARED_MEMORY_128B_SWIZZLE_TEXT, operationText: 'Shared_Memory_128B_Swizzle', inputName: 'Logical Offsets' },
    { title: 'Sliced Layout', specsText: SLICED_LAYOUT_TEXT, operationText: 'Sliced_Layout', inputName: 'Logical Offsets' },
] as const;
// sync-linear-layout-examples:end

/**
 * Builds one baked linear-layout catalog entry from compose-layout source text
 * and initializes the sidebar-only state that is not stored in the source files.
 *
 * @param title - Label shown for the baked example tab in the demo UI.
 * @param specsText - Compose-layout specification text copied into the Specs editor.
 * @param operationText - Compose operation expression copied into the Operation editor.
 * @param inputName - Name of the input tensor used when evaluating the example; defaults to the standard demo input name.
 * @returns An ExampleState whose title is the catalog label and whose state contains the supplied source text, matched preset selection, empty tensor visibility, disabled output propagation, default H/S/L mapping, and default color ranges.
 * @noThrows The helper only assembles state from provided strings, asks the preset matcher for an optional selection, and creates default color ranges; it does not parse the compose-layout text or read external state.
 * @example
 * const example = bakedExample(
 *     'Tiny transpose',
 *     'T: i j -> j i',
 *     'T',
 *     'activations',
 * );
 *
 * console.assert(example.title === 'Tiny transpose');
 * console.assert(example.state.specsText === 'T: i j -> j i');
 * console.assert(example.state.operationText === 'T');
 * console.assert(example.state.inputName === 'activations');
 * console.assert(example.state.propagateOutputs === false);
 */
function bakedExample(
    title: string,
    specsText: string,
    operationText: string,
    inputName = DEFAULT_INPUT_NAME,
): ExampleState {
    const state = {
        specsText,
        operationText,
        inputName,
        presetSelection: matchedComposeLayoutPresetSelection({ specsText, operationText, inputName }),
        visibleTensors: {},
        propagateOutputs: false,
        mapping: { H: 'none', S: 'none', L: 'none' },
        ranges: defaultColorRanges(),
    };
    return {
        title,
        state,
    };
}

const BAKED_EXAMPLES: ExampleState[] = [
    ...BAKED_EXAMPLE_DEFINITIONS.map(({ title, specsText, operationText, inputName }) => (
        bakedExample(title, specsText, operationText, inputName)
    )),
];

/**
 * Returns the startup compose-layout state used by the static demo.
 *
 * This is intentionally derived from the baked catalog instead of duplicating
 * text fields, otherwise the default tab can drift from the examples synced out
 * of demo_linear_layout.py.
 *
 * @returns An auto-colored clone of the first baked example's ComposeLayoutState, suitable for initializing the linear-layout sidebar or a new default tab without sharing mutable state with the catalog.
 * @noThrows The function reads the first in-module baked example and clones/colors that state; it accepts no caller input and performs no compose-layout parsing.
 * @example
 * const state = defaultComposeLayoutState();
 * const document = createComposeLayoutDocument(state);
 *
 * console.assert(typeof state.specsText === 'string');
 * console.assert(typeof state.operationText === 'string');
 * console.assert(document !== undefined);
 */
export function defaultComposeLayoutState(): ComposeLayoutState {
    return autoColoredComposeLayoutState(cloneComposeLayoutState(BAKED_EXAMPLES[0]!.state));
}

/**
 * Builds the renderable compose-layout sidebar state used when the user opens a new blank linear-layout tab.
 *
 * The state contains the default empty spec text, operation name, input tensor name, empty tensor visibility, disabled output propagation, and the auto-color mapping/ranges for `Layout_1`.
 *
 * @returns A fresh `ComposeLayoutState` that can be stored on a new tab before the user has chosen a baked example or edited the layout text.
 * @noThrows Uses module constants and deterministic auto-color defaults only, so it has no validation, parsing, storage, or I/O path that is expected to throw.
 * @example
 * const state = emptyComposeLayoutState();
 *
 * console.assert(state.operationText === 'Layout_1');
 * console.assert(state.inputName === 'A');
 * console.assert(state.propagateOutputs === false);
 * console.assert(Object.keys(state.visibleTensors).length === 0);
 */
export function emptyComposeLayoutState(): ComposeLayoutState {
    const autoColor = autoColorLayoutState(DEFAULT_EMPTY_SPEC_TEXT, 'Layout_1');
    return {
        specsText: DEFAULT_EMPTY_SPEC_TEXT,
        operationText: 'Layout_1',
        inputName: DEFAULT_INPUT_NAME,
        presetSelection: emptyComposeLayoutPresetSelection(),
        visibleTensors: {},
        propagateOutputs: false,
        mapping: autoColor.mapping,
        ranges: autoColor.ranges,
    };
}

/**
 * Returns the built-in linear-layout examples as freshly cloned, auto-colored sidebar states.
 *
 * Widget code can copy one of these entries into a tab and then mutate its text, visibility map, mapping, or color ranges without modifying the shared baked-example definitions.
 *
 * @returns Example entries with the baked title and an independent `ComposeLayoutState` for each sidebar preset.
 * @noThrows Reads in-memory baked examples and clones plain state objects; it does not parse user text, access storage, or perform I/O.
 * @example
 * const [first] = bakedComposeLayoutExamples();
 * const again = bakedComposeLayoutExamples();
 *
 * console.assert(typeof first.title === 'string');
 * first.state.visibleTensors.debug = true;
 * console.assert(again[0].state.visibleTensors.debug === undefined);
 */
export function bakedComposeLayoutExamples(): ExampleState[] {
    return BAKED_EXAMPLES.map(({ title, state }) => ({
        title,
        state: autoColoredComposeLayoutState(cloneComposeLayoutState(state)),
    }));
}

/**
 * Copies a compose-layout sidebar state so tab snapshots and saved viewer metadata do not share editable nested objects.
 *
 * The clone preserves the layout/spec text, selected preset, tensor visibility, output-propagation flag, channel mapping, and H/S/L color ranges while allocating new objects and arrays for mutable fields.
 *
 * @param state - A validated compose-layout state from a live sidebar, saved tab, or snapshot metadata.
 * @returns An independent `ComposeLayoutState` with the same field values; callers can mutate the clone for another tab without changing the original state.
 * @noThrows Expects `state` to already satisfy `ComposeLayoutState`, including `ranges.H`, `ranges.S`, and `ranges.L` arrays, so the function only performs property reads, object spreads, and array copies.
 * @example
 * const original = emptyComposeLayoutState();
 * const copy = cloneComposeLayoutState(original);
 *
 * copy.visibleTensors.output = true;
 * copy.ranges.H[0] = 42;
 *
 * console.assert(original.visibleTensors.output === undefined);
 * console.assert(original.ranges.H[0] !== 42);
 */
export function cloneComposeLayoutState(state: ComposeLayoutState): ComposeLayoutState {
    // state is copied across tabs and snapshots, so clone nested structures to
    // keep one tab's widget edits from mutating another tab's saved state.
    return {
        specsText: state.specsText,
        operationText: state.operationText,
        inputName: state.inputName,
        presetSelection: cloneComposeLayoutPresetSelection(state.presetSelection),
        visibleTensors: { ...state.visibleTensors },
        propagateOutputs: state.propagateOutputs ?? false,
        mapping: { ...state.mapping },
        ranges: {
            H: [...state.ranges.H],
            S: [...state.ranges.S],
            L: [...state.ranges.L],
        },
    };
}

/**
 * Checks whether unknown saved tab metadata has the fields needed to restore the compose-layout sidebar.
 *
 * The guard accepts objects with string layout/spec/input fields, optional preset and propagation fields, a tensor-visibility object, string H/S/L channel mappings, and array H/S/L color ranges.
 *
 * @param value - Unknown data read from viewer manifest metadata, browser storage, or another persisted tab source.
 * @returns `true` when `value` has the compose-layout state shape and is narrowed to `ComposeLayoutState`; otherwise `false` so callers can fall back to a default state.
 * @noThrows Malformed persisted values are handled with type checks and optional property access, so missing or wrong-typed fields are reported as `false` instead of throwing.
 * @example
 * const saved = {
 *   specsText: '',
 *   operationText: 'Layout_1',
 *   inputName: 'A',
 *   visibleTensors: {},
 *   mapping: { H: 'operation', S: 'input', L: 'output' },
 *   ranges: { H: [0, 360], S: [40, 90], L: [35, 70] },
 * };
 *
 * console.assert(isComposeLayoutState(saved) === true);
 * console.assert(isComposeLayoutState({ specsText: '', mapping: {} }) === false);
 */
export function isComposeLayoutState(value: unknown): value is ComposeLayoutState {
    if (!value || typeof value !== 'object') return false;
    const record = value as ComposeLayoutState;
    return typeof record.specsText === 'string'
        && typeof record.operationText === 'string'
        && typeof record.inputName === 'string'
        && (record.presetSelection === undefined || isComposeLayoutPresetSelection(record.presetSelection))
        && !!record.visibleTensors
        && typeof record.visibleTensors === 'object'
        && (record.propagateOutputs === undefined || typeof record.propagateOutputs === 'boolean')
        && ['H', 'S', 'L'].every((channel) => typeof record.mapping?.[channel as ComposeChannel] === 'string')
        && ['H', 'S', 'L'].every((channel) => Array.isArray(record.ranges?.[channel as ComposeChannel]));
}

/**
 * Test whether an unknown viewer manifest value is the compose-layout metadata
 * saved on linear-layout tabs.
 *
 * The guard accepts metadata versions 1 and 3, requires the saved specs and
 * operation text strings, and checks that label, bit-count, and tensor fields
 * are arrays before session restore code treats the value as layout metadata.
 *
 * @param value - Unknown `viewer.composeLayoutMeta` value read from a loaded tab manifest.
 * @returns `true` when `value` has the required compose-layout metadata fields and can be narrowed to `ComposeLayoutMeta`; otherwise `false`.
 * @noThrows Only performs null, `typeof`, equality, and `Array.isArray` checks against the unknown value, so malformed metadata is rejected instead of parsed or dereferenced deeply.
 * @example
 * const meta = {
 *   version: 3,
 *   specsText: 'Pair: [A,B] -> [C]\nA: [[1]]\nB: [[2]]',
 *   operationText: 'Pair',
 *   rootInputLabels: ['A', 'B'],
 *   rootInputBitCounts: [1, 1],
 *   finalOutputLabels: ['C'],
 *   finalOutputBitCounts: [2],
 *   tensors: [],
 * };
 *
 * expect(isComposeLayoutMeta(meta)).toBe(true);
 * expect(isComposeLayoutMeta({ version: 3, specsText: 'Pair' })).toBe(false);
 */
export function isComposeLayoutMeta(value: unknown): value is ComposeLayoutMeta {
    if (!value || typeof value !== 'object') return false;
    const record = value as Record<string, unknown>;
    // version 1 is accepted only for older saved sessions; new documents write v3.
    return (record.version === 1 || record.version === 3)
        && typeof record.specsText === 'string'
        && typeof record.operationText === 'string'
        && (record.inputName === undefined || typeof record.inputName === 'string')
        && (record.injective === undefined || typeof record.injective === 'boolean')
        && Array.isArray(record.rootInputLabels)
        && Array.isArray(record.rootInputBitCounts)
        && Array.isArray(record.finalOutputLabels)
        && Array.isArray(record.finalOutputBitCounts)
        && Array.isArray(record.tensors);
}

/**
 * Convert metadata from older python/demo linear-layout sessions into the
 * compose-layout editor state used by the current sidebar.
 *
 * Old sessions stored input dimensions, bases, and color axes before the
 * compose-layout editor existed. This migration normalizes legacy names such as
 * thread/warp/register to the current short labels, formats compose-layout spec
 * text, and preserves legacy color choices when present.
 *
 * @param raw - Unknown legacy session payload, typically an object with fields such as `name`, `input_dims`, `output_dims`, `bases`, `color_axes`, or `color_ranges`.
 * @param fallbackTitle - Tab title to use when the legacy payload does not contain a usable layout name; defaults to `Layout_1`.
 * @returns A `ComposeLayoutState` that can be installed in the linear-layout sidebar, including generated `specsText`, `operationText`, default input name, visibility state, and migrated or automatically generated color mapping/ranges.
 * @noThrows The legacy parser treats missing or malformed legacy fields as absent and fills editor defaults, so unsupported saved-session shapes migrate to a usable fallback state rather than throwing.
 * @example
 * const state = composeLayoutStateFromLegacySpec({
 *   name: 'Legacy',
 *   input_dims: ['warp', 'thread', 'register'],
 *   output_dims: ['Y', 'X'],
 *   bases: [[[1]], [[2]], [[4]]],
 * }, 'Saved tab');
 *
 * expect(state.operationText).toBe('Legacy');
 * expect(state.specsText).toContain('Legacy: [T,W,R] -> [Y,X]');
 * expect(state.inputName).toBe('input');
 */
export function composeLayoutStateFromLegacySpec(raw: unknown, fallbackTitle = 'Layout_1'): ComposeLayoutState {
    const legacy = parseLegacySpec(raw, fallbackTitle);
    const labelMap = new Map<string, string>();
    // old demos used semantic names like thread/warp/register.  Normalize them
    // to the short labels used by current presets so color mappings survive.
    const inputEntries = legacy.inputs.map((name, axis) => {
        const label = canonicalLegacyLabel(name, axis);
        labelMap.set(name, label);
        return { label, bases: legacy.bases[axis] ?? [] };
    });
    inputEntries.sort((left, right) => legacyAxisOrder(left.label) - legacyAxisOrder(right.label));
    const outputs = legacy.outputs.map((name, axis) => canonicalLegacyOutputLabel(name, axis));
    const specsText = formatSpecsText([{
        name: sanitizeIdentifier(legacy.name || fallbackTitle || 'Layout_1'),
        inputs: inputEntries.map((entry) => entry.label),
        outputs,
        bases: inputEntries.map((entry) => entry.bases),
    }]);
    const operationText = sanitizeIdentifier(legacy.name || fallbackTitle || 'Layout_1');
    const autoColor = autoColorLayoutState(specsText, operationText);
    return {
        specsText,
        operationText,
        inputName: DEFAULT_INPUT_NAME,
        presetSelection: emptyComposeLayoutPresetSelection(),
        visibleTensors: {},
        propagateOutputs: false,
        mapping: Object.keys(legacy.colorAxes).length > 0 ? legacyMapping(labelMap, legacy.colorAxes) : autoColor.mapping,
        ranges: Object.keys(legacy.colorRanges).length > 0 ? legacyRanges(legacy.colorRanges) : autoColor.ranges,
    };
}

/**
 * Derive the default H/S/L color-channel assignments for a compose-layout
 * operation.
 *
 * The operation is evaluated so the color picker can rank the actual propagated
 * labels by bit width. When output propagation is enabled, the labels come from
 * the final output space; otherwise they come from the root input space.
 *
 * @param specsText - Compose-layout specification text containing the layout declarations and basis matrices to evaluate.
 * @param operationText - Operation name or expression to evaluate against `specsText` before choosing color labels.
 * @param propagateOutputs - Whether color channels should target propagated final-output labels instead of root-input labels.
 * @returns The color `mapping` selected for the largest available labels plus the default color `ranges` used by the linear-layout color widget.
 * @throws If `specsText` cannot be parsed or `operationText` does not resolve to a valid compose-layout operation.
 * @example
 * const colors = autoColorLayoutState([
 *   'Pair: [A,B] -> [C]',
 *   'A: [[1],[2],[4]]',
 *   'B: [[8],[16]]',
 * ].join('\n'), 'Pair');
 *
 * expect(colors.mapping.H).toBe('B');
 * expect(colors.mapping.L).toBe('A');
 * expect(colors.ranges.H).toEqual([0, 1]);
 *
 * @example
 * expect(() => autoColorLayoutState('Pair: [A] -> [B]', 'Missing')).toThrow();
 */
export function autoColorLayoutState(
    specsText: string,
    operationText: string,
    propagateOutputs = false,
): Pick<ComposeLayoutState, 'mapping' | 'ranges'> {
    // auto-color needs the evaluated operation, not just the raw specs, because
    // propagateOutputs changes whether labels come from the root or final space.
    const runtime = buildComposeRuntime({
        specsText,
        operationText,
        inputName: DEFAULT_INPUT_NAME,
        visibleTensors: {},
    });
    return {
        mapping: autoColorMapping(...propagationLabels(runtime, propagateOutputs)),
        ranges: defaultColorRanges(),
    };
}

/**
 * Return a compose-layout editor state with automatically regenerated color
 * mapping and color ranges.
 *
 * The helper preserves the supplied editor fields, such as spec text, operation
 * text, preset selection, visibility flags, and propagation settings, while
 * replacing stale manual color settings with defaults derived from the current
 * operation.
 *
 * @param state - Compose-layout editor state whose `specsText` and `operationText` should be evaluated for automatic color selection.
 * @returns A new `ComposeLayoutState` with the same non-color fields as `state` and refreshed `mapping` and `ranges` fields for the color widget.
 * @throws If `state.specsText` cannot be parsed or `state.operationText` does not resolve to a valid compose-layout operation.
 * @example
 * const recolored = autoColoredComposeLayoutState({
 *   ...emptyComposeLayoutState(),
 *   specsText: 'Pair: [A,B] -> [C]\nA: [[1]]\nB: [[2],[4]]',
 *   operationText: 'Pair',
 *   mapping: { H: null, S: null, L: null },
 * });
 *
 * expect(recolored.operationText).toBe('Pair');
 * expect(recolored.mapping.H).toBe('B');
 */
function autoColoredComposeLayoutState(state: ComposeLayoutState): ComposeLayoutState {
    const autoColor = autoColorLayoutState(state.specsText, state.operationText);
    return {
        ...state,
        mapping: autoColor.mapping,
        ranges: autoColor.ranges,
    };
}

/**
 * Converts the sidebar's compose-layout text into the runtime model consumed by
 * the linear-layout widgets and viewer synchronization layer.
 *
 * The runtime is the boundary between user-editable strings and viewer data: it
 * parses named layout specs, evaluates the operation AST over GF(2), builds the
 * render chain, creates matrix-preview metadata, and emits Python reconstruction
 * code for the same layout operations.
 *
 * @param state - Sidebar fields containing layout specification text, operation
 * text, the root input tensor display name, and per-tensor visibility flags.
 * `specsText` must define at least one named layout and `operationText` must
 * reference those names using the compose-layout expression syntax.
 * @returns The evaluated compose runtime: root/final labels and shapes, visible
 * tensor metadata, render-chain metadata, matrix previews, generated Python, and
 * coordinate mappings used by hover, selection, propagation, and document
 * creation.
 * @throws Error when no layout specification is present, the operation text is
 * empty, the operation references an unknown layout name, or an expression such
 * as `inv(...)`, product, or composition is not valid for the parsed layouts.
 * @example
 * const runtime = buildComposeRuntime({
 *   specsText: [
 *     'Identity: [X] -> [Y]',
 *     'X: [[1]]',
 *   ].join('\n'),
 *   operationText: 'Identity',
 *   inputName: 'lane',
 *   visibleTensors: {},
 * });
 *
 * expect(runtime.inputLabels).toEqual(['X']);
 * expect(runtime.finalOutputLabels).toEqual(['Y']);
 * expect(runtime.tensors.map((tensor) => tensor.title)).toEqual(['lane', 'Identity']);
 * @example
 * expect(() => buildComposeRuntime({
 *   specsText: 'Stretch: [X] -> [Y]\nX: [[0],[0]]',
 *   operationText: 'inv(Stretch)',
 *   inputName: 'input',
 *   visibleTensors: {},
 * })).toThrow('Stretch is not bijective, so inv(Stretch) is invalid.');
 */
export function buildComposeRuntime(state: Pick<ComposeLayoutState, 'specsText' | 'operationText' | 'inputName' | 'visibleTensors'>): ComposeRuntime {
    const specs = parseLayoutSpecs(state.specsText);
    if (specs.length === 0) {
        throw new Error('At least one layout specification is required.');
    }
    const operationText = state.operationText.trim();
    if (!operationText) throw new Error('Layout Operation is required.');
    const operation = parseOperation(operationText);
    const specLayouts = new Map(specs.map((spec) => [spec.name, namedLayout(spec)]));
    const codeLines: CodeLine[] = [];
    const tempRefs = new WeakMap<object, EvaluatedLayout>();
    let tempIndex = 0;
    // evaluation is memoized by AST node so matrix blocks and generated python
    // reuse the same temporary names for shared subexpressions.
    /**
 * Evaluates one parsed compose-layout expression and records the Python line
 * needed to reconstruct any temporary layout created by that expression.
 *
 * Named expressions resolve to parsed layout specs. Inverses, products, and
 * applications recursively evaluate their operands, create a temporary code
 * reference, and append the matching Python initialization line for the render
 * chain.
 *
 * @param expr - A parsed `LayoutExpr` node from the operation AST. Name nodes
 * must refer to an entry in `specLayouts`; compound nodes must contain operands
 * whose labels and bit counts make the requested inverse, product, or compose
 * operation legal.
 * @returns The evaluated layout for `expr`, including its GF(2) matrix,
 * input/output labels and bit counts, expression text, and stable `codeRef` used
 * by matrix previews and generated Python.
 * @throws Error when a name node does not match a parsed layout spec, or when a
 * recursive inverse, product, or compose operation is invalid for the operand
 * layouts.
 * @example
 * const layout = evaluate({ kind: 'name', name: 'Identity' });
 *
 * expect(layout.exprText).toBe('Identity');
 * expect(layout.codeRef).toBe('Identity');
 * @example
 * expect(() => evaluate({ kind: 'name', name: 'Missing' }))
 *   .toThrow('Unknown layout Missing.');
 */
    const evaluate = (expr: LayoutExpr): EvaluatedLayout => {
        const cached = tempRefs.get(expr as object);
        if (cached) return cached;
        let layout: EvaluatedLayout;
        switch (expr.kind) {
            case 'name': {
                const named = specLayouts.get(expr.name);
                if (!named) throw new Error(`Unknown layout ${expr.name}.`);
                layout = named;
                break;
            }
            case 'inverse': {
                const inner = evaluate(expr.expr);
                const inverse = invertLayout(inner, exprToString(expr));
                const codeRef = `layout_tmp${tempIndex += 1}`;
                layout = { ...inverse, codeRef };
                codeLines.push({
                    title: `${codeRef} = ${layout.exprText}`,
                    layout,
                    line: `${codeRef} = ${inner.codeRef}.invert()`,
                });
                break;
            }
            case 'product': {
                const left = evaluate(expr.left);
                const right = evaluate(expr.right);
                const productResult = productLayout(left, right, exprToString(expr));
                const codeRef = `layout_tmp${tempIndex += 1}`;
                layout = { ...productResult, codeRef };
                codeLines.push({
                    title: `${codeRef} = ${layout.exprText}`,
                    layout,
                    line: `${codeRef} = ${left.codeRef} * ${right.codeRef}`,
                });
                break;
            }
            case 'apply': {
                const outer = evaluate(expr.outer);
                const inner = evaluate(expr.inner);
                // user syntax is Outer(Inner), while the triton python method is
                // inner.compose(outer); keeping the order explicit prevents codegen
                // from silently reversing composition.
                const composed = composeLayouts(inner, outer, exprToString(expr));
                const codeRef = `layout_tmp${tempIndex += 1}`;
                layout = { ...composed, codeRef };
                codeLines.push({
                    title: `${codeRef} = ${layout.exprText}`,
                    layout,
                    line: `${codeRef} = ${inner.codeRef}.compose(${outer.codeRef})`,
                });
                break;
            }
        }
        tempRefs.set(expr as object, layout);
        return layout;
    };
    const renderChain = renderSteps(operation, evaluate);
    const finalLayout = renderChain.at(-1)?.layout;
    if (!finalLayout) throw new Error('Unable to evaluate Layout Operation.');
    const inputLabels = finalLayout.inputs.slice();
    const inputBitCounts = finalLayout.inputBitCounts.slice();
    const inputShape = shapeFromBitCounts(inputBitCounts);
    const rootCount = product(inputShape);
    const finalOutputLabels = finalLayout.outputs.slice();
    const finalOutputBitCounts = finalLayout.outputBitCounts.slice();
    const finalOutputShape = shapeFromBitCounts(finalOutputBitCounts);
    const injective = gf2Rank(finalLayout.matrix) === sum(finalLayout.inputBitCounts);
    const inputName = state.inputName.trim() || DEFAULT_INPUT_NAME;
    const visibleTensors = state.visibleTensors ?? {};
    // every intermediate tensor is indexed by the same root input space; this is
    // what lets selection, hover popups, and multi-input display stay aligned.
    const finalCoords = Array.from({ length: rootCount }, (_entry, rootIndex) => (
        mapCoord(unravelIndex(rootIndex, inputShape), inputBitCounts, finalLayout.matrix, finalOutputBitCounts)
    ));
    const rootTensor: ComposeTensorMeta = {
        id: 'compose-root',
        title: inputName,
        exprText: inputName,
        kind: 'root',
        axisLabels: inputLabels.slice(),
        shape: inputShape.slice(),
        rootToTensor: Array.from({ length: rootCount }, (_entry, index) => unravelIndex(index, inputShape)),
        tensorToFinal: finalCoords.map((coord) => coord.slice()),
        visible: visibleTensors['compose-root'] ?? true,
    };
    const tensors = [
        rootTensor,
        ...renderChain.map(({ layout, exprText }, index) => {
            const shape = shapeFromBitCounts(layout.outputBitCounts);
            const rootToTensor = Array.from({ length: rootCount }, (_entry, rootIndex) => (
                mapCoord(unravelIndex(rootIndex, inputShape), inputBitCounts, layout.matrix, layout.outputBitCounts)
            ));
            const tensorToFinal = Array.from({ length: product(shape) }, () => null as number[] | null);
            rootToTensor.forEach((coord, rootIndex) => {
                // non-injective layouts overwrite this slot with the last root,
                // while cellRootIndexes in createComposeLayoutDocument preserves
                // every root for hover and multi-input display.
                tensorToFinal[flatIndex(coord, shape)] = finalCoords[rootIndex]!.slice();
            });
            return {
                id: `compose-step-${index + 1}`,
                title: exprText,
                exprText,
                kind: 'step' as const,
                axisLabels: layout.outputs.slice(),
                shape,
                rootToTensor,
                tensorToFinal,
                visible: visibleTensors[`compose-step-${index + 1}`] ?? true,
            };
        }),
    ];
    const matrixBlocks = [
        ...specs.map((spec) => matrixBlock(spec.name, namedLayout(spec))),
        ...codeLines.map(({ title, layout }) => matrixBlock(title, layout)),
    ];
    // generated code follows the exact evaluated expression, including temporaries,
    // so users can copy from the widget and reproduce the same matrix preview.
    const pythonCode = [
        ...specs.flatMap((spec) => pythonNamedLayout(spec)),
        ...(specs.length ? [''] : []),
        ...codeLines.map(({ line }) => line),
    ].filter((line, index, lines) => !(line === '' && lines[index - 1] === '')).join('\n').trim();
    const meta: ComposeLayoutMeta = {
        version: 3,
        specsText: state.specsText,
        operationText,
        inputName,
        injective,
        rootInputLabels: inputLabels.slice(),
        rootInputBitCounts: inputBitCounts.slice(),
        finalOutputLabels: finalOutputLabels.slice(),
        finalOutputBitCounts: finalOutputBitCounts.slice(),
        tensors: tensors.map((tensor) => ({
            ...tensor,
            rootToTensor: tensor.rootToTensor.map((coord) => coord.slice()),
            tensorToFinal: tensor.tensorToFinal.map((coord) => coord ? coord.slice() : null),
            axisLabels: tensor.axisLabels.slice(),
            shape: tensor.shape.slice(),
        })),
    };
    return {
        specs,
        inputLabels,
        inputBitCounts,
        inputShape,
        finalOutputLabels,
        finalOutputBitCounts,
        finalOutputShape,
        injective,
        tensors,
        matrixBlocks,
        pythonCode,
        meta,
    };
}

/**
 * Builds the loaded viewer tab for one evaluated compose-layout sidebar state.
 *
 * This is where abstract layout metadata becomes concrete tensors: dense buffers
 * store displayed root or propagated-output ids, RGB buffers store the current
 * color mapping, marker coordinates identify holes, and compose metadata lets
 * hover, selection, tensor-view presets, and saved snapshots round-trip through
 * the viewer.
 *
 * @param state - Complete compose-layout sidebar state, including specs,
 * operation text, input name, output propagation, color mapping/ranges, and
 * tensor visibility. At least one tensor in the evaluated render chain must be
 * visible.
 * @param viewer - Optional viewer snapshot whose persisted settings seed the
 * document manifest, for example when recreating a tab from saved state.
 * @param title - Optional tab title. When omitted, the document title falls back
 * to the last visible tensor title and then to the operation text.
 * @param tensorViews - Optional per-tensor view snapshots. Entries whose ids
 * match generated tensor ids are copied into the corresponding manifest tensor
 * view settings.
 * @returns A `LoadedBundleDocument` with id `compose-layout`, a manifest for the
 * visible tensors, dense Float32 tensor buffers keyed by tensor id, and embedded
 * compose-layout state/meta used by the extension after the tab loads.
 * @throws Error when runtime construction fails for invalid compose-layout text
 * or when the visibility settings hide every tensor in the render chain.
 * @example
 * const document = createComposeLayoutDocument({
 *   ...defaultComposeLayoutState(),
 *   specsText: 'Identity: [X] -> [Y]\nX: [[1]]',
 *   operationText: 'Identity',
 *   inputName: 'lane',
 * }, { showSelectionPanel: false }, 'Identity tab');
 *
 * expect(document.id).toBe('compose-layout');
 * expect(document.title).toBe('Identity tab');
 * expect([...document.tensors.keys()]).toContain('compose-root');
 * expect(document.manifest.viewer.composeLayoutMeta.operationText).toBe('Identity');
 * @example
 * expect(() => createComposeLayoutDocument({
 *   ...defaultComposeLayoutState(),
 *   visibleTensors: { 'compose-root': false, 'compose-0': false },
 * })).toThrow('At least one tensor in the render chain must stay visible.');
 */
export function createComposeLayoutDocument(
    state: ComposeLayoutState,
    viewer?: Partial<ViewerSnapshot>,
    title?: string,
    tensorViews?: Record<string, TensorViewSnapshot>,
): LoadedBundleDocument {
    const runtime = buildComposeRuntime(state);
    const visibleTensors = runtime.tensors.filter((tensor) => tensor.visible);
    if (visibleTensors.length === 0) {
        throw new Error('At least one tensor in the render chain must stay visible.');
    }
    const colorBuffers = composeTensorColorBuffers(runtime, state);
    const tensors = new Map<string, Float32Array>();
    const manifest = createBundleManifest({
        viewer: persistedViewerSettings(viewer),
        tensors: visibleTensors.map((tensor) => {
            const data = new Float32Array(product(tensor.shape)).fill(-1);
            const cellRootIndexes = Array.from({ length: product(tensor.shape) }, () => [] as number[]);
            tensor.rootToTensor.forEach((coord, rootIndex) => {
                cellRootIndexes[flatIndex(coord, tensor.shape)]!.push(rootIndex);
            });
            // the dense data buffer stores the displayed root id.  Empty cells
            // stay -1 so the viewer can mark holes without inventing values.
            cellRootIndexes.forEach((rootIndexes, flat) => {
                const rootIndex = rootIndexes[0];
                if (rootIndex === undefined) return;
                data[flat] = rootIndex;
            });
            const rgb = colorBuffers.get(tensor.id) ?? new Float32Array(product(tensor.shape) * 3);
            tensor.tensorToFinal.forEach((coord, flat) => {
                // output propagation switches displayed values from input-root
                // ids to final-output ids so labels/colors share one coordinate space.
                if (coord && state.propagateOutputs) data[flat] = flatIndex(coord, runtime.finalOutputShape);
            });
            const markerCoords = Array.from({ length: data.length }, (_entry, index) => index)
                .filter((index) => data[index] < 0)
                .map((index) => unravelIndex(index, tensor.shape));
            tensors.set(tensor.id, data);
            return {
                id: tensor.id,
                name: tensor.title,
                dtype: 'float32' as const,
                shape: tensor.shape,
                axisLabels: tensor.axisLabels,
                view: tensorViews?.[tensor.id]
                    ? {
                        editor: tensorViews[tensor.id]!.editor,
                        hiddenIndices: tensorViews[tensor.id]!.hiddenIndices.slice(),
                    }
                    : undefined,
                colorInstructions: [{ mode: 'rgb' as const, kind: 'dense' as const, values: Array.from(rgb) }],
                markerCoords,
            };
        }),
    });
    const viewerState = manifest.viewer as ViewerSnapshot & {
        composeLayoutState?: ComposeLayoutState;
        composeLayoutMeta?: ComposeLayoutMeta;
    };
    viewerState.composeLayoutState = cloneComposeLayoutState(state);
    viewerState.composeLayoutMeta = runtime.meta;
    return {
        id: 'compose-layout',
        title: title ?? visibleTensors.at(-1)?.title ?? runtime.meta.operationText,
        manifest,
        tensors,
    };
}

/**
 * Returns the RGB color assigned to each root input coordinate by the active
 * hue/saturation/lightness mapping.
 *
 * The result is ordered by flattened root index, so callers can use the array
 * directly for hover highlights, ghost layers, and dense tensor color buffers
 * that need to stay aligned with root-input membership.
 *
 * @param inputLabels - Root tensor axis labels in shape order. Mapping channels
 * in `state.mapping` refer to these labels to choose which coordinate controls
 * hue, saturation, or lightness.
 * @param inputShape - Root tensor extents corresponding to `inputLabels`; the
 * product of these dimensions determines the number of RGB triples returned.
 * @param state - Color configuration containing the selected label-to-H/S/L
 * mapping and numeric ranges used to convert root coordinates into RGB values.
 * @returns RGB triples in flattened root-coordinate order; entry `i` is the
 * color for `unravelIndex(i, inputShape)`.
 * @noThrows The function only builds an in-memory label map, enumerates root
 * coordinates, and delegates each coordinate to the color mapper; it performs no
 * parsing, I/O, or explicit validation that would create an expected throw path
 * for normal compose-layout state.
 * @example
 * const colors = rootColorsForLayoutState(['X'], [2], {
 *   mapping: { hue: 'X', saturation: undefined, lightness: undefined },
 *   ranges: {
 *     hue: [0, 360],
 *     saturation: [100, 100],
 *     lightness: [50, 50],
 *   },
 * });
 *
 * expect(colors).toHaveLength(2);
 * expect(colors[0]).toHaveLength(3);
 * expect(colors[1]).toHaveLength(3);
 */
export function rootColorsForLayoutState(
    inputLabels: string[],
    inputShape: number[],
    state: Pick<ComposeLayoutState, 'mapping' | 'ranges'>,
): Array<[number, number, number]> {
    const rootLabelToAxis = new Map(inputLabels.map((label, axis) => [label, axis]));
    return Array.from({ length: product(inputShape) }, (_entry, rootIndex) => (
        rgbColorForRootCoord(
            unravelIndex(rootIndex, inputShape),
            inputShape,
            rootLabelToAxis,
            state.mapping,
            state.ranges,
        )
    ));
}

/**
 * Builds the per-tensor RGB payloads that tint visible compose-layout cells by their propagated root coordinate.
 *
 * @param runtime - Compose runtime containing the rendered tensor metadata, tensor shapes, visibility flags, and input/output propagation spaces.
 * @param state - Color configuration for the render pass: label-to-channel mapping, numeric channel ranges, and whether colors propagate from final outputs instead of inputs.
 * @returns A map from each visible tensor id to a dense `Float32Array` of `r,g,b` triples in that tensor's flat viewer storage order; hidden tensors are omitted.
 * @noThrows For a normalized compose runtime and color state, this only allocates arrays and copies precomputed coordinate/color values; visibility filtering and null propagation holes are handled without throwing.
 * @example
 * const buffers = composeTensorColorBuffers(runtime, {
 *   mapping: autoColorMapping(['M', 'N'], [2, 2]),
 *   ranges: defaultColorRanges(),
 *   propagateOutputs: false,
 * });
 *
 * const lhsRgb = buffers.get('lhs');
 * expect(lhsRgb).toBeInstanceOf(Float32Array);
 * expect(lhsRgb).toHaveLength(product(runtime.tensors.find((tensor) => tensor.id === 'lhs')!.shape) * 3);
 * expect(buffers.has('hidden-intermediate')).toBe(false);
 */
export function composeTensorColorBuffers(
    runtime: ComposeRuntime,
    state: Pick<ComposeLayoutState, 'mapping' | 'ranges' | 'propagateOutputs'>,
): Map<string, Float32Array> {
    const [labels, shape] = propagationLabels(runtime, state.propagateOutputs);
    const labelToAxis = new Map(labels.map((label, axis) => [label, axis]));
    const colors = Array.from({ length: product(shape) }, (_entry, index) => (
        rgbColorForRootCoord(
            unravelIndex(index, shape),
            shape,
            labelToAxis,
            state.mapping,
            state.ranges,
        )
    ));
    return new Map(runtime.tensors
        .filter((tensor) => tensor.visible)
        .map((tensor) => {
            const rgb = new Float32Array(product(tensor.shape) * 3);
            // map each rendered cell into the active propagation space.  null
            // coordinates represent holes in non-surjective intermediate layouts.
            propagationCoordsForTensor(runtime, tensor, state.propagateOutputs).forEach((coord, flat) => {
                if (!coord) return;
                rgb.set(colors[flatIndex(coord, shape)]!, flat * 3);
            });
            return [tensor.id, rgb] as const;
        }));
}

/**
 * Selects the axis labels and extents for the coordinate space currently used by propagated colors and cell text.
 *
 * @param runtime - Compose runtime that stores both the original input labels/shape and the final output labels/shape.
 * @param propagateOutputs - When `true`, use the final output coordinate space; when `false`, use the original input coordinate space.
 * @returns A `[labels, shape]` tuple whose arrays identify the active propagation axes and their extents for color mapping and widget labels.
 * @noThrows The function only returns one of two label/shape pairs already present on the runtime and performs no parsing, indexing, or allocation that can fail for a valid runtime object.
 * @example
 * const runtime = {
 *   inputLabels: ['X'],
 *   inputShape: [2],
 *   finalOutputLabels: ['Y'],
 *   finalOutputShape: [4],
 * } as ComposeRuntime;
 *
 * expect(propagationLabels(runtime, false)).toEqual([['X'], [2]]);
 * expect(propagationLabels(runtime, true)).toEqual([['Y'], [4]]);
 */
export function propagationLabels(
    runtime: ComposeRuntime,
    propagateOutputs: boolean,
): [string[], number[]] {
    if (propagateOutputs) return [runtime.finalOutputLabels, runtime.finalOutputShape];
    return [runtime.inputLabels, runtime.inputShape];
}

/**
 * Produces a tensor-cell lookup into the active propagation coordinate space used to fill color and text buffers.
 *
 * @param runtime - Compose runtime whose `inputShape` is used to decode root indices when input-space propagation is active.
 * @param tensor - Tensor metadata containing the tensor shape plus either `tensorToFinal` output coordinates or `rootToTensor` input-to-cell coordinates.
 * @param propagateOutputs - When `true`, return the tensor's precomputed final-output coordinates; when `false`, reverse `rootToTensor` into tensor storage order.
 * @returns An array indexed by the tensor's flat cell offset; each entry is the propagated coordinate for that cell, or `null` for holes that do not correspond to a propagated root/output coordinate.
 * @noThrows For tensor metadata produced by compose-runtime construction, all coordinates are already bounds-checked and the function only reshapes those coordinate tables into viewer storage order.
 * @example
 * const coords = propagationCoordsForTensor(runtime, tensor, false);
 *
 * expect(coords).toHaveLength(product(tensor.shape));
 * expect(coords[flatIndex([0, 0], tensor.shape)]).toEqual([0]);
 * expect(coords.some((coord) => coord === null)).toBe(true);
 *
 * expect(propagationCoordsForTensor(runtime, tensor, true)).toBe(tensor.tensorToFinal);
 */
function propagationCoordsForTensor(
    runtime: ComposeRuntime,
    tensor: ComposeTensorMeta,
    propagateOutputs: boolean,
): Array<number[] | null> {
    if (propagateOutputs) return tensor.tensorToFinal;
    const coords = Array.from({ length: product(tensor.shape) }, () => null as number[] | null);
    // input propagation reverses rootToTensor into a cell-indexed lookup so
    // color buffers can be filled in viewer storage order.
    tensor.rootToTensor.forEach((tensorCoord, rootIndex) => {
        coords[flatIndex(tensorCoord, tensor.shape)] = unravelIndex(rootIndex, runtime.inputShape);
    });
    return coords;
}

/**
 * Parses sidebar compose-layout operation text into the expression tree evaluated by layout composition.
 *
 * @param source - Complete operation-language source string from the layout state, after the caller has applied any required trimming or empty-input checks.
 * @returns The `LayoutExpr` AST for the requested compose operation, ready to evaluate against named layouts and emit viewer metadata.
 * @throws Error when `source` contains malformed operation syntax, trailing tokens, or an incomplete expression rejected by `OperationParser.parse()` or `OperationParser.finish()`.
 * @example
 * const expr = parseOperation('L');
 * expect(expr).toMatchObject({ kind: 'name', name: 'L' });
 *
 * expect(() => parseOperation('L )')).toThrow(Error);
 */
function parseOperation(source: string): LayoutExpr {
    const parser = new OperationParser(source);
    const expr = parser.parse();
    parser.finish();
    return expr;
}

/**
 * Parses compose-layout operation expressions into the small AST used by layout
 * evaluation, generated Python, and matrix previews.
 *
 * The grammar keeps precedence intentionally explicit: function application such
 * as `A(B)` binds tighter than product `A * B`, and `inv(...)` is the only unary
 * form.
 *
 * @example
 * const parser = new OperationParser('inv(row_major) * swizzle(tile)');
 * const expr = parser.parse();
 * parser.finish();
 * expect(expr).toEqual({
 *   kind: 'product',
 *   left: { kind: 'inverse', expr: { kind: 'name', name: 'row_major' } },
 *   right: {
 *     kind: 'apply',
 *     outer: { kind: 'name', name: 'swizzle' },
 *     inner: { kind: 'name', name: 'tile' },
 *   },
 * });
 */
class OperationParser {
    private index = 0;

        /**
 * Creates a parser positioned at the start of one compose-layout operation
 * expression.
 *
 * @param source - Operation text from the linear-layout editor, using layout
 * names, `*` products, function application with parentheses, and `inv(...)`.
 * @noThrows Stores the source string and initializes the cursor only; syntax is
 * validated later by {@link parse} and {@link finish}.
 * @example
 * const parser = new OperationParser('row_major * inv(tile)');
 * const expr = parser.parse();
 * parser.finish();
 * expect(expr.kind).toBe('product');
 */
public constructor(private readonly source: string) {}

        /**
 * Reads the next complete layout operation expression from the current cursor
 * position and returns its AST.
 *
 * @returns A `LayoutExpr` tree whose nodes represent layout names, products,
 * nested application calls, and inverses for the operation evaluator.
 * @throws Error when the expression starts where no layout name, `inv(...)`, or
 * parenthesized expression is present, or when a nested form is missing its
 * closing parenthesis.
 * @example
 * const parser = new OperationParser('outer(inner) * inv(base)');
 * expect(parser.parse()).toEqual({
 *   kind: 'product',
 *   left: {
 *     kind: 'apply',
 *     outer: { kind: 'name', name: 'outer' },
 *     inner: { kind: 'name', name: 'inner' },
 *   },
 *   right: { kind: 'inverse', expr: { kind: 'name', name: 'base' } },
 * });
 *
 * @example
 * const parser = new OperationParser('* base');
 * expect(() => parser.parse()).toThrow('Expected a layout name, "inv(...)", or parenthesized expression.');
 */
public parse(): LayoutExpr {
        this.skipWhitespace();
        return this.parseProduct();
    }

        /**
 * Verifies that only whitespace remains after the parsed layout operation.
 *
 * Call this after {@link parse} to reject trailing characters that would
 * otherwise be ignored by the expression parser.
 *
 * @returns Nothing when the parser cursor has reached the end of the operation
 * text after optional whitespace.
 * @throws Error when any non-whitespace token remains after the parsed
 * expression, such as an extra layout name or unmatched punctuation.
 * @example
 * const parser = new OperationParser('base trailing');
 * parser.parse();
 * expect(() => parser.finish()).toThrow('Unexpected token "t" in Layout Operation.');
 */
public finish(): void {
        this.skipWhitespace();
        if (this.index < this.source.length) {
            throw new Error(`Unexpected token ${JSON.stringify(this.source[this.index])} in Layout Operation.`);
        }
    }

        /**
 * Parses `*`-separated compose-layout expressions, preserving left-to-right product composition after each application expression operand.
 *
 * @returns A LayoutExpr tree whose root is either the first application operand or a `product` node chaining each `*` operand for later layout evaluation.
 * @throws Error when a product operand is missing, an operand contains malformed `inv(...)` syntax, or a nested parenthesized/application expression is missing its closing `)`.
 * @example
 * const parser = new OperationParser('row * col * tile');
 * const expr = (parser as unknown as { parseProduct(): LayoutExpr }).parseProduct();
 * expect(expr).toEqual({
 *   kind: 'product',
 *   left: { kind: 'product', left: { kind: 'name', name: 'row' }, right: { kind: 'name', name: 'col' } },
 *   right: { kind: 'name', name: 'tile' },
 * });
 *
 * @example
 * const parser = new OperationParser('row *');
 * expect(() => (parser as unknown as { parseProduct(): LayoutExpr }).parseProduct()).toThrow(
 *   'Expected a layout name, "inv(...)", or parenthesized expression.',
 * );
 */
private parseProduct(): LayoutExpr {
        let expr = this.parseApplication();
        while (true) {
            this.skipWhitespace();
            if (!this.consume('*')) return expr;
            expr = {
                kind: 'product',
                left: expr,
                right: this.parseApplication(),
            };
        }
    }

        /**
 * Parses function-style layout application syntax such as `outer(inner)` and folds repeated applications into nested `apply` expressions.
 *
 * @returns A LayoutExpr tree whose root is either the unary expression being applied or an `apply` node that pairs the outer layout expression with the inner product expression supplied in parentheses.
 * @throws Error when an application opens with `(` but the inner expression is missing or the closing `)` is absent.
 * @example
 * const parser = new OperationParser('swizzle(row * col)');
 * const expr = (parser as unknown as { parseApplication(): LayoutExpr }).parseApplication();
 * expect(expr).toEqual({
 *   kind: 'apply',
 *   outer: { kind: 'name', name: 'swizzle' },
 *   inner: {
 *     kind: 'product',
 *     left: { kind: 'name', name: 'row' },
 *     right: { kind: 'name', name: 'col' },
 *   },
 * });
 *
 * @example
 * const parser = new OperationParser('swizzle(row');
 * expect(() => (parser as unknown as { parseApplication(): LayoutExpr }).parseApplication()).toThrow();
 */
private parseApplication(): LayoutExpr {
        let expr = this.parseUnary();
        while (true) {
            this.skipWhitespace();
            if (!this.consume('(')) return expr;
            const inner = this.parseProduct();
            this.skipWhitespace();
            this.expect(')');
            expr = {
                kind: 'apply',
                outer: expr,
                inner,
            };
        }
    }

        /**
 * Parses the unary inverse form `inv(...)`; when no inverse prefix is present, delegates to primary layout-name or parenthesized expression parsing.
 *
 * @returns An `inverse` LayoutExpr wrapping the product expression inside `inv(...)`, or the primary expression found at the current parser position.
 * @throws Error when `inv` is not followed by a parenthesized expression, the inverse body is empty, the closing `)` is missing, or no primary expression is available.
 * @example
 * const parser = new OperationParser('inv(row * col)');
 * const expr = (parser as unknown as { parseUnary(): LayoutExpr }).parseUnary();
 * expect(expr).toEqual({
 *   kind: 'inverse',
 *   expr: {
 *     kind: 'product',
 *     left: { kind: 'name', name: 'row' },
 *     right: { kind: 'name', name: 'col' },
 *   },
 * });
 *
 * @example
 * const parser = new OperationParser('inv(row');
 * expect(() => (parser as unknown as { parseUnary(): LayoutExpr }).parseUnary()).toThrow();
 */
private parseUnary(): LayoutExpr {
        this.skipWhitespace();
        if (this.peekWord('inv')) {
            this.index += 3;
            this.skipWhitespace();
            this.expect('(');
            const expr = this.parseProduct();
            this.skipWhitespace();
            this.expect(')');
            return { kind: 'inverse', expr };
        }
        return this.parsePrimary();
    }

        /**
 * Parses a primary compose-layout term: either a layout name token or a parenthesized product expression used to group composition precedence.
 *
 * @returns A `name` LayoutExpr for the layout identifier at the current position, or the grouped expression contained inside matching parentheses.
 * @throws Error when the current position does not contain a layout name, `inv(...)`, or `(` expression `)` primary, or when a parenthesized expression has no matching closing `)`.
 * @example
 * const parser = new OperationParser('(row * col)');
 * const expr = (parser as unknown as { parsePrimary(): LayoutExpr }).parsePrimary();
 * expect(expr).toEqual({
 *   kind: 'product',
 *   left: { kind: 'name', name: 'row' },
 *   right: { kind: 'name', name: 'col' },
 * });
 *
 * @example
 * const parser = new OperationParser('* row');
 * expect(() => (parser as unknown as { parsePrimary(): LayoutExpr }).parsePrimary()).toThrow(
 *   'Expected a layout name, "inv(...)", or parenthesized expression.',
 * );
 */
private parsePrimary(): LayoutExpr {
        this.skipWhitespace();
        if (this.consume('(')) {
            const expr = this.parseProduct();
            this.skipWhitespace();
            this.expect(')');
            return expr;
        }
        const name = this.parseName();
        if (!name) throw new Error('Expected a layout name, "inv(...)", or parenthesized expression.');
        return { kind: 'name', name };
    }

        /**
 * Reads the next layout identifier from the operation string after ignoring leading whitespace.
 *
 * A valid identifier starts with an ASCII letter or underscore and may continue with ASCII letters,
 * digits, or underscores. The parser index advances past the matched identifier only when a name is
 * found.
 *
 * @returns The matched layout name token, or null when the next non-whitespace character cannot start an identifier.
 * @noThrows Invalid name starts are reported as null so higher-level productions can decide whether to throw a contextual parse error.
 * @example
 * const parser = new OperationParser('  swizzle_2 * transpose');
 * expect(parser['parseName']()).toBe('swizzle_2');
 * expect(parser['index']).toBe(11);
 *
 * @example
 * const parser = new OperationParser('  * transpose');
 * expect(parser['parseName']()).toBeNull();
 * // The helper only skipped whitespace; it did not consume the operator.
 * expect(parser['index']).toBe(2);
 */
private parseName(): string | null {
        this.skipWhitespace();
        const match = this.source.slice(this.index).match(/^[A-Za-z_][A-Za-z0-9_]*/);
        if (!match) return null;
        this.index += match[0].length;
        return match[0];
    }

        /**
 * Tests whether a reserved parser word begins at the current source index without consuming it.
 *
 * The match must end before another identifier character, so `inv` is recognized in `inv(row)` but
 * not in a layout name such as `inverse_layout`.
 *
 * @param word - Reserved layout-operation word to check at the current parser index, such as `inv`.
 * @returns True when the exact word is present and followed by a non-identifier character; otherwise false.
 * @noThrows Failed keyword matches are normal parser lookahead results and are returned as false without mutating parser state.
 * @example
 * const parser = new OperationParser('inv(row_major)');
 * expect(parser['peekWord']('inv')).toBe(true);
 * expect(parser['index']).toBe(0);
 *
 * @example
 * const parser = new OperationParser('inverse_layout(row_major)');
 * expect(parser['peekWord']('inv')).toBe(false);
 */
private peekWord(word: string): boolean {
        return this.source.slice(this.index, this.index + word.length) === word
            && !/[A-Za-z0-9_]/.test(this.source[this.index + word.length] ?? '');
    }

        /**
 * Advances the parser index over insignificant whitespace before reading the next layout-operation token.
 *
 * Spaces, tabs, and line breaks are skipped until the index reaches a non-whitespace character or the
 * end of the source string.
 *
 * @returns Nothing; the parser index is updated in place to the next non-whitespace position.
 * @noThrows Reaching the end of the source is a valid stopping condition, so whitespace scanning has no parse-error branch.
 * @example
 * const parser = new OperationParser(' \n\trow_major');
 * parser['skipWhitespace']();
 * expect(parser['index']).toBe(3);
 * expect(parser['source'][parser['index']]).toBe('r');
 */
private skipWhitespace(): void {
        while (/\s/.test(this.source[this.index] ?? '')) this.index += 1;
    }

        /**
 * Consumes one expected punctuation character from the current layout-operation position.
 *
 * Parser productions use this for syntax markers such as `*`, `(`, and `)`. On success the index
 * advances by one character; on mismatch the source is left unchanged.
 *
 * @param char - Literal character expected at the current parser index.
 * @returns True when the current source character matched and was consumed; false when it did not match.
 * @noThrows Mismatches are ordinary parser control flow for optional syntax, so this helper reports them as false instead of throwing.
 * @example
 * const parser = new OperationParser('* rhs');
 * expect(parser['consume']('*')).toBe(true);
 * expect(parser['index']).toBe(1);
 *
 * @example
 * const parser = new OperationParser('rhs');
 * expect(parser['consume']('*')).toBe(false);
 * expect(parser['index']).toBe(0);
 */
private consume(char: string): boolean {
        if (this.source[this.index] !== char) return false;
        this.index += 1;
        return true;
    }

        /**
 * Consumes one required delimiter or operator character from the layout-operation expression.
 *
 * @param char - Exact single-character token that must appear at the parser's current position, such as `(`, `)`, `*`, or `,`.
 * @returns Nothing; on success the parser cursor advances past `char`.
 * @throws Error when the next unconsumed character is not `char`, with a message identifying the missing token in the layout operation.
 * @example
 * ```ts
 * // While parsing `outer(inner)`, the parser requires the opening call delimiter.
 * this.expect('(');
 *
 * // If the cursor is instead at `inner`, the parser reports the missing delimiter:
 * expect(() => this.expect('(')).toThrow('Expected "(" in Layout Operation.');
 * ```
 */
private expect(char: string): void {
        if (!this.consume(char)) {
            throw new Error(`Expected ${JSON.stringify(char)} in Layout Operation.`);
        }
    }
}

/**
 * Converts a named linear-layout preset into the evaluated form used by composition, previews, generated code, and tab metadata.
 *
 * @param spec - Normalized named layout specification containing the preset name, input labels, output labels, and per-output basis vectors.
 * @returns An evaluated `name` layout whose labels are copied from `spec`, whose bit counts are derived from `spec.bases`, and whose matrix encodes the basis mapping.
 * @noThrows The function only derives arrays and a matrix from an already-normalized `NamedLayoutSpec`; it performs no label compatibility checks itself.
 * @example
 * ```ts
 * const layout = namedLayout({
 *     name: 'identity',
 *     inputs: ['row'],
 *     outputs: ['row'],
 *     bases: [[[1]]],
 * });
 *
 * expect(layout.kind).toBe('name');
 * expect(layout.exprText).toBe('identity');
 * expect(layout.inputs).toEqual(['row']);
 * expect(layout.outputs).toEqual(['row']);
 * expect(layout.inputBitCounts).toEqual([1]);
 * expect(layout.outputBitCounts).toEqual([1]);
 * ```
 */
function namedLayout(spec: NamedLayoutSpec): EvaluatedLayout {
    const inputBitCounts = spec.bases.map((bases) => bases.length);
    const outputBitCounts = outputBitCountsFromBases(spec.bases, spec.outputs.length);
    return {
        kind: 'name',
        exprText: spec.name,
        codeRef: spec.name,
        inputs: spec.inputs.slice(),
        outputs: spec.outputs.slice(),
        inputBitCounts,
        outputBitCounts,
        matrix: matrixFromBases(spec.bases, inputBitCounts, outputBitCounts),
        spec,
    };
}

/**
 * Applies an outer evaluated layout to an inner evaluated layout by multiplying their matrices across the shared output/input axes.
 *
 * @param inner - Layout evaluated first; its `outputs` labels must exactly match `outer.inputs` in order.
 * @param outer - Layout evaluated second; its input labels define the bridge axes consumed from `inner.outputs`.
 * @param exprText - Source compose expression to store on the returned layout for UI labels, generated code references, and diagnostics.
 * @returns An `apply` layout that maps `inner.inputs` directly to `outer.outputs`, preserving the outer output bit counts and the inner input bit counts.
 * @throws Error when `inner.outputs` and `outer.inputs` are not the same ordered label list.
 * @example
 * ```ts
 * const composed = composeLayouts(innerLayout, outerLayout, 'outer(inner)');
 *
 * expect(composed.kind).toBe('apply');
 * expect(composed.exprText).toBe('outer(inner)');
 * expect(composed.inputs).toEqual(innerLayout.inputs);
 * expect(composed.outputs).toEqual(outerLayout.outputs);
 *
 * expect(() => composeLayouts(
 *     { ...innerLayout, outputs: ['lane'] },
 *     { ...outerLayout, exprText: 'outer', inputs: ['row'] },
 *     'outer(inner)',
 * )).toThrow('outer expects [row] but received [lane].');
 * ```
 */
function composeLayouts(inner: EvaluatedLayout, outer: EvaluatedLayout, exprText: string): EvaluatedLayout {
    if (!sameLabels(inner.outputs, outer.inputs)) {
        throw new Error(`${outer.exprText} expects [${outer.inputs.join(',')}] but received [${inner.outputs.join(',')}].`);
    }
    // axes may use fewer bits before or after composition.  Pad both matrices
    // to the bridge width so gf(2) multiplication lines up by axis and bit.
    const bridgeBitCounts = inner.outputBitCounts.map((bits, axis) => Math.max(bits, outer.inputBitCounts[axis] ?? 0));
    const matrix = multiplyMatrices(
        expandInputColumns(outer.matrix, outer.inputBitCounts, bridgeBitCounts),
        expandOutputRows(inner.matrix, inner.outputBitCounts, bridgeBitCounts),
    );
    const layout: EvaluatedLayout = {
        kind: 'apply',
        exprText,
        codeRef: exprText,
        inputs: inner.inputs.slice(),
        outputs: outer.outputs.slice(),
        inputBitCounts: inner.inputBitCounts.slice(),
        outputBitCounts: outer.outputBitCounts.slice(),
        matrix,
    };
    return layout;
}

/**
 * Builds the product of two independent layouts by embedding both matrices into a shared axis list and combining disjoint bit lanes.
 *
 * @param left - First evaluated layout; its input and output axes appear first and provide the initial bit lanes in the product.
 * @param right - Second evaluated layout; axes not already present are appended, while shared labels receive bit offsets after the left layout's lanes.
 * @param exprText - Source product expression to attach to the returned layout for UI labels, generated code references, and diagnostics.
 * @returns A `product` layout with merged input/output labels, summed bit counts for shared axes, and a matrix containing both layouts in non-overlapping lanes.
 * @noThrows The function does not reject overlapping labels; it handles them by calculating right-hand bit offsets and embedding both matrices into the merged axis space.
 * @example
 * ```ts
 * const product = productLayout(leftLayout, rightLayout, 'left * right');
 *
 * expect(product.kind).toBe('product');
 * expect(product.exprText).toBe('left * right');
 * expect(product.inputs).toEqual([
 *     ...leftLayout.inputs,
 *     ...rightLayout.inputs.filter((label) => !leftLayout.inputs.includes(label)),
 * ]);
 * expect(product.outputs).toEqual([
 *     ...leftLayout.outputs,
 *     ...rightLayout.outputs.filter((label) => !leftLayout.outputs.includes(label)),
 * ]);
 * ```
 */
function productLayout(left: EvaluatedLayout, right: EvaluatedLayout, exprText: string): EvaluatedLayout {
    const inputs = [...left.inputs, ...right.inputs.filter((label) => !left.inputs.includes(label))];
    const inputBitCounts = inputs.map((label) => (
        (left.inputBitCounts[left.inputs.indexOf(label)] ?? 0)
        + (right.inputBitCounts[right.inputs.indexOf(label)] ?? 0)
    ));
    const outputs = [...left.outputs, ...right.outputs.filter((label) => !left.outputs.includes(label))];
    const outputBitCounts = outputs.map((label) => (
        (left.outputBitCounts[left.outputs.indexOf(label)] ?? 0)
        + (right.outputBitCounts[right.outputs.indexOf(label)] ?? 0)
    ));
    const rightInputBitOffsets = inputs.map((label) => left.inputBitCounts[left.inputs.indexOf(label)] ?? 0);
    const rightOutputBitOffsets = outputs.map((label) => left.outputBitCounts[left.outputs.indexOf(label)] ?? 0);
    // product combines independent layouts by embedding each matrix into a
    // shared axis list, then OR-ing their disjoint bit lanes together.
    const matrix = mergeProductMatrices(
        embedProductMatrix(
            left,
            inputs,
            inputBitCounts,
            new Array(inputs.length).fill(0),
            outputs,
            outputBitCounts,
            new Array(outputs.length).fill(0),
        ),
        embedProductMatrix(
            right,
            inputs,
            inputBitCounts,
            rightInputBitOffsets,
            outputs,
            outputBitCounts,
            rightOutputBitOffsets,
        ),
    );
    const layout: EvaluatedLayout = {
        kind: 'product',
        exprText,
        codeRef: exprText,
        inputs,
        outputs,
        inputBitCounts,
        outputBitCounts,
        matrix,
    };
    return layout;
}

/**
 * Builds the evaluated layout for a compose-layout `inv(...)` expression by inverting a bijective layout's matrix and swapping its input/output axes.
 *
 * @param layout - Evaluated layout being inverted; it must be bijective and have a square matrix that can be inverted.
 * @param exprText - Stable expression text for the inverse, such as `inv(Layout1)`, stored on the returned layout and used in matrix labels/errors.
 * @returns An inverse `EvaluatedLayout` whose inputs are the original outputs, whose outputs are the original inputs, and whose matrix is the numeric inverse used by later composition and matrix previews.
 * @throws Error when `layout` is not bijective, or when its matrix cannot be inverted.
 * @example
 * const layout = {
 *   kind: 'name',
 *   exprText: 'Swap',
 *   codeRef: 'Swap',
 *   inputs: ['i'],
 *   outputs: ['o'],
 *   inputBitCounts: [1],
 *   outputBitCounts: [1],
 *   matrix: [[0, 1], [1, 0]],
 * } satisfies EvaluatedLayout;
 *
 * expect(invertLayout(layout, 'inv(Swap)')).toMatchObject({
 *   kind: 'inverse',
 *   exprText: 'inv(Swap)',
 *   inputs: ['o'],
 *   outputs: ['i'],
 *   matrix: [[0, 1], [1, 0]],
 * });
 *
 * expect(() => invertLayout({ ...layout, outputs: ['o0', 'o1'] }, 'inv(Wide)'))
 *   .toThrow('Swap is not bijective, so inv(Swap) is invalid.');
 */
function invertLayout(layout: EvaluatedLayout, exprText: string): EvaluatedLayout {
    if (!isBijective(layout)) {
        throw new Error(`${layout.exprText} is not bijective, so inv(${layout.exprText}) is invalid.`);
    }
    const inverse = invertSquareMatrix(layout.matrix);
    if (!inverse) {
        throw new Error(`Unable to invert ${layout.exprText}.`);
    }
    return {
        kind: 'inverse',
        exprText,
        codeRef: exprText,
        inputs: layout.outputs.slice(),
        outputs: layout.inputs.slice(),
        inputBitCounts: layout.outputBitCounts.slice(),
        outputBitCounts: layout.inputBitCounts.slice(),
        matrix: inverse,
    };
}

/**
 * Expands a layout operation into the ordered intermediate expressions that should be rendered as user-visible tensors and matrix blocks.
 *
 * @param expr - Parsed compose-layout expression tree; nested `apply` nodes create intermediate render steps, while products and inverses remain part of the expression evaluated at that step.
 * @param evaluate - Callback that converts each selected expression node into its `EvaluatedLayout` for the runtime metadata.
 * @returns Expression text and evaluated layout pairs in display order, ending with the full operation result.
 * @noThrows The traversal has no validation or throw branch of its own for a valid `LayoutExpr`; exceptions from the supplied `evaluate` callback are intentionally propagated.
 * @example
 * const inner = { kind: 'name', name: 'Layout1' } satisfies LayoutExpr;
 * const expr = { kind: 'apply', outer: { kind: 'name', name: 'Layout2' }, inner } satisfies LayoutExpr;
 * const evaluate = (node: LayoutExpr) => ({ exprText: exprToString(node) }) as EvaluatedLayout;
 *
 * expect(renderSteps(expr, evaluate).map((step) => step.exprText)).toEqual([
 *   'Layout1',
 *   'Layout2(Layout1)',
 * ]);
 */
function renderSteps(
    expr: LayoutExpr,
    evaluate: (expr: LayoutExpr) => EvaluatedLayout,
): Array<{ exprText: string; layout: EvaluatedLayout }> {
    // only application creates user-visible intermediate tensors; product and
    // inverse remain part of the expression that feeds each step.
    if (expr.kind !== 'apply') {
        return [{ exprText: exprToString(expr), layout: evaluate(expr) }];
    }
    return [
        ...renderSteps(expr.inner, evaluate),
        { exprText: exprToString(expr), layout: evaluate(expr) },
    ];
}

/**
 * Formats a parsed compose-layout expression into the stable text used for tensor titles, matrix labels, generated temporaries, and error messages.
 *
 * @param expr - Layout expression tree containing names, inverse nodes, products, or function-application composition nodes.
 * @param parentPrecedence - Precedence of the surrounding expression; callers normally omit it, and recursive calls use it to add parentheses only when needed.
 * @returns Compose-layout expression text with `inv(...)`, `*`, and application syntax preserved without unnecessary parentheses.
 * @noThrows For a valid `LayoutExpr` discriminated union, formatting only reads fields and concatenates strings; it performs no parsing, matrix work, or validation.
 * @example
 * const expr = {
 *   kind: 'product',
 *   left: { kind: 'name', name: 'A' },
 *   right: {
 *     kind: 'apply',
 *     outer: { kind: 'name', name: 'B' },
 *     inner: { kind: 'inverse', expr: { kind: 'name', name: 'C' } },
 *   },
 * } satisfies LayoutExpr;
 *
 * expect(exprToString(expr)).toBe('A * B(inv(C))');
 * expect(exprToString(expr, 2)).toBe('(A * B(inv(C)))');
 */
function exprToString(expr: LayoutExpr, parentPrecedence = 0): string {
    // stable expression text is used as tensor titles and matrix labels; avoid
    // extra parentheses except where precedence would change the operation.
    const precedence = expr.kind === 'product' ? 1 : expr.kind === 'apply' ? 2 : 3;
    let text = '';
    switch (expr.kind) {
        case 'name':
            text = expr.name;
            break;
        case 'inverse':
            text = `inv(${exprToString(expr.expr, 0)})`;
            break;
        case 'product':
            text = `${exprToString(expr.left, precedence)} * ${exprToString(expr.right, precedence + 1)}`;
            break;
        case 'apply':
            text = `${exprToString(expr.outer, precedence)}(${exprToString(expr.inner, 0)})`;
            break;
    }
    return precedence < parentPrecedence ? `(${text})` : text;
}

/**
 * Converts an evaluated layout into the matrix-preview block consumed by the linear-layout UI and session metadata.
 *
 * @param title - Preset name or expression text to show above this matrix in the preview, such as `Layout1` or `Layout2(Layout1)`.
 * @param layout - Evaluated layout whose output labels become rows, input labels become columns, and matrix entries become preview values.
 * @returns A `MatrixBlock` with derived bit labels and a defensive copy of the matrix values so preview rendering cannot mutate the evaluated layout.
 * @noThrows For a valid `EvaluatedLayout`, the conversion only derives labels and copies arrays; it does not validate dimensions or perform matrix algebra.
 * @example
 * const layout = {
 *   kind: 'name',
 *   exprText: 'Layout1',
 *   codeRef: 'Layout1',
 *   inputs: ['src'],
 *   outputs: ['dst'],
 *   inputBitCounts: [1],
 *   outputBitCounts: [1],
 *   matrix: [[1, 0], [0, 1]],
 * } satisfies EvaluatedLayout;
 *
 * const block = matrixBlock('Layout1', layout);
 * expect(block).toEqual({
 *   title: 'Layout1',
 *   rows: ['dst[0]'],
 *   columns: ['src[0]'],
 *   values: [[1, 0], [0, 1]],
 * });
 * expect(block.values).not.toBe(layout.matrix);
 */
function matrixBlock(title: string, layout: EvaluatedLayout): MatrixBlock {
    return {
        title,
        rows: bitLabels(layout.outputs, layout.outputBitCounts),
        columns: bitLabels(layout.inputs, layout.inputBitCounts),
        values: layout.matrix.map((row) => row.slice()),
    };
}

/**
 * Expands each layout axis label into one row or column label per bit for the matrix preview.
 *
 * @param labels - Axis names from a layout's input or output labels, in the same order as the bit counts.
 * @param bitCounts - Number of bits to display for each axis; a missing count produces no entries for that axis.
 * @returns Matrix-axis descriptors whose labels use the `axisLabel:bitIndex` form and whose `axis` field points back to the source axis.
 * @noThrows Builds labels with array iteration and treats missing bit counts as zero, so malformed length mismatches do not create an explicit throw path.
 * @example
 * bitLabels(['m', 'n'], [2, 1]);
 * // returns [
 * //   { label: 'm:0', axis: 0 },
 * //   { label: 'm:1', axis: 0 },
 * //   { label: 'n:0', axis: 1 },
 * // ]
 */
function bitLabels(labels: string[], bitCounts: number[]): MatrixAxis[] {
    return labels.flatMap((label, axis) => (
        Array.from({ length: bitCounts[axis] ?? 0 }, (_entry, bit) => ({ label: `${label}:${bit}`, axis }))
    ));
}

/**
 * Emits the Python assignment that recreates a named linear layout with `LinearLayout.from_bases`.
 *
 * @param spec - Normalized named-layout specification containing the Python variable name, input axis labels, per-input basis vectors, and output labels.
 * @returns Source lines that the sidebar can concatenate into the generated Python snippet copied into notebooks.
 * @noThrows Formats already-normalized spec fields with `JSON.stringify` and array mapping; it performs no validation that would intentionally reject a spec.
 * @example
 * pythonNamedLayout({
 *   name: 'swizzle',
 *   inputs: ['m'],
 *   bases: [[[1, 0], [0, 1]]],
 *   outputs: ['row', 'col'],
 * });
 * // returns [
 * //   'swizzle = LinearLayout.from_bases(',
 * //   '    [',
 * //   '        ("m", [[1,0],[0,1]]),',
 * //   '    ],',
 * //   '    ["row","col"],',
 * //   ')',
 * // ]
 */
function pythonNamedLayout(spec: NamedLayoutSpec): string[] {
    // emit verbose multiline code because contributors copy it into python
    // notebooks while checking new presets.
    if (spec.inputs.length === 0) {
        return [
            `${spec.name} = LinearLayout.from_bases(`,
            '    [],',
            `    ${JSON.stringify(spec.outputs)},`,
            ')',
        ];
    }
    return [
        `${spec.name} = LinearLayout.from_bases(`,
        '    [',
        ...spec.inputs.map((label, axis) => `        (${JSON.stringify(label)}, ${JSON.stringify(spec.bases[axis] ?? [])}),`),
        '    ],',
        `    ${JSON.stringify(spec.outputs)},`,
        ')',
    ];
}

/**
 * Expands integer basis-vector coordinates into the GF(2) matrix used for previews and layout composition.
 *
 * @param bases - Per-input-axis basis vectors; `bases[inputAxis][inputBit][outputAxis]` stores the output coordinate bits set by that input bit.
 * @param inputBitCounts - Bit width of each input axis, used to compute the matrix column offsets.
 * @param outputBitCounts - Bit width of each output axis, used to compute the matrix row offsets.
 * @returns A row-major matrix with one row per output bit and one column per input bit, where `1` means that input bit contributes to that output bit.
 * @noThrows Allocates a zero matrix and ignores missing output bit counts as zero; the function does not deliberately validate or throw for ragged basis data.
 * @example
 * matrixFromBases(
 *   [
 *     [[1], [2]],
 *   ],
 *   [2],
 *   [2],
 * );
 * // returns [
 * //   [1, 0],
 * //   [0, 1],
 * // ]
 */
function matrixFromBases(
    bases: number[][][],
    inputBitCounts: number[],
    outputBitCounts: number[],
): number[][] {
    const rowCount = sum(outputBitCounts);
    const columnCount = sum(inputBitCounts);
    const matrix = Array.from({ length: rowCount }, () => new Array(columnCount).fill(0));
    const inputOffsets = offsets(inputBitCounts);
    const outputOffsets = offsets(outputBitCounts);
    // basis vectors encode output coordinates as integers.  Expanding them to a
    // gf(2) matrix makes composition/product/inversion all use one algebra.
    bases.forEach((axisBases, inputAxis) => {
        axisBases.forEach((basis, bit) => {
            const column = inputOffsets[inputAxis]! + bit;
            basis.forEach((component, outputAxis) => {
                const outputBits = outputBitCounts[outputAxis] ?? 0;
                for (let outputBit = 0; outputBit < outputBits; outputBit += 1) {
                    if (((component >> outputBit) & 1) === 0) continue;
                    matrix[outputOffsets[outputAxis]! + outputBit]![column] = 1;
                }
            });
        });
    });
    return matrix;
}

/**
 * Maps one input tensor coordinate through a layout matrix and packs the resulting bits into output-axis coordinates.
 *
 * @param inputCoord - Coordinate values for the root/input tensor axes, ordered to match `inputBitCounts`.
 * @param inputBitCounts - Bit width for each input coordinate axis, used to unpack `inputCoord` into the matrix column vector.
 * @param matrix - GF(2) layout matrix whose columns correspond to input bits and rows correspond to output bits.
 * @param outputBitCounts - Bit width for each output axis, used to pack multiplied output bits back into coordinate values.
 * @returns Output coordinate values in layout-output axis order; callers use them for tensor metadata, hover, selection, and multi-input alignment.
 * @noThrows Delegates to bit packing and GF(2) multiplication helpers without adding validation or explicit error branches; well-formed dimensions map deterministically.
 * @example
 * mapCoord(
 *   [3],
 *   [2],
 *   [
 *     [1, 0],
 *     [0, 1],
 *   ],
 *   [2],
 * );
 * // returns [3]
 */
function mapCoord(
    inputCoord: number[],
    inputBitCounts: number[],
    matrix: number[][],
    outputBitCounts: number[],
): number[] {
    // one coordinate maps by expanding to input bits, multiplying the gf(2)
    // matrix, then packing output bits back into axis coordinates.
    const inputBits = bitsFromCoord(inputCoord, inputBitCounts);
    const outputBits = multiplyMatrixVector(matrix, inputBits);
    return coordFromBits(outputBits, outputBitCounts);
}

/**
 * Expands an axis coordinate into the flattened little-endian bit vector used by linear-layout matrices.
 *
 * @param coord - Per-axis coordinate values, such as `[x, y]`; a missing axis is treated as coordinate `0`.
 * @param bitCounts - Number of low-order bits to emit for each axis, in the same axis order as `coord`.
 * @returns Flattened bits grouped by axis, with each axis emitted least-significant bit first for matrix multiplication.
 * @noThrows Missing coordinate entries are zero-filled and the function only reads arrays and builds the derived bit list.
 * @example
 * bitsFromCoord([5, 2], [3, 2]);
 * // Returns [1, 0, 1, 0, 1]: x=5 -> 101b, y=2 -> 10b, both least-significant bit first.
 */
function bitsFromCoord(coord: number[], bitCounts: number[]): number[] {
    return bitCounts.flatMap((count, axis) => (
        Array.from({ length: count }, (_entry, bit) => ((coord[axis] ?? 0) >> bit) & 1)
    ));
}

/**
 * Packs the flattened little-endian layout bit vector back into per-axis coordinate values.
 *
 * @param bits - Flattened matrix output bits grouped by axis, with each axis stored least-significant bit first.
 * @param bitCounts - Number of bits to consume for each output axis, in output coordinate order.
 * @returns Coordinate array whose entries are reconstructed from the corresponding axis bit groups.
 * @noThrows Missing bit entries are treated as `0`, so short bit vectors decode to coordinates with zero-filled high bits.
 * @example
 * coordFromBits([1, 0, 1, 0, 1], [3, 2]);
 * // Returns [5, 2].
 */
function coordFromBits(bits: number[], bitCounts: number[]): number[] {
    const result = new Array(bitCounts.length).fill(0);
    const axisOffsets = offsets(bitCounts);
    bitCounts.forEach((count, axis) => {
        let value = 0;
        for (let bit = 0; bit < count; bit += 1) {
            value |= (bits[axisOffsets[axis]! + bit] ?? 0) << bit;
        }
        result[axis] = value;
    });
    return result;
}

/**
 * Applies a linear-layout matrix to an input bit vector using GF(2) arithmetic.
 *
 * @param matrix - Matrix rows that define each output bit; row cells are combined with vector bits using AND then XOR.
 * @param vector - Input bit vector aligned with the matrix columns; missing vector columns are treated as `0`.
 * @returns Output bit vector containing one GF(2) dot-product result for each matrix row.
 * @noThrows Short vectors are zero-filled during column reads, and the function performs only array iteration and bitwise operations.
 * @example
 * multiplyMatrixVector([
 *   [1, 0, 1],
 *   [0, 1, 1],
 * ], [1, 1, 0]);
 * // Returns [1, 1].
 */
function multiplyMatrixVector(matrix: number[][], vector: number[]): number[] {
    // linear layouts operate over bits, so addition is xor and multiplication is and.
    return matrix.map((row) => row.reduce((value, cell, column) => value ^ (cell & (vector[column] ?? 0)), 0));
}

/**
 * Composes two linear-layout matrices using GF(2) multiplication.
 *
 * @param left - Left matrix whose column count must equal the number of rows in `right`.
 * @param right - Right matrix whose rows provide the shared dimension and whose columns become the result columns.
 * @returns Composed matrix with `left.length` rows and `right[0]?.length ?? 0` columns, where each cell is the XOR of ANDed shared-dimension bits.
 * @throws Error when the left matrix column count differs from the right matrix row count.
 * @example
 * multiplyMatrices([
 *   [1, 1],
 *   [0, 1],
 * ], [
 *   [1, 0],
 *   [1, 1],
 * ]);
 * // Returns [
 * //   [0, 1],
 * //   [1, 1],
 * // ].
 *
 * @example
 * multiplyMatrices([[1, 0, 1]], [[1, 0], [0, 1]]);
 * // Throws Error: "Incompatible layout matrices: 1x3 cannot multiply 2x2."
 */
function multiplyMatrices(left: number[][], right: number[][]): number[][] {
    const rows = left.length;
    const shared = left[0]?.length ?? 0;
    const columns = right[0]?.length ?? 0;
    if (shared !== right.length) {
        throw new Error(`Incompatible layout matrices: ${rows}x${shared} cannot multiply ${right.length}x${columns}.`);
    }
    return Array.from({ length: rows }, (_entry, row) => (
        Array.from({ length: columns }, (_cell, column) => {
            let value = 0;
            for (let index = 0; index < shared; index += 1) {
                value ^= (left[row]![index] ?? 0) & (right[index]![column] ?? 0);
            }
            return value;
        })
    ));
}

/**
 * Combines two product-layout matrices that have already been embedded into the same output and input axes.
 *
 * Product composition assigns the left and right layouts to disjoint bit lanes, so each cell is merged with
 * bitwise OR rather than GF(2) xor.
 *
 * @param left - Embedded matrix for the left product operand, indexed as output bit rows by input bit columns.
 * @param right - Embedded matrix for the right product operand using the same row and column positions as `left`.
 * @returns A matrix with `left`'s shape whose entries contain every lane set by either operand; missing `right` cells are treated as zero.
 * @noThrows The merge performs only array mapping and numeric bitwise OR; for well-formed embedded matrices there is no validation branch or explicit error path.
 * @example
 * const left = [
 *   [1, 0, 0],
 *   [0, 0, 0],
 * ];
 * const right = [
 *   [0, 2, 0],
 *   [0, 0, 4],
 * ];
 *
 * mergeProductMatrices(left, right);
 * // => [
 * //   [1, 2, 0],
 * //   [0, 0, 4],
 * // ]
 */
function mergeProductMatrices(left: number[][], right: number[][]): number[][] {
    // product matrices occupy disjoint bit lanes after embedding, so OR is the
    // merge operation rather than xor.
    return left.map((row, rowIndex) => row.map((value, columnIndex) => value | (right[rowIndex]?.[columnIndex] ?? 0)));
}

/**
 * Places one evaluated layout matrix into the shared axis matrix used for product composition.
 *
 * Rows and columns whose labels appear in the target axis lists are copied into the target bit ranges; labels that
 * are not present in the target lists are ignored.
 *
 * @param layout - Evaluated operand layout containing `inputs`, `outputs`, per-axis bit counts, and the source matrix to copy.
 * @param targetInputs - Product input axis labels, in the column-axis order of the returned matrix.
 * @param targetInputBitCounts - Bit width for each `targetInputs` axis; the sum determines the returned column count.
 * @param sharedInputOffset - Per-target-input offset used when this operand occupies a later bit lane within a shared input label.
 * @param targetOutputs - Product output axis labels, in the row-axis order of the returned matrix.
 * @param targetOutputBitCounts - Bit width for each `targetOutputs` axis; the sum determines the returned row count.
 * @param sharedOutputOffset - Per-target-output offset used when this operand occupies a later bit lane within a shared output label.
 * @returns A zero-filled target matrix with the operand's matching labeled row and column blocks copied into their product-axis positions.
 * @noThrows Missing labels are skipped and missing offset entries default to zero, so a well-formed evaluated layout is embedded without an explicit exception path.
 * @example
 * const layout = {
 *   inputs: ['a'],
 *   inputBitCounts: [2],
 *   outputs: ['b'],
 *   outputBitCounts: [2],
 *   matrix: [
 *     [1, 0],
 *     [0, 1],
 *   ],
 * } as EvaluatedLayout;
 *
 * embedProductMatrix(
 *   layout,
 *   ['a', 'extra'],
 *   [2, 1],
 *   [0, 0],
 *   ['b'],
 *   [2],
 *   [0],
 * );
 * // => [
 * //   [1, 0, 0],
 * //   [0, 1, 0],
 * // ]
 */
function embedProductMatrix(
    layout: EvaluatedLayout,
    targetInputs: string[],
    targetInputBitCounts: number[],
    sharedInputOffset: number[],
    targetOutputs: string[],
    targetOutputBitCounts: number[],
    sharedOutputOffset: number[],
): number[][] {
    // shared axis lists let A:[T]->[M] * B:[R]->[N] become [T,R]->[M,N]
    // without special cases for overlapping labels.
    const matrix = Array.from({ length: sum(targetOutputBitCounts) }, () => new Array(sum(targetInputBitCounts)).fill(0));
    const targetInputAxes = new Map(targetInputs.map((label, axis) => [label, axis]));
    const targetOutputAxes = new Map(targetOutputs.map((label, axis) => [label, axis]));
    const targetInputOffsets = offsets(targetInputBitCounts);
    const targetOutputOffsets = offsets(targetOutputBitCounts);
    const inputOffsets = offsets(layout.inputBitCounts);
    const outputOffsets = offsets(layout.outputBitCounts);
    layout.outputs.forEach((label, outputAxis) => {
        const targetOutputAxis = targetOutputAxes.get(label);
        if (targetOutputAxis === undefined) return;
        const outputBase = targetOutputOffsets[targetOutputAxis]! + (sharedOutputOffset[targetOutputAxis] ?? 0);
        for (let outputBit = 0; outputBit < (layout.outputBitCounts[outputAxis] ?? 0); outputBit += 1) {
            const row = matrix[outputBase + outputBit]!;
            layout.inputs.forEach((inputLabel, inputAxis) => {
                const targetInputAxis = targetInputAxes.get(inputLabel);
                if (targetInputAxis === undefined) return;
                const inputBase = targetInputOffsets[targetInputAxis]! + (sharedInputOffset[targetInputAxis] ?? 0);
                for (let inputBit = 0; inputBit < (layout.inputBitCounts[inputAxis] ?? 0); inputBit += 1) {
                    row[inputBase + inputBit] = layout.matrix[outputOffsets[outputAxis]! + outputBit]?.[inputOffsets[inputAxis]! + inputBit] ?? 0;
                }
            });
        }
    });
    return matrix;
}

/**
 * Computes the inverse of a binary layout matrix using Gaussian elimination over GF(2).
 *
 * The inverse is used for `inv(...)` layouts after the caller has checked that the layout is bijective.
 *
 * @param matrix - Candidate binary matrix whose rows are output bits and columns are input bits; entries are interpreted as GF(2) values.
 * @returns The GF(2) inverse matrix, or `null` when the input is not square or no pivot exists for one of the columns.
 * @noThrows Non-square and singular matrices are reported as `null`; the elimination code has no explicit exception branch for those cases.
 * @example
 * invertSquareMatrix([
 *   [1, 1],
 *   [0, 1],
 * ]);
 * // => [
 * //   [1, 1],
 * //   [0, 1],
 * // ]
 *
 * invertSquareMatrix([
 *   [1, 0],
 *   [1, 0],
 * ]);
 * // => null
 */
function invertSquareMatrix(matrix: number[][]): number[][] | null {
    const size = matrix.length;
    if (size !== (matrix[0]?.length ?? 0)) return null;
    const rows = matrix.map((row, index) => [
        ...row.slice(),
        ...Array.from({ length: size }, (_entry, column) => column === index ? 1 : 0),
    ]);
    // gaussian elimination over gf(2); if any pivot is missing, inverse layout
    // would map multiple inputs to the same output and must be rejected.
    let pivotRow = 0;
    for (let column = 0; column < size; column += 1) {
        const candidate = rows.findIndex((row, index) => index >= pivotRow && row[column] === 1);
        if (candidate === -1) return null;
        if (candidate !== pivotRow) {
            const swap = rows[pivotRow]!;
            rows[pivotRow] = rows[candidate]!;
            rows[candidate] = swap;
        }
        rows.forEach((row, index) => {
            if (index === pivotRow || row[column] === 0) return;
            for (let offset = column; offset < row.length; offset += 1) {
                row[offset] = row[offset]! ^ rows[pivotRow]![offset]!;
            }
        });
        pivotRow += 1;
    }
    return rows.map((row) => row.slice(size));
}

/**
 * Calculates the rank of a layout matrix over GF(2) for injectivity and bijectivity checks.
 *
 * Row elimination treats `1` entries as pivots and combines rows with xor, matching the binary arithmetic used by
 * compose-layout matrices.
 *
 * @param matrix - Binary matrix with output-bit rows and input-bit columns to reduce over GF(2).
 * @returns The number of independent pivot columns found; callers compare it with the input bit count to decide whether a layout is injective.
 * @noThrows Empty matrices return rank `0`, and the reduction has no explicit validation or error branch for ordinary numeric rows.
 * @example
 * gf2Rank([
 *   [1, 0, 1],
 *   [0, 1, 1],
 *   [1, 1, 0],
 * ]);
 * // => 2
 *
 * gf2Rank([]);
 * // => 0
 */
function gf2Rank(matrix: number[][]): number {
    if (matrix.length === 0) return 0;
    const rows = matrix.map((row) => row.slice());
    const columnCount = rows[0]?.length ?? 0;
    let rank = 0;
    // rank over gf(2) is the injectivity test for the input bit space.
    for (let column = 0; column < columnCount; column += 1) {
        const pivot = rows.findIndex((row, index) => index >= rank && row[column] === 1);
        if (pivot === -1) continue;
        if (pivot !== rank) {
            const swap = rows[rank]!;
            rows[rank] = rows[pivot]!;
            rows[pivot] = swap;
        }
        rows.forEach((row, index) => {
            if (index === rank || row[column] === 0) return;
            for (let offset = column; offset < columnCount; offset += 1) {
                row[offset] = row[offset]! ^ rows[rank]![offset]!;
            }
        });
        rank += 1;
    }
    return rank;
}

/**
 * Checks whether an evaluated linear layout can be inverted.
 *
 * A layout is bijective only when it maps the same number of input and output
 * bits and its GF(2) matrix has full rank for those bits.
 *
 * @param layout - Evaluated layout with per-axis input bit counts, per-axis output bit counts, and the GF(2) transform matrix to test.
 * @returns `true` when `inv(...)` can use the layout as a one-to-one bit mapping; otherwise `false`.
 * @noThrows Reads numeric arrays and computes their rank without performing validation or throwing its own errors.
 * @example
 * const layout = {
 *   inputBitCounts: [1, 1],
 *   outputBitCounts: [2],
 *   matrix: [
 *     [1, 0],
 *     [0, 1],
 *   ],
 * } as EvaluatedLayout;
 *
 * isBijective(layout); // true
 */
function isBijective(layout: EvaluatedLayout): boolean {
    const inputBits = sum(layout.inputBitCounts);
    const outputBits = sum(layout.outputBitCounts);
    return inputBits === outputBits && gf2Rank(layout.matrix) === inputBits;
}

/**
 * Infers the bit width needed for each named output axis in a compose-layout basis specification.
 *
 * Each output width is the position of the highest set bit that appears for that
 * output axis across all input-axis basis vectors.
 *
 * @param bases - Basis vectors grouped by input axis, then input bit; each vector stores one numeric mask per output axis.
 * @param outputCount - Number of output axes declared by the layout spec.
 * @returns One inferred bit count per output axis, suitable for `EvaluatedLayout.outputBitCounts`.
 * @noThrows Missing basis entries are treated as zero masks, so sparse or shorter vectors simply contribute no bits for that output axis.
 * @example
 * const bases = [
 *   [[1, 0], [2, 0]],
 *   [[0, 4]],
 * ];
 *
 * outputBitCountsFromBases(bases, 2); // [2, 3]
 */
function outputBitCountsFromBases(bases: number[][][], outputCount: number): number[] {
    // output bit width is inferred from the largest set bit in the basis vectors;
    // explicit output sizes are not stored in compose-layout specs.
    return Array.from({ length: outputCount }, (_entry, outputAxis) => {
        let bits = 0;
        bases.forEach((axisBases) => {
            axisBases.forEach((basis) => {
                const value = basis[outputAxis] ?? 0;
                bits = Math.max(bits, value <= 0 ? 0 : Math.floor(Math.log2(value)) + 1);
            });
        });
        return bits;
    });
}

/**
 * Removes unused trailing output bits from each output axis count.
 *
 * Rows are read in output-axis order according to `outputBitCounts`; the returned
 * count for an axis stops after the highest row in that axis block that contains
 * a nonzero matrix entry.
 *
 * @param matrix - GF(2) layout matrix whose rows are grouped by output axis and bit position.
 * @param outputBitCounts - Current row count for each output axis before trimming unused high bits.
 * @returns Trimmed per-axis output bit counts that still include every nonzero output row.
 * @noThrows Missing matrix rows are treated as empty rows, so absent rows are considered unused rather than exceptional.
 * @example
 * const matrix = [
 *   [1, 0],
 *   [0, 0],
 *   [0, 1],
 *   [0, 0],
 * ];
 *
 * trimOutputBitCounts(matrix, [2, 2]); // [1, 1]
 */
function trimOutputBitCounts(matrix: number[][], outputBitCounts: number[]): number[] {
    let rowOffset = 0;
    return outputBitCounts.map((bitCount) => {
        let used = 0;
        for (let bit = 0; bit < bitCount; bit += 1) {
            const row = matrix[rowOffset + bit] ?? [];
            if (row.some((value) => value !== 0)) used = bit + 1;
        }
        rowOffset += bitCount;
        return used;
    });
}

/**
 * Copies the matrix rows that remain live after per-axis output bit counts are trimmed.
 *
 * Rows are selected from the original matrix using offsets derived from
 * `currentBitCounts`, then emitted in the same output-axis order with only the
 * first `nextBitCounts[axis]` rows for each axis.
 *
 * @param matrix - GF(2) layout matrix whose rows are grouped by output axis under the current bit counts.
 * @param currentBitCounts - Existing number of matrix rows allocated to each output axis.
 * @param nextBitCounts - Trimmed number of rows to keep for each output axis.
 * @returns A new matrix containing copies of the retained rows, aligned to the trimmed output bit counts.
 * @noThrows Copies rows from a matrix whose row layout was produced by the linear-layout evaluator; callers provide compatible bit-count arrays.
 * @example
 * const matrix = [
 *   [1, 0],
 *   [0, 0],
 *   [0, 1],
 *   [1, 1],
 * ];
 *
 * trimMatrixRows(matrix, [2, 2], [1, 2]);
 * // [
 * //   [1, 0],
 * //   [0, 1],
 * //   [1, 1],
 * // ]
 */
function trimMatrixRows(matrix: number[][], currentBitCounts: number[], nextBitCounts: number[]): number[][] {
    const currentOffsets = offsets(currentBitCounts);
    // keep only the still-live rows for each axis so matrix row layout stays aligned
    // with nextBitCounts; otherwise later axes read stale rows from trimmed-away bits.
    return nextBitCounts.flatMap((bitCount, axis) => (
        Array.from({ length: bitCount }, (_entry, bit) => matrix[currentOffsets[axis]! + bit]!.slice())
    ));
}

/**
 * Builds prefix offsets for a sequence of per-axis bit counts or widths.
 *
 * @param values - Ordered counts whose starting positions are needed in a flattened bit or matrix row/column layout.
 * @returns One offset per input entry; each offset is the sum of all earlier entries, so callers can locate that axis in the flattened layout.
 * @noThrows The helper only iterates over the provided numeric array and accumulates totals; it performs no validation and has no explicit error branch.
 * @example
 * const inputBitCounts = [2, 0, 3];
 * offsets(inputBitCounts); // [0, 2, 2]
 */
function offsets(values: number[]): number[] {
    let total = 0;
    return values.map((value) => {
        const offset = total;
        total += value;
        return offset;
    });
}

/**
 * Converts per-axis coordinate bit counts into tensor extents for the linear-layout preview tensors.
 *
 * @param bitCounts - Bit width for each logical axis in layout order; a zero-bit axis is represented as a single coordinate.
 * @returns Tensor shape where each nonzero bit count becomes `2 ** bits` and each zero-bit axis becomes `1`.
 * @noThrows The mapping uses arithmetic on each supplied count and does not validate or throw for any normal numeric array input.
 * @example
 * const outputBitCounts = [0, 1, 3];
 * shapeFromBitCounts(outputBitCounts); // [1, 2, 8]
 */
function shapeFromBitCounts(bitCounts: number[]): number[] {
    return bitCounts.map((bits) => bits === 0 ? 1 : 2 ** bits);
}

/**
 * Totals a numeric sequence such as the bit counts that determine matrix width or rank checks.
 *
 * @param values - Numeric counts to add in order.
 * @returns The arithmetic total of all entries; an empty array returns `0`.
 * @noThrows The reducer supplies an initial total of `0`, so empty arrays do not trigger the native reduce empty-array error path.
 * @example
 * const inputBitCounts = [2, 0, 3];
 * sum(inputBitCounts); // 5
 */
function sum(values: number[]): number {
    return values.reduce((total, value) => total + value, 0);
}

/**
 * Checks whether two layout label lists describe the same axes in the same order.
 *
 * @param leftLabels - First ordered axis-label list, such as the outputs emitted by an inner composed layout.
 * @param rightLabels - Second ordered axis-label list, such as the inputs expected by the next layout.
 * @returns `true` only when both arrays have the same length and every label matches at the same index.
 * @noThrows The comparison only reads array lengths and performs indexed string equality checks; it has no explicit error branch.
 * @example
 * sameLabels(['m', 'n'], ['m', 'n']); // true
 * sameLabels(['m', 'n'], ['n', 'm']); // false
 */
function sameLabels(leftLabels: string[], rightLabels: string[]): boolean {
    return leftLabels.length === rightLabels.length
        && leftLabels.every((label, index) => label === rightLabels[index]);
}

/**
 * Rebuilds a layout matrix so its output rows match the bridge bit widths used
 * while composing two linear layouts.
 *
 * @param matrix - Source matrix whose rows are grouped by output axis bits in
 * the order described by `currentBitCounts`.
 * @param currentBitCounts - Number of output bit rows currently present for
 * each source axis.
 * @param targetBitCounts - Number of output bit rows required for each bridge
 * axis in the composed layout.
 * @returns A new matrix with copied rows for existing bits and all-zero rows
 * for target high bits that are wider than the source matrix.
 * @noThrows Missing source axes or high bits are represented as zero rows, and
 * rows are copied with `slice` instead of requiring validation or mutation.
 * @example
 * const expanded = expandOutputRows(
 *     [
 *         [1, 0],
 *         [0, 1],
 *     ],
 *     [1, 1],
 *     [2, 1],
 * );
 * // expanded === [
 * //     [1, 0],
 * //     [0, 0],
 * //     [0, 1],
 * // ]
 */
function expandOutputRows(
    matrix: number[][],
    currentBitCounts: number[],
    targetBitCounts: number[],
): number[][] {
    // composition can require bridge axes wider than either side of the source
    // matrix; missing high bits are zero rows.
    const currentOffsets = offsets(currentBitCounts);
    return targetBitCounts.flatMap((targetBits, axis) => (
        Array.from({ length: targetBits }, (_entry, bit) => {
            const currentBits = currentBitCounts[axis] ?? 0;
            if (bit >= currentBits) return new Array(matrix[0]?.length ?? 0).fill(0);
            return matrix[currentOffsets[axis]! + bit]!.slice();
        })
    ));
}

/**
 * Rebuilds a layout matrix so its input columns match the bridge bit widths
 * used while composing two linear layouts.
 *
 * @param matrix - Source matrix whose columns are grouped by input axis bits in
 * the order described by `currentBitCounts`.
 * @param currentBitCounts - Number of input bit columns currently present for
 * each source axis.
 * @param targetBitCounts - Number of input bit columns required for each bridge
 * axis in the composed layout.
 * @returns A new matrix with copied columns for existing bits and zero-valued
 * columns for target high bits that are wider than the source matrix.
 * @noThrows Missing source axes, missing columns, and high target bits are
 * converted to zero values instead of being dereferenced as required entries.
 * @example
 * const expanded = expandInputColumns(
 *     [
 *         [1, 2],
 *         [3, 4],
 *     ],
 *     [1, 1],
 *     [2, 1],
 * );
 * // expanded === [
 * //     [1, 0, 2],
 * //     [3, 0, 4],
 * // ]
 */
function expandInputColumns(
    matrix: number[][],
    currentBitCounts: number[],
    targetBitCounts: number[],
): number[][] {
    // composition can require bridge axes wider than either side of the source
    // matrix; missing high bits are zero columns.
    const currentOffsets = offsets(currentBitCounts);
    return matrix.map((row) => targetBitCounts.flatMap((targetBits, axis) => (
        Array.from({ length: targetBits }, (_entry, bit) => {
            const currentBits = currentBitCounts[axis] ?? 0;
            if (bit >= currentBits) return 0;
            return row[currentOffsets[axis]! + bit] ?? 0;
        })
    )));
}

/**
 * Evaluates the compose color mapping at one tensor coordinate and converts the
 * mapped H/S/L channel values into an RGB tuple for viewer cell coloring.
 *
 * @param coord - Root or propagated tensor coordinate being colored, with one
 * entry per axis in `shape`.
 * @param shape - Tensor axis extents used to normalize coordinate positions
 * when a channel is mapped across an axis.
 * @param labelToAxis - Map from compose axis labels to their numeric axis
 * indexes in `coord` and `shape`.
 * @param mapping - H, S, and L channel assignments, where each value names the
 * axis label that drives that color channel.
 * @param ranges - Numeric range text for each H, S, and L channel control.
 * @returns The red, green, and blue byte values used by the viewer for the
 * specified coordinate.
 * @throws Error when any mapped channel range contains non-numeric start or end
 * text, as reported by `channelValue`.
 * @example
 * const color = rgbColorForRootCoord(
 *     [1, 0],
 *     [3, 1],
 *     new Map([['x', 0], ['y', 1]]),
 *     { H: 'x', S: 'y', L: 'y' },
 *     { H: ['0', '120'], S: ['1', '1'], L: ['0.5', '0.5'] },
 * );
 * // color is the RGB tuple for hue halfway between 0 and 120 at x = 1.
 *
 * expect(() => rgbColorForRootCoord(
 *     [0],
 *     [2],
 *     new Map([['x', 0]]),
 *     { H: 'x', S: 'x', L: 'x' },
 *     { H: ['red', '120'], S: ['1', '1'], L: ['0.5', '0.5'] },
 * )).toThrow('H range must contain numbers.');
 */
function rgbColorForRootCoord(
    coord: number[],
    shape: number[],
    labelToAxis: Map<string, number>,
    mapping: Record<ComposeChannel, ComposeMappingValue>,
    ranges: Record<ComposeChannel, [string, string]>,
): [number, number, number] {
    // H/S/L controls are stored as axis labels, so the same mapping works for
    // root-space and propagated output-space coloring.
    const hue = channelValue('H', coord, shape, labelToAxis.get(mapping.H) ?? null, ranges);
    const saturation = channelValue('S', coord, shape, labelToAxis.get(mapping.S) ?? null, ranges);
    const lightness = channelValue('L', coord, shape, labelToAxis.get(mapping.L) ?? null, ranges);
    return hsvToRgb(hue, saturation, lightness);
}

/**
 * Converts one compose color channel range into the channel value for a tensor
 * coordinate, interpolating across the mapped axis when that axis has length.
 *
 * @param channel - Color channel whose range should be read from `ranges`.
 * @param coord - Tensor coordinate being colored; `coord[axis]` supplies the
 * interpolation position when `axis` is not null.
 * @param shape - Tensor axis extents; `shape[axis] - 1` is the interpolation
 * denominator for multi-position axes.
 * @param axis - Axis index mapped to the channel, or null when the channel uses
 * the range start as a constant value.
 * @param ranges - Start and end values for each channel, stored as strings from
 * the compose color controls.
 * @returns The numeric channel value: the range start for an unmapped or
 * single-position axis, otherwise the linear interpolation between start and end.
 * @throws Error when the selected channel range start or end cannot be parsed
 * as a finite number.
 * @example
 * const value = channelValue(
 *     'H',
 *     [2],
 *     [5],
 *     0,
 *     { H: ['0', '100'], S: ['1', '1'], L: ['0.5', '0.5'] },
 * );
 * // value === 50
 *
 * expect(() => channelValue(
 *     'H',
 *     [0],
 *     [2],
 *     0,
 *     { H: ['blue', '100'], S: ['1', '1'], L: ['0.5', '0.5'] },
 * )).toThrow('H range must contain numbers.');
 */
function channelValue(
    channel: ComposeChannel,
    coord: number[],
    shape: number[],
    axis: number | null,
    ranges: Record<ComposeChannel, [string, string]>,
): number {
    const [startText, endText] = ranges[channel];
    const start = Number(startText);
    const end = Number(endText);
    if (!Number.isFinite(start) || !Number.isFinite(end)) {
        throw new Error(`${channel} range must contain numbers.`);
    }
    if (axis === null) return start;
    if ((shape[axis] ?? 1) <= 1) return start;
    const position = (coord[axis] ?? 0) / ((shape[axis] ?? 1) - 1);
    return start + ((end - start) * position);
}

/**
 * Converts a normalized HSV color used by linear-layout coloring into an RGB channel tuple.
 *
 * @param hue - Hue position on the color wheel; values are normalized modulo 1, so negative values and values greater than 1 wrap around.
 * @param saturation - HSV saturation multiplier, normally in the range 0 to 1, where 0 produces gray and 1 produces the full hue.
 * @param value - HSV value/brightness channel; the returned RGB channels are scaled by this number.
 * @returns Three RGB channel values as `[red, green, blue]`, in the same numeric scale as `value`, for writing into color buffers.
 * @noThrows Uses only numeric arithmetic and a fixed switch over the computed hue sector, so valid JavaScript number inputs have no expected throw path.
 * @example
 * hsvToRgb(0, 1, 1);
 * // => [1, 0, 0]
 *
 * @example
 * hsvToRgb(1 / 3, 1, 0.5);
 * // => [0, 0.5, 0]
 */
function hsvToRgb(hue: number, saturation: number, value: number): [number, number, number] {
    const scaledHue = ((((hue % 1) + 1) % 1) * 6);
    const sector = Math.floor(scaledHue);
    const fraction = scaledHue - sector;
    const p = value * (1 - saturation);
    const q = value * (1 - (fraction * saturation));
    const t = value * (1 - ((1 - fraction) * saturation));
    switch (sector % 6) {
        case 0:
            return [value, t, p];
        case 1:
            return [q, value, p];
        case 2:
            return [p, value, t];
        case 3:
            return [p, q, value];
        case 4:
            return [t, p, value];
        default:
            return [value, p, q];
    }
}

/**
 * Converts a tensor coordinate into the row-major storage offset used by dense viewer buffers.
 *
 * @param coord - Per-axis zero-based coordinate inside the tensor, with one entry for each axis in `shape`.
 * @param shape - Tensor extents for the same axes as `coord`; each extent is used as the stride base for the following axis.
 * @returns The zero-based flat array index for the coordinate, suitable for indexing tensor data, color buffers, and root-index arrays.
 * @noThrows The helper performs a reduction over caller-provided arrays and does not validate rank or bounds; linear-layout callers provide matching in-bounds coordinates from runtime tensor metadata.
 * @example
 * flatIndex([1, 2, 3], [2, 3, 4]);
 * // => 23
 */
function flatIndex(coord: number[], shape: number[]): number {
    return coord.reduce((index, value, axis) => (index * shape[axis]!) + value, 0);
}

/**
 * Copies the viewer UI settings that should survive when a linear-layout bundle manifest is rebuilt.
 *
 * @param viewer - Optional partial viewer snapshot captured from the current tab or saved session.
 * @returns A new partial viewer snapshot containing display and panel settings, with `dimensionMappingScheme` defaulted to `'contiguous'`; tensor snapshots are omitted because tensors are regenerated from the evaluated layout runtime.
 * @noThrows Only checks for an undefined snapshot and reads optional properties into a new object, so missing viewer fields fall through as `undefined` instead of throwing.
 * @example
 * persistedViewerSettings(undefined);
 * // => { dimensionMappingScheme: 'contiguous' }
 *
 * @example
 * persistedViewerSettings({
 *   displayMode: 'heatmap',
 *   showInspectorPanel: true,
 *   dimensionMappingScheme: 'blocked',
 *   tensors: []
 * });
 * // => {
 * //   displayMode: 'heatmap',
 * //   heatmap: undefined,
 * //   dimensionBlockGapMultiple: undefined,
 * //   displayGaps: undefined,
 * //   logScale: undefined,
 * //   collapseHiddenAxes: undefined,
 * //   dimensionMappingScheme: 'blocked',
 * //   showDimensionLines: undefined,
 * //   showTensorNames: undefined,
 * //   showInspectorPanel: true,
 * //   showSelectionPanel: undefined,
 * //   showHoverDetailsPanel: undefined
 * // }
 */
function persistedViewerSettings(viewer: Partial<ViewerSnapshot> | undefined): Partial<ViewerSnapshot> {
    if (!viewer) return { dimensionMappingScheme: 'contiguous' };
    // omit tensor snapshots here because createBundleManifest rebuilds tensors
    // from the evaluated runtime; preserving stale tensor ids would corrupt tabs.
    return {
        displayMode: viewer.displayMode,
        heatmap: viewer.heatmap,
        dimensionBlockGapMultiple: viewer.dimensionBlockGapMultiple,
        displayGaps: viewer.displayGaps,
        logScale: viewer.logScale,
        collapseHiddenAxes: viewer.collapseHiddenAxes,
        dimensionMappingScheme: viewer.dimensionMappingScheme ?? 'contiguous',
        showDimensionLines: viewer.showDimensionLines,
        showTensorNames: viewer.showTensorNames,
        showInspectorPanel: viewer.showInspectorPanel,
        showSelectionPanel: viewer.showSelectionPanel,
        showHoverDetailsPanel: viewer.showHoverDetailsPanel,
    };
}

/**
 * Converts a layout title into an identifier safe for generated Python and compose-layout operation text.
 *
 * @param value - User-visible layout, operation, or preset name that may contain spaces, punctuation, or leading digits.
 * @returns The sanitized identifier with non-alphanumeric characters collapsed to underscores and edge underscores trimmed; names that do not start with a letter or underscore are prefixed with `Layout_`, and an empty result becomes `Layout_1`.
 * @noThrows Uses deterministic string replacement and prefix checks on the provided string, with no parsing or external state that can throw under normal string inputs.
 * @example
 * sanitizeIdentifier('Blocked Layout v2');
 * // => 'Blocked_Layout_v2'
 *
 * @example
 * sanitizeIdentifier('123 tiles!');
 * // => 'Layout_123_tiles'
 *
 * @example
 * sanitizeIdentifier('!!!');
 * // => 'Layout_1'
 */
function sanitizeIdentifier(value: string): string {
    // generated python and operation text both need identifiers, not arbitrary titles.
    const cleaned = value.replace(/[^A-Za-z0-9_]+/g, '_').replace(/^_+|_+$/g, '');
    return /^[A-Za-z_]/.test(cleaned) ? cleaned : `Layout_${cleaned || '1'}`;
}

/**
 * Reads a saved pre-compose linear-layout metadata object and extracts the pieces
 * needed to rebuild a current compose-layout spec.
 *
 * @param raw - Legacy tab metadata object from saved viewer state; supported
 *   shapes include `input_dims` or `bases` for input axes, `output_dims` or
 *   `out_dims` for output axes, optional `color_axes`, optional `color_ranges`,
 *   and an optional non-empty string `name`.
 * @param fallbackTitle - Layout title to use when the legacy object has no
 *   non-empty string `name`.
 * @returns Canonical legacy layout parts: the migrated layout name, input axis
 *   names, output axis names, per-input basis matrices, and color metadata that
 *   the migration step can map into current compose-layout state.
 * @throws Error when `raw` is null, undefined, or not an object, because there is
 *   no legacy layout metadata to upgrade.
 * @example
 * const spec = parseLegacySpec({
 *   name: 'mma_tile',
 *   input_dims: [{ name: 'thread', bases: [[1, 0], [0, 1]] }],
 *   output_dims: ['m', 'n'],
 *   color_axes: { H: 'thread' },
 *   color_ranges: { H: [0, 31] },
 * }, 'Layout_1');
 *
 * expect(spec).toEqual({
 *   name: 'mma_tile',
 *   inputs: ['thread'],
 *   outputs: ['m', 'n'],
 *   bases: [[[1, 0], [0, 1]]],
 *   colorAxes: { H: 'thread' },
 *   colorRanges: { H: [0, 31] },
 * });
 *
 * @example
 * expect(() => parseLegacySpec(null, 'Layout_1'))
 *   .toThrow('Unable to upgrade legacy layout state.');
 */
function parseLegacySpec(raw: unknown, fallbackTitle: string): {
    name: string;
    inputs: string[];
    outputs: string[];
    bases: number[][][];
    colorAxes: Record<string, string>;
    colorRanges: Record<string, [number, number]>;
} {
    if (!raw || typeof raw !== 'object') {
        throw new Error('Unable to upgrade legacy layout state.');
    }
    const record = raw as Record<string, unknown>;
    // legacy specs appeared in several shapes while the demo evolved, so keep
    // this migration tolerant and canonicalize only after the raw pieces are read.
    const basesEntries = Array.isArray(record.input_dims)
        ? record.input_dims
        : Array.isArray(record.bases)
            ? record.bases
            : [];
    const inputs: string[] = [];
    const bases: number[][][] = [];
    basesEntries.forEach((entry, axis) => {
        const pair = Array.isArray(entry) ? entry : [entry?.name, entry?.bases];
        inputs.push(typeof pair[0] === 'string' ? pair[0] : `Axis${axis + 1}`);
        bases.push(Array.isArray(pair[1]) ? pair[1] as number[][] : []);
    });
    const outputsSource = Array.isArray(record.output_dims)
        ? record.output_dims
        : Array.isArray(record.out_dims)
            ? record.out_dims
            : [];
    const outputs = outputsSource.length
        ? outputsSource.map((entry, axis) => {
            if (typeof entry === 'string') return entry;
            if (Array.isArray(entry) && typeof entry[0] === 'string') return entry[0];
            if (entry && typeof entry === 'object' && typeof (entry as { name?: unknown }).name === 'string') {
                return (entry as { name: string }).name;
            }
            return `Axis${axis + 1}`;
        })
        : Array.from({ length: Math.max(1, ...bases.flatMap((row) => row.map((basis) => basis.length))) }, (_entry, axis) => String.fromCharCode(65 + axis));
    const colorAxes = record.color_axes && typeof record.color_axes === 'object'
        ? record.color_axes as Record<string, string>
        : {};
    const colorRanges = record.color_ranges && typeof record.color_ranges === 'object'
        ? Object.fromEntries(Object.entries(record.color_ranges).flatMap(([key, value]) => (
            Array.isArray(value) && value.length === 2 ? [[key, [Number(value[0]), Number(value[1])]]] : []
        ))) as Record<string, [number, number]>
        : {};
    return {
        name: typeof record.name === 'string' && record.name.trim() ? record.name : fallbackTitle,
        inputs,
        outputs,
        bases,
        colorAxes,
        colorRanges,
    };
}

/**
 * Converts a legacy input-axis name into the short label used by current
 * compose-layout presets, preserving hardware aliases such as thread, warp, and
 * register so migrated color mappings still target T, W, and R.
 *
 * @param name - Legacy input dimension name read from saved layout metadata.
 * @param axis - Zero-based position of that input dimension in the legacy spec;
 *   nonzero positions receive a numeric suffix when the name is not a known
 *   hardware-axis alias.
 * @returns The canonical input label used in the migrated compose spec: a known
 *   legacy alias, the first letter of `name` uppercased, or `A` when the name has
 *   no letters.
 * @noThrows The function only lowercases and pattern-matches the provided string;
 *   unmatched names are handled by deterministic label fallbacks instead of
 *   raising migration errors.
 * @example
 * expect(canonicalLegacyLabel('thread', 0)).toBe('T');
 * expect(canonicalLegacyLabel('lane_id', 0)).toBe('L');
 * expect(canonicalLegacyLabel('lane_id', 2)).toBe('L2');
 */
function canonicalLegacyLabel(name: string, axis: number): string {
    // preserve the old hardware-axis labels where possible so saved color
    // mappings still point at T/W/R after migration.
    const lowered = name.toLowerCase();
    if (lowered in LEGACY_AXIS_ALIASES) return LEGACY_AXIS_ALIASES[lowered as keyof typeof LEGACY_AXIS_ALIASES];
    const match = name.match(/[A-Za-z]/);
    const base = (match?.[0] ?? 'A').toUpperCase();
    return axis === 0 ? base : `${base}${axis}`;
}

/**
 * Normalizes a legacy output-axis name to the uppercase axis label expected by
 * current compose-layout output lists.
 *
 * @param name - Legacy output dimension name read from `output_dims` or
 *   `out_dims` metadata.
 * @param axis - Zero-based output position used to choose an alphabetic fallback
 *   when the legacy name contains no letters.
 * @returns The migrated output label: single-letter names with optional numeric
 *   suffixes are uppercased as-is, other names use their first letter, and
 *   non-alphabetic names fall back to `A + axis`.
 * @noThrows The function handles every string by regex matching and alphabetic
 *   fallback generation, so malformed legacy labels still migrate to a label.
 * @example
 * expect(canonicalLegacyOutputLabel('m0', 0)).toBe('M0');
 * expect(canonicalLegacyOutputLabel('column', 1)).toBe('C');
 * expect(canonicalLegacyOutputLabel('123', 2)).toBe('C');
 */
function canonicalLegacyOutputLabel(name: string, axis: number): string {
    const match = name.match(/^([A-Za-z])([0-9]*)$/);
    if (match) return `${match[1]!.toUpperCase()}${match[2] ?? ''}`;
    const firstLetter = name.match(/[A-Za-z]/)?.[0];
    return (firstLetter ?? String.fromCharCode(65 + axis)).toUpperCase();
}

/**
 * Provides the stable input-axis sort priority used when migrating old
 * linear-layout specs into the current T/W/R-oriented preset order.
 *
 * @param label - Canonical legacy input label, typically produced by
 *   `canonicalLegacyLabel`.
 * @returns Sort rank for migrated inputs: T first, W second, R third, and every
 *   other label after the hardware-axis labels.
 * @noThrows Unknown labels are intentionally assigned the catch-all rank `3`, so
 *   sorting can continue even when a saved spec contains custom axis names.
 * @example
 * const labels = ['X', 'R', 'T', 'W'];
 * labels.sort((left, right) => legacyAxisOrder(left) - legacyAxisOrder(right));
 *
 * expect(labels).toEqual(['T', 'W', 'R', 'X']);
 */
function legacyAxisOrder(label: string): number {
    if (label === 'T') return 0;
    if (label === 'W') return 1;
    if (label === 'R') return 2;
    return 3;
}

/**
 * Converts saved pre-compose color-axis metadata into the current H/S/L compose-channel mapping.
 *
 * Starts from the historic thread, warp, and register defaults, then applies any saved axis-to-channel overrides whose channel name resolves to H, S, or L.
 *
 * @param labelMap - Lookup from legacy axis names, such as `thread`, `warp`, `register`, or preset-specific axes, to the label text stored in restored tensor metadata.
 * @param colorAxes - Saved legacy color overrides keyed by axis name with channel names such as `h`, `S`, or `L`; entries for other channels are ignored.
 * @returns H/S/L mapping for restored compose color controls, using mapped labels when present and a canonical legacy label for a known override axis that has no saved label.
 * @noThrows The conversion only reads map/object entries, normalizes channel names, and falls back for unknown or missing axes instead of rejecting them.
 * @example
 * const labelMap = new Map([
 *   ['thread', 'thread_id'],
 *   ['warp', 'warp_id'],
 *   ['register', 'reg_id'],
 *   ['lane', 'lane_id'],
 * ]);
 *
 * legacyMapping(labelMap, { lane: 'h', ignored: 'alpha' });
 * // => { H: 'lane_id', S: 'warp_id', L: 'reg_id' }
 */
function legacyMapping(
    labelMap: Map<string, string>,
    colorAxes: Record<string, string>,
): Record<ComposeChannel, ComposeMappingValue> {
    // preserve the old thread/warp/register color defaults, then apply any
    // saved channel overrides that reference known legacy axes.
    const mapping: Record<ComposeChannel, ComposeMappingValue> = {
        H: labelMap.get('thread') ?? 'none',
        S: labelMap.get('warp') ?? 'none',
        L: labelMap.get('register') ?? 'none',
    };
    Object.entries(colorAxes).forEach(([axisName, channelName]) => {
        const channel = String(channelName).toUpperCase() as ComposeChannel;
        if (!['H', 'S', 'L'].includes(channel)) return;
        mapping[channel] = labelMap.get(axisName) ?? canonicalLegacyLabel(axisName, 0);
    });
    return mapping;
}

/**
 * Converts saved numeric legacy color ranges into the string ranges used by compose H/S/L controls.
 *
 * Missing channels are filled from the built-in compose defaults so restored sessions always have complete H, S, and L range state.
 *
 * @param colorRanges - Saved legacy range metadata keyed by compose channel, where each provided tuple is the numeric minimum and maximum for that channel.
 * @returns Complete H/S/L range object with every endpoint converted to a string for sidebar form state.
 * @noThrows The function only checks optional channel properties, stringifies tuple entries, and copies defaults for absent channels.
 * @example
 * const ranges = legacyRanges({ H: [0, 31] });
 *
 * ranges.H;
 * // => ['0', '31']
 * ranges.S;
 * // => [...DEFAULT_COLOR_RANGES.S]
 */
function legacyRanges(colorRanges: Record<string, [number, number]>): Record<ComposeChannel, [string, string]> {
    return {
        H: colorRanges.H ? [String(colorRanges.H[0]), String(colorRanges.H[1])] : [...DEFAULT_COLOR_RANGES.H],
        S: colorRanges.S ? [String(colorRanges.S[0]), String(colorRanges.S[1])] : [...DEFAULT_COLOR_RANGES.S],
        L: colorRanges.L ? [String(colorRanges.L[0]), String(colorRanges.L[1])] : [...DEFAULT_COLOR_RANGES.L],
    };
}

/**
 * Assigns propagated input-axis labels to compose color channels by ranking axes from largest to smallest.
 *
 * Larger tensor dimensions receive the earlier auto-color channels, and equal-sized axes prefer the later axis index to preserve historic hardware-layout defaults.
 *
 * @param inputLabels - Axis labels in tensor-axis order from propagated linear-layout metadata.
 * @param inputShape - Tensor dimension sizes in the same order as `inputLabels`; a missing size is treated as `1` for ranking.
 * @returns H/S/L mapping for automatic color controls, using `'none'` for any channel that has no ranked input axis.
 * @noThrows Ranking uses array mapping, numeric comparison, and optional indexing; missing shape entries and missing labels are handled with defaults.
 * @example
 * autoColorMapping(['batch', 'row', 'col'], [2, 128, 64]);
 * // => { H: 'row', S: 'col', L: 'batch' }
 */
function autoColorMapping(
    inputLabels: string[],
    inputShape: number[],
): Record<ComposeChannel, ComposeMappingValue> {
    // largest axes get the most perceptually distinct channels first; ties prefer
    // later axes to match the historic hardware-layout defaults.
    const rankedAxes = inputLabels
        .map((label, axis) => ({ label, axis, size: inputShape[axis] ?? 1 }))
        .sort((left, right) => (right.size - left.size) || (right.axis - left.axis));
    const mapping: Record<ComposeChannel, ComposeMappingValue> = { H: 'none', S: 'none', L: 'none' };
    AUTO_COLOR_CHANNELS.forEach((channel, axis) => {
        mapping[channel] = rankedAxes[axis]?.label ?? 'none';
    });
    return mapping;
}

/**
 * Creates fresh H/S/L compose color-range state from the built-in defaults.
 *
 * Each channel tuple is cloned so sidebar state can mutate its copy without changing `DEFAULT_COLOR_RANGES` or other tabs.
 *
 * @returns New per-channel string range tuples for H, S, and L compose color controls.
 * @noThrows The function only copies static default tuples into a new object.
 * @example
 * const ranges = defaultColorRanges();
 *
 * ranges.H;
 * // => [...DEFAULT_COLOR_RANGES.H]
 * ranges.H === DEFAULT_COLOR_RANGES.H;
 * // => false
 */
function defaultColorRanges(): Record<ComposeChannel, [string, string]> {
    return {
        H: [...DEFAULT_COLOR_RANGES.H],
        S: [...DEFAULT_COLOR_RANGES.S],
        L: [...DEFAULT_COLOR_RANGES.L],
    };
}
