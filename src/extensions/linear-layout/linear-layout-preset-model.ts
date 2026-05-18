import { COMPOSE_LAYOUT_PRESET_FAMILIES } from './presets/index.js';
import type {
    ComposeLayoutPresetDefinition,
    ComposeLayoutPresetFacetValue,
    ComposeLayoutPresetFieldDefinition,
} from './presets/types.js';
import {
    formatSpecsText,
    parseLayoutSpecs,
    parseSignature,
    stripLayoutComment,
} from './linear-layout-parser.js';

/**
 * Maps each compose-layout preset selector field key to the currently selected facet value in the sidebar.
 *
 * Empty strings represent fields the user has not chosen yet; completed selections can be matched against the preset catalog to produce layout text.
 *
 * @example
 * const selection: ComposeLayoutPresetSelection = {
 *     instruction: 'mma',
 *     shape: 'm16n8k16',
 *     trans: 'nt',
 *     major: 'row',
 * };
 */
export type ComposeLayoutPresetSelection = Record<string, string>;

/**
 * Normalized metadata for one preset selector shown by the compose-layout preset widget.
 *
 * The model guarantees a stable `id` and an array of available string `values`, even when the source preset field definition omitted or merged values.
 *
 * @example
 * const field: ComposeLayoutPresetField = {
 *     id: 'compose-layout-preset-instruction',
 *     key: 'instruction',
 *     label: 'Instruction',
 *     values: ['mma', 'wgmma'],
 * };
 */
export type ComposeLayoutPresetField = Required<Omit<ComposeLayoutPresetFieldDefinition, 'values'>> & {
    id: string;
    values: string[];
};

/**
 * Compose-layout editor contents produced when a preset selection resolves to a single catalog entry.
 *
 * The sidebar writes these strings into the layout specs editor, operation editor, and selected input name so the rendered tensor view matches the chosen preset.
 *
 * @example
 * const state: ComposeLayoutPresetState = {
 *     specsText: 'M = 16\nN = 8\nK = 16',
 *     operationText: 'mma(M, N, K)',
 *     inputName: 'A',
 * };
 */
export type ComposeLayoutPresetState = {
    specsText: string;
    operationText: string;
    inputName: string;
};

/**
 * Normalized compose-layout preset consumed by the linear-layout sidebar.
 *
 * Each preset combines the selector values shown in the preset controls, the
 * facet metadata used for filtering, and the compose-layout state that should be
 * loaded when a user chooses the preset.
 *
 * @example
 * const preset: ComposeLayoutPreset = {
 *     title: 'MMA row-major f32',
 *     facets: { family: ['mma'], precision: ['f32'] },
 *     gpuArchs: ['sm_90'],
 *     instruction: 'mma',
 *     matrixSize: '16x8x16',
 *     dtype: 'f32',
 *     operand: 'A',
 *     trans: 'n',
 *     major: 'row',
 *     state: {
 *         specsText: 'A: 16x16 row-major',
 *         operationText: 'mma',
 *         inputName: 'A',
 *     },
 * };
 *
 * console.assert(preset.state.inputName === 'A');
 */
export type ComposeLayoutPreset = {
    title: string;
    facets: Record<string, string[]>;
    gpuArchs: string[];
    instruction: string;
    matrixSize: string;
    dtype: string;
    operand: string;
    trans: string;
    major: string;
    state: ComposeLayoutPresetState;
};

/**
 * Available selector values for the compose-layout preset sidebar.
 *
 * The object is keyed by canonical field names and also exposes the legacy
 * plural aliases still read by older widget code.
 *
 * @example
 * const options: ComposeLayoutPresetOptions = {
 *     gpuArchs: ['sm_80', 'sm_90'],
 *     instructions: ['mma', 'wgmma'],
 *     matrixSizes: ['16x8x16'],
 *     dtypes: ['f16', 'f32'],
 *     operands: ['A', 'B'],
 *     transes: ['n', 't'],
 *     majors: ['row', 'col'],
 *     gpuArch: ['sm_80', 'sm_90'],
 *     instruction: ['mma', 'wgmma'],
 *     matrixSize: ['16x8x16'],
 *     dtype: ['f16', 'f32'],
 *     operand: ['A', 'B'],
 *     trans: ['n', 't'],
 *     major: ['row', 'col'],
 * };
 *
 * console.assert(options.gpuArchs.includes('sm_90'));
 */
export type ComposeLayoutPresetOptions = Record<string, string[]> & {
    gpuArchs: string[];
    instructions: string[];
    matrixSizes: string[];
    dtypes: string[];
    operands: string[];
    transes: string[];
    majors: string[];
};

const LEGACY_PRESET_FIELD_KEYS = ['gpuArch', 'instruction', 'matrixSize', 'dtype', 'operand', 'trans', 'major'] as const;

const PRESET_FIELD_OPTION_ALIASES = {
    gpuArch: 'gpuArchs',
    instruction: 'instructions',
    matrixSize: 'matrixSizes',
    dtype: 'dtypes',
    operand: 'operands',
    trans: 'transes',
    major: 'majors',
} as const;

const PRESET_DEFINITIONS: readonly ComposeLayoutPresetDefinition[] = COMPOSE_LAYOUT_PRESET_FAMILIES.flatMap((family) => family.presets);

// field metadata is merged from family declarations plus facet keys found in
// preset data; this lets contributed families add fields without widget edits.
const COMPOSE_LAYOUT_PRESET_FIELDS = mergedPresetFields([
    ...COMPOSE_LAYOUT_PRESET_FAMILIES.flatMap((family) => family.fields ?? []),
    ...PRESET_DEFINITIONS.flatMap((definition) => (
        Object.keys(definition.facets ?? {}).map((key) => inferredPresetFieldDefinition(key))
    )),
]);

const COMPOSE_LAYOUT_PRESET_CATALOG = PRESET_DEFINITIONS.map((definition) => composeLayoutPreset(definition));

/**
 * Creates the blank preset-filter state used before the user chooses any
 * compose-layout selector values.
 *
 * @returns A selection object containing every registered preset field key with
 * an empty string value, so widgets can bind controls without checking for
 * missing keys.
 * @noThrows Reads module-level preset field metadata and constructs a plain
 * object; it does not parse user input or perform validation that would raise an
 * application error.
 * @example
 * const selection = emptyComposeLayoutPresetSelection();
 *
 * console.assert(selection.gpuArch === '');
 * console.assert(selection.instruction === '');
 */
export function emptyComposeLayoutPresetSelection(): ComposeLayoutPresetSelection {
    return Object.fromEntries(COMPOSE_LAYOUT_PRESET_FIELDS.map((field) => [field.key, '']));
}

/**
 * Copies persisted or externally supplied preset selection data into the current
 * compose-layout field set.
 *
 * Unknown keys are ignored, non-string field values are reset to empty strings,
 * and a legacy `category` string is treated as the instruction selection when no
 * `instruction` value is present.
 *
 * @param selection - Saved sidebar selection data from tab/session state, or
 * `undefined` when no preset selection was persisted.
 * @returns A fresh selection object with every registered preset field present;
 * recognized string values are preserved and missing, unknown, or non-string
 * values become empty strings.
 * @noThrows Malformed persisted data is sanitized with type checks instead of
 * being parsed or validated as a strict schema.
 * @example
 * const selection = cloneComposeLayoutPresetSelection({
 *     gpuArch: 'sm_90',
 *     matrixSize: 128 as unknown as string,
 *     category: 'wgmma',
 *     ignoredWidgetKey: 'unused',
 * });
 *
 * console.assert(selection.gpuArch === 'sm_90');
 * console.assert(selection.matrixSize === '');
 * console.assert(selection.instruction === 'wgmma');
 * console.assert(!('ignoredWidgetKey' in selection));
 */
export function cloneComposeLayoutPresetSelection(
    selection: ComposeLayoutPresetSelection | undefined,
): ComposeLayoutPresetSelection {
    const record = selection as (Record<string, unknown> & { category?: unknown }) | undefined;
    const cloned = emptyComposeLayoutPresetSelection();
    COMPOSE_LAYOUT_PRESET_FIELDS.forEach((field) => {
        const value = record?.[field.key];
        cloned[field.key] = typeof value === 'string' ? value : '';
    });
    if (!cloned.instruction && typeof record?.category === 'string') cloned.instruction = record.category;
    return cloned;
}

/**
 * Checks whether persisted sidebar preset-selection data can be copied back into
 * the compose-layout editor.
 *
 * @param value - Unknown value read from saved linear-layout tab metadata or browser storage.
 * @returns `true` when `value` is a non-null object whose selection entries are all strings; `false` for nullish values, primitives, arrays with non-string entries, or records containing non-string values.
 * @noThrows The guard inspects only `typeof`, nullability, and `Object.values` on values that have already been proven object-like, so malformed persisted data is reported as `false` instead of throwing.
 * @example
 * const saved = { instruction: 'mma', gpuArch: 'sm_90', layout: '' };
 *
 * if (isComposeLayoutPresetSelection(saved)) {
 *     saved.instruction.toUpperCase(); // narrowed to string-valued preset selection data
 * }
 *
 * isComposeLayoutPresetSelection(saved); // true
 * isComposeLayoutPresetSelection({ instruction: 'mma', gpuArch: 90 }); // false
 */
export function isComposeLayoutPresetSelection(value: unknown): value is ComposeLayoutPresetSelection {
    if (!value || typeof value !== 'object') return false;
    return Object.values(value as Record<string, unknown>).every((entry) => typeof entry === 'string');
}

/**
 * Provides the preset selector field definitions in the order the linear-layout
 * sidebar renders them.
 *
 * @returns Field metadata for the preset widget, including each field key, DOM id, label, dependency keys, allowed values, and whether the field is required. The returned field objects contain fresh `dependsOn` and `values` arrays so widget code can derive temporary state without mutating module constants.
 * @noThrows The data is assembled from in-module preset field constants with shallow object and array copies only; it performs no parsing, I/O, or user-input validation.
 * @example
 * const fields = composeLayoutPresetFields();
 *
 * fields.map((field) => field.key); // render order for preset controls
 * fields.every((field) => Array.isArray(field.dependsOn) && Array.isArray(field.values)); // true
 *
 * const again = composeLayoutPresetFields();
 * fields[0].dependsOn === again[0].dependsOn; // false
 */
export function composeLayoutPresetFields(): ComposeLayoutPresetField[] {
    return COMPOSE_LAYOUT_PRESET_FIELDS.map((field) => ({
        ...field,
        dependsOn: [...field.dependsOn],
        values: [...field.values],
    }));
}

/**
 * Exposes the canonical compose-layout preset catalog used by model tests and
 * preset-matching helpers.
 *
 * @returns The shared preset catalog containing each preset's selection facets, supported GPU architectures, and compose-layout editor state. Callers that need mutable UI copies should use `composeLayoutPresets()` instead.
 * @noThrows The catalog is a module-level constant and this accessor simply returns that reference without parsing preset text or reading external state.
 * @example
 * const catalog = composeLayoutPresetCatalog();
 *
 * catalog.length > 0; // true when preset data is registered
 * catalog.every((preset) => typeof preset.state.specsText === 'string'); // true
 * composeLayoutPresetCatalog() === catalog; // true, this is the shared catalog reference
 */
export function composeLayoutPresetCatalog(): ComposeLayoutPreset[] {
    return COMPOSE_LAYOUT_PRESET_CATALOG;
}

/**
 * Builds UI-safe copies of the compose-layout presets for sidebar filtering and
 * temporary selection state.
 *
 * @returns A preset list with each preset object copied, including cloned facet arrays, GPU architecture arrays, and editor state objects. Sidebar callers can sort, filter, or annotate the returned objects without changing the canonical catalog.
 * @noThrows The function clones the in-memory preset catalog with object spreads, array spreads, and `Object.entries`; it performs no user-input parsing or external I/O.
 * @example
 * const uiPresets = composeLayoutPresets();
 * const catalog = composeLayoutPresetCatalog();
 *
 * uiPresets.length === catalog.length; // true
 * uiPresets[0] === catalog[0]; // false
 * uiPresets[0].gpuArchs === catalog[0].gpuArchs; // false
 * uiPresets[0].state === catalog[0].state; // false
 */
export function composeLayoutPresets(): ComposeLayoutPreset[] {
    return composeLayoutPresetCatalog().map((preset) => ({
        ...preset,
        facets: Object.fromEntries(Object.entries(preset.facets).map(([key, values]) => [key, [...values]])),
        gpuArchs: [...preset.gpuArchs],
        state: { ...preset.state },
    }));
}

/**
 * Reconstructs the preset dropdown selection for compose-layout editor text that matches a shipped preset.
 *
 * The match uses canonicalized specs text plus the operation text and input-space name, so saved tabs can regain
 * their preset selectors even when comments or formatting differ from the catalog entry.
 *
 * @param state - Compose-layout editor state containing `specsText`, `operationText`, and `inputName` from a saved tab or rendered metadata.
 * @returns Selector facet values for the matching preset, or an empty preset selection when the editor state is custom or unmatched.
 * @noThrows Matching is performed with string canonicalization and catalog lookup; unmatched editor states are represented as an empty selection instead of an exception.
 * @example
 * const selection = matchedComposeLayoutPresetSelection({
 *   specsText: [
 *     'ldmatrix_m8n8_x1_b16: [R, C] -> [T, R32]',
 *     'input: [R, C] -> ldmatrix_m8n8_x1_b16',
 *   ].join('\n'),
 *   operationText: 'input',
 *   inputName: 'Input Space',
 * });
 *
 * expect(selection).toMatchObject({
 *   gpuArch: 'sm_75',
 *   instruction: 'ldmatrix',
 *   matrixSize: 'm8n8',
 *   dtype: 'b16',
 *   trans: 'no',
 * });
 */
export function matchedComposeLayoutPresetSelection(
    state: ComposeLayoutPresetState,
): ComposeLayoutPresetSelection {
    const canonicalSpecsText = canonicalLayoutSpecsText(state.specsText);
    // presets are matched by canonicalized text, operation, and input name so
    // adding comments to a preset does not strand an already-loaded editor state.
    const preset = composeLayoutPresetCatalog().find((entry) => canonicalLayoutSpecsText(entry.state.specsText) === canonicalSpecsText
        && entry.state.operationText === state.operationText
        && entry.state.inputName === state.inputName);
    if (!preset) return emptyComposeLayoutPresetSelection();
    return Object.fromEntries(COMPOSE_LAYOUT_PRESET_FIELDS.map((field) => [
        field.key,
        preset.facets[field.key]?.[0] ?? '',
    ]));
}

/**
 * Builds the preset selector option lists that remain valid for the current compose-layout facet path.
 *
 * Each field is computed while ignoring that field's current value, which lets the sidebar still offer valid
 * recovery choices when a user-entered value is misspelled or no longer compatible with the other facets.
 *
 * @param selection - Partial preset facet selection from the sidebar, such as GPU architecture, instruction, matrix size, dtype, operand, transpose, and major layout.
 * @returns Option arrays for each selector field, filtered by the other selected facets and mirrored through legacy plural aliases used by callers.
 * @noThrows Undefined and partial selections are cloned into an empty/default selection, and incompatible facets simply produce shorter or empty option arrays.
 * @example
 * const options = composeLayoutPresetOptions({
 *   gpuArch: 'sm_75',
 *   instruction: 'ldmatrix',
 *   matrixSize: 'm8n8',
 *   dtype: '',
 *   operand: '',
 *   trans: '',
 *   major: '',
 * });
 *
 * expect(options.transes).toEqual(['no', 'yes']);
 * expect(options.instructions).toEqual(['ldmatrix']);
 */
export function composeLayoutPresetOptions(
    selection: ComposeLayoutPresetSelection | undefined,
): ComposeLayoutPresetOptions {
    const current = cloneComposeLayoutPresetSelection(selection);
    const presets = composeLayoutPresetCatalog();
    const options: ComposeLayoutPresetOptions = {
        gpuArchs: [],
        instructions: [],
        matrixSizes: [],
        dtypes: [],
        operands: [],
        transes: [],
        majors: [],
    };
    COMPOSE_LAYOUT_PRESET_FIELDS.forEach((field) => {
        // when computing choices for one field, ignore that field's current
        // value; otherwise a typo would hide the valid options needed to recover.
        const values = uniquePresetFacetValues(filteredPresets(
            presets,
            Object.fromEntries(Object.entries(current).map(([key, value]) => [key, key === field.key ? '' : value])),
        ), field);
        options[field.key] = values;
        const alias = PRESET_FIELD_OPTION_ALIASES[field.key as keyof typeof PRESET_FIELD_OPTION_ALIASES];
        if (alias) options[alias] = values;
    });
    return options;
}

/**
 * Sanitizes a compose-layout preset selection after a sidebar facet changes.
 *
 * Invalid facet values are cleared, and any field with exactly one remaining compatible option is filled so the
 * preset picker can advance through forced choices without requiring another user click.
 *
 * @param selection - Undefined, partial, or user-edited preset facet values from the compose-layout preset controls.
 * @returns A complete selection-shaped object with impossible facet values removed and singleton compatible facets populated.
 * @noThrows The function treats missing and inconsistent selections as data to normalize; catalog filtering returns empty choices instead of throwing.
 * @example
 * const normalized = normalizeComposeLayoutPresetSelection({
 *   gpuArch: 'sm_75',
 *   instruction: 'ldmatrix',
 *   matrixSize: 'not-a-shipped-size',
 *   dtype: '',
 *   operand: '',
 *   trans: '',
 *   major: '',
 * });
 *
 * expect(normalized).toMatchObject({
 *   gpuArch: 'sm_75',
 *   instruction: 'ldmatrix',
 *   matrixSize: '',
 * });
 */
export function normalizeComposeLayoutPresetSelection(
    selection: ComposeLayoutPresetSelection | undefined,
): ComposeLayoutPresetSelection {
    const current = cloneComposeLayoutPresetSelection(selection);
    const normalized = { ...current };
    COMPOSE_LAYOUT_PRESET_FIELDS.forEach((field) => {
        normalized[field.key] = normalizedPresetField(current[field.key] ?? '', composeLayoutPresetOptions(normalized)[field.key] ?? []);
    });
    return normalized;
}

/**
 * Resolves a fully specified compose-layout preset selection to the catalog preset it names.
 *
 * This is used after the dropdown facets identify one shipped instruction layout so callers can load that preset's
 * specs text, operation text, title, and input-space metadata into the editor.
 *
 * @param selection - Preset facet values chosen in the sidebar, including architecture, instruction, matrix size, dtype, operand, transpose, and major layout.
 * @returns The single catalog preset whose facets are complete for the selection, or `null` when the selection is incomplete, ambiguous, or not in the shipped catalog.
 * @noThrows Failed preset resolution is a normal picker state and is reported as `null`; the function only clones the selection and filters the in-memory catalog.
 * @example
 * const preset = composeLayoutPresetForSelection({
 *   gpuArch: 'sm_75',
 *   instruction: 'ldmatrix',
 *   matrixSize: 'm8n8',
 *   dtype: 'b16',
 *   operand: '',
 *   trans: 'no',
 *   major: '',
 * });
 *
 * expect(preset?.title).toBe('ldmatrix_m8n8_x1_b16');
 *
 * expect(composeLayoutPresetForSelection({
 *   gpuArch: 'sm_75',
 *   instruction: 'ldmatrix',
 *   matrixSize: '',
 *   dtype: '',
 *   operand: '',
 *   trans: '',
 *   major: '',
 * })).toBeNull();
 */
export function composeLayoutPresetForSelection(
    selection: ComposeLayoutPresetSelection | undefined,
): ComposeLayoutPreset | null {
    const current = cloneComposeLayoutPresetSelection(selection);
    const matches = filteredPresets(composeLayoutPresetCatalog(), current).filter((preset) => (
        COMPOSE_LAYOUT_PRESET_FIELDS.every((field) => presetFieldIsComplete(preset, current, field.key))
    ));
    return matches.length === 1 ? matches[0]! : null;
}

/**
 * Builds the preset selector field list from family-declared fields and facet keys discovered in preset data.
 *
 * Definitions with the same key are folded together in input order, then the resulting fields are sorted by numeric
 * order and finally by key so the sidebar renders a stable selector order.
 *
 * @param definitions - Field declarations contributed by preset families or inferred from preset facet names.
 * @returns Normalized selector fields with duplicate keys merged, dependency/value lists de-duplicated, and display order applied.
 * @noThrows The function only iterates the provided array and delegates per-field normalization; it performs no parsing, lookup, or validation that is expected to throw for preset metadata.
 * @example
 * const fields = mergedPresetFields([
 *     { key: 'mmaShape', label: 'MMA shape', order: 20, values: ['m16n8k16'] },
 *     { key: 'architecture', label: 'Architecture', order: 10, values: ['sm90'] },
 *     { key: 'mmaShape', label: 'Ignored duplicate label', order: 5, required: true, values: ['m16n8k16', 'm32n8k16'] },
 * ]);
 *
 * fields.map((field) => ({ key: field.key, order: field.order, values: field.values, required: field.required }));
 * // => [
 * //   { key: 'mmaShape', order: 5, values: ['m16n8k16', 'm32n8k16'], required: true },
 * //   { key: 'architecture', order: 10, values: ['sm90'], required: false },
 * // ]
 */
function mergedPresetFields(definitions: readonly ComposeLayoutPresetFieldDefinition[]): ComposeLayoutPresetField[] {
    const fields = new Map<string, ComposeLayoutPresetField>();
    definitions.forEach((definition) => {
        fields.set(definition.key, normalizePresetFieldDefinition(definition, fields.get(definition.key)));
    });
    return Array.from(fields.values()).sort((left, right) => left.order - right.order || left.key.localeCompare(right.key));
}

/**
 * Converts one preset field declaration into the normalized selector metadata used by the linear-layout preset UI.
 *
 * When a normalized field for the same key already exists, its id, label, and placeholder stay authoritative while
 * the new declaration can lower the sort order, mark the field required, and add dependency or option values.
 *
 * @param definition - New field declaration from a preset family or inferred preset facet, including its key and optional display/options metadata.
 * @param current - Previously normalized field for the same key, when another declaration already contributed metadata.
 * @returns A selector field with a generated DOM-safe id, merged dependency/value arrays without duplicates, the lowest order, and a unioned required flag.
 * @noThrows The normalization is a deterministic object/array/string transformation over already-loaded preset metadata and does not parse compose-layout syntax or access external state.
 * @example
 * const field = normalizePresetFieldDefinition(
 *     { key: 'mmaShape', label: 'Ignored label', order: 5, required: true, values: ['m16n8k16', 'm32n8k16'] },
 *     { id: 'linear-layout-preset-mma-shape', key: 'mmaShape', label: 'MMA shape', order: 20, required: false, dependsOn: ['architecture'], values: ['m16n8k16'] },
 * );
 *
 * field;
 * // => {
 * //   id: 'linear-layout-preset-mma-shape',
 * //   key: 'mmaShape',
 * //   label: 'MMA shape',
 * //   placeholder: undefined,
 * //   order: 5,
 * //   required: true,
 * //   dependsOn: ['architecture'],
 * //   values: ['m16n8k16', 'm32n8k16'],
 * // }
 */
function normalizePresetFieldDefinition(
    definition: ComposeLayoutPresetFieldDefinition,
    current?: ComposeLayoutPresetField,
): ComposeLayoutPresetField {
    const values = [...new Set([...(current?.values ?? []), ...(definition.values ?? [])])];
    const dependsOn = [...new Set([...(current?.dependsOn ?? []), ...(definition.dependsOn ?? [])])];
    return {
        key: definition.key,
        id: current?.id ?? `linear-layout-preset-${definition.key.replace(/([a-z0-9])([A-Z])/g, '$1-$2').replace(/[^a-z0-9]+/gi, '-').toLowerCase()}`,
        label: current?.label ?? definition.label,
        placeholder: current?.placeholder ?? definition.placeholder,
        order: Math.min(current?.order ?? definition.order, definition.order),
        required: Boolean(current?.required || definition.required),
        dependsOn,
        values,
    };
}

/**
 * Creates fallback selector metadata for a preset facet key that was present in preset data but not declared by a family.
 *
 * The generated label is human-readable, the placeholder prompts for that label, and the default order places inferred
 * fields after explicitly ordered family fields.
 *
 * @param key - Facet property name from a compose-layout preset definition, such as a camelCase, snake_case, or kebab-case selector key.
 * @returns A field declaration using the original facet key, a title-cased label, a matching text-entry placeholder, and default order 100.
 * @noThrows The function only applies regular-expression string replacements to the supplied key and constructs a plain metadata object.
 * @example
 * inferredPresetFieldDefinition('mmaShape');
 * // => {
 * //   key: 'mmaShape',
 * //   label: 'Mma Shape',
 * //   placeholder: 'Type mma shape',
 * //   order: 100,
 * // }
 *
 * inferredPresetFieldDefinition('operand-layout');
 * // => {
 * //   key: 'operand-layout',
 * //   label: 'Operand Layout',
 * //   placeholder: 'Type operand layout',
 * //   order: 100,
 * // }
 */
function inferredPresetFieldDefinition(key: string): ComposeLayoutPresetFieldDefinition {
    const label = key
        .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
        .replace(/[_-]+/g, ' ')
        .replace(/\b\w/g, (char) => char.toUpperCase());
    return {
        key,
        label,
        placeholder: `Type ${label.toLowerCase()}`,
        order: 100,
    };
}

/**
 * Extracts the title text for a preset from the first line of its annotated compose-layout specs.
 *
 * If the first line contains a colon, only the text before the colon is used; otherwise the trimmed first line is used
 * unchanged. This lets presets default their sidebar title to the operation signature rather than the full spec body.
 *
 * @param specsText - Annotated compose-layout specification text, with the operation signature expected on the first line.
 * @returns Trimmed operation title derived from the first line, suitable for a preset card or generated fallback title.
 * @noThrows The function performs only string splitting, trimming, and slicing on the supplied specs text and has no validation or parsing branch.
 * @example
 * presetOperationText('mma.sync.aligned.m16n8k16: A[16, 16] x B[16, 8]\nC[16, 8]');
 * // => 'mma.sync.aligned.m16n8k16'
 *
 * presetOperationText('ldmatrix.x4.trans\n# loads four matrix fragments');
 * // => 'ldmatrix.x4.trans'
 */
function presetOperationText(specsText: string): string {
    const signature = specsText.split('\n', 1)[0]?.trim() ?? '';
    const colonIndex = signature.indexOf(':');
    return (colonIndex === -1 ? signature : signature.slice(0, colonIndex)).trim();
}

/**
 * Builds the explanatory comment used beside a compose-layout axis label in preset editor text.
 *
 * @param label - Axis token from a parsed compose-layout signature, such as `T`, `R`, `R32`, `C`, or a custom label.
 * @param signature - Parsed signature inputs and outputs; `R` is described as a row axis when the layout also mentions column input `C` or packed output `R32`.
 * @returns A single `LABEL = meaning` note that can be emitted as a `# ...` comment in generated preset specs.
 * @noThrows Uses fixed label comparisons and array membership checks; unknown labels fall back to a generic axis description instead of failing.
 * @example
 * axisComment('T', { inputs: ['T', 'R'], outputs: ['R32'] });
 * // Returns: 'T = thread (AKA lane)'
 *
 * axisComment('R', { inputs: ['T', 'R'], outputs: ['R32'] });
 * // Returns: 'R = row'
 *
 * axisComment('Q', { inputs: ['Q'], outputs: [] });
 * // Returns: 'Q = Q axis'
 */
function axisComment(label: string, signature: { inputs: string[]; outputs: string[] }): string {
    if (label === 'T') return 'T = thread (AKA lane)';
    if (label === 'R') {
        return signature.inputs.includes('C') || signature.outputs.includes('R32')
            ? 'R = row'
            : 'R = register';
    }
    if (label === 'R32') return 'R32 = packed 32-bit register';
    if (label === 'C') return 'C = column';
    if (label === 'W') return 'W = warp';
    if (label === 'Y') return 'Y = y-position';
    if (label === 'X') return 'X = x-position';
    if (label === 'M') return 'M = row';
    if (label === 'N') return 'N = column';
    if (label === 'K') return 'K = reduction dimension';
    if (label === 'O') return 'O = logical offset';
    if (label === 'H') return 'H = higher-order tile axis';
    if (label === 'L') return 'L = line';
    if (label === 'B') return 'B = byte offset';
    return `${label} = ${label} axis`;
}

/**
 * Inserts deterministic `# ...` comments after a preset's signature line so loaded compose-layout specs explain their axis labels.
 *
 * @param specsText - Raw compose-layout specs text whose first line contains the operation name and signature, followed by row definitions.
 * @param comments - Additional preset notes to append after the generated axis comments; duplicates are removed while preserving first occurrence order.
 * @returns The specs text with normalized newlines and generated comments placed between the first line and the remaining layout rows.
 * @noThrows For preset specs that already parse in this model, the function only normalizes newlines, derives label comments, de-duplicates strings, and joins text lines.
 * @example
 * annotatedLayoutSpecsText('mma: T,R -> R32\nT: 0,1', ['uses packed accumulator']);
 * // Returns:
 * // 'mma: T,R -> R32\n# T = thread (AKA lane)\n# R = row\n# R32 = packed 32-bit register\n# uses packed accumulator\nT: 0,1'
 */
function annotatedLayoutSpecsText(specsText: string, comments: string[] = []): string {
    const lines = specsText.replace(/\r\n/g, '\n').split('\n');
    const signature = parseSignature(stripLayoutComment(lines[0] ?? '').trim());
    const labelComments = [...signature.inputs, ...signature.outputs].map((label) => axisComment(label, signature));
    // comments become part of the loaded editor text, so keep generated axis
    // notes deterministic and deduplicated to avoid preset matching churn.
    return [
        lines[0] ?? '',
        ...Array.from(new Set([...labelComments, ...comments])).map((comment) => `# ${comment}`),
        ...lines.slice(1),
    ].join('\n');
}

/**
 * Normalizes every sidebar selector facet for a compose-layout preset definition.
 *
 * @param definition - Preset catalog entry, including optional `facets` metadata and legacy scalar selector fields used by older NVIDIA presets.
 * @returns A record keyed by every known compose-layout preset field; each value is the normalized list of selectable values, or an empty array when the preset does not participate in that selector.
 * @noThrows Iterates the fixed preset-field registry and delegates missing or legacy fields to fallback normalization, so absent facet data becomes `[]` rather than an exception.
 * @example
 * const facets = presetDefinitionFacets({
 *   name: 'mma.sync.aligned.m16n8k16',
 *   signature: 'T,R -> R32',
 *   rows: [],
 *   facets: { gpuArch: 'sm_90', dataType: ['f16', 'f32'] },
 * });
 *
 * facets.gpuArch;
 * // Returns: ['sm_90']
 * facets.dataType;
 * // Returns: ['f16', 'f32']
 */
function presetDefinitionFacets(definition: ComposeLayoutPresetDefinition): Record<string, string[]> {
    // every normalized preset receives every known field key.  Empty arrays mean
    // the field is irrelevant for that preset and should remain blank.
    return Object.fromEntries(COMPOSE_LAYOUT_PRESET_FIELDS.map((field) => [
        field.key,
        presetDefinitionFacetValues(definition, field.key),
    ]));
}

/**
 * Reads and normalizes the selector values for one compose-layout preset field.
 *
 * @param definition - Preset catalog entry that may provide modern `facets` values or legacy scalar fields for selector metadata.
 * @param key - Selector field key to resolve, such as `gpuArch` or another key registered in the compose-layout preset field list.
 * @returns The normalized string values for that selector; returns an empty array when the key is not present on the preset and has no legacy fallback.
 * @noThrows Optional facet lookup and unknown keys are handled with empty-array fallback, and legacy fields are normalized only when the key is in the known legacy field list.
 * @example
 * const definition = {
 *   name: 'mma.sync.aligned.m16n8k16',
 *   signature: 'T,R -> R32',
 *   rows: [],
 *   facets: { gpuArch: 'sm_90', dataType: ['f16', 'f32'] },
 * };
 *
 * presetDefinitionFacetValues(definition, 'gpuArch');
 * // Returns: ['sm_90']
 *
 * presetDefinitionFacetValues(definition, 'notASelector');
 * // Returns: []
 */
function presetDefinitionFacetValues(definition: ComposeLayoutPresetDefinition, key: string): string[] {
    const facet = definition.facets?.[key];
    if (facet !== undefined) return normalizePresetFacetValue(facet);
    // legacy scalar fields keep old NVIDIA presets working while new families
    // can describe selector behavior entirely through facets.
    if (LEGACY_PRESET_FIELD_KEYS.includes(key as typeof LEGACY_PRESET_FIELD_KEYS[number])) {
        return normalizePresetFacetValue(definition[key as typeof LEGACY_PRESET_FIELD_KEYS[number]] ?? '');
    }
    return [];
}

/**
 * Converts a catalog facet value into the string list used by preset selectors and matching.
 *
 * @param value - A preset facet stored either as one scalar value or as an array of facet values from a catalog definition.
 * @returns Stringified facet entries with empty string results removed, preserving the original order for dropdown options and filter checks.
 * @noThrows The helper only wraps scalars in an array, calls `String` on each entry, and filters empty results; well-formed catalog facet values do not require parsing or external lookups.
 * @example
 * normalizePresetFacetValue(['sm_90', '', 'sm_100']);
 * // Returns ['sm_90', 'sm_100'].
 *
 * normalizePresetFacetValue('f16');
 * // Returns ['f16'].
 */
function normalizePresetFacetValue(value: ComposeLayoutPresetFacetValue): string[] {
    const values = Array.isArray(value) ? value : [value];
    return values.map((entry) => String(entry)).filter(Boolean);
}

/**
 * Reads the first value for a preset facet when populating legacy scalar preset fields.
 *
 * @param facets - Facet map produced from a preset definition, where each key contains the normalized selector values for that preset.
 * @param key - Facet name to expose as a scalar field such as `instruction`, `matrixSize`, or `dtype`.
 * @returns The first facet value for `key`, or an empty string when the preset does not define that facet or its value list is empty.
 * @noThrows The lookup uses optional property and array access and falls back to `''` for missing data instead of indexing through undefined.
 * @example
 * const facets = { instruction: ['mma'], dtype: ['f16', 'bf16'] };
 * presetFacetScalar(facets, 'dtype');
 * // Returns 'f16'.
 *
 * presetFacetScalar(facets, 'major');
 * // Returns ''.
 */
function presetFacetScalar(facets: Record<string, string[]>, key: string): string {
    return facets[key]?.[0] ?? '';
}

/**
 * Builds the sidebar preset record and editor state from one linear-layout catalog definition.
 *
 * @param definition - Catalog entry that either provides ready-to-use `specsText` or a named ISA-table shape with `name`, `signature`, and row bases, plus optional facets, comments, title, and input name.
 * @returns A `ComposeLayoutPreset` containing the display title, normalized facet map, GPU architecture list, legacy scalar selector fields, and the compose-layout text/operation/input state loaded into the editor.
 * @noThrows For well-formed catalog data this helper only normalizes strings, joins named-definition rows, annotates comments, and copies facet values; it does not parse or evaluate the compose-layout program.
 * @example
 * const preset = composeLayoutPreset({
 *   title: 'SM90 f16 MMA',
 *   specsText: 'mma: (M, N) -> (M, N)',
 *   facets: { gpuArch: 'sm_90', instruction: 'mma', dtype: ['f16'] },
 * });
 *
 * preset.title;
 * // Returns 'SM90 f16 MMA'.
 * preset.gpuArchs;
 * // Returns ['sm_90'].
 * preset.dtype;
 * // Returns 'f16'.
 * preset.state.inputName;
 * // Returns 'Hardware Layout'.
 */
function composeLayoutPreset(definition: ComposeLayoutPresetDefinition): ComposeLayoutPreset {
    const inputName = definition.inputName ?? 'Hardware Layout';
    const facets = presetDefinitionFacets(definition);
    const gpuArchs = facets.gpuArch ?? [];
    if ('specsText' in definition) {
        const specsText = annotatedLayoutSpecsText(definition.specsText, definition.comments);
        const operationText = presetOperationText(specsText);
        return {
            title: definition.title ?? operationText,
            facets,
            gpuArchs,
            instruction: presetFacetScalar(facets, 'instruction'),
            matrixSize: presetFacetScalar(facets, 'matrixSize'),
            dtype: presetFacetScalar(facets, 'dtype'),
            operand: presetFacetScalar(facets, 'operand'),
            trans: presetFacetScalar(facets, 'trans'),
            major: presetFacetScalar(facets, 'major'),
            state: { specsText, operationText, inputName },
        };
    }
    // named definitions are the compact path for ISA-table-style presets: the
    // source file stores row data, and this model builds the editor notation.
    const specsText = annotatedLayoutSpecsText(
        [`${definition.name}: ${definition.signature}`, ...definition.rows.map(([label, bases]) => `${label}: ${bases}`)].join('\n'),
        definition.comments,
    );
    return {
        title: definition.title ?? definition.name,
        facets,
        gpuArchs,
        instruction: presetFacetScalar(facets, 'instruction'),
        matrixSize: presetFacetScalar(facets, 'matrixSize'),
        dtype: presetFacetScalar(facets, 'dtype'),
        operand: presetFacetScalar(facets, 'operand'),
        trans: presetFacetScalar(facets, 'trans'),
        major: presetFacetScalar(facets, 'major'),
        state: {
            specsText,
            operationText: definition.name,
            inputName,
        },
    };
}

/**
 * Selects presets whose facet lists contain every non-empty value chosen in the sidebar filters.
 *
 * @param presets - Catalog presets to test, in the order they should remain displayed after filtering.
 * @param filters - Current selector values keyed by facet name; empty strings mean that facet is not constrained.
 * @returns The original preset objects that match all active facet filters, preserving their input order for option generation and preset lookup.
 * @noThrows Filtering is an in-memory pass over preset facet maps; missing facet keys are handled with optional lookup and simply fail active filters.
 * @example
 * const presets = [
 *   { title: 'SM90 f16', facets: { gpuArch: ['sm_90'], dtype: ['f16'] } } as ComposeLayoutPreset,
 *   { title: 'SM80 f32', facets: { gpuArch: ['sm_80'], dtype: ['f32'] } } as ComposeLayoutPreset,
 * ];
 *
 * filteredPresets(presets, { gpuArch: 'sm_90', dtype: '' }).map((preset) => preset.title);
 * // Returns ['SM90 f16'].
 */
function filteredPresets(
    presets: ComposeLayoutPreset[],
    filters: ComposeLayoutPresetSelection,
): ComposeLayoutPreset[] {
    return presets.filter((preset) => Object.entries(filters).every(([key, value]) => {
        if (!value) return true;
        return preset.facets[key]?.includes(value) ?? false;
    }));
}

/**
 * Keeps a preset facet selection only when it is still one of the available choices, and auto-selects the sole remaining choice.
 *
 * @param value - Current selection string for one compose-layout preset facet, such as an operation family or input shape.
 * @param options - Valid option labels currently available for that facet after the other preset filters have been applied.
 * @returns The unchanged selection when it appears in `options`, the only option when exactly one choice remains, or an empty string when the selection must be cleared.
 * @noThrows Uses only string-array membership and length checks; invalid or missing choices are represented by returning an empty string rather than by throwing.
 * @example
 * normalizedPresetField('mma', ['mma', 'wgmma']);
 * // => 'mma'
 *
 * normalizedPresetField('typo', ['wgmma']);
 * // => 'wgmma'
 *
 * normalizedPresetField('typo', ['mma', 'wgmma']);
 * // => ''
 */
function normalizedPresetField(value: string, options: string[]): string {
    if (options.includes(value)) return value;
    return options.length === 1 ? options[0] ?? '' : '';
}

/**
 * Collects the distinct values a filtered preset list contributes for one selector facet while preserving catalog display order.
 *
 * @param presets - Compose-layout presets that remain after applying the other selector fields.
 * @param field - Selector metadata whose `key` names the facet in each preset and whose `values` array defines the preferred UI order.
 * @returns Unique facet values for the selector: known catalog values first in `field.values` order, followed by contributed values that were not declared in the field metadata.
 * @noThrows Missing facet arrays are treated as empty arrays, and the function only performs array flattening, set construction, and filtering on the supplied preset metadata.
 * @example
 * const presets = [
 *   { facets: { family: ['wgmma', 'experimental'] } },
 *   { facets: { family: ['mma'] } },
 *   { facets: { family: ['wgmma'] } },
 * ] as ComposeLayoutPreset[];
 * const field = { key: 'family', values: ['mma', 'wgmma'] } as ComposeLayoutPresetField;
 *
 * uniquePresetFacetValues(presets, field);
 * // => ['mma', 'wgmma', 'experimental']
 */
function uniquePresetFacetValues(presets: ComposeLayoutPreset[], field: ComposeLayoutPresetField): string[] {
    const values = new Set(presets.flatMap((preset) => preset.facets[field.key] ?? []));
    // catalog-provided values define display order; contributed values still
    // appear, but after known values so existing UX stays stable.
    return [
        ...field.values.filter((value) => values.has(value)),
        ...Array.from(values).filter((value) => !field.values.includes(value)),
    ];
}

/**
 * Checks whether the current selector value satisfies the facet requirements declared by one compose-layout preset.
 *
 * @param preset - Candidate preset whose `facets` map lists the allowed values for each selector key.
 * @param selection - Current user selector state keyed by facet name.
 * @param key - Facet key to validate, such as an instruction family, operation, or input-shape selector.
 * @returns `true` when the preset has no values for the facet and the user left it blank, or when the selected value is one of the preset's allowed values.
 * @noThrows Missing facet and selection entries are normalized to empty arrays or empty strings before comparison.
 * @example
 * const preset = { facets: { family: ['mma', 'wgmma'], dtype: [] } } as ComposeLayoutPreset;
 *
 * presetFieldIsComplete(preset, { family: 'wgmma' }, 'family');
 * // => true
 *
 * presetFieldIsComplete(preset, { family: 'unknown' }, 'family');
 * // => false
 *
 * presetFieldIsComplete(preset, { dtype: '' }, 'dtype');
 * // => true
 */
function presetFieldIsComplete(
    preset: ComposeLayoutPreset,
    selection: ComposeLayoutPresetSelection,
    key: string,
): boolean {
    const values = preset.facets[key] ?? [];
    const selected = selection[key] ?? '';
    return values.length === 0 ? selected === '' : values.includes(selected);
}

/**
 * Converts compose-layout specs text to the same canonical formatting used when matching editor state against preset catalog entries.
 *
 * @param specsText - Raw compose-layout specs text from a preset definition or the sidebar editor.
 * @returns Formatted specs text when parsing succeeds; otherwise the original text with Windows newlines converted to `\n` and surrounding whitespace trimmed.
 * @noThrows Parser or formatter failures are caught so malformed in-progress editor text can still be compared and displayed as normalized raw text.
 * @example
 * canonicalLayoutSpecsText('  not valid compose layout\r\n');
 * // => 'not valid compose layout'
 */
function canonicalLayoutSpecsText(specsText: string): string {
    try {
        return formatSpecsText(parseLayoutSpecs(specsText));
    } catch {
        return specsText.replace(/\r\n/g, '\n').trim();
    }
}
