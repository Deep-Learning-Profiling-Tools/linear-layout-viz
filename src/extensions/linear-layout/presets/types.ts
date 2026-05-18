/**
 * Legacy selector fields that older in-tree preset definitions may place at the
 * top level before catalog normalization copies them into explicit `facets`.
 *
 * New preset families should prefer `facets`: the selector UI only needs to
 * know which facet values identify a preset, so keeping that data explicit
 * prevents widget code from growing instruction-specific branches.
 *
 * @example
 * const fields: ComposeLayoutPresetFields = {
 *     gpuArch: 'sm_80',
 *     instruction: 'mma',
 *     matrixSize: 'm16n8k16',
 *     dtype: 'f16',
 *     operand: 'A',
 *     facets: {
 *         gpuArch: ['sm_80', 'sm_90'],
 *         instruction: 'mma',
 *         matrixSize: 'm16n8k16',
 *         dtype: 'f16',
 *         operand: 'A',
 *     },
 * };
 */
export type ComposeLayoutPresetFields = {
    gpuArch?: string;
    instruction?: string;
    matrixSize?: string;
    dtype?: string;
    operand?: string;
    trans?: string;
    major?: string;
    comments?: string[];
    inputName?: string;
    title?: string;
    facets?: ComposeLayoutPresetFacets;
};

/**
 * Selector value stored on a compose-layout preset facet.
 *
 * A string means the preset matches one selector option; a readonly string array
 * means the same preset is available for several concrete options, such as
 * multiple GPU architectures.
 *
 * @example
 * const singleArch: ComposeLayoutPresetFacetValue = 'sm_90';
 * const compatibleArchs: ComposeLayoutPresetFacetValue = ['sm_80', 'sm_90'];
 */
export type ComposeLayoutPresetFacetValue = string | readonly string[];

/**
 * Selector values that must match before a preset can be chosen.
 *
 * Arrays mean one preset is valid for several selector values, such as a layout
 * that applies to multiple GPU archs. The invariant tests expand these arrays
 * to make sure every concrete selection still resolves to exactly one preset.
 *
 * @example
 * const facets: ComposeLayoutPresetFacets = {
 *     gpuArch: ['sm_80', 'sm_90'],
 *     instruction: 'wgmma',
 *     operand: 'A',
 * };
 *
 * // The preset is considered for either sm_80 or sm_90 when the other
 * // selector values also match.
 * facets.gpuArch;
 */
export type ComposeLayoutPresetFacets = Record<string, ComposeLayoutPresetFacetValue>;

/**
 * Metadata for one text field or dropdown shown by the preset widget.
 *
 * The key must match a facet key on preset definitions. Labels and placeholders
 * are displayed in the sidebar, while ordering and dependencies control when the
 * selector appears.
 *
 * @example
 * const gpuArchField: ComposeLayoutPresetFieldDefinition = {
 *     key: 'gpuArch',
 *     label: 'GPU Arch',
 *     placeholder: 'Type GPU arch',
 *     order: 10,
 *     required: true,
 *     values: ['sm_80', 'sm_90'],
 * };
 *
 * const operandField: ComposeLayoutPresetFieldDefinition = {
 *     key: 'operand',
 *     label: 'Operand',
 *     placeholder: 'Choose operand',
 *     order: 40,
 *     dependsOn: ['instruction'],
 * };
 */
export type ComposeLayoutPresetFieldDefinition = {
    key: string;
    label: string;
    placeholder: string;
    /** lower values render earlier; keep shared hardware concepts before family-specific fields. */
    order: number;
    /** required fields are always visible and must be filled before a preset resolves. */
    required?: boolean;
    /** field-level visibility dependency; value-specific hiding comes from filtered facet options. */
    dependsOn?: string[];
    /** preferred display order for known values; unknown contributed values append after this list. */
    values?: readonly string[];
};

/**
 * Independently reviewable group of compose-layout presets, usually for one GPU
 * instruction family.
 *
 * Families keep instruction-specific preset data and any extra selector fields
 * together so the registry can add them without changing the preset widget.
 *
 * @example
 * const family: ComposeLayoutPresetFamily = {
 *     fields: [
 *         {
 *             key: 'gpuArch',
 *             label: 'GPU Arch',
 *             placeholder: 'Type GPU arch',
 *             order: 10,
 *             required: true,
 *             values: ['sm_90'],
 *         },
 *     ],
 *     presets: [
 *         {
 *             name: 'example_wgmma_A',
 *             facets: { gpuArch: 'sm_90', instruction: 'wgmma', operand: 'A' },
 *             signature: '[T,R] -> [M,K]',
 *             rows: [['T', '[[1,0]]']],
 *             comments: ['Example preset shape; real rows should come from verified hardware mapping.'],
 *         },
 *     ],
 * };
 */
export type ComposeLayoutPresetFamily = {
    fields?: readonly ComposeLayoutPresetFieldDefinition[];
    presets: readonly ComposeLayoutPresetDefinition[];
};

/**
 * Preset catalog entry that stores the complete compose-layout specification as authored text.
 * Use this branch for instruction families whose operation, comments, or basis rows are clearer
 * when kept as one hand-written `specsText` block instead of generated from `name`, `signature`,
 * and `rows` fields.
 *
 * @example
 * const preset: ComposeLayoutPresetTextDefinition = {
 *     facets: {
 *         gpuArch: ['sm90', 'sm100'],
 *         instruction: 'swizzle',
 *         dtype: 'b16',
 *         major: 'MN',
 *     },
 *     inputName: 'Logical Offsets',
 *     specsText: `# swizzle_128B_MN_major_b16
 * swizzle_128B_MN_major_b16 = [M,N] -> [M,N]
 * M: [[1,0],[2,0]]
 * N: [[0,1],[0,2]]`,
 * };
 *
 * preset.specsText.includes('swizzle_128B_MN_major_b16'); // true
 */
export type ComposeLayoutPresetTextDefinition = ComposeLayoutPresetFields & {
    /** complete specs text for layouts that are clearer as hand-written notation. */
    specsText: string;
};

/**
 * Preset catalog entry whose compose-layout specification is generated from an instruction name,
 * signature, optional comments, and labeled basis rows. This is the common form for MMA, WGMMA,
 * ldmatrix, and stmatrix presets where selector facets and compact row tables should live together.
 *
 * @example
 * const preset: ComposeLayoutPresetNamedDefinition = {
 *     name: 'WGMMA_m64n8_D_b32',
 *     facets: {
 *         gpuArch: 'sm90',
 *         instruction: 'wgmma',
 *         matrixSize: 'm64n8',
 *         dtype: 'b32',
 *         operand: 'D',
 *     },
 *     signature: '[W,L] -> [M,N]',
 *     comments: ['Accumulator layout for one WGMMA tile.'],
 *     inputName: 'Hardware Layout',
 *     rows: [
 *         ['W', '[[1,0],[2,0]]'],
 *         ['L', '[[0,1],[0,2]]'],
 *     ],
 * };
 *
 * preset.rows[0]; // ['W', '[[1,0],[2,0]]']
 */
export type ComposeLayoutPresetNamedDefinition = ComposeLayoutPresetFields & {
    /** layout name used both in the generated signature and as the operation text. */
    name: string;
    signature: string;
    comments?: string[];
    /** basis rows kept as strings so preset files stay compact and close to ISA tables. */
    rows: Array<[label: string, bases: string]>;
};

/**
 * Union accepted by instruction-family preset files in the catalog. A preset can either provide
 * hand-written `specsText` directly or provide a named row table that the linear-layout preset
 * model normalizes into the compose-layout text shown in the sidebar.
 *
 * @example
 * const presets: ComposeLayoutPresetDefinition[] = [
 *     {
 *         name: 'AMD_mfma_m32n32k8_A_f16',
 *         facets: {
 *             gpuArch: ['gfx90a', 'gfx942'],
 *             instruction: 'mfma',
 *             matrixSize: 'm32n32k8',
 *             dtype: 'f16',
 *             operand: 'A',
 *         },
 *         signature: '[T,R] -> [M,K]',
 *         rows: [
 *             ['T', '[[1,0],[2,0]]'],
 *             ['R', '[[0,1],[0,2]]'],
 *         ],
 *         inputName: 'Hardware Layout',
 *     },
 *     {
 *         facets: { instruction: 'swizzle', dtype: 'b16', major: 'MN' },
 *         inputName: 'Logical Offsets',
 *         specsText: 'swizzle_128B_MN_major_b16 = [M,N] -> [M,N]',
 *     },
 * ];
 *
 * presets.length; // 2
 */
export type ComposeLayoutPresetDefinition = ComposeLayoutPresetTextDefinition | ComposeLayoutPresetNamedDefinition;

/**
 * Shared ldmatrix/stmatrix table row that is expanded into concrete compose-layout presets for
 * each supported transpose mode. The transfer generators copy the instruction metadata into
 * selector facets and choose `rowsByTrans.no` or `rowsByTrans.yes` when building the row table.
 *
 * @example
 * const layout: MatrixTransferLayoutDefinition = {
 *     name: 'ldmatrix_m8n8_x1_b16',
 *     gpuArch: 'sm75',
 *     instruction: 'ldmatrix',
 *     matrixSize: 'm8n8.x1',
 *     dtype: 'b16',
 *     operand: 'A',
 *     inputName: 'Shared Memory Layout',
 *     rowsByTrans: {
 *         no: [
 *             ['lane', '[[1,0],[2,0]]'],
 *             ['byte', '[[0,1],[0,2]]'],
 *         ],
 *         yes: [
 *             ['lane', '[[0,1],[0,2]]'],
 *             ['byte', '[[1,0],[2,0]]'],
 *         ],
 *     },
 * };
 *
 * layout.rowsByTrans.yes?.[0][0]; // 'lane'
 */
export type MatrixTransferLayoutDefinition = {
    name: string;
    gpuArch: string;
    instruction: 'ldmatrix' | 'stmatrix';
    matrixSize: string;
    dtype: string;
    operand: string;
    inputName: string;
    rowsByTrans: Partial<Record<'no' | 'yes', Array<[label: string, bases: string]>>>;
};
