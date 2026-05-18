import {
    GPU_ARCHS_SM70_PLUS,
    GPU_ARCHS_SM75_PLUS,
    GPU_ARCHS_SM80_PLUS,
    GPU_ARCHS_SM90_PLUS,
    GPU_ARCHS_SM120_ONLY,
} from './gpu-archs.js';
import type { ComposeLayoutPresetDefinition } from './types.js';

// this file is deliberately data-heavy: mma layout behavior should be added by
// editing preset rows, not by changing preset filtering or widget logic.
// the model layer reads the facets below and automatically derives the visible
// selector fields, option lists, and unique selection matching.
// `T` means thread id bits.
// `R` means register-lane bits.
// `M`, `N`, and `K` are the logical matrix axes used by the instruction name.
// row-major and col-major suffixes are represented as operand facet values so
// the preset widget can disambiguate them without instruction-specific code.
// old text presets are kept for the sm_70 m8n8k4 family because those were
// authored before named row presets existed.
// new entries should prefer named presets with `signature` and `rows` so the
// notation stays compact and close to ISA tables.
// b1, b4, b8, b16, and b32 are layout-width aliases, not complete dtype
// semantics; entry comments explain the concrete semantic families when needed.
// `mmaGpuArchs` is the only architecture inference in this file.
// do not duplicate a preset just to list another compatible architecture.
// if a future instruction has different selector fields, add a new preset
// family under presets/index.ts instead of putting widget branches here.
// matrix-size groups below are ordered by first supported architecture, then by
// operand order A/B/C/D, which makes external review diff-friendly.
// accumulator C and output D often share the same layout; both entries remain
// explicit because they are different selector outcomes in the UI.
// f64 support moves later than f16/f32 for several shapes; the architecture
// helper encodes that rule so data rows stay readable.
// sm_120-only f16 m16n8k32 is intentionally narrow because earlier chips expose
// different forms for the same nominal dimensions.
// when adding an AMD-style family later, do not extend this file unless the
// instruction is genuinely NVIDIA mma-compatible.
// add comments to unusual rows when the basis vector contains a zero placeholder
// or a cross-axis term that would surprise a reader copying from ISA docs.
// the checker enforces comment density here because preset files are the most
// likely external contribution surface.
// keep this comment as a reading guide rather than a source of truth; the data
// rows and invariant tests are the source of truth.
// examples of selector outcomes:
// gpuArch=sm_70, instruction=mma, matrixSize=m8n8k4, dtype=f16, operand=A-row-major.
// gpuArch=sm_75, instruction=mma, matrixSize=m8n8k16, dtype=b8, operand=B.
// gpuArch=sm_80, instruction=mma, matrixSize=m16n8k8, dtype=tf32, operand=A.
// gpuArch=sm_90, instruction=mma, matrixSize=m16n8k4, dtype=f64, operand=D.
// gpuArch=sm_120, instruction=mma, matrixSize=m16n8k32, dtype=f16, operand=A.
// review checklist for new rows:
// confirm the name encodes the same matrix size, operand, and dtype as facets.
// confirm every input label in the signature has exactly one row.
// confirm each basis vector has one entry per output axis.
// confirm C/D accumulator rows are both present when the ISA exposes both.
// confirm the architecture helper returns the broadest correct compatible set.
// confirm comments explain dtype aliases that would otherwise be ambiguous.
// confirm generated specs still pass parser and preset invariant tests.
// confirm the row order follows the bit significance used by the ISA table.
// confirm examples in ARCHITECTURE.md do not need a new field explanation.
// group map:
// sm_70 m8n8k4 f16/f32 text presets cover original 8x8 tensor-core shapes.
// sm_70 m8n8k4 f64 text presets use a later architecture range in the helper.
// sm_75 m8n8k16 b8/s32 covers byte element input with s32 accumulators.
// sm_75 m8n8k32 b4/s32 covers nibble element input with s32 accumulators.
// sm_80 m8n8k128 b1/s32 covers bit element input with s32 accumulators.
// sm_80 m16n8k4 tf32/f64 covers the first 16x8 input/output variants.
// sm_80 m16n8k8 b16/tf32/f64 covers mixed element widths for 16x8x8.
// sm_80 m16n8k16 b8/b16/f16/f64 covers byte and half-width 16x8x16 forms.
// sm_80 m16n8k32 b4/b8/f16 covers wider K forms with several dtype aliases.
// sm_80 m16n8k64 b2/b4 covers two-bit and four-bit element forms.
// sm_80 m16n8k128 b1/b2 covers bit and two-bit element forms.
// sm_80 m16n8k256 b1 covers the widest K bit form in this table.
// C and D rows remain separate even when their basis rows are equal.
// A and B rows may target [M,K], [K,N], or [N,K] depending on ISA layout.
// comments array entries become notation comments, not TypeScript comments.
// facets are normalized by linear-layout-preset-model.ts before the widget sees
// them, so do not pre-expand selector aliases in this file.
// the raw definitions are intentionally `satisfies`-checked before facet
// expansion to catch malformed rows while preserving literal selector values.
// the final export maps every raw row to a normalized preset with gpu arch
// compatibility attached.
// avoid helper functions here unless they remove repeated row logic across at
// least three groups; otherwise keep the notation local to the data row.
// rows copied from documents should stay as strings to avoid changing layout
// meaning through formatter rewrites.
// if a row needs a computed expression, prefer moving the whole family to a
// generator like swizzle.ts rather than mixing computation into this table.
// the goal is that an external contributor can add one row and rely on tests to
// tell them whether selectors, parser notation, and matching are coherent.
// common review mistakes:
// forgetting to add both C and D accumulator entries.
// using a title that does not match the generated layout name.
// listing one concrete gpuArch where an array-valued facet is expected.
// copying row-major A rows into B rows without swapping output axes.
// omitting comments for dtype aliases such as b4 or b16.
// treating `comments` as a TypeScript review note instead of user-visible text.
// adding a selector value that is not reachable from DEFAULT_PRESET_FIELDS.
// changing `mmaGpuArchs` in a way that narrows existing presets.
// mixing AMD terminology into NVIDIA mma data.
// adding a helper for two entries when repeated rows would be clearer inline.
// removing old text presets before saved snapshots have a migration path.
// putting a widget dependency in this data file.

/**
 * Builds the hand-written compose-layout text block stored on MMA presets whose
 * signature and basis rows are clearer as notation than as structured row data.
 *
 * @param signature - First line of the preset text, including the layout name and input-to-output axis signature.
 * @param rows - Compose-layout row definitions, such as `T: [[1,0],[2,0]]`, appended in display order after the signature.
 * @returns Newline-delimited preset text that can be assigned to a preset's `specsText` field.
 * @noThrows The helper only concatenates caller-provided strings with newline separators; it performs no parsing or validation.
 * @example
 * layoutSpecText('MMA_m8n8k4_A_row_major_f16: [T,R] -> [M,K]', [
 *     'T: [[1,0],[2,0],[0,0],[0,0],[4,0]]',
 *     'R: [[0,1],[0,2]]',
 * ]);
 * // 'MMA_m8n8k4_A_row_major_f16: [T,R] -> [M,K]\nT: [[1,0],[2,0],[0,0],[0,0],[4,0]]\nR: [[0,1],[0,2]]'
 */
function layoutSpecText(signature: string, rows: string[]): string {
    return [signature, ...rows].join('\n');
}

const MMA_M8N8K4_A_ROW_MAJOR_F16_TEXT = layoutSpecText('MMA_m8n8k4_A_row_major_f16: [T,R] -> [M,K]', [
    'T: [[1,0],[2,0],[0,0],[0,0],[4,0]]',
    'R: [[0,1],[0,2]]',
]);
const MMA_M8N8K4_A_COL_MAJOR_F16_TEXT = layoutSpecText('MMA_m8n8k4_A_col_major_f16: [T,R] -> [M,K]', [
    'T: [[0,1],[0,2],[0,0],[0,0],[4,0]]',
    'R: [[1,0],[2,0]]',
]);
const MMA_M8N8K4_B_ROW_MAJOR_F16_TEXT = layoutSpecText('MMA_m8n8k4_B_row_major_f16: [T,R] -> [K,N]', [
    'T: [[1,0],[2,0],[0,0],[0,0],[0,4]]',
    'R: [[0,1],[0,2]]',
]);
const MMA_M8N8K4_B_COL_MAJOR_F16_TEXT = layoutSpecText('MMA_m8n8k4_B_col_major_f16: [T,R] -> [K,N]', [
    'T: [[0,1],[0,2],[0,0],[0,0],[0,4]]',
    'R: [[1,0],[2,0]]',
]);
const MMA_M8N8K4_C_F16_TEXT = layoutSpecText('MMA_m8n8k4_C_f16: [T,R] -> [M,N]', [
    'T: [[1,0],[2,0],[0,0],[0,0],[4,0]]',
    'R: [[0,1],[0,2],[0,4]]',
]);
const MMA_M8N8K4_D_F16_TEXT = layoutSpecText('MMA_m8n8k4_D_f16: [T,R] -> [M,N]', [
    'T: [[1,0],[2,0],[0,0],[0,0],[4,0]]',
    'R: [[0,1],[0,2],[0,4]]',
]);
const MMA_M8N8K4_C_F32_TEXT = layoutSpecText('MMA_m8n8k4_C_f32: [T,R] -> [M,N]', [
    'T: [[1,0],[0,2],[0,0],[0,0],[4,0]]',
    'R: [[0,1],[2,0],[0,4]]',
]);
const MMA_M8N8K4_D_F32_TEXT = layoutSpecText('MMA_m8n8k4_D_f32: [T,R] -> [M,N]', [
    'T: [[1,0],[0,2],[0,0],[0,0],[4,0]]',
    'R: [[0,1],[2,0],[0,4]]',
]);
const MMA_M8N8K4_A_F64_TEXT = layoutSpecText('MMA_m8n8k4_A_f64: [T,R] -> [M,K]', [
    'T: [[0,1],[0,2],[1,0],[2,0],[4,0]]',
    'R: []',
]);
const MMA_M8N8K4_B_F64_TEXT = layoutSpecText('MMA_m8n8k4_B_f64: [T,R] -> [K,N]', [
    'T: [[1,0],[2,0],[0,1],[0,2],[0,4]]',
    'R: []',
]);
const MMA_M8N8K4_C_F64_TEXT = layoutSpecText('MMA_m8n8k4_C_f64: [T,R] -> [M,N]', [
    'T: [[0,2],[0,4],[1,0],[2,0],[4,0]]',
    'R: [[0,1]]',
]);
const MMA_M8N8K4_D_F64_TEXT = layoutSpecText('MMA_m8n8k4_D_f64: [T,R] -> [M,N]', [
    'T: [[0,2],[0,4],[1,0],[2,0],[4,0]]',
    'R: [[0,1]]',
]);
const MMA_M8N8K16_A_B8_TEXT = layoutSpecText('MMA_m8n8k16_A_b8: [T,R] -> [M,K]', [
    'T: [[0,4],[0,8],[1,0],[2,0],[0,0]]',
    'R: [[0,1],[0,2]]',
]);
const MMA_M8N8K16_B_B8_TEXT = layoutSpecText('MMA_m8n8k16_B_b8: [T,R] -> [N,K]', [
    'T: [[0,4],[0,8],[1,0],[2,0],[0,0]]',
    'R: [[1,0],[8,0]]',
]);
const MMA_M8N8K16_C_S32_TEXT = layoutSpecText('MMA_m8n8k16_C_s32: [T,R] -> [M,N]', [
    'T: [[1,0],[2,0],[0,1],[0,2],[0,0]]',
    'R: [[4,0]]',
]);
const MMA_M8N8K16_D_S32_TEXT = layoutSpecText('MMA_m8n8k16_D_s32: [T,R] -> [M,N]', [
    'T: [[1,0],[2,0],[0,1],[0,2],[0,0]]',
    'R: [[4,0]]',
]);

const MMA_RAW_PRESET_DEFINITIONS = [
    {
        title: 'MMA_m8n8k4_A_row_major_f16',
        gpuArch: 'sm_70',
        instruction: 'mma',
        matrixSize: 'm8n8k4',
        dtype: 'f16',
        operand: 'A-row-major',
        specsText: MMA_M8N8K4_A_ROW_MAJOR_F16_TEXT,
        inputName: 'Hardware Layout',
    },
    {
        title: 'MMA_m8n8k4_A_col_major_f16',
        gpuArch: 'sm_70',
        instruction: 'mma',
        matrixSize: 'm8n8k4',
        dtype: 'f16',
        operand: 'A-col-major',
        specsText: MMA_M8N8K4_A_COL_MAJOR_F16_TEXT,
        inputName: 'Hardware Layout',
    },
    {
        title: 'MMA_m8n8k4_B_row_major_f16',
        gpuArch: 'sm_70',
        instruction: 'mma',
        matrixSize: 'm8n8k4',
        dtype: 'f16',
        operand: 'B-row-major',
        specsText: MMA_M8N8K4_B_ROW_MAJOR_F16_TEXT,
        inputName: 'Hardware Layout',
    },
    {
        title: 'MMA_m8n8k4_B_col_major_f16',
        gpuArch: 'sm_70',
        instruction: 'mma',
        matrixSize: 'm8n8k4',
        dtype: 'f16',
        operand: 'B-col-major',
        specsText: MMA_M8N8K4_B_COL_MAJOR_F16_TEXT,
        inputName: 'Hardware Layout',
    },
    {
        title: 'MMA_m8n8k4_C_f16',
        gpuArch: 'sm_70',
        instruction: 'mma',
        matrixSize: 'm8n8k4',
        dtype: 'f16',
        operand: 'C',
        specsText: MMA_M8N8K4_C_F16_TEXT,
        inputName: 'Hardware Layout',
    },
    {
        title: 'MMA_m8n8k4_D_f16',
        gpuArch: 'sm_70',
        instruction: 'mma',
        matrixSize: 'm8n8k4',
        dtype: 'f16',
        operand: 'D',
        specsText: MMA_M8N8K4_D_F16_TEXT,
        inputName: 'Hardware Layout',
    },
    {
        title: 'MMA_m8n8k4_C_f32',
        gpuArch: 'sm_70',
        instruction: 'mma',
        matrixSize: 'm8n8k4',
        dtype: 'f32',
        operand: 'C',
        specsText: MMA_M8N8K4_C_F32_TEXT,
        inputName: 'Hardware Layout',
    },
    {
        title: 'MMA_m8n8k4_D_f32',
        gpuArch: 'sm_70',
        instruction: 'mma',
        matrixSize: 'm8n8k4',
        dtype: 'f32',
        operand: 'D',
        specsText: MMA_M8N8K4_D_F32_TEXT,
        inputName: 'Hardware Layout',
    },
    {
        title: 'MMA_m8n8k4_A_f64',
        gpuArch: 'sm_70',
        instruction: 'mma',
        matrixSize: 'm8n8k4',
        dtype: 'f64',
        operand: 'A',
        specsText: MMA_M8N8K4_A_F64_TEXT,
        inputName: 'Hardware Layout',
    },
    {
        title: 'MMA_m8n8k4_B_f64',
        gpuArch: 'sm_70',
        instruction: 'mma',
        matrixSize: 'm8n8k4',
        dtype: 'f64',
        operand: 'B',
        specsText: MMA_M8N8K4_B_F64_TEXT,
        inputName: 'Hardware Layout',
    },
    {
        title: 'MMA_m8n8k4_C_f64',
        gpuArch: 'sm_70',
        instruction: 'mma',
        matrixSize: 'm8n8k4',
        dtype: 'f64',
        operand: 'C',
        specsText: MMA_M8N8K4_C_F64_TEXT,
        inputName: 'Hardware Layout',
    },
    {
        title: 'MMA_m8n8k4_D_f64',
        gpuArch: 'sm_70',
        instruction: 'mma',
        matrixSize: 'm8n8k4',
        dtype: 'f64',
        operand: 'D',
        specsText: MMA_M8N8K4_D_F64_TEXT,
        inputName: 'Hardware Layout',
    },
    {
        title: 'MMA_m8n8k16_A_b8',
        gpuArch: 'sm_75',
        instruction: 'mma',
        matrixSize: 'm8n8k16',
        dtype: 'b8',
        operand: 'A',
        comments: ['b8 represents 8-bit elements (u8/s8).'],
        specsText: MMA_M8N8K16_A_B8_TEXT,
        inputName: 'Hardware Layout',
    },
    {
        title: 'MMA_m8n8k16_B_b8',
        gpuArch: 'sm_75',
        instruction: 'mma',
        matrixSize: 'm8n8k16',
        dtype: 'b8',
        operand: 'B',
        comments: ['b8 represents 8-bit elements (u8/s8).'],
        specsText: MMA_M8N8K16_B_B8_TEXT,
        inputName: 'Hardware Layout',
    },
    {
        title: 'MMA_m8n8k16_C_s32',
        gpuArch: 'sm_75',
        instruction: 'mma',
        matrixSize: 'm8n8k16',
        dtype: 's32',
        operand: 'C',
        specsText: MMA_M8N8K16_C_S32_TEXT,
        inputName: 'Hardware Layout',
    },
    {
        title: 'MMA_m8n8k16_D_s32',
        gpuArch: 'sm_75',
        instruction: 'mma',
        matrixSize: 'm8n8k16',
        dtype: 's32',
        operand: 'D',
        specsText: MMA_M8N8K16_D_S32_TEXT,
        inputName: 'Hardware Layout',
    },
    {
        name: 'MMA_m8n8k32_A_b4',
        gpuArch: 'sm_75',
        instruction: 'mma',
        matrixSize: 'm8n8k32',
        dtype: 'b4',
        operand: 'A',
        comments: ['b4 represents 4-bit elements (u4/s4).'],
        signature: '[T,R] -> [M,K]',
        rows: [['T', '[[0,8],[0,16],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[0,2],[0,4]]']],
    },
    {
        name: 'MMA_m8n8k32_B_b4',
        gpuArch: 'sm_75',
        instruction: 'mma',
        matrixSize: 'm8n8k32',
        dtype: 'b4',
        operand: 'B',
        comments: ['b4 represents 4-bit elements (u4/s4).'],
        signature: '[T,R] -> [K,N]',
        rows: [['T', '[[8,0],[16,0],[0,1],[0,2],[0,4]]'], ['R', '[[1,0],[2,0],[4,0]]']],
    },
    {
        name: 'MMA_m8n8k32_C_s32',
        gpuArch: 'sm_75',
        instruction: 'mma',
        matrixSize: 'm8n8k32',
        dtype: 's32',
        operand: 'C',
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1]]']],
    },
    {
        name: 'MMA_m8n8k32_D_s32',
        gpuArch: 'sm_75',
        instruction: 'mma',
        matrixSize: 'm8n8k32',
        dtype: 's32',
        operand: 'D',
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1]]']],
    },
    {
        name: 'MMA_m8n8k128_A_b1',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm8n8k128',
        dtype: 'b1',
        operand: 'A',
        signature: '[T,R] -> [M,K]',
        rows: [['T', '[[0,32],[0,64],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[0,2],[0,4],[0,8],[0,16]]']],
    },
    {
        name: 'MMA_m8n8k128_B_b1',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm8n8k128',
        dtype: 'b1',
        operand: 'B',
        signature: '[T,R] -> [K,N]',
        rows: [['T', '[[32,0],[64,0],[0,1],[0,2],[0,4]]'], ['R', '[[1,0],[2,0],[4,0],[8,0],[16,0]]']],
    },
    {
        name: 'MMA_m8n8k128_C_s32',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm8n8k128',
        dtype: 's32',
        operand: 'C',
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1]]']],
    },
    {
        name: 'MMA_m8n8k128_D_s32',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm8n8k128',
        dtype: 's32',
        operand: 'D',
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1]]']],
    },
    {
        name: 'MMA_m16n8k4_A_tf32',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k4',
        dtype: 'tf32',
        operand: 'A',
        signature: '[T,R] -> [M,K]',
        rows: [['T', '[[0,1],[0,2],[1,0],[2,0],[4,0]]'], ['R', '[[8,0]]']],
    },
    {
        name: 'MMA_m16n8k4_A_f64',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k4',
        dtype: 'f64',
        operand: 'A',
        signature: '[T,R] -> [M,K]',
        rows: [['T', '[[0,1],[0,2],[1,0],[2,0],[4,0]]'], ['R', '[[8,0]]']],
    },
    {
        name: 'MMA_m16n8k4_B_tf32',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k4',
        dtype: 'tf32',
        operand: 'B',
        signature: '[T] -> [K,N]',
        rows: [['T', '[[1,0],[2,0],[0,1],[0,2],[0,4]]']],
    },
    {
        name: 'MMA_m16n8k4_B_f64',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k4',
        dtype: 'f64',
        operand: 'B',
        signature: '[T] -> [K,N]',
        rows: [['T', '[[1,0],[2,0],[0,1],[0,2],[0,4]]']],
    },
    {
        name: 'MMA_m16n8k4_C_f32',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k4',
        dtype: 'f32',
        operand: 'C',
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k4_D_f32',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k4',
        dtype: 'f32',
        operand: 'D',
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k4_C_f64',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k4',
        dtype: 'f64',
        operand: 'C',
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k4_D_f64',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k4',
        dtype: 'f64',
        operand: 'D',
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k8_A_b16',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k8',
        dtype: 'b16',
        operand: 'A',
        comments: ['b16 represents 16-bit elements (f16/bf16).'],
        signature: '[T,R] -> [M,K]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k8_A_tf32',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k8',
        dtype: 'tf32',
        operand: 'A',
        signature: '[T,R] -> [M,K]',
        rows: [['T', '[[0,1],[0,2],[1,0],[2,0],[4,0]]'], ['R', '[[8,0],[0,4]]']],
    },
    {
        name: 'MMA_m16n8k8_A_f64',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k8',
        dtype: 'f64',
        operand: 'A',
        signature: '[T,R] -> [M,K]',
        rows: [['T', '[[0,1],[0,2],[1,0],[2,0],[4,0]]'], ['R', '[[8,0],[0,4]]']],
    },
    {
        name: 'MMA_m16n8k8_B_b16',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k8',
        dtype: 'b16',
        operand: 'B',
        comments: ['b16 represents 16-bit elements (f16/bf16).'],
        signature: '[T,R] -> [K,N]',
        rows: [['T', '[[2,0],[4,0],[0,1],[0,2],[0,4]]'], ['R', '[[1,0]]']],
    },
    {
        name: 'MMA_m16n8k8_B_tf32',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k8',
        dtype: 'tf32',
        operand: 'B',
        signature: '[T,R] -> [K,N]',
        rows: [['T', '[[1,0],[2,0],[0,1],[0,2],[0,4]]'], ['R', '[[4,0]]']],
    },
    {
        name: 'MMA_m16n8k8_B_f64',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k8',
        dtype: 'f64',
        operand: 'B',
        signature: '[T,R] -> [K,N]',
        rows: [['T', '[[1,0],[2,0],[0,1],[0,2],[0,4]]'], ['R', '[[4,0]]']],
    },
    {
        name: 'MMA_m16n8k8_C_b16',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k8',
        dtype: 'b16',
        operand: 'C',
        comments: ['b16 represents 16-bit elements (f16/bf16).'],
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k8_D_b16',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k8',
        dtype: 'b16',
        operand: 'D',
        comments: ['b16 represents 16-bit elements (f16/bf16).'],
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k8_C_b32',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k8',
        dtype: 'b32',
        operand: 'C',
        comments: ['b32 represents 32-bit elements (f32/tf32).'],
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k8_D_b32',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k8',
        dtype: 'b32',
        operand: 'D',
        comments: ['b32 represents 32-bit elements (f32/tf32).'],
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k8_C_f64',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k8',
        dtype: 'f64',
        operand: 'C',
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k8_D_f64',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k8',
        dtype: 'f64',
        operand: 'D',
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k16_A_b16',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k16',
        dtype: 'b16',
        operand: 'A',
        comments: ['b16 represents 16-bit elements (f16/bf16).'],
        signature: '[T,R] -> [M,K]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0],[0,8]]']],
    },
    {
        name: 'MMA_m16n8k16_A_f64',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k16',
        dtype: 'f64',
        operand: 'A',
        signature: '[T,R] -> [M,K]',
        rows: [['T', '[[0,1],[0,2],[1,0],[2,0],[4,0]]'], ['R', '[[8,0],[0,4],[0,8]]']],
    },
    {
        name: 'MMA_m16n8k16_A_b8',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k16',
        dtype: 'b8',
        operand: 'A',
        comments: ['b8 represents 8-bit elements (u8/s8/e4m3/e5m2).'],
        signature: '[T,R] -> [M,K]',
        rows: [['T', '[[0,4],[0,8],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[0,2],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k16_B_b16',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k16',
        dtype: 'b16',
        operand: 'B',
        comments: ['b16 represents 16-bit elements (f16/bf16).'],
        signature: '[T,R] -> [K,N]',
        rows: [['T', '[[2,0],[4,0],[0,1],[0,2],[0,4]]'], ['R', '[[1,0],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k16_B_f64',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k16',
        dtype: 'f64',
        operand: 'B',
        signature: '[T,R] -> [K,N]',
        rows: [['T', '[[1,0],[2,0],[0,1],[0,2],[0,4]]'], ['R', '[[4,0],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k16_B_b8',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k16',
        dtype: 'b8',
        operand: 'B',
        comments: ['b8 represents 8-bit elements (u8/s8/e4m3/e5m2).'],
        signature: '[T,R] -> [K,N]',
        rows: [['T', '[[4,0],[8,0],[0,1],[0,2],[0,4]]'], ['R', '[[1,0],[2,0]]']],
    },
    {
        name: 'MMA_m16n8k16_C_f16',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k16',
        dtype: 'f16',
        operand: 'C',
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k16_D_f16',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k16',
        dtype: 'f16',
        operand: 'D',
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k16_C_b32',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k16',
        dtype: 'b32',
        operand: 'C',
        comments: ['b32 represents 32-bit elements (f32/s32).'],
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k16_D_b32',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k16',
        dtype: 'b32',
        operand: 'D',
        comments: ['b32 represents 32-bit elements (f32/s32).'],
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k16_C_f64',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k16',
        dtype: 'f64',
        operand: 'C',
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k16_D_f64',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k16',
        dtype: 'f64',
        operand: 'D',
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k32_A_b4',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k32',
        dtype: 'b4',
        operand: 'A',
        comments: ['b4 represents 4-bit elements (u4/s4/e2m1).'],
        signature: '[T,R] -> [M,K]',
        rows: [['T', '[[0,8],[0,16],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[0,2],[0,4],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k32_A_b8',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k32',
        dtype: 'b8',
        operand: 'A',
        comments: ['b8 represents 8-bit elements (u8/s8/e4m3/e5m2/e3m2/e2m3).'],
        signature: '[T,R] -> [M,K]',
        rows: [['T', '[[0,4],[0,8],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[0,2],[8,0],[0,16]]']],
    },
    {
        name: 'MMA_m16n8k32_B_b4',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k32',
        dtype: 'b4',
        operand: 'B',
        comments: ['b4 represents 4-bit elements (u4/s4/e2m1).'],
        signature: '[T,R] -> [K,N]',
        rows: [['T', '[[8,0],[16,0],[0,1],[0,2],[0,4]]'], ['R', '[[1,0],[2,0],[4,0]]']],
    },
    {
        name: 'MMA_m16n8k32_B_b8',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k32',
        dtype: 'b8',
        operand: 'B',
        comments: ['b8 represents 8-bit elements (u8/s8/e4m3/e5m2/e3m2/e2m3).'],
        signature: '[T,R] -> [K,N]',
        rows: [['T', '[[4,0],[8,0],[0,1],[0,2],[0,4]]'], ['R', '[[1,0],[2,0],[16,0]]']],
    },
    {
        name: 'MMA_m16n8k32_C_f16',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k32',
        dtype: 'f16',
        operand: 'C',
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k32_D_f16',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k32',
        dtype: 'f16',
        operand: 'D',
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k32_C_b32',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k32',
        dtype: 'b32',
        operand: 'C',
        comments: ['b32 represents 32-bit elements (f32/s32).'],
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k32_D_b32',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k32',
        dtype: 'b32',
        operand: 'D',
        comments: ['b32 represents 32-bit elements (f32/s32).'],
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k64_A_b4',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k64',
        dtype: 'b4',
        operand: 'A',
        comments: ['b4 represents 4-bit elements (u4/s4/e2m1).'],
        signature: '[T,R] -> [M,K]',
        rows: [['T', '[[0,8],[0,16],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[0,2],[0,4],[8,0],[0,32]]']],
    },
    {
        name: 'MMA_m16n8k64_B_b4',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k64',
        dtype: 'b4',
        operand: 'B',
        comments: ['b4 represents 4-bit elements (u4/s4/e2m1).'],
        signature: '[T,R] -> [K,N]',
        rows: [['T', '[[8,0],[16,0],[0,1],[0,2],[0,4]]'], ['R', '[[1,0],[2,0],[4,0],[32,0]]']],
    },
    {
        name: 'MMA_m16n8k64_C_b32',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k64',
        dtype: 'b32',
        operand: 'C',
        comments: ['b32 represents 32-bit elements (f32/s32).'],
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k64_D_b32',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k64',
        dtype: 'b32',
        operand: 'D',
        comments: ['b32 represents 32-bit elements (f32/s32).'],
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k128_A_b1',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k128',
        dtype: 'b1',
        operand: 'A',
        signature: '[T,R] -> [M,K]',
        rows: [['T', '[[0,32],[0,64],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[0,2],[0,4],[0,8],[0,16],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k128_B_b1',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k128',
        dtype: 'b1',
        operand: 'B',
        signature: '[T,R] -> [K,N]',
        rows: [['T', '[[32,0],[64,0],[0,1],[0,2],[0,4]]'], ['R', '[[1,0],[2,0],[4,0],[8,0],[16,0]]']],
    },
    {
        name: 'MMA_m16n8k128_C_s32',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k128',
        dtype: 's32',
        operand: 'C',
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k128_D_s32',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k128',
        dtype: 's32',
        operand: 'D',
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k256_B_b1',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k256',
        dtype: 'b1',
        operand: 'B',
        signature: '[T,R] -> [K,N]',
        rows: [['T', '[[32,0],[64,0],[0,1],[0,2],[0,4]]'], ['R', '[[1,0],[2,0],[4,0],[8,0],[16,0],[128,0]]']],
    },
    {
        name: 'MMA_m16n8k256_C_s32',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k256',
        dtype: 's32',
        operand: 'C',
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k256_D_s32',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k256',
        dtype: 's32',
        operand: 'D',
        signature: '[T,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[8,0]]']],
    },
    {
        name: 'MMA_m16n8k256_A_b1',
        gpuArch: 'sm_80',
        instruction: 'mma',
        matrixSize: 'm16n8k256',
        dtype: 'b1',
        operand: 'A',
        signature: '[T,R] -> [M,K]',
        rows: [['T', '[[0,32],[0,64],[1,0],[2,0],[4,0]]'], ['R', '[[0,1],[0,2],[0,4],[0,8],[0,16],[8,0],[0,128]]']],
    },
] satisfies ComposeLayoutPresetDefinition[];

/**
 * Infers the NVIDIA SM architecture selector values for an MMA preset from its
 * matrix shape and element-width family so the catalog can keep compatibility
 * rules in one place.
 *
 * @param preset - MMA preset definition whose legacy `matrixSize` and optional `dtype` fields identify the instruction variant.
 * @returns Read-only GPU architecture values used for the preset's `facets.gpuArch` selector.
 * @noThrows Missing `matrixSize` or `dtype` fields are treated as empty strings, and the helper only compares those strings against known MMA cases.
 * @example
 * mmaGpuArchs({ matrixSize: 'm8n8k4', dtype: 'f64' } as ComposeLayoutPresetDefinition);
 * // GPU_ARCHS_SM80_PLUS
 *
 * mmaGpuArchs({ matrixSize: 'm16n8k32', dtype: 'f16' } as ComposeLayoutPresetDefinition);
 * // GPU_ARCHS_SM120_ONLY
 */
function mmaGpuArchs(preset: ComposeLayoutPresetDefinition): readonly string[] {
    const matrixSize = preset.matrixSize ?? '';
    const dtype = preset.dtype ?? '';
    if (matrixSize === 'm8n8k4') return dtype === 'f64' ? GPU_ARCHS_SM80_PLUS : GPU_ARCHS_SM70_PLUS;
    if (matrixSize === 'm8n8k16' || matrixSize === 'm8n8k32' || matrixSize === 'm8n8k128') return GPU_ARCHS_SM75_PLUS;
    if (matrixSize === 'm16n8k4') return dtype === 'f64' ? GPU_ARCHS_SM90_PLUS : GPU_ARCHS_SM80_PLUS;
    if (matrixSize === 'm16n8k8') return dtype === 'b16' ? GPU_ARCHS_SM75_PLUS : dtype === 'f64' ? GPU_ARCHS_SM90_PLUS : GPU_ARCHS_SM80_PLUS;
    if (matrixSize === 'm16n8k16') return dtype === 'f64' ? GPU_ARCHS_SM90_PLUS : GPU_ARCHS_SM80_PLUS;
    if (matrixSize === 'm16n8k32') return dtype === 'f16' ? GPU_ARCHS_SM120_ONLY : GPU_ARCHS_SM80_PLUS;
    return GPU_ARCHS_SM80_PLUS;
}

export const MMA_PRESET_DEFINITIONS: ComposeLayoutPresetDefinition[] = MMA_RAW_PRESET_DEFINITIONS.map((preset) => ({
    ...preset,
    facets: {
        gpuArch: mmaGpuArchs(preset),
        instruction: preset.instruction ?? 'mma',
        matrixSize: preset.matrixSize ?? '',
        dtype: preset.dtype ?? '',
        operand: preset.operand ?? '',
    },
}));
