import { GPU_ARCHS_SM75_PLUS, GPU_ARCHS_SM100_PLUS } from './gpu-archs.js';
import type { ComposeLayoutPresetDefinition, MatrixTransferLayoutDefinition } from './types.js';

// ldmatrix presets start from one table of memory-transfer layouts.
// each table row describes the matrix shape, dtype, input display name, and the
// basis rows for transpose/no-transpose modes.
// the exported preset list expands that table into selector facets so the
// preset widget can filter purely from data.
// `R` and `C` are logical shared-memory row/column coordinates.
// `T` is the hardware thread coordinate emitted by the instruction.
// `R32` is the register-lane coordinate used by tensor-core operands.
// sm_75-plus entries cover the original b16 forms.
// sm_100-plus entries cover newer b8/b4 forms.
// transpose variants only exist when the source table supplies rows for that
// mode; missing rows intentionally hide that selector choice.
// comments attached to generated presets are shown in the notation panel after
// the layout is selected.
// if future architectures add a new transfer shape, add one table row here and
// only split it out if the row-generation rule stops being shared.

export const MATRIX_TRANSFER_LAYOUTS: MatrixTransferLayoutDefinition[] = [
    {
        name: 'ldmatrix_m8n8_x1_b16',
        gpuArch: 'sm_75',
        instruction: 'ldmatrix',
        matrixSize: 'm8n8.x1',
        dtype: 'b16',
        operand: '',
        inputName: 'Shared Memory',
        rowsByTrans: {
            no: [['R', '[[4,0],[8,0],[16,0]]'], ['C', '[[0,0],[1,0],[2,0]]']],
            yes: [['R', '[[0,0],[1,0],[2,0]]'], ['C', '[[4,0],[8,0],[16,0]]']],
        },
    },
    {
        name: 'ldmatrix_m8n8_x2_b16',
        gpuArch: 'sm_75',
        instruction: 'ldmatrix',
        matrixSize: 'm8n8.x2',
        dtype: 'b16',
        operand: '',
        inputName: 'Shared Memory',
        rowsByTrans: {
            no: [['R', '[[4,0],[8,0],[16,0],[0,1]]'], ['C', '[[0,0],[1,0],[2,0]]']],
            yes: [['R', '[[0,0],[1,0],[2,0],[0,1]]'], ['C', '[[4,0],[8,0],[16,0]]']],
        },
    },
    {
        name: 'ldmatrix_m8n8_x4_b16',
        gpuArch: 'sm_75',
        instruction: 'ldmatrix',
        matrixSize: 'm8n8.x4',
        dtype: 'b16',
        operand: '',
        inputName: 'Shared Memory',
        rowsByTrans: {
            no: [['R', '[[4,0],[8,0],[16,0],[0,1],[0,2]]'], ['C', '[[0,0],[1,0],[2,0]]']],
            yes: [['R', '[[0,0],[1,0],[2,0],[0,1],[0,2]]'], ['C', '[[4,0],[8,0],[16,0]]']],
        },
    },
    {
        name: 'ldmatrix_m16n16_x1_b8',
        gpuArch: 'sm_100',
        instruction: 'ldmatrix',
        matrixSize: 'm16n16.x1',
        dtype: 'b8',
        operand: '',
        inputName: 'Shared Memory',
        rowsByTrans: {
            yes: [['R', '[[0,0],[0,0],[1,0],[2,0]]'], ['C', '[[4,0],[8,0],[16,0],[0,1]]']],
        },
    },
    {
        name: 'ldmatrix_m16n16_x2_b8',
        gpuArch: 'sm_100',
        instruction: 'ldmatrix',
        matrixSize: 'm16n16.x2',
        dtype: 'b8',
        operand: '',
        inputName: 'Shared Memory',
        rowsByTrans: {
            yes: [['R', '[[0,0],[0,0],[1,0],[2,0]]'], ['C', '[[4,0],[8,0],[16,0],[0,1],[0,2]]']],
        },
    },
    {
        name: 'ldmatrix_m8n16_x1_b4',
        gpuArch: 'sm_100',
        instruction: 'ldmatrix',
        matrixSize: 'm8n16.x1',
        dtype: 'b4',
        operand: '',
        inputName: 'Shared Memory',
        rowsByTrans: {
            no: [['R', '[[4,0],[8,0],[16,0]]'], ['C', '[[0,0],[0,0],[1,0],[2,0]]']],
        },
    },
    {
        name: 'ldmatrix_m8n16_x2_b4',
        gpuArch: 'sm_100',
        instruction: 'ldmatrix',
        matrixSize: 'm8n16.x2',
        dtype: 'b4',
        operand: '',
        inputName: 'Shared Memory',
        rowsByTrans: {
            no: [['R', '[[4,0],[8,0],[16,0],[0,1]]'], ['C', '[[0,0],[0,0],[1,0],[2,0]]']],
        },
    },
    {
        name: 'ldmatrix_m8n16_x4_b4',
        gpuArch: 'sm_100',
        instruction: 'ldmatrix',
        matrixSize: 'm8n16.x4',
        dtype: 'b4',
        operand: '',
        inputName: 'Shared Memory',
        rowsByTrans: {
            no: [['R', '[[4,0],[8,0],[16,0],[0,1],[0,2]]'], ['C', '[[0,0],[0,0],[1,0],[2,0]]']],
        },
    },
];

export const LDMATRIX_PRESET_DEFINITIONS: ComposeLayoutPresetDefinition[] = MATRIX_TRANSFER_LAYOUTS.flatMap((layout) => (
    (['no', 'yes'] as const)
        .filter((trans): trans is 'no' | 'yes' => Boolean(layout.rowsByTrans[trans]))
        .map((trans) => ({
            gpuArch: layout.gpuArch,
            instruction: 'ldmatrix',
            matrixSize: layout.matrixSize,
            dtype: layout.dtype,
            operand: '',
            trans,
            facets: {
                gpuArch: layout.dtype === 'b16' ? GPU_ARCHS_SM75_PLUS : GPU_ARCHS_SM100_PLUS,
                instruction: 'ldmatrix',
                matrixSize: layout.matrixSize,
                dtype: layout.dtype,
                operand: '',
                trans,
            },
            inputName: layout.inputName,
            signature: '[R, C] -> [T, R32]',
            name: trans === 'yes' ? layout.name.replace(/_([^_]+)$/, '_trans_$1') : layout.name,
            rows: layout.rowsByTrans[trans]!,
            comments: trans === 'yes'
                ? [
                    'Consecutive columns need not be contiguous in memory; each row address points to the start of a matrix column.',
                    'trans = yes means the matrix is loaded in column-major format.',
                ]
                : ['Consecutive rows need not be contiguous in memory; each row address points to the start of a matrix row.'],
        }))
));
