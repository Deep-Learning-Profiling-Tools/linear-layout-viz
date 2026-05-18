import { GPU_ARCHS_SM90_PLUS, GPU_ARCHS_SM100_PLUS } from './gpu-archs.js';
import { MATRIX_TRANSFER_LAYOUTS } from './ldmatrix.js';
import type { ComposeLayoutPresetDefinition } from './types.js';

// stmatrix mirrors ldmatrix layout definitions but shifts the architecture
// support window and swaps the instruction name in generated preset ids.
// sharing the source table keeps transpose/no-transpose notation synchronized.
// future store-specific exceptions should be represented as data before adding
// widget branches.

export const STMATRIX_PRESET_DEFINITIONS: ComposeLayoutPresetDefinition[] = MATRIX_TRANSFER_LAYOUTS.flatMap((layout) => (
    (['no', 'yes'] as const)
        .filter((trans): trans is 'no' | 'yes' => Boolean(layout.rowsByTrans[trans]))
        .map((trans) => ({
            gpuArch: layout.gpuArch === 'sm_75' ? 'sm_90' : layout.gpuArch,
            instruction: 'stmatrix',
            matrixSize: layout.matrixSize,
            dtype: layout.dtype,
            operand: '',
            trans,
            facets: {
                gpuArch: layout.dtype === 'b16' ? GPU_ARCHS_SM90_PLUS : GPU_ARCHS_SM100_PLUS,
                instruction: 'stmatrix',
                matrixSize: layout.matrixSize,
                dtype: layout.dtype,
                operand: '',
                trans,
            },
            inputName: layout.inputName,
            signature: '[R, C] -> [T, R32]',
            name: trans === 'yes'
                ? layout.name.replace(/^ldmatrix/, 'stmatrix').replace(/_([^_]+)$/, '_trans_$1')
                : layout.name.replace(/^ldmatrix/, 'stmatrix'),
            rows: layout.rowsByTrans[trans]!,
            comments: trans === 'yes'
                ? [
                    'Consecutive columns need not be contiguous in memory; each row address points to the start of a matrix column.',
                    'trans = yes means the matrix is stored in column-major format.',
                ]
                : ['Consecutive rows need not be contiguous in memory; each row address points to the start of a matrix row.'],
        }))
));
