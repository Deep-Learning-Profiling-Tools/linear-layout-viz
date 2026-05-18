import { PRESET_GPU_ARCHS } from './gpu-archs.js';
import { LDMATRIX_PRESET_DEFINITIONS } from './ldmatrix.js';
import { MMA_PRESET_DEFINITIONS } from './mma.js';
import { STMATRIX_PRESET_DEFINITIONS } from './stmatrix.js';
import { SWIZZLE_PRESET_DEFINITIONS } from './swizzle.js';
import type { ComposeLayoutPresetFamily, ComposeLayoutPresetFieldDefinition } from './types.js';
import { WGMMA_PRESET_DEFINITIONS } from './wgmma.js';
export { PRESET_GPU_ARCHS } from './gpu-archs.js';

// these fields are shared by every current preset family.
// families can still contribute additional facet fields, but common hardware
// selectors stay here so the preset widget renders them in one stable order.
// required fields are always visible; optional fields appear only when their
// dependency path is active and matching presets have values for that field.
// the model layer merges these definitions with facet keys found in data, so a
// new architecture family can add metadata without changing widget code.
// keep `order` sparse enough that external families can insert fields between
// existing concepts without renumbering this file.
const DEFAULT_PRESET_FIELDS = [
    {
        key: 'gpuArch',
        label: 'GPU Arch',
        placeholder: 'Type GPU arch',
        order: 10,
        required: true,
        values: PRESET_GPU_ARCHS,
    },
    {
        key: 'instruction',
        label: 'Instruction',
        placeholder: 'Type instruction',
        order: 20,
        required: true,
    },
    {
        key: 'matrixSize',
        label: 'Matrix Size',
        placeholder: 'Type matrix size',
        order: 30,
        required: true,
    },
    {
        key: 'dtype',
        label: 'DType',
        placeholder: 'Type dtype',
        order: 40,
        required: true,
    },
    {
        key: 'operand',
        label: 'Operand',
        placeholder: 'Type operand',
        order: 50,
        dependsOn: ['instruction'],
    },
    {
        key: 'trans',
        label: 'Transpose',
        placeholder: 'Type transpose',
        order: 60,
        dependsOn: ['instruction'],
    },
    {
        key: 'major',
        label: 'Major',
        placeholder: 'Type major',
        order: 70,
        dependsOn: ['instruction'],
    },
] satisfies ComposeLayoutPresetFieldDefinition[];

const NVIDIA_PRESET_DEFINITIONS = [
    ...MMA_PRESET_DEFINITIONS,
    ...SWIZZLE_PRESET_DEFINITIONS,
    ...LDMATRIX_PRESET_DEFINITIONS,
    ...STMATRIX_PRESET_DEFINITIONS,
    ...WGMMA_PRESET_DEFINITIONS,
];

export const COMPOSE_LAYOUT_PRESET_FAMILIES = [
    { fields: DEFAULT_PRESET_FIELDS, presets: NVIDIA_PRESET_DEFINITIONS },
] satisfies ComposeLayoutPresetFamily[];
