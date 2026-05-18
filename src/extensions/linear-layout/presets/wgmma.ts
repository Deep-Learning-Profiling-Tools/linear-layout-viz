import { GPU_ARCHS_WGMMA } from './gpu-archs.js';
import type { ComposeLayoutPresetDefinition } from './types.js';

// wgmma presets are tensor-core layouts for warp-group instructions.
// the selector facets are explicit data because the widget should not know
// instruction-specific rules.
// all current entries target sm_90a through GPU_ARCHS_WGMMA.
// `T` is the per-thread coordinate inside the warp group.
// `W` is the warp coordinate inside the warp group.
// `R` is the per-thread register coordinate.
// output accumulator presets share the same D operand forms across A/B input
// families, so they live in WGMMA_D_PRESETS and are spread into the raw list.
// dtype aliases like b16 and b32 intentionally describe element bit width, not
// the exact semantic type, because several ISA dtypes share one layout.
// comments attached to entries are shown to users when the preset is loaded.
// keep new shape families as data rows here before adding any preset-model code;
// the model already handles array-valued facets and shared fields.
// if an entry applies to several architectures, change the gpu-arch helper data
// rather than duplicating the same layout under multiple names.
// names match the notation block names so copy/paste from rendered specs stays
// reversible during documentation review.

const WGMMA_D_PRESETS: ComposeLayoutPresetDefinition[] = [
    {
        name: 'WGMMA_m64n8_D_b32',
        gpuArch: 'sm_90',
        instruction: 'wgmma',
        matrixSize: 'm64n8',
        dtype: 'b32',
        operand: 'D',
        comments: ['b32 represents 32-bit accumulators (f32).'],
        signature: '[T,W,R] -> [M,N]',
        rows: [['T', '[[0,1],[0,2],[1,0],[2,0],[4,0]]'], ['W', '[[16,0],[32,0]]'], ['R', '[[8,0],[0,4]]']],
    },
    {
        name: 'WGMMA_m64n16_D_b16',
        gpuArch: 'sm_90',
        instruction: 'wgmma',
        matrixSize: 'm64n16',
        dtype: 'b16',
        operand: 'D',
        comments: ['b16 represents 16-bit accumulators (f16).'],
        signature: '[T,W,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['W', '[[16,0],[32,0]]'], ['R', '[[0,1],[8,0],[0,8]]']],
    },
    {
        name: 'WGMMA_m64n16_D_b32',
        gpuArch: 'sm_90',
        instruction: 'wgmma',
        matrixSize: 'm64n16',
        dtype: 'b32',
        operand: 'D',
        comments: ['b32 represents 32-bit accumulators (f32).'],
        signature: '[T,W,R] -> [M,N]',
        rows: [['T', '[[0,1],[0,2],[1,0],[2,0],[4,0]]'], ['W', '[[16,0],[32,0]]'], ['R', '[[8,0],[0,4],[0,8]]']],
    },
    {
        name: 'WGMMA_m64n32_D_b16',
        gpuArch: 'sm_90',
        instruction: 'wgmma',
        matrixSize: 'm64n32',
        dtype: 'b16',
        operand: 'D',
        comments: ['b16 represents 16-bit accumulators (f16).'],
        signature: '[T,W,R] -> [M,N]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['W', '[[16,0],[32,0]]'], ['R', '[[0,1],[8,0],[0,8],[0,16]]']],
    },
    {
        name: 'WGMMA_m64n32_D_b32',
        gpuArch: 'sm_90',
        instruction: 'wgmma',
        matrixSize: 'm64n32',
        dtype: 'b32',
        operand: 'D',
        comments: ['b32 represents 32-bit accumulators (f32/s32).'],
        signature: '[T,W,R] -> [M,N]',
        rows: [['T', '[[0,1],[0,2],[1,0],[2,0],[4,0]]'], ['W', '[[16,0],[32,0]]'], ['R', '[[8,0],[0,4],[0,8],[0,16]]']],
    },
    {
        name: 'WGMMA_m64n64_D_b32',
        gpuArch: 'sm_90',
        instruction: 'wgmma',
        matrixSize: 'm64n64',
        dtype: 'b32',
        operand: 'D',
        comments: ['b32 represents 32-bit accumulators (f32/s32).'],
        signature: '[T,W,R] -> [M,N]',
        rows: [['T', '[[0,1],[0,2],[1,0],[2,0],[4,0]]'], ['W', '[[16,0],[32,0]]'], ['R', '[[8,0],[0,4],[0,8],[0,16],[0,32]]']],
    },
    {
        name: 'WGMMA_m64n128_D_b32',
        gpuArch: 'sm_90',
        instruction: 'wgmma',
        matrixSize: 'm64n128',
        dtype: 'b32',
        operand: 'D',
        comments: ['b32 represents 32-bit accumulators (f32/s32).'],
        signature: '[T,W,R] -> [M,N]',
        rows: [['T', '[[0,1],[0,2],[1,0],[2,0],[4,0]]'], ['W', '[[16,0],[32,0]]'], ['R', '[[8,0],[0,4],[0,8],[0,16],[0,32],[0,64]]']],
    },
    {
        name: 'WGMMA_m64n256_D_b32',
        gpuArch: 'sm_90',
        instruction: 'wgmma',
        matrixSize: 'm64n256',
        dtype: 'b32',
        operand: 'D',
        comments: ['b32 represents 32-bit accumulators (s32).'],
        signature: '[T,W,R] -> [M,N]',
        rows: [['T', '[[0,1],[0,2],[1,0],[2,0],[4,0]]'], ['W', '[[16,0],[32,0]]'], ['R', '[[8,0],[0,4],[0,8],[0,16],[0,32],[0,64],[0,128]]']],
    },
];

const WGMMA_RAW_PRESET_DEFINITIONS = [
    {
        name: 'WGMMA_m64k8_A_b32',
        gpuArch: 'sm_90',
        instruction: 'wgmma',
        matrixSize: 'm64k8',
        dtype: 'b32',
        operand: 'A',
        comments: ['b32 represents 32-bit elements (tf32).'],
        signature: '[T,W,R] -> [M,K]',
        rows: [['T', '[[0,1],[0,2],[1,0],[2,0],[4,0]]'], ['W', '[[16,0],[32,0]]'], ['R', '[[8,0],[0,4]]']],
    },
    {
        name: 'WGMMA_m64k16_A_b16',
        gpuArch: 'sm_90',
        instruction: 'wgmma',
        matrixSize: 'm64k16',
        dtype: 'b16',
        operand: 'A',
        comments: ['b16 represents 16-bit elements (f16/bf16).'],
        signature: '[T,W,R] -> [M,K]',
        rows: [['T', '[[0,2],[0,4],[1,0],[2,0],[4,0]]'], ['W', '[[16,0],[32,0]]'], ['R', '[[0,1],[8,0],[0,8]]']],
    },
    {
        name: 'WGMMA_m64k32_A_b8',
        gpuArch: 'sm_90',
        instruction: 'wgmma',
        matrixSize: 'm64k32',
        dtype: 'b8',
        operand: 'A',
        comments: ['b8 represents 8-bit elements (u8/s8/e4m3/e5m2).'],
        signature: '[T,W,R] -> [M,K]',
        rows: [['T', '[[0,4],[0,8],[1,0],[2,0],[4,0]]'], ['W', '[[16,0],[32,0]]'], ['R', '[[0,1],[0,2],[8,0],[0,16]]']],
    },
    {
        name: 'WGMMA_m64k256_A_b1',
        gpuArch: 'sm_90',
        instruction: 'wgmma',
        matrixSize: 'm64k256',
        dtype: 'b1',
        operand: 'A',
        signature: '[T,W,R] -> [M,K]',
        rows: [['T', '[[0,32],[0,64],[1,0],[2,0],[4,0]]'], ['W', '[[16,0],[32,0]]'], ['R', '[[0,1],[0,2],[0,4],[0,8],[0,16],[8,0],[0,128]]']],
    },
    ...WGMMA_D_PRESETS,
] satisfies ComposeLayoutPresetDefinition[];

export const WGMMA_PRESET_DEFINITIONS: ComposeLayoutPresetDefinition[] = WGMMA_RAW_PRESET_DEFINITIONS.map((preset) => ({
    ...preset,
    facets: {
        gpuArch: GPU_ARCHS_WGMMA,
        instruction: preset.instruction ?? 'wgmma',
        matrixSize: preset.matrixSize ?? '',
        dtype: preset.dtype ?? '',
        operand: preset.operand ?? '',
    },
}));
