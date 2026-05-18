import { GPU_ARCHS_SWIZZLE } from './gpu-archs.js';
import type { ComposeLayoutPresetDefinition } from './types.js';

// swizzle presets are generated because the major variants differ only by which
// output axis receives the contiguous and cross terms.
// the generated rows are intentionally kept in the same notation contributors
// use in hand-written preset files.
// b8 through b128 reuse the same 128-byte tile rule with fewer leading vectors
// as element width increases.

/**
 * Generates the JSON basis-vector row used by swizzle presets for the logical
 * offset input, orienting the contiguous and cross-lane bases by major order.
 *
 * @param leadingVectors - Number of leading power-of-two contiguous bases to emit before the three cross bases.
 * @param major - Swizzle orientation; `MN-major` places contiguous powers on M, while `K-major` places them on K.
 * @returns JSON-encoded array of `[M, K]` basis pairs suitable for the preset row `['O', value]`.
 * @noThrows The helper performs deterministic array construction and JSON serialization of numeric pairs without parsing or validation branches.
 * @example
 * swizzleBases(3, 'MN-major');
 * // '[[1,0],[2,0],[4,0],[1,1],[2,2],[4,4]]'
 *
 * swizzleBases(3, 'K-major');
 * // '[[0,1],[0,2],[0,4],[1,1],[2,2],[4,4]]'
 */
function swizzleBases(leadingVectors: number, major: 'MN-major' | 'K-major'): string {
    const contiguousBases = Array.from({ length: leadingVectors }, (_, index) => major === 'MN-major'
        ? [2 ** index, 0]
        : [0, 2 ** index]);
    const crossBases = Array.from({ length: 3 }, (_, index) => major === 'MN-major'
        ? [2 ** (leadingVectors - 3 + index), 2 ** index]
        : [2 ** index, 2 ** (leadingVectors - 3 + index)]);
    return JSON.stringify([...contiguousBases, ...crossBases]);
}

const SWIZZLE_DTYPES = [
    ['b8', 7],
    ['b16', 6],
    ['b32', 5],
    ['b64', 4],
    ['b128', 3],
] as const;

export const SWIZZLE_PRESET_DEFINITIONS: ComposeLayoutPresetDefinition[] = SWIZZLE_DTYPES.flatMap(([dtype, leadingVectors]) => ([
    {
        name: `swizzle_128B_MN_major_${dtype}`,
        facets: {
            gpuArch: GPU_ARCHS_SWIZZLE,
            instruction: 'swizzle',
            matrixSize: '128B',
            dtype,
            major: 'MN-major',
        },
        comments: ['bX means each element is X bits wide.'],
        signature: '[O] -> [M, K]',
        rows: [['O', swizzleBases(leadingVectors, 'MN-major')]],
        inputName: 'Logical Offsets',
    },
    {
        name: `swizzle_128B_K_major_${dtype}`,
        facets: {
            gpuArch: GPU_ARCHS_SWIZZLE,
            instruction: 'swizzle',
            matrixSize: '128B',
            dtype,
            major: 'K-major',
        },
        comments: ['bX means each element is X bits wide.'],
        signature: '[O] -> [M, K]',
        rows: [['O', swizzleBases(leadingVectors, 'K-major')]],
        inputName: 'Logical Offsets',
    },
])) satisfies ComposeLayoutPresetDefinition[];
