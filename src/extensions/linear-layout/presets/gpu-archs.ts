// architecture selector values are deliberately centralized.
// preset families import these ranges instead of spelling support matrices by
// hand, which keeps future NVIDIA/AMD family edits local to one file.
// architecture suffixes such as "a" and "f" are treated as distinct UI values
// because instruction availability can differ between otherwise adjacent chips.
export const PRESET_GPU_ARCHS = [
    'sm_70',
    'sm_75',
    'sm_80',
    'sm_90',
    'sm_90a',
    'sm_100',
    'sm_100a',
    'sm_100f',
    'sm_110',
    'sm_110a',
    'sm_110f',
    'sm_120',
    'sm_120a',
    'sm_120f',
] as const;

export const GPU_ARCHS_SM70_PLUS = [...PRESET_GPU_ARCHS];
export const GPU_ARCHS_SM75_PLUS = ['sm_75', 'sm_80', 'sm_90', 'sm_90a', 'sm_100', 'sm_100a', 'sm_100f', 'sm_110', 'sm_110a', 'sm_110f', 'sm_120', 'sm_120a', 'sm_120f'];
export const GPU_ARCHS_SM80_PLUS = ['sm_80', 'sm_90', 'sm_90a', 'sm_100', 'sm_100a', 'sm_100f', 'sm_110', 'sm_110a', 'sm_110f', 'sm_120', 'sm_120a', 'sm_120f'];
export const GPU_ARCHS_SM90_PLUS = ['sm_90', 'sm_90a', 'sm_100', 'sm_100a', 'sm_100f', 'sm_110', 'sm_110a', 'sm_110f', 'sm_120', 'sm_120a', 'sm_120f'];
export const GPU_ARCHS_SM100_PLUS = ['sm_100', 'sm_100a', 'sm_100f', 'sm_110', 'sm_110a', 'sm_110f', 'sm_120', 'sm_120a', 'sm_120f'];
export const GPU_ARCHS_SM120_ONLY = ['sm_120', 'sm_120a', 'sm_120f'];
export const GPU_ARCHS_SWIZZLE = ['sm_90a', 'sm_100a', 'sm_100f', 'sm_110a', 'sm_110f', 'sm_120a', 'sm_120f'];
export const GPU_ARCHS_WGMMA = ['sm_90a'];
