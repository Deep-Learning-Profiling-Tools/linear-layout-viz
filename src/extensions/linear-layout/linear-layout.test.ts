import { describe, expect, it } from 'vitest';
import {
    autoColorLayoutState,
    buildComposeRuntime,
    composeLayoutPresetCatalog,
    composeLayoutPresetFields,
    composeLayoutPresetForSelection,
    composeLayoutPresetOptions,
    composeLayoutStateFromLegacySpec,
    createComposeLayoutDocument,
    defaultComposeLayoutState,
    matchedComposeLayoutPresetSelection,
    normalizeComposeLayoutPresetSelection,
    propagationLabels,
    type ComposeLayoutState,
} from './linear-layout.js';
import { COMPOSE_LAYOUT_PRESET_FAMILIES } from './presets/index.js';
import {
    coordsForRootIndexes,
    linearLayoutDisplayModel,
    linearLayoutMultiInputModel,
    linearLayoutSelectionMapForMeta,
    rootIndexesForCoords,
} from './linear-layout-multi-input.js';
import { linearLayoutHoverPopupEntries } from './linear-layout-viewer-sync.js';

const DEFAULT_RANGES = {
    H: ['0', '0.8'],
    S: ['1', '0.2'],
    L: ['1', '0.2'],
} as const;

function composeState(specsText: string, operationText: string): ComposeLayoutState {
    return {
        specsText,
        operationText,
        inputName: 'Input Space',
        presetSelection: matchedComposeLayoutPresetSelection({ specsText, operationText, inputName: 'Input Space' }),
        visibleTensors: {},
        propagateOutputs: false,
        mapping: { H: 'none', S: 'none', L: 'none' } as const,
        ranges: {
            H: [DEFAULT_RANGES.H[0], DEFAULT_RANGES.H[1]],
            S: [DEFAULT_RANGES.S[0], DEFAULT_RANGES.S[1]],
            L: [DEFAULT_RANGES.L[0], DEFAULT_RANGES.L[1]],
        },
    };
}

function presetName(preset: (typeof COMPOSE_LAYOUT_PRESET_FAMILIES)[number]['presets'][number]): string {
    if (preset.title) return preset.title;
    return 'name' in preset ? preset.name : preset.specsText.split(':', 1)[0] ?? 'unnamed preset';
}

function concretePresetSelections(
    preset: ReturnType<typeof composeLayoutPresetCatalog>[number],
): Record<string, string>[] {
    return composeLayoutPresetFields().reduce((selections, field) => (
        selections.flatMap((selection) => {
            const values = preset.facets[field.key]?.length ? preset.facets[field.key]! : [''];
            return values.map((value) => ({ ...selection, [field.key]: value }));
        })
    ), [{}] as Record<string, string>[]);
}

describe('compose layout helpers', () => {
    it('builds the default blocked-layout example', () => {
        const state = defaultComposeLayoutState();
        const runtime = buildComposeRuntime(state);
        const document = createComposeLayoutDocument(state);

        expect(runtime.tensors.map((tensor) => ({ title: tensor.title, shape: tensor.shape }))).toEqual([
            { title: 'Hardware Layout', shape: [32, 4, 4] },
            { title: 'Blocked_Layout', shape: [16, 32] },
        ]);
        expect(document.title).toBe('Blocked_Layout');
        expect(document.manifest.viewer.dimensionMappingScheme).toBe('contiguous');
        expect(document.manifest.tensors.map((tensor) => ({
            name: tensor.name,
            shape: tensor.shape,
            axisLabels: tensor.axisLabels,
        }))).toEqual([
            { name: 'Hardware Layout', shape: [32, 4, 4], axisLabels: ['T', 'W', 'R'] },
            { name: 'Blocked_Layout', shape: [16, 32], axisLabels: ['Y', 'X'] },
        ]);
        expect(state.propagateOutputs).toBe(false);
        expect(state.mapping).toEqual({ H: 'T', S: 'W', L: 'R' });
        expect(state.ranges).toEqual({ H: ['0', '0.8'], S: ['1', '0.2'], L: ['1', '0.2'] });
        expect(state.presetSelection).toEqual({
            gpuArch: '',
            instruction: '',
            matrixSize: '',
            dtype: '',
            operand: '',
            trans: '',
            major: '',
        });
    });

    it('matches the shipped ldmatrix preset from its editor state', () => {
        const selection = matchedComposeLayoutPresetSelection({
            specsText: [
                'ldmatrix_m8n8_x1_b16: [R, C] -> [T, R32]',
                '# R32 = packed 32-bit register',
                '# Consecutive rows need not be contiguous in memory; each row address points to the start of a matrix row.',
                'R: [[4,0],[8,0],[16,0]]',
                'C: [[0,0],[1,0],[2,0]]',
            ].join('\n'),
            operationText: 'ldmatrix_m8n8_x1_b16',
            inputName: 'Shared Memory',
        });

        expect(selection).toEqual({
            gpuArch: 'sm_75',
            instruction: 'ldmatrix',
            matrixSize: 'm8n8.x1',
            dtype: 'b16',
            operand: '',
            trans: 'no',
            major: '',
        });
        const preset = composeLayoutPresetForSelection(selection);
        expect(preset?.title).toBe('ldmatrix_m8n8_x1_b16');
        expect(preset?.state.specsText).toContain('# R = row');
        expect(preset?.state.specsText).toContain('# C = column');
        expect(preset?.state.specsText).toContain('# T = thread (AKA lane)');
        expect(preset?.state.specsText).toContain('# R32 = packed 32-bit register');
        expect(preset?.state.specsText).toContain('# Consecutive rows need not be contiguous in memory; each row address points to the start of a matrix row.');
        expect(preset?.state.inputName).toBe('Shared Memory');
    });

    it('emits transposed m16n16 ldmatrix specs', () => {
        const preset = composeLayoutPresetForSelection({
            gpuArch: 'sm_100',
            instruction: 'ldmatrix',
            matrixSize: 'm16n16.x1',
            dtype: 'b8',
            operand: '',
            trans: 'yes',
            major: '',
        });

        expect(preset?.title).toBe('ldmatrix_m16n16_x1_trans_b8');
        expect(preset?.state.specsText).toContain('R: [[0,0],[0,0],[1,0],[2,0]]');
        expect(preset?.state.specsText).toContain('C: [[4,0],[8,0],[16,0],[0,1]]');
    });

    it('emits distinct transposed m8n8 ldmatrix specs', () => {
        expect(composeLayoutPresetOptions({
            gpuArch: 'sm_75',
            instruction: 'ldmatrix',
            matrixSize: 'm8n8.x1',
            dtype: 'b16',
            operand: '',
            trans: '',
            major: '',
        }).transes).toEqual(['no', 'yes']);

        const preset = composeLayoutPresetForSelection({
            gpuArch: 'sm_75',
            instruction: 'ldmatrix',
            matrixSize: 'm8n8.x1',
            dtype: 'b16',
            operand: '',
            trans: 'yes',
            major: '',
        });

        expect(preset?.title).toBe('ldmatrix_m8n8_x1_trans_b16');
        expect(preset?.state.specsText).toContain('R: [[0,0],[1,0],[2,0]]');
        expect(preset?.state.specsText).toContain('C: [[4,0],[8,0],[16,0]]');
        expect(preset?.state.specsText).not.toContain('R: [[4,0],[8,0],[16,0]]');
    });

    it('filters preset dropdown options by the current selection path', () => {
        const sm80Options = composeLayoutPresetOptions({
            gpuArch: 'sm_80',
            instruction: '',
            matrixSize: '',
            dtype: '',
            operand: '',
            trans: '',
            major: '',
        });
        expect(sm80Options).toMatchObject({
            instructions: ['mma', 'ldmatrix'],
        });
        expect(sm80Options.instruction).toEqual(['mma', 'ldmatrix']);
        expect(composeLayoutPresetOptions({
            gpuArch: '',
            instruction: '',
            matrixSize: '',
            dtype: '',
            operand: '',
            trans: '',
            major: '',
        })).toMatchObject({
            gpuArchs: ['sm_70', 'sm_75', 'sm_80', 'sm_90', 'sm_90a', 'sm_100', 'sm_100a', 'sm_100f', 'sm_110', 'sm_110a', 'sm_110f', 'sm_120', 'sm_120a', 'sm_120f'],
            instructions: ['mma', 'swizzle', 'ldmatrix', 'stmatrix', 'wgmma'],
        });
        expect(composeLayoutPresetOptions({
            gpuArch: '',
            instruction: 'ldmatrix',
            matrixSize: '',
            dtype: '',
            operand: '',
            trans: '',
            major: '',
        })).toMatchObject({
            matrixSizes: ['m8n8.x1', 'm8n8.x2', 'm8n8.x4', 'm16n16.x1', 'm16n16.x2', 'm8n16.x1', 'm8n16.x2', 'm8n16.x4'],
            dtypes: ['b16', 'b8', 'b4'],
            operands: [],
            transes: ['no', 'yes'],
        });

        expect(composeLayoutPresetOptions({
            gpuArch: 'sm_75',
            instruction: 'ldmatrix',
            matrixSize: 'm8n8.x1',
            dtype: 'b16',
            operand: '',
            trans: '',
            major: '',
        })).toMatchObject({
            operands: [],
            transes: ['no', 'yes'],
        });

        expect(composeLayoutPresetOptions({
            gpuArch: 'sm_70',
            instruction: 'mma',
            matrixSize: 'm8n8k4',
            dtype: 'f16',
            operand: '',
            trans: '',
            major: '',
        })).toMatchObject({
            operands: ['A-row-major', 'A-col-major', 'B-row-major', 'B-col-major', 'C', 'D'],
        });

        expect(composeLayoutPresetOptions({
            gpuArch: 'sm_90a',
            instruction: 'wgmma',
            matrixSize: 'm64k32',
            dtype: '',
            operand: '',
            trans: '',
            major: '',
        })).toMatchObject({
            dtypes: ['b8'],
            operands: ['A'],
        });

        expect(composeLayoutPresetOptions({
            gpuArch: 'sm_90a',
            instruction: 'wgmma',
            matrixSize: 'm64n32',
            dtype: '',
            operand: 'D',
            trans: '',
            major: '',
        })).toMatchObject({
            dtypes: ['b16', 'b32'],
            operands: ['D'],
        });

        expect(composeLayoutPresetOptions({
            gpuArch: 'sm_90a',
            instruction: 'swizzle',
            matrixSize: '128B',
            dtype: '',
            operand: '',
            trans: '',
            major: '',
        })).toMatchObject({
            dtypes: ['b8', 'b16', 'b32', 'b64', 'b128'],
            operands: [],
            majors: ['MN-major', 'K-major'],
        });

    });

    it('exposes preset selector fields from catalog metadata', () => {
        expect(composeLayoutPresetFields().map(({ key }) => key)).toEqual([
            'gpuArch',
            'instruction',
            'matrixSize',
            'dtype',
            'operand',
            'trans',
            'major',
        ]);
        expect(composeLayoutPresetFields().find(({ key }) => key === 'operand')?.dependsOn).toEqual(['instruction']);
    });

    it('requires shipped preset definitions to declare gpu arch facets', () => {
        const missing = COMPOSE_LAYOUT_PRESET_FAMILIES.flatMap((family) => (
            family.presets
                .filter((preset) => !preset.facets?.gpuArch)
                .map((preset) => presetName(preset))
        ));

        expect(missing).toEqual([]);
    });

    it('keeps every concrete preset selection unique and resolvable', () => {
        const requiredFields = composeLayoutPresetFields().filter((field) => field.required).map((field) => field.key);
        const missingRequired = composeLayoutPresetCatalog().flatMap((preset) => (
            requiredFields
                .filter((key) => !preset.facets[key]?.length)
                .map((key) => `${preset.title}:${key}`)
        ));
        const seen = new Map<string, string>();
        const duplicates: string[] = [];
        const unresolved: string[] = [];
        composeLayoutPresetCatalog().forEach((preset) => {
            concretePresetSelections(preset).forEach((selection) => {
                const key = composeLayoutPresetFields().map((field) => `${field.key}:${selection[field.key] ?? ''}`).join('|');
                const previous = seen.get(key);
                if (previous && previous !== preset.title) duplicates.push(`${previous} / ${preset.title}: ${key}`);
                seen.set(key, preset.title);
                if (composeLayoutPresetForSelection(selection)?.title !== preset.title) unresolved.push(`${preset.title}: ${key}`);
            });
        });

        expect(missingRequired).toEqual([]);
        expect(duplicates).toEqual([]);
        expect(unresolved).toEqual([]);
    });

    it('matches presets across every supported architecture in the family list', () => {
        const selection = {
            gpuArch: 'sm_120',
            instruction: 'mma',
            matrixSize: 'm8n8k4',
            dtype: 'f16',
            operand: 'A-row-major',
            trans: '',
            major: '',
        };

        expect(composeLayoutPresetForSelection(selection)?.title).toBe('MMA_m8n8k4_A_row_major_f16');
        expect(composeLayoutPresetOptions({
            gpuArch: 'sm_110',
            instruction: '',
            matrixSize: '',
            dtype: '',
            operand: '',
            trans: '',
            major: '',
        }).instructions).toEqual(['mma', 'ldmatrix', 'stmatrix']);
    });

    it('autofills singleton preset fields during normalization', () => {
        expect(normalizeComposeLayoutPresetSelection({
            gpuArch: '',
            instruction: 'wgmma',
            matrixSize: '',
            dtype: '',
            operand: '',
            trans: '',
            major: '',
        })).toEqual({
            gpuArch: 'sm_90a',
            instruction: 'wgmma',
            matrixSize: '',
            dtype: '',
            operand: '',
            trans: '',
            major: '',
        });
    });

    it('matches shipped mma presets with row-major and col-major operand variants', () => {
        const selection = matchedComposeLayoutPresetSelection({
            specsText: [
                'MMA_m8n8k4_A_row_major_f16: [T,R] -> [M,K]',
                'T: [[1,0],[2,0],[0,0],[0,0],[4,0]]',
                'R: [[0,1],[0,2]]',
            ].join('\n'),
            operationText: 'MMA_m8n8k4_A_row_major_f16',
            inputName: 'Hardware Layout',
        });

        expect(selection).toEqual({
            gpuArch: 'sm_70',
            instruction: 'mma',
            matrixSize: 'm8n8k4',
            dtype: 'f16',
            operand: 'A-row-major',
            trans: '',
            major: '',
        });
        expect(composeLayoutPresetForSelection(selection)?.title).toBe('MMA_m8n8k4_A_row_major_f16');
    });

    it('matches the corrected m8n8k4 B row-major f16 preset', () => {
        const selection = matchedComposeLayoutPresetSelection({
            specsText: [
                'MMA_m8n8k4_B_row_major_f16: [T,R] -> [K,N]',
                'T: [[1,0],[2,0],[0,0],[0,0],[0,4]]',
                'R: [[0,1],[0,2]]',
            ].join('\n'),
            operationText: 'MMA_m8n8k4_B_row_major_f16',
            inputName: 'Hardware Layout',
        });

        expect(selection).toEqual({
            gpuArch: 'sm_70',
            instruction: 'mma',
            matrixSize: 'm8n8k4',
            dtype: 'f16',
            operand: 'B-row-major',
            trans: '',
            major: '',
        });
        expect(composeLayoutPresetForSelection(selection)?.title).toBe('MMA_m8n8k4_B_row_major_f16');
    });

    it('matches merged mma presets by storage width', () => {
        const selection = matchedComposeLayoutPresetSelection({
            specsText: [
                'MMA_m16n8k16_A_b8: [T,R] -> [M,K]',
                '# b8 represents 8-bit elements (u8/s8/e4m3/e5m2).',
                'T: [[0,4],[0,8],[1,0],[2,0],[4,0]]',
                'R: [[0,1],[0,2],[8,0]]',
            ].join('\n'),
            operationText: 'MMA_m16n8k16_A_b8',
            inputName: 'Hardware Layout',
        });

        expect(selection).toEqual({
            gpuArch: 'sm_80',
            instruction: 'mma',
            matrixSize: 'm16n8k16',
            dtype: 'b8',
            operand: 'A',
            trans: '',
            major: '',
        });
        expect(composeLayoutPresetForSelection(selection)?.state.specsText)
            .toContain('# b8 represents 8-bit elements (u8/s8/e4m3/e5m2).');
    });

    it('matches shipped wgmma presets from their editor state', () => {
        const selection = matchedComposeLayoutPresetSelection({
            specsText: [
                'WGMMA_m64k32_A_b8: [T,W,R] -> [M,K]',
                'T: [[0,4],[0,8],[1,0],[2,0],[4,0]]',
                'W: [[16,0],[32,0]]',
                'R: [[0,1],[0,2],[8,0],[0,16]]',
            ].join('\n'),
            operationText: 'WGMMA_m64k32_A_b8',
            inputName: 'Hardware Layout',
        });

        expect(selection).toEqual({
            gpuArch: 'sm_90a',
            instruction: 'wgmma',
            matrixSize: 'm64k32',
            dtype: 'b8',
            operand: 'A',
            trans: '',
            major: '',
        });
        const preset = composeLayoutPresetForSelection(selection);
        expect(preset?.title).toBe('WGMMA_m64k32_A_b8');
        expect(preset?.state.specsText).toContain('# b8 represents 8-bit elements (u8/s8/e4m3/e5m2).');
    });

    it('matches packed-16 wgmma accumulator layouts separately from 32-bit accumulators', () => {
        const selection = matchedComposeLayoutPresetSelection({
            specsText: [
                'WGMMA_m64n16_D_b16: [T,W,R] -> [M,N]',
                '# b16 represents 16-bit accumulators (f16).',
                'T: [[0,2],[0,4],[1,0],[2,0],[4,0]]',
                'W: [[16,0],[32,0]]',
                'R: [[0,1],[8,0],[0,8]]',
            ].join('\n'),
            operationText: 'WGMMA_m64n16_D_b16',
            inputName: 'Hardware Layout',
        });

        expect(selection).toEqual({
            gpuArch: 'sm_90a',
            instruction: 'wgmma',
            matrixSize: 'm64n16',
            dtype: 'b16',
            operand: 'D',
            trans: '',
            major: '',
        });
    });

    it('surfaces ldmatrix transpose as preset metadata', () => {
        expect(normalizeComposeLayoutPresetSelection({
            gpuArch: 'sm_100',
            instruction: 'ldmatrix',
            matrixSize: 'm16n16.x1',
            dtype: 'b8',
            operand: '',
            trans: '',
            major: '',
        })).toEqual({
            gpuArch: 'sm_100',
            instruction: 'ldmatrix',
            matrixSize: 'm16n16.x1',
            dtype: 'b8',
            operand: '',
            trans: 'yes',
            major: '',
        });

        expect(normalizeComposeLayoutPresetSelection({
            gpuArch: 'sm_100',
            instruction: 'ldmatrix',
            matrixSize: 'm8n16.x1',
            dtype: 'b4',
            operand: '',
            trans: '',
            major: '',
        })).toEqual({
            gpuArch: 'sm_100',
            instruction: 'ldmatrix',
            matrixSize: 'm8n16.x1',
            dtype: 'b4',
            operand: '',
            trans: 'no',
            major: '',
        });
    });

    it('matches the shipped swizzle preset from its editor state', () => {
        const selection = matchedComposeLayoutPresetSelection({
            specsText: [
                'swizzle_128B_MN_major_b32: [O] -> [M, K]',
                'O: [[1,0],[2,0],[4,0],[8,0],[16,0],[4,1],[8,2],[16,4]]',
            ].join('\n'),
            operationText: 'swizzle_128B_MN_major_b32',
            inputName: 'Logical Offsets',
        });

        expect(selection).toEqual({
            gpuArch: 'sm_90a',
            instruction: 'swizzle',
            matrixSize: '128B',
            dtype: 'b32',
            operand: '',
            trans: '',
            major: 'MN-major',
        });
        const preset = composeLayoutPresetForSelection(selection);
        expect(preset?.title).toBe('swizzle_128B_MN_major_b32');
        expect(preset?.state.specsText).toContain('# bX means each element is X bits wide.');
        expect(preset?.state.specsText).toContain('O: [[1,0],[2,0],[4,0],[8,0],[16,0],[4,1],[8,2],[16,4]]');
    });

    it('matches shipped K-major swizzle presets from their editor state', () => {
        const selection = matchedComposeLayoutPresetSelection({
            specsText: [
                'swizzle_128B_K_major_b64: [O] -> [M, K]',
                'O: [[0,1],[0,2],[0,4],[0,8],[1,2],[2,4],[4,8]]',
            ].join('\n'),
            operationText: 'swizzle_128B_K_major_b64',
            inputName: 'Logical Offsets',
        });

        expect(selection).toEqual({
            gpuArch: 'sm_90a',
            instruction: 'swizzle',
            matrixSize: '128B',
            dtype: 'b64',
            operand: '',
            trans: '',
            major: 'K-major',
        });
        expect(composeLayoutPresetForSelection(selection)?.title).toBe('swizzle_128B_K_major_b64');
    });

    it('matches the shipped stmatrix preset from its editor state', () => {
        const selection = matchedComposeLayoutPresetSelection({
            specsText: [
                'stmatrix_m8n8_x4_b16: [R, C] -> [T, R32]',
                '# R32 = packed 32-bit register',
                '# Consecutive rows need not be contiguous in memory; each row address points to the start of a matrix row.',
                'R: [[4,0],[8,0],[16,0],[0,1],[0,2]]',
                'C: [[0,0],[1,0],[2,0]]',
            ].join('\n'),
            operationText: 'stmatrix_m8n8_x4_b16',
            inputName: 'Shared Memory',
        });

        expect(selection).toEqual({
            gpuArch: 'sm_90',
            instruction: 'stmatrix',
            matrixSize: 'm8n8.x4',
            dtype: 'b16',
            operand: '',
            trans: 'no',
            major: '',
        });
        expect(composeLayoutPresetForSelection(selection)?.title).toBe('stmatrix_m8n8_x4_b16');
    });

    it('accepts python-style comments in layout specs', () => {
        const runtime = buildComposeRuntime(composeState([
            '# comment before the signature',
            'Layout_1: [T,W,R] -> [A,B] # signature comment',
            '# comment before T',
            'T: [[0,1],[0,2]] # basis comment',
            'W: []',
            'R: [[1,0],[2,0]]',
        ].join('\n'), 'Layout_1'));

        expect(runtime.tensors.at(-1)).toMatchObject({
            title: 'Layout_1',
            axisLabels: ['A', 'B'],
            shape: [4, 4],
        });
    });

    it('renders composition chains left to right and emits compose code in the same order', () => {
        const state = composeState([
            'Layout1: [T] -> [A]',
            'T: [[1]]',
            '',
            'Layout2: [A] -> [B]',
            'A: [[1]]',
        ].join('\n'), 'Layout2(Layout1)');
        const runtime = buildComposeRuntime(state);

        expect(runtime.tensors.map((tensor) => tensor.title)).toEqual([
            'Input Space',
            'Layout1',
            'Layout2(Layout1)',
        ]);
        expect(runtime.tensors.map((tensor) => tensor.shape)).toEqual([[2], [2], [2]]);
        expect(runtime.matrixBlocks.map((block) => block.title)).toEqual([
            'Layout1',
            'Layout2',
            'layout_tmp1 = Layout2(Layout1)',
        ]);
        expect(runtime.pythonCode).toContain('layout_tmp1 = Layout1.compose(Layout2)');
    });

    it('normalizes legacy warp-thread-register ordering to thread-warp-register', () => {
        const state = composeLayoutStateFromLegacySpec({
            name: 'Legacy',
            input_dims: [
                ['warp', []],
                ['thread', []],
                ['register', []],
            ],
            out_dims: ['x'],
        });

        expect(state.specsText.startsWith('Legacy: [T,W,R] -> [X]')).toBe(true);
    });

    it('composes layouts with matching labels even when intermediate bit widths differ', () => {
        const state = composeState([
            'Layout1: [T] -> [A]',
            'T: [[1]]',
            '',
            'Layout2: [A] -> [B]',
            'A: [[1],[2]]',
        ].join('\n'), 'Layout2(Layout1)');
        const runtime = buildComposeRuntime(state);

        expect(runtime.tensors.map((tensor) => tensor.title)).toEqual([
            'Input Space',
            'Layout1',
            'Layout2(Layout1)',
        ]);
        expect(runtime.tensors.map((tensor) => tensor.shape)).toEqual([[2], [2], [4]]);
    });

    it('composes swizzles after tiles without dropping reachable coordinates', () => {
        const runtime = buildComposeRuntime(composeState([
            'Tile: [T,W] -> [Y,X]',
            'T: [[0,1],[0,2]]',
            'W: [[1,0]]',
            '',
            'Swizzle: [Y,X] -> [S,B]',
            'Y: [[1,1],[2,2]]',
            'X: [[0,1],[0,2]]',
        ].join('\n'), 'Swizzle(Tile)'));

        expect(runtime.tensors.at(-1)).toMatchObject({
            axisLabels: ['S', 'B'],
            shape: [4, 4],
            rootToTensor: [[0, 0], [1, 1], [0, 1], [1, 0], [0, 2], [1, 3], [0, 3], [1, 2]],
        });
    });

    it('supports compositions that become non-injective after width alignment', () => {
        const runtime = buildComposeRuntime(composeState([
            'L: [X] -> [X]',
            'X: [[0],[1]]',
        ].join('\n'), 'L(L)'));

        expect(runtime.tensors.at(-1)).toMatchObject({
            title: 'L(L)',
            shape: [2],
            rootToTensor: [[0], [0], [0], [0]],
        });
    });

    it('tracks final-output coords for non-injective layouts', () => {
        const runtime = buildComposeRuntime(composeState([
            'L1: [X] -> [Y]',
            'X: [[0],[1]]',
            '',
            'L2: [Y] -> [Z]',
            'Y: [[0],[1]]',
        ].join('\n'), 'L2(L1)'));

        expect(runtime.injective).toBe(false);
        expect(runtime.meta.finalOutputLabels).toEqual(['Z']);
        expect(runtime.tensors[0]?.tensorToFinal).toEqual([[0], [0], [0], [0]]);
        expect(runtime.tensors[1]?.tensorToFinal).toEqual([[0], [0]]);
        expect(runtime.tensors[2]?.tensorToFinal).toEqual([[0], null]);
    });

    it('switches propagated labels between inputs and outputs', () => {
        const runtime = buildComposeRuntime(composeState([
            'L: [X] -> [Y]',
            'X: [[1]]',
        ].join('\n'), 'L'));

        expect(propagationLabels(runtime, false)).toEqual([['X'], [2]]);
        expect(propagationLabels(runtime, true)).toEqual([['Y'], [2]]);
    });

    it('supports products as atomic render steps', () => {
        const state = composeState([
            'Left: [T] -> [A]',
            'T: [[1]]',
            '',
            'Right: [U] -> [B]',
            'U: [[1]]',
        ].join('\n'), 'Left * Right');
        const runtime = buildComposeRuntime(state);

        expect(runtime.inputLabels).toEqual(['T', 'U']);
        expect(runtime.tensors.map((tensor) => ({ title: tensor.title, shape: tensor.shape }))).toEqual([
            { title: 'Input Space', shape: [2, 2] },
            { title: 'Left * Right', shape: [2, 2] },
        ]);
        expect(runtime.pythonCode).toContain('layout_tmp1 = Left * Right');
    });

    it('supports products with duplicate inputs and distinct outputs', () => {
        const runtime = buildComposeRuntime(composeState([
            'Left: [A,B] -> [C,D]',
            'A: [[1,0],[2,0]]',
            'B: [[0,1],[0,2]]',
            '',
            'Right: [A,B] -> [C2,D2]',
            'A: [[1,0],[2,0]]',
            'B: [[0,1],[0,2]]',
        ].join('\n'), 'Left * Right'));

        expect(runtime.inputLabels).toEqual(['A', 'B']);
        expect(runtime.inputShape).toEqual([16, 16]);
        expect(runtime.tensors.at(-1)).toMatchObject({
            axisLabels: ['C', 'D', 'C2', 'D2'],
            shape: [4, 4, 4, 4],
        });
        expect(runtime.matrixBlocks.at(-1)?.values).toEqual([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]);
    });

    it('supports products with distinct inputs and duplicate outputs', () => {
        const runtime = buildComposeRuntime(composeState([
            'Left: [A] -> [C]',
            'A: [[1]]',
            '',
            'Right: [B] -> [C]',
            'B: [[1]]',
        ].join('\n'), 'Left * Right'));

        expect(runtime.inputLabels).toEqual(['A', 'B']);
        expect(runtime.inputShape).toEqual([2, 2]);
        expect(runtime.tensors.at(-1)).toMatchObject({
            axisLabels: ['C'],
            shape: [4],
            rootToTensor: [[0], [2], [1], [3]],
        });
    });

    it('supports products with duplicate inputs and duplicate outputs', () => {
        const runtime = buildComposeRuntime(composeState([
            'Left: [A] -> [C]',
            'A: [[1]]',
            '',
            'Right: [A] -> [C]',
            'A: [[1]]',
        ].join('\n'), 'Left * Right'));

        expect(runtime.inputLabels).toEqual(['A']);
        expect(runtime.inputShape).toEqual([4]);
        expect(runtime.tensors.at(-1)).toMatchObject({
            axisLabels: ['C'],
            shape: [4],
            rootToTensor: [[0], [1], [2], [3]],
        });
    });

    it('accepts non-injective specs', () => {
        const runtime = buildComposeRuntime(composeState([
            'Bad: [T] -> [A]',
            'T: [[0],[0]]',
        ].join('\n'), 'Bad'));

        expect(runtime.tensors.at(-1)).toMatchObject({
            title: 'Bad',
            shape: [1],
            rootToTensor: [[0], [0], [0], [0]],
        });
    });

    it('rejects inverse on injective-but-non-bijective layouts', () => {
        const state = composeState([
            'Stretch: [T] -> [A]',
            'T: [[2]]',
        ].join('\n'), 'inv(Stretch)');

        expect(() => buildComposeRuntime(state)).toThrow('Stretch is not bijective, so inv(Stretch) is invalid.');
    });

    it('accepts labeled basis rows with flexible whitespace and reordered inputs', () => {
        const runtime = buildComposeRuntime(composeState([
            'Blocked_Layout: [T,W,R] -> [Y,X]',
            'R:[[1,0],[2,0]]',
            'W : [[0,8],[0,16]]',
            'T   :[[4,0],[8,0],[0,1],[0,2],[0,4]]',
        ].join('\n'), 'Blocked_Layout'));

        expect(runtime.tensors.at(-1)).toMatchObject({
            title: 'Blocked_Layout',
            shape: [16, 32],
            axisLabels: ['Y', 'X'],
        });
    });

    it('rejects unlabeled basis rows', () => {
        const state = composeState([
            'Bad: [T] -> [A]',
            '[[1]]',
        ].join('\n'), 'Bad');

        expect(() => buildComposeRuntime(state)).toThrow('Layout Bad basis rows must use "<label>: <json>" syntax.');
    });

    it('maps largest inputs to H, then L, then S', () => {
        expect(autoColorLayoutState([
            'Blocked_Layout: [T,W,R] -> [Y,X]',
            'T: [[4,0],[8,0],[0,1],[0,2],[0,4]]',
            'W: [[0,8],[0,16]]',
            'R: [[1,0],[2,0]]',
        ].join('\n'), 'Blocked_Layout').mapping).toEqual({ H: 'T', S: 'W', L: 'R' });
    });

    it('leaves unused color channels unmapped when fewer than three inputs exist', () => {
        expect(autoColorLayoutState([
            'Pair: [A,B] -> [C]',
            'A: [[1],[2],[4]]',
            'B: [[8]]',
        ].join('\n'), 'Pair').mapping).toEqual({ H: 'A', S: 'none', L: 'B' });
    });

    it('prefers later dimensions when equal sizes compete for the next color axis', () => {
        expect(autoColorLayoutState([
            'Square: [Y,X] -> [A]',
            'Y: [[1]]',
            'X: [[2]]',
        ].join('\n'), 'Square').mapping).toEqual({ H: 'X', S: 'none', L: 'Y' });
    });

    it('renders the first member of a non-injective cell by default', () => {
        const document = createComposeLayoutDocument(composeState([
            'L: [X] -> [X]',
            'X: [[0],[1]]',
        ].join('\n'), 'L(L)'));

        expect(Array.from(document.tensors.get('compose-step-2') ?? [])).toEqual([0, -1]);
    });

    it('derives multi-input display layers from the focused tensor membership', () => {
        const document = createComposeLayoutDocument(composeState([
            'L: [X] -> [X]',
            'X: [[0],[1]]',
        ].join('\n'), 'L(L)'));
        const mapping = linearLayoutSelectionMapForMeta(document)!;
        const ctx = {
            viewer: {
                getState: () => ({ activeTensorId: 'compose-step-2' }),
                getTensorStatus: (tensorId: string) => {
                    const tensor = document.manifest.tensors.find((entry) => entry.id === tensorId)!;
                    return {
                        id: tensor.id,
                        name: tensor.name,
                        rank: tensor.shape.length,
                        shape: tensor.shape.slice(),
                        axisLabels: tensor.axisLabels ?? [],
                        dtype: tensor.dtype,
                        hasData: true,
                        valueRange: null,
                    };
                },
                getTensorView: (tensorId: string) => document.manifest.tensors.find((entry) => entry.id === tensorId)!.view,
            },
            state: {
                linearLayoutMultiInputState: { 'compose-step-2': 2 },
                linearLayoutSelectionMaps: new Map(),
            },
        } as never;

        expect(linearLayoutMultiInputModel(ctx, mapping)).toEqual({
            focusedTensorId: 'compose-step-2',
            size: 4,
            value: 2,
        });

        const display = linearLayoutDisplayModel(ctx, mapping);
        expect(Array.from(display.rootIndexes)).toEqual([2]);
        expect(display.displayedRootIndexByTensor.get('compose-root')).toEqual([null, null, 2, null]);
        expect(display.displayedRootIndexByTensor.get('compose-step-1')).toEqual([null, 2]);
        expect(display.displayedRootIndexByTensor.get('compose-step-2')).toEqual([2, null]);
    });

    it('keeps multi-input disabled by default until the slider leaves -1', () => {
        const document = createComposeLayoutDocument(composeState([
            'L: [X] -> [X]',
            'X: [[1],[0]]',
        ].join('\n'), 'L'));
        const mapping = linearLayoutSelectionMapForMeta(document)!;
        const ctx = {
            viewer: {
                getState: () => ({ activeTensorId: 'compose-step-1' }),
                getTensorStatus: (tensorId: string) => {
                    const tensor = document.manifest.tensors.find((entry) => entry.id === tensorId)!;
                    return {
                        id: tensor.id,
                        name: tensor.name,
                        rank: tensor.shape.length,
                        shape: tensor.shape.slice(),
                        axisLabels: tensor.axisLabels ?? [],
                        dtype: tensor.dtype,
                        hasData: true,
                        valueRange: null,
                    };
                },
                getTensorView: (tensorId: string) => document.manifest.tensors.find((entry) => entry.id === tensorId)!.view,
            },
            state: {
                linearLayoutMultiInputState: {},
                linearLayoutSelectionMaps: new Map(),
            },
        } as never;

        expect(linearLayoutMultiInputModel(ctx, mapping)).toEqual({
            focusedTensorId: 'compose-step-1',
            size: 2,
            value: -1,
        });
        expect(Array.from(linearLayoutDisplayModel(ctx, mapping).rootIndexes)).toEqual([0, 1, 2, 3]);
        expect(linearLayoutDisplayModel(ctx, mapping).ghostRootIndexesByTensor.get('compose-step-1')).toEqual([
            { coord: [0], rootIndex: 2, layer: 1 },
            { coord: [1], rootIndex: 3, layer: 1 },
        ]);
    });

    it('switches displayed members for a many-to-one tensor when multi-input changes', () => {
        const document = createComposeLayoutDocument(composeState([
            'L: [X] -> [X]',
            'X: [[1],[0]]',
        ].join('\n'), 'L'));
        const mapping = linearLayoutSelectionMapForMeta(document)!;
        const makeCtx = (value: number) => ({
            viewer: {
                getState: () => ({ activeTensorId: 'compose-step-1' }),
                getTensorStatus: (tensorId: string) => {
                    const tensor = document.manifest.tensors.find((entry) => entry.id === tensorId)!;
                    return {
                        id: tensor.id,
                        name: tensor.name,
                        rank: tensor.shape.length,
                        shape: tensor.shape.slice(),
                        axisLabels: tensor.axisLabels ?? [],
                        dtype: tensor.dtype,
                        hasData: true,
                        valueRange: null,
                    };
                },
                getTensorView: (tensorId: string) => document.manifest.tensors.find((entry) => entry.id === tensorId)!.view,
            },
            state: {
                linearLayoutMultiInputState: { 'compose-step-1': value },
                linearLayoutSelectionMaps: new Map(),
            },
        }) as never;

        expect(linearLayoutDisplayModel(makeCtx(0), mapping).displayedRootIndexByTensor.get('compose-step-1')).toEqual([0, 1]);
        expect(linearLayoutDisplayModel(makeCtx(1), mapping).displayedRootIndexByTensor.get('compose-step-1')).toEqual([2, 3]);
    });

    it('selects every input root in an output cell regardless of the displayed member', () => {
        const document = createComposeLayoutDocument(composeState([
            'L: [X] -> [X]',
            'X: [[1],[0]]',
        ].join('\n'), 'L'));
        const mapping = linearLayoutSelectionMapForMeta(document)!;
        const selectedRoots = rootIndexesForCoords(mapping, 'compose-step-1', [[0]]);

        expect(Array.from(selectedRoots)).toEqual([0, 2]);
        expect(coordsForRootIndexes(mapping, 'compose-root', selectedRoots, null)).toEqual([[0], [2]]);
    });

    it('lists every hovered input cell with text and color for many-to-one outputs', () => {
        const document = createComposeLayoutDocument(composeState([
            'L: [X] -> [X]',
            'X: [[1],[0]]',
        ].join('\n'), 'L'));
        const mapping = linearLayoutSelectionMapForMeta(document)!;
        const entries = linearLayoutHoverPopupEntries({
            state: {
                linearLayoutCellTextState: { X: true },
                linearLayoutState: composeState([
                    'L: [X] -> [X]',
                    'X: [[1],[0]]',
                ].join('\n'), 'L'),
            },
        } as never, {
            tensorId: 'compose-step-1',
            tensorName: 'L',
            viewCoord: [0],
            layoutCoord: [0],
            tensorCoord: [0],
            value: 0,
            colorSource: 'custom',
        }, mapping);

        expect(entries).toEqual([
            { text: 'X:0', color: 'rgb(255 0 0)' },
            { text: 'X:2', color: 'rgb(255 0 0)' },
        ]);
    });
});
