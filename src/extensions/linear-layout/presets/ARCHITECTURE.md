The preset catalog is split by instruction because contributors usually know one instruction family well and should be able to edit only that slice.

Each file in this directory exports raw preset definitions for one instruction family:

- `mma.ts`
- `wgmma.ts`
- `ldmatrix.ts`
- `stmatrix.ts`
- `swizzle.ts`
- `index.ts`

Those files are intentionally data-heavy and light on control flow. The goal is to keep instruction knowledge declarative: names, signatures, comments, filter facets, and basis rows live next to each other. `linear-layout.ts` turns that data into UI presets without knowing which instruction families exist.

`types.ts` defines the shared preset shapes. `index.ts` is the catalog registry. It imports instruction-family preset arrays, attaches field metadata, and exports the families consumed by the app.

If a contributor wants to add a new instruction family, the expected workflow is:

1. Add one new file in this directory that exports `ComposeLayoutPresetDefinition[]`.
2. Put architecture compatibility and selector values in each preset's `facets`.
3. Add any new selector fields to a family entry in `index.ts`.
4. Import and append that family in `index.ts`.

The preset widget reads field metadata from the catalog, so new fields appear automatically once they have matching facets. This avoids one giant preset file, reduces merge conflicts, and keeps instruction-specific review local to the family file.

Field metadata in `index.ts` has a small contract:

- `key`: the selector concept and the matching key in each preset's `facets`.
- `label`: the visible label shown in the preset selector.
- `placeholder`: the empty-input hint for that selector.
- `order`: where the selector appears; lower values are rendered first.
- `required`: show the selector even before any other selection is made.
- `dependsOn`: for optional selectors, wait until these other field keys have selected values before showing it. This checks only that a dependency is selected, not which value it has.
- `values`: preferred display order for known values; facet values not listed here still appear after the preferred values.

Value-specific visibility comes from the presets themselves. After the dependency fields are selected, the widget computes which options still exist for each optional field by filtering presets against the current selection. For example, `operand`, `trans`, and `major` can all depend on `instruction`: when `instruction` is `mma`, only MMA presets provide `operand` values, so `operand` appears; when `instruction` is `swizzle`, only swizzle presets provide `major` values, so `major` appears instead.

For example, an AMD MFMA family would start with a new `amd.ts` file:

```ts
import type { ComposeLayoutPresetDefinition } from './types.js';

export const AMD_PRESET_DEFINITIONS = [
    {
        name: 'AMD_mfma_m32n32k8_A_f16',
        facets: {
            gpuArch: ['gfx90a', 'gfx942'],
            instruction: 'mfma',
            matrixSize: 'm32n32k8',
            dtype: 'f16',
            operand: 'A',
        },
        signature: '[T,R] -> [M,K]',
        rows: [
            ['T', '[[1,0],[2,0]]'],
            ['R', '[[0,1],[0,2]]'],
        ],
        inputName: 'Hardware Layout',
        comments: ['mfma layout example; replace basis rows with verified AMD mapping.'],
    },
] satisfies ComposeLayoutPresetDefinition[];
```

Register the family in `index.ts`. Reuse existing fields when possible and add new fields only for real selector concepts:

```ts
import { AMD_PRESET_DEFINITIONS } from './amd.js';
import type { ComposeLayoutPresetFamily, ComposeLayoutPresetFieldDefinition } from './types.js';

const AMD_PRESET_FIELDS = [
    {
        key: 'gpuArch',
        label: 'GPU Arch',
        placeholder: 'Type GPU arch',
        order: 10,
        required: true,
        values: ['gfx90a', 'gfx942'],
    },
] satisfies ComposeLayoutPresetFieldDefinition[];

export const COMPOSE_LAYOUT_PRESET_FAMILIES = [
    // existing families...
    { fields: AMD_PRESET_FIELDS, presets: AMD_PRESET_DEFINITIONS },
] satisfies ComposeLayoutPresetFamily[];
```

If AMD needs a selector concept that NVIDIA presets do not use, add a field for that concept and put the matching value in each preset's `facets`:

```ts
const AMD_PRESET_FIELDS = [
    {
        key: 'gpuArch',
        label: 'GPU Arch',
        placeholder: 'Type GPU arch',
        order: 10,
        required: true,
        values: ['gfx90a', 'gfx942'],
    },
    {
        key: 'waveSize',
        label: 'Wave Size',
        placeholder: 'Type wave size',
        order: 80,
        dependsOn: ['instruction'],
    },
] satisfies ComposeLayoutPresetFieldDefinition[];
```

```ts
facets: {
    gpuArch: 'gfx942',
    instruction: 'mfma',
    matrixSize: 'm32n32k8',
    dtype: 'f16',
    operand: 'A',
    waveSize: 'wave64',
},
```
