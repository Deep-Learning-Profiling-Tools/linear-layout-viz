/**
 * Parsed representation of one named compose-layout basis block from the specs editor.
 *
 * `inputs` and `outputs` preserve the labels from the signature line, while `bases[inputIndex][bitIndex][outputIndex]` stores the numeric contribution of each input bit to each output label.
 *
 * @example
 * const spec: NamedLayoutSpec = {
 *   name: 'mma',
 *   inputs: ['A', 'B'],
 *   outputs: ['M', 'N'],
 *   bases: [
 *     [[1, 0], [0, 1]],
 *     [[1, 1]],
 *   ],
 * };
 * console.assert(spec.inputs[0] === 'A');
 * console.assert(spec.bases[0][1][1] === 1);
 */
export type NamedLayoutSpec = {
    name: string;
    inputs: string[];
    outputs: string[];
    bases: number[][][];
};

/**
 * Parses the linear-layout specs editor notation into named basis blocks.
 *
 * The notation uses a signature line such as `mma: [A,B] -> [M,N]` followed by one `<input label>: <json>` basis row for each input label. Blank lines and `#` comments are ignored before syntax parsing.
 *
 * @param text - Raw contents of the layout specs textarea, including signature lines, labeled JSON basis rows, blank lines, and optional `#` comments.
 * @returns Parsed layout specs in editor order; each spec contains the signature name, input labels, output labels, and basis rows reordered to match the signature input order.
 * @throws Error when a required basis row is missing, a row does not use `<label>: <json>` syntax, a row names an input label not present in the signature, an input label is duplicated within a layout, the basis JSON is invalid for the output count, or two layouts use the same name.
 * @example
 * const specs = parseLayoutSpecs(`
 * mma: [A,B] -> [M,N]
 * A: [[1,0],[0,1]] # A contributes two bits
 * B: [[1,1]]
 * `);
 *
 * console.assert(specs[0]?.name === 'mma');
 * console.assert(specs[0]?.inputs.join(',') === 'A,B');
 * console.assert(specs[0]?.bases[0]?.[1]?.[1] === 1);
 *
 * @example
 * try {
 *   parseLayoutSpecs(`
 *   mma: [A] -> [M]
 *   Z: [[1]]
 *   `);
 * } catch (error) {
 *   console.assert(error instanceof Error);
 *   console.assert(error.message === 'Layout mma received basis row for unknown input label Z.');
 * }
 */
export function parseLayoutSpecs(text: string): NamedLayoutSpec[] {
    const lines = text.replace(/\r\n/g, '\n').split('\n');
    const specs: NamedLayoutSpec[] = [];
    let index = 0;
    while (index < lines.length) {
        while (index < lines.length && !stripLayoutComment(lines[index]!).trim()) index += 1;
        if (index >= lines.length) break;
        const signatureLine = stripLayoutComment(lines[index]!).trim();
        const signature = parseSignature(signatureLine);
        index += 1;
        const basisByLabel = new Map<string, number[][]>();
        // read rows by label instead of position so the canonical formatter can
        // preserve signature order while users keep spec rows grouped naturally.
        for (let axis = 0; axis < signature.inputs.length; axis += 1) {
            while (index < lines.length && !stripLayoutComment(lines[index]!).trim()) index += 1;
            const line = stripLayoutComment(lines[index] ?? '').trim();
            if (!line) {
                throw new Error(`Layout ${signature.name} is missing basis row for ${signature.inputs[axis]}.`);
            }
            const match = line.match(/^([A-Za-z][0-9]*)\s*:\s*(.+)$/);
            if (!match) {
                throw new Error(`Layout ${signature.name} basis rows must use "<label>: <json>" syntax.`);
            }
            const axisLabel = match[1]!;
            if (!signature.inputs.includes(axisLabel)) {
                throw new Error(`Layout ${signature.name} received basis row for unknown input label ${axisLabel}.`);
            }
            if (basisByLabel.has(axisLabel)) {
                throw new Error(`Layout ${signature.name} has duplicate basis row for ${axisLabel}.`);
            }
            basisByLabel.set(axisLabel, parseBasisRow(match[2]!, signature.outputs.length, axisLabel));
            index += 1;
        }
        const bases = signature.inputs.map((axisLabel) => {
            const basis = basisByLabel.get(axisLabel);
            if (!basis) throw new Error(`Layout ${signature.name} is missing basis row for ${axisLabel}.`);
            return basis;
        });
        specs.push({
            name: signature.name,
            inputs: signature.inputs,
            outputs: signature.outputs,
            bases,
        });
        while (index < lines.length && !stripLayoutComment(lines[index]!).trim()) index += 1;
    }
    const duplicate = duplicateValue(specs.map((spec) => spec.name));
    if (duplicate) throw new Error(`Layout names must be unique; received duplicate ${duplicate}.`);
    return specs;
}

/**
 * Removes the trailing `#` comment portion from a single linear-layout specs line before the parser trims and interprets it.
 *
 * @param line - One raw line from the specs textarea, such as a signature or labeled basis row that may end with a `#` explanatory comment.
 * @returns The portion of `line` before the first `#`, preserving any whitespace that appeared before the comment marker.
 * @noThrows The implementation only applies a regular-expression replacement to the provided string and does not parse JSON or validate layout syntax.
 * @example
 * console.assert(stripLayoutComment('A: [[1,0]] # row for A') === 'A: [[1,0]] ');
 * console.assert(stripLayoutComment('mma: [A] -> [M]') === 'mma: [A] -> [M]');
 */
export function stripLayoutComment(line: string): string {
    return line.replace(/#.*$/, '');
}

/**
 * Parses the signature line that starts a compose-layout block, such as
 * `mma: [A,B] -> [C,D]`.
 *
 * @param line - Trimmed signature text containing a layout identifier, an input label list, `->`, and an output label list.
 * @returns The layout name together with ordered input and output axis labels used to read the following basis rows.
 * @throws Error when the line does not match `<name>: [labels] -> [labels]`, when a label is not shaped like `T`, `A0`, or `B12`, or when an input or output label is repeated.
 * @example
 * parseSignature('mma: [A0,B12] -> [C]');
 * // => { name: 'mma', inputs: ['A0', 'B12'], outputs: ['C'] }
 *
 * @example
 * parseSignature('mma: [A,A] -> [C]');
 * // throws Error('Layout mma has duplicate input label A.')
 */
export function parseSignature(line: string): { name: string; inputs: string[]; outputs: string[] } {
    const match = line.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\[(.*)\]\s*->\s*\[(.*)\]\s*$/);
    if (!match) {
        throw new Error('Each specification must start with "<name>: [labels] -> [labels]".');
    }
    const name = match[1]!;
    const inputs = parseLabelList(match[2] ?? '', `${name} inputs`);
    const outputs = parseLabelList(match[3] ?? '', `${name} outputs`);
    const duplicateInput = duplicateValue(inputs);
    if (duplicateInput) throw new Error(`Layout ${name} has duplicate input label ${duplicateInput}.`);
    const duplicateOutput = duplicateValue(outputs);
    if (duplicateOutput) throw new Error(`Layout ${name} has duplicate output label ${duplicateOutput}.`);
    return { name, inputs, outputs };
}

/**
 * Serializes parsed layout specs into the canonical compose-layout text used when matching presets.
 *
 * @param specs - Parsed layout specs with a name, ordered input labels, ordered output labels, and one basis matrix per input axis.
 * @returns Canonical text with one signature line per spec and JSON basis rows below it, separated by blank lines between specs.
 * @noThrows Formatting only joins existing arrays and JSON-stringifies basis rows from parsed `NamedLayoutSpec` objects; normal spec data is not validated here.
 * @example
 * formatSpecsText([{
 *     name: 'mma',
 *     inputs: ['A', 'B'],
 *     outputs: ['C'],
 *     bases: [[[1]], [[0]]],
 * }]);
 * // => 'mma: [A,B] -> [C]\nA: [[1]]\nB: [[0]]'
 */
export function formatSpecsText(specs: NamedLayoutSpec[]): string {
    return specs.map((spec) => [
        `${spec.name}: [${spec.inputs.join(',')}] -> [${spec.outputs.join(',')}]`,
        ...spec.bases.map((row, axis) => `${spec.inputs[axis]}: ${JSON.stringify(row)}`),
    ].join('\n')).join('\n\n');
}

/**
 * Splits a comma-separated compose-layout label list into validated axis labels.
 *
 * @param source - Text from inside a signature bracket pair, such as `A0, B12`; blank or whitespace-only text represents no labels.
 * @param label - Human-readable context included in validation errors, such as `mma inputs` or `mma outputs`.
 * @returns Trimmed labels in source order, or an empty array when the list is blank.
 * @throws Error when any entry is not a letter followed by optional digits, for example `_A`, `A_B`, or `1A`.
 * @example
 * parseLabelList('A0, B12, T', 'mma inputs');
 * // => ['A0', 'B12', 'T']
 *
 * @example
 * parseLabelList('A_B', 'mma inputs');
 * // throws Error('mma inputs may only contain labels like T, A0, or B12 (received "A_B").')
 */
function parseLabelList(source: string, label: string): string[] {
    const trimmed = source.trim();
    if (!trimmed) return [];
    return trimmed.split(',').map((entry) => {
        const value = entry.trim();
        if (!/^[A-Za-z][0-9]*$/.test(value)) {
            throw new Error(`${label} may only contain labels like T, A0, or B12 (received ${JSON.stringify(value)}).`);
        }
        return value;
    });
}

/**
 * Parses one input-axis basis row from compose-layout text into non-negative integer output contributions.
 *
 * @param line - JSON text after the axis label, expected to be an array of basis vectors such as `[[1,0],[0,1]]`.
 * @param outputCount - Required length of each basis vector, matching the number of output labels in the signature.
 * @param axisLabel - Input axis label used to identify the failing row in error messages.
 * @returns Basis vectors for the input axis, with each vector containing one non-negative integer contribution per output axis.
 * @throws Error when the row is not valid JSON, is not a JSON array, contains a non-array basis vector, has a vector whose length differs from `outputCount`, or contains a negative or fractional value.
 * @example
 * parseBasisRow('[[1,0],[0,1]]', 2, 'A');
 * // => [[1, 0], [0, 1]]
 *
 * @example
 * parseBasisRow('[[1]]', 2, 'A');
 * // throws Error('A basis 1 must have length 2.')
 */
function parseBasisRow(line: string, outputCount: number, axisLabel: string): number[][] {
    let parsed: unknown;
    try {
        parsed = JSON.parse(line);
    } catch {
        throw new Error(`${axisLabel} bases must be valid JSON.`);
    }
    if (!Array.isArray(parsed)) throw new Error(`${axisLabel} bases must be a JSON array.`);
    return parsed.map((basis, basisIndex) => {
        if (!Array.isArray(basis)) {
            throw new Error(`${axisLabel} basis ${basisIndex + 1} must be an array.`);
        }
            if (basis.length !== outputCount) {
                throw new Error(`${axisLabel} basis ${basisIndex + 1} must have length ${outputCount}.`);
            }
            // bases are integer bit contributions; negative or fractional values
            // would break the gf(2) matrix conversion in linear-layout.ts.
            return basis.map((value, outputAxis) => {
            if (!Number.isInteger(value) || Number(value) < 0) {
                throw new Error(`${axisLabel} basis ${basisIndex + 1}[${outputAxis + 1}] must be a non-negative integer.`);
            }
            return Number(value);
        });
    });
}

/**
 * Finds the first string that appears more than once while preserving the caller's list order.
 *
 * @param values - Ordered string tokens collected from parsed linear-layout input, such as dimension names or selector values.
 * @returns The first entry whose value has already appeared earlier in `values`, or `null` when every string is unique.
 * @noThrows Uses only local `Set` membership checks over the provided array and does not parse, allocate external resources, or reject any string contents.
 * @example
 * duplicateValue(['m', 'n', 'k', 'n']);
 * // Returns 'n'.
 *
 * duplicateValue(['m', 'n', 'k']);
 * // Returns null.
 */
function duplicateValue(values: string[]): string | null {
    const seen = new Set<string>();
    for (const value of values) {
        if (seen.has(value)) return value;
        seen.add(value);
    }
    return null;
}
