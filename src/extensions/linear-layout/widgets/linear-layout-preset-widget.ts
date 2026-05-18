import { escapeInfo, infoButton } from '@tensor-viz/viewer-demo/extension-api';
import {
    cloneComposeLayoutPresetSelection,
    composeLayoutPresetFields,
    composeLayoutPresets,
    composeLayoutPresetForSelection,
    composeLayoutPresetOptions,
    normalizeComposeLayoutPresetSelection,
    type ComposeLayoutPresetField,
    type ComposeLayoutPresetOptions,
    type ComposeLayoutPresetSelection,
} from '../linear-layout.js';
import type { LinearLayoutUiContext } from '../linear-layout-state.js';
import { applyLinearLayoutSpec } from './linear-layout-widget-actions.js';

let clearPresetOutsideClickHandler: (() => void) | null = null;

/**
 * Builds the collapsible help panel shown above the preset selector, including usage instructions, supported instruction names, and sample GPU layout selections.
 *
 * @returns An HTML string for the preset widget's usage guide; callers insert it into the sidebar before the preset search fields.
 * @noThrows The helper reads the preset catalog and escapes the instruction list before interpolation; it accepts no caller-provided input and has no error branch.
 * @example
 * const html = linearLayoutPresetHelpHtml();
 * expect(html).toContain('<details class="usage-guide">');
 * expect(html).toContain('Load a preset layout');
 * expect(html).toContain('Instruction: mma');
 */
function linearLayoutPresetHelpHtml(): string {
    const instructions = composeLayoutPresetOptions(undefined).instruction.join(', ');
    return `
      <details class="usage-guide">
        <summary>How do I use this?</summary>
        <div class="usage-guide-body">
          <div class="usage-guide-step">
            <span>Load a preset layout by typing into text fields and selecting values from the dropdowns. Once the text <strong>No preset matches the current selection yet.</strong> is replaced with <strong>Selected preset: &lt;preset&gt;</strong>, click <strong>Load Preset</strong> to visualize the specified layout.</span>
          </div>
          <div class="usage-guide-step">
            <span>Currently supported instructions: <strong>${escapeInfo(instructions || 'none')}</strong>.</span>
          </div>
          <div class="usage-guide-subtitle">Examples</div>
          <div class="usage-guide-examples">
            <div class="usage-guide-example">
              <code>GPU Arch: sm_80
Instruction: mma
Matrix Size: m16n8k16
DType: b16
Operand: A</code>
            </div>
            <div class="usage-guide-example">
              <code>GPU Arch: sm_100a
Instruction: swizzle
Matrix Size: 128B
DType: b128
Major: K-major</code>
            </div>
            <div class="usage-guide-example">
              <code>GPU Arch: sm_75
Instruction: ldmatrix
Matrix Size: m8n8.x1
DType: b16
Transpose: no</code>
            </div>
            <div class="usage-guide-example">
              <code>GPU Arch: sm_100
Instruction: stmatrix
Matrix Size: m16n16.x2
DType: b8
Transpose: yes</code>
            </div>
            <div class="usage-guide-example">
              <code>GPU Arch: sm_90a
Instruction: wgmma
Matrix Size: m64n64
DType: b32
Operand: D</code>
            </div>
          </div>
        </div>
      </details>
    `;
}

/**
 * Renders one text-search row in the preset chooser, with the field label, escaped current value, placeholder text, and matching valid and invalid preset options.
 *
 * @param field - Preset catalog field definition that supplies the DOM id, display label, placeholder, and selection key for this row.
 * @param value - Current text typed for this preset field, before filtering and after any value stored in the preset selection state.
 * @param validOptions - Catalog option labels that are valid for the current partial preset selection.
 * @param invalidOptions - Option labels known to the catalog but unavailable for the current partial selection, rendered separately so the UI can explain mismatches.
 * @param selection - Current preset selection object used to mark the option that is already selected for this field key.
 * @returns An HTML fragment for a `<div class="preset-field">` containing the input and filtered option list for insertion into the preset sidebar.
 * @noThrows Rendering is synchronous string construction over supplied metadata and arrays; the helper does not parse the selection or perform failing catalog lookups.
 * @example
 * const html = presetSearchField(
 *   { id: 'preset-instruction', key: 'instruction', label: 'Instruction', placeholder: 'mma' },
 *   'wm',
 *   ['mma', 'wgmma'],
 *   ['ldmatrix'],
 *   { instruction: 'wgmma' },
 * );
 * expect(html).toContain('data-preset-field="preset-instruction"');
 * expect(html).toContain('value="wm"');
 * expect(html).toContain('wgmma');
 * expect(html).not.toContain('ldmatrix');
 */
function presetSearchField(
    field: ComposeLayoutPresetField,
    value: string,
    validOptions: string[],
    invalidOptions: string[],
    selection: ComposeLayoutPresetSelection,
): string {
    const validMatches = filteredPresetOptions(validOptions, value);
    const invalidMatches = filteredPresetOptions(invalidOptions, value);
    return `
      <div class="preset-field" data-preset-field="${field.id}">
        <span class="meta-label">${field.label}</span>
        <input id="${field.id}" type="text" value="${escapeInfo(value)}" placeholder="${escapeInfo(field.placeholder)}" autocomplete="off" />
        <div class="preset-option-list">
          ${presetOptionsHtml(field.key, field.id, validMatches, invalidMatches, selection)}
        </div>
      </div>
    `;
}

/**
 * Filters preset option labels for the chooser by comparing each option against a lowercase, alphanumeric-only fuzzy search query.
 *
 * @param options - Preset option labels from the catalog, such as GPU architecture, instruction, matrix size, dtype, or operand values.
 * @param query - User-entered search text; punctuation, spaces, and case are ignored before matching.
 * @returns The original option array when the normalized query is empty, otherwise only the labels that satisfy the preset fuzzy matcher.
 * @noThrows The helper performs only string normalization and array filtering, so valid string arrays and a string query do not enter an expected error path.
 * @example
 * expect(filteredPresetOptions(['m16n8k16', 'm64n64', '128B'], 'M16-N8')).toEqual(['m16n8k16']);
 * expect(filteredPresetOptions(['mma', 'wgmma'], '   ')).toEqual(['mma', 'wgmma']);
 */
function filteredPresetOptions(options: string[], query: string): string[] {
    const normalizedQuery = query.toLowerCase().replace(/[^a-z0-9]/g, '');
    if (!normalizedQuery) return options;
    return options.filter((option) => fuzzyPresetMatch(option, normalizedQuery));
}

/**
 * Tests whether a preset option should remain visible for a normalized search query.
 *
 * A match succeeds when the option, after lowercasing and removing non-alphanumeric
 * characters, either contains the query as a contiguous substring or contains all
 * query characters in order as a fuzzy subsequence.
 *
 * @param option - Human-readable preset option label from the preset catalog, such as `MMA 16x8x16`.
 * @param normalizedQuery - Already lowercased alphanumeric search text from the input box, such as `mma168`.
 * @returns `true` when the option should be shown in the preset dropdown for the query; otherwise `false`.
 * @noThrows The function only normalizes and scans string arguments supplied by TypeScript callers, with no parsing, DOM access, or external state lookup.
 * @example
 * fuzzyPresetMatch('MMA 16x8x16', 'mma168'); // true
 * fuzzyPresetMatch('Swizzle 32B', 'mma'); // false
 */
function fuzzyPresetMatch(option: string, normalizedQuery: string): boolean {
    const normalizedOption = option.toLowerCase().replace(/[^a-z0-9]/g, '');
    if (normalizedOption.includes(normalizedQuery)) return true;
    let queryIndex = 0;
    for (const char of normalizedOption) {
        if (char === normalizedQuery[queryIndex]) queryIndex += 1;
        if (queryIndex === normalizedQuery.length) return true;
    }
    return false;
}

/**
 * Installs the preset text-input handlers that keep typed facet values and dropdowns in sync.
 *
 * Focus opens the matching dropdown, input events copy the element value into
 * `linearLayoutState.presetSelection[field]`, blur normalizes the full selection,
 * and Enter clicks the first visible option for that input when one exists.
 *
 * @param ctx - Linear-layout widget context containing the preset widget root and mutable sidebar state.
 * @param input - Preset facet `<input>` element to bind, or `null` when that field is not currently rendered.
 * @param field - Key in `linearLayoutState.presetSelection` that receives this input's typed value.
 * @returns Nothing; subsequent DOM events on `input` update preset selection and refresh the preset controls.
 * @noThrows A missing input is ignored with optional chaining, and the installed handlers guard absent option buttons before using them.
 * @example
 * const input = ctx.linearLayoutPresetWidget.querySelector<HTMLInputElement>('#linear-layout-preset-instruction');
 * bindPresetInput(ctx, input, 'instruction');
 * input!.value = 'mma';
 * input!.dispatchEvent(new Event('input'));
 * // ctx.state.linearLayoutState.presetSelection.instruction is now 'mma'.
 */
function bindPresetInput(
    ctx: LinearLayoutUiContext,
    input: HTMLInputElement | null,
    field: string,
): void {
    input?.addEventListener('focus', () => {
        syncPresetControls(ctx, input.id);
    });
    input?.addEventListener('input', () => {
        ctx.state.linearLayoutState.presetSelection = {
            ...ctx.state.linearLayoutState.presetSelection,
            [field]: input.value,
        };
        syncPresetControls(ctx, input.id);
    });
    input?.addEventListener('blur', () => {
        // blur is the commit point for typed text.  Normalizing here lets users
        // type partial values without immediately clearing their in-progress input.
        ctx.state.linearLayoutState.presetSelection = normalizeComposeLayoutPresetSelection(
            ctx.state.linearLayoutState.presetSelection,
        );
        syncPresetControls(ctx, null);
    });
    input?.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter') return;
        const firstOption = ctx.linearLayoutPresetWidget.querySelector<HTMLButtonElement>(
            `[data-preset-input="${input.id}"][data-preset-value]`,
        );
        if (!firstOption) return;
        event.preventDefault();
        firstOption.click();
    });
}

/**
 * Binds rendered preset option buttons so choosing a dropdown entry updates the selection model.
 *
 * Each `[data-preset-input][data-preset-value]` button keeps focus on the text input during
 * mousedown and, on click, maps its input id back to a preset field before applying the button's
 * value and validity flag to `linearLayoutState.presetSelection`.
 *
 * @param ctx - Linear-layout widget context containing the preset widget root, rendered option buttons, and mutable preset-selection state.
 * @returns Nothing; click and mousedown listeners are attached to the currently rendered preset option buttons.
 * @noThrows Buttons whose `data-preset-input` no longer maps to a known preset field are ignored, and missing dataset values fall back to empty strings.
 * @example
 * bindPresetOptions(ctx);
 * const option = ctx.linearLayoutPresetWidget.querySelector<HTMLButtonElement>('[data-preset-input="linear-layout-preset-instruction"][data-preset-value="mma"]')!;
 * option.click();
 * // ctx.state.linearLayoutState.presetSelection reflects the clicked `mma` instruction option.
 */
function bindPresetOptions(ctx: LinearLayoutUiContext): void {
    ctx.linearLayoutPresetWidget.querySelectorAll<HTMLButtonElement>('[data-preset-input][data-preset-value]').forEach((button) => {
        button.addEventListener('mousedown', (event) => {
            event.preventDefault();
        });
        button.addEventListener('click', () => {
            const inputId = button.dataset.presetInput ?? '';
            const field = composeLayoutPresetFields().find((candidate) => candidate.id === inputId)?.key ?? null;
            if (!field) return;
            ctx.state.linearLayoutState.presetSelection = presetSelectionForOption(
                ctx.state.linearLayoutState.presetSelection,
                field,
                button.dataset.presetValue ?? '',
                button.dataset.presetValidity === 'invalid',
            );
            syncPresetControls(ctx, null);
        });
    });
}

/**
 * Refreshes the preset chooser DOM from the current preset selection.
 *
 * The sync updates visible field inputs, rebuilds each field's filtered option list,
 * rebinds option buttons, opens or closes dropdowns, writes the selected-preset summary,
 * and enables the Load Preset button only when the selection resolves to a catalog preset.
 * If no input is active and the visible field set changed, the whole preset widget is rerendered.
 *
 * @param ctx - Linear-layout widget context containing the preset widget root and current `linearLayoutState.presetSelection`.
 * @param activeInputId - Id of the focused preset input whose typed value must be preserved, or `null` when no preset input is active.
 * @returns Nothing; the preset widget's inputs, option lists, summary text, dropdown visibility, and load-button disabled state are updated in place.
 * @noThrows Missing optional DOM nodes are skipped with null checks, so partially rendered preset widgets are refreshed on a best-effort basis rather than treated as errors.
 * @example
 * ctx.state.linearLayoutState.presetSelection = normalizeComposeLayoutPresetSelection(undefined);
 * syncPresetControls(ctx, null);
 * const summary = ctx.linearLayoutPresetWidget.querySelector('#linear-layout-preset-summary')!;
 * // summary.textContent is either a selected preset title or "No preset matches the current selection yet.".
 */
function syncPresetControls(ctx: LinearLayoutUiContext, activeInputId: string | null): void {
    const presetOptions = composeLayoutPresetOptions(ctx.state.linearLayoutState.presetSelection);
    if (activeInputId === null && renderedPresetFieldIds(ctx.linearLayoutPresetWidget) !== visiblePresetFields(ctx.state.linearLayoutState.presetSelection, presetOptions).map((field) => field.id).join(',')) {
        // field visibility depends on selected facets, so re-render only when the
        // field set changes; otherwise update inputs in place to preserve focus.
        renderLinearLayoutPresetWidget(ctx);
        return;
    }
    const preset = composeLayoutPresetForSelection(ctx.state.linearLayoutState.presetSelection);
    const renderedFields = composeLayoutPresetFields().filter((field) => (
        ctx.linearLayoutPresetWidget.querySelector<HTMLElement>(`[data-preset-field="${field.id}"]`) !== null
    ));
    renderedFields.forEach((field) => {
        const input = ctx.linearLayoutPresetWidget.querySelector<HTMLInputElement>(`#${CSS.escape(field.id)}`);
        const list = ctx.linearLayoutPresetWidget.querySelector<HTMLElement>(`[data-preset-field="${field.id}"] .preset-option-list`);
        if (!input || !list) return;
        if (field.id !== activeInputId) input.value = ctx.state.linearLayoutState.presetSelection[field.key] ?? '';
        const validOptions = presetFieldOptions(presetOptions, field.key);
        list.innerHTML = presetOptionsHtml(
            field.key,
            field.id,
            filteredPresetOptions(validOptions, input.value),
            filteredPresetOptions(invalidPresetFieldOptions(field.key, validOptions), input.value),
            {
                ...ctx.state.linearLayoutState.presetSelection,
                [field.key]: input.value,
            },
        );
    });
    bindPresetOptions(ctx);
    setPresetDropdownVisibility(ctx.linearLayoutPresetWidget, activeInputId);
    const summary = ctx.linearLayoutPresetWidget.querySelector<HTMLElement>('#linear-layout-preset-summary');
    if (summary) {
        summary.innerHTML = preset
            ? `Selected preset: <span class="inline-code">${escapeInfo(preset.title)}</span>`
            : 'No preset matches the current selection yet.';
    }
    const loadPreset = ctx.linearLayoutPresetWidget.querySelector<HTMLButtonElement>('#linear-layout-load-preset');
    if (loadPreset) loadPreset.disabled = preset === null;
}

/**
 * Builds the tooltip text shown on a conflicting preset option, naming any
 * already-selected preset fields that would be cleared if the user chooses it.
 *
 * @param field - Preset field key being changed, such as an instruction-family or layout-mode selector key from `composeLayoutPresetFields()`.
 * @param value - Option label/value the user is hovering or about to choose from the conflicting option list.
 * @param selection - Current preset selection map before the conflicting option is applied.
 * @returns Sentence for the option's `data-info` attribute; includes cleared field labels when the recovery selection drops existing values.
 * @noThrows The helper only derives a normalized selection from preset catalog metadata and formats labels into a string; missing conflicts simply produce the generic clear-warning text.
 * @example
 * const message = invalidPresetOptionInfo('instruction', 'swizzle', {
 *   instruction: 'mma',
 *   element: 'f16',
 *   shape: 'm16n8k16',
 * });
 * // message is either the generic warning or a label-specific warning, for example:
 * // 'Selecting this clears conflicting fields: Shape'
 */
function invalidPresetOptionInfo(field: string, value: string, selection: ComposeLayoutPresetSelection): string {
    const nextSelection = presetSelectionForOption(selection, field, value, true);
    const clearedFields = composeLayoutPresetFields()
        .filter(({ key }) => key !== field && selection[key] && !nextSelection[key])
        .map(({ label }) => label);
    if (clearedFields.length === 0) {
        return 'Selecting this clears conflicting fields.';
    }
    return `Selecting this clears conflicting fields: ${clearedFields.join(', ')}`;
}

/**
 * Renders the autocomplete option list for one preset input, separating options
 * that still match the current preset path from options that would start a new
 * compatible path and clear conflicts.
 *
 * @param field - Preset field key whose option list is being rendered.
 * @param inputId - DOM id of the text input that should receive clicks from the generated option buttons.
 * @param validOptions - Option values that match the current partial preset selection.
 * @param invalidOptions - Catalog option values for the same field that conflict with the current selection but are still offered as recovery choices.
 * @param selection - Current preset selection map used to explain what each invalid option would clear.
 * @returns HTML fragment for the option list, including valid buttons, an optional conflict divider, invalid buttons with `data-info`, or a `mapping-empty` span when no options match.
 * @noThrows Rendering is deterministic string assembly over caller-provided arrays; option text and conflict messages are escaped before insertion into attributes or button text.
 * @example
 * const html = presetOptionsHtml(
 *   'instruction',
 *   'preset-instruction',
 *   ['mma'],
 *   ['swizzle'],
 *   { instruction: '', shape: 'm16n8k16' },
 * );
 * // html contains a valid button with data-preset-value="mma",
 * // a "conflicts with current selection" divider,
 * // and an invalid button with data-preset-validity="invalid".
 *
 * @example
 * presetOptionsHtml('instruction', 'preset-instruction', [], [], {});
 * // '<span class="mapping-empty">no matches</span>'
 */
function presetOptionsHtml(
    field: string,
    inputId: string,
    validOptions: string[],
    invalidOptions: string[],
    selection: ComposeLayoutPresetSelection,
): string {
    if (validOptions.length === 0 && invalidOptions.length === 0) return '<span class="mapping-empty">no matches</span>';
    return [
        ...validOptions.map((option) => (
            `<button class="preset-option" type="button" data-preset-input="${inputId}" data-preset-value="${escapeInfo(option)}" data-preset-validity="valid">${escapeInfo(option)}</button>`
        )),
        ...(invalidOptions.length > 0
            ? [`<div class="preset-option-divider"><span>conflicts with current selection</span>${infoButton("You didn't make an error. You can still choose one of these options, and the preset widget will clear any conflicting fields automatically.")}</div>`]
            : []),
        ...invalidOptions.map((option) => (
            `<button class="preset-option preset-option-invalid" type="button" data-preset-input="${inputId}" data-preset-value="${escapeInfo(option)}" data-preset-validity="invalid" data-info="${escapeInfo(invalidPresetOptionInfo(field, option, selection))}">${escapeInfo(option)}</button>`
        )),
    ].join('');
}

/**
 * Finds preset catalog options for a field that are not valid for the current
 * partial selection so the UI can show them as conflict-recovery choices.
 *
 * @param field - Preset field key to read from the complete preset catalog.
 * @param validOptions - Options for the same field that already match the current partial selection and should not be repeated in the invalid list.
 * @returns Catalog option values for `field` excluding every value present in `validOptions`.
 * @noThrows The helper only reads the preset catalog with no active selection and filters strings; an unknown field naturally yields whatever the catalog lookup returns for that key.
 * @example
 * const invalid = invalidPresetFieldOptions('instruction', ['mma']);
 * // invalid contains instruction options from the preset catalog other than 'mma',
 * // which the widget renders below the conflict divider.
 */
function invalidPresetFieldOptions(field: string, validOptions: string[]): string[] {
    const allOptions = presetFieldOptions(composeLayoutPresetOptions(undefined), field);
    return allOptions.filter((option) => !validOptions.includes(option));
}

/**
 * Applies a clicked preset option and returns the normalized selection the
 * sidebar should store, preserving compatible fields and clearing values that
 * would silently block a matching preset.
 *
 * @param selection - Current preset selection map from the linear-layout sidebar before the option click.
 * @param field - Preset field key being changed by the clicked option.
 * @param value - Option value selected by the user for `field`.
 * @param invalid - Whether the clicked option came from the conflict list instead of the valid-options list.
 * @returns Normalized preset selection after the click; valid options merge into the existing selection, while invalid options start a new path, copy only fields that still match a preset, and clear fields hidden by the new path.
 * @noThrows The function delegates normalization and catalog checks to preset helpers and has no explicit error branch; incompatible fields are represented by empty selection values instead of thrown errors.
 * @example
 * const next = presetSelectionForOption(
 *   { instruction: 'mma', element: 'f16', shape: 'm16n8k16' },
 *   'instruction',
 *   'swizzle',
 *   true,
 * );
 * // next.instruction === 'swizzle'; fields incompatible with the swizzle path,
 * // such as an mma-only shape, are cleared rather than kept as blockers.
 *
 * @example
 * const next = presetSelectionForOption(
 *   { instruction: 'mma', element: 'f16' },
 *   'shape',
 *   'm16n8k16',
 *   false,
 * );
 * // next keeps the existing instruction and element and records shape: 'm16n8k16'.
 */
function presetSelectionForOption(
    selection: ComposeLayoutPresetSelection,
    field: string,
    value: string,
    invalid: boolean,
) {
    if (!invalid) {
        return normalizeComposeLayoutPresetSelection({
            ...selection,
            [field]: value,
        });
    }
    const next = cloneComposeLayoutPresetSelection(undefined);
    next[field] = value;
    // invalid options are shown intentionally: selecting one starts a new valid
    // path and copies over only fields that still match some preset.
    composeLayoutPresetFields().forEach(({ key }) => {
        if (key === field) return;
        const candidate = selection[key];
        if (!candidate) return;
        if (presetMatches({
            ...next,
            [key]: candidate,
        })) next[key] = candidate;
    });
    const visibleKeys = new Set(visiblePresetFields(next, composeLayoutPresetOptions(next)).map(({ key }) => key));
    // hidden fields must be cleared or they can silently block preset matching
    // after an instruction switch such as mma -> swizzle.
    composeLayoutPresetFields().forEach(({ key }) => {
        if (!visibleKeys.has(key)) next[key] = '';
    });
    return normalizeComposeLayoutPresetSelection(next);
}

/**
 * Checks whether the partial preset facet selection can still resolve to at
 * least one compose-layout preset in the catalog.
 *
 * Empty selection values are ignored so partially filled search fields do not
 * reject otherwise valid preset families.
 *
 * @param filters - Map of preset facet keys, such as instruction family or
 *   layout variant, to the values currently selected in the preset chooser.
 * @returns `true` when some catalog preset contains every non-empty selected
 *   facet value; otherwise `false`, which callers use to drop stale facet
 *   values while normalizing the selection.
 * @noThrows Reads the in-memory preset catalog and performs optional facet
 *   lookups only; ordinary selection objects and missing facet keys fall back
 *   to `false` instead of throwing.
 * @example
 * ```ts
 * const canKeepSelection = presetMatches({ instruction: 'mma' });
 * // true when the preset catalog contains at least one preset tagged with
 * // the `mma` instruction facet.
 * expect(canKeepSelection).toBe(true);
 *
 * const impossibleSelection = presetMatches({ instruction: 'not-a-preset-family' });
 * expect(impossibleSelection).toBe(false);
 * ```
 */
function presetMatches(filters: ComposeLayoutPresetSelection): boolean {
    return composeLayoutPresets().some((preset) => Object.keys(filters).every((key) => {
        const value = filters[key];
        if (!value) return true;
        return preset.facets[key]?.includes(value) ?? false;
    }));
}

/**
 * Opens the preset option list for the active search field and closes every
 * other preset option list inside the widget root.
 *
 * @param root - Preset widget container that owns `.preset-option-list` nodes
 *   nested below elements with `data-preset-field` attributes.
 * @param activeInputId - Field id that should have its option list marked with
 *   `is-open`, or `null` to close all preset dropdowns.
 * @returns Nothing; the function mutates each option list's `classList` in the
 *   supplied DOM subtree.
 * @noThrows Uses scoped DOM queries and optional closest-field lookup, so lists
 *   without a matching `data-preset-field` wrapper are simply closed.
 * @example
 * ```ts
 * const root = document.createElement('div');
 * root.innerHTML = `
 *   <div data-preset-field="instruction"><div class="preset-option-list"></div></div>
 *   <div data-preset-field="variant"><div class="preset-option-list is-open"></div></div>
 * `;
 *
 * setPresetDropdownVisibility(root, 'instruction');
 *
 * expect(root.querySelector('[data-preset-field="instruction"] .preset-option-list')
 *   ?.classList.contains('is-open')).toBe(true);
 * expect(root.querySelector('[data-preset-field="variant"] .preset-option-list')
 *   ?.classList.contains('is-open')).toBe(false);
 * ```
 */
function setPresetDropdownVisibility(root: HTMLElement, activeInputId: string | null): void {
    root.querySelectorAll<HTMLElement>('.preset-option-list').forEach((list) => {
        list.classList.toggle('is-open', list.closest<HTMLElement>('[data-preset-field]')?.dataset.presetField === activeInputId);
    });
}

/**
 * Reads the available autocomplete values for one preset facet from the
 * filtered compose-layout preset option map.
 *
 * @param options - Option map produced for the current preset selection, keyed
 *   by preset field name.
 * @param field - Preset field key whose valid values should populate the
 *   corresponding search field.
 * @returns The option strings for `field`; returns an empty array when the
 *   filtered preset catalog has no values for that field.
 * @noThrows Performs only an indexed lookup with an empty-array fallback for
 *   missing keys.
 * @example
 * ```ts
 * const options = {
 *   instruction: ['mma', 'swizzle'],
 *   variant: ['m16n8k16'],
 * };
 *
 * expect(presetFieldOptions(options, 'instruction')).toEqual(['mma', 'swizzle']);
 * expect(presetFieldOptions(options, 'missingFacet')).toEqual([]);
 * ```
 */
function presetFieldOptions(options: ComposeLayoutPresetOptions, field: string): string[] {
    return options[field] ?? [];
}

/**
 * Chooses which preset selector fields should be rendered for the current
 * facet selection.
 *
 * Required catalog fields are always visible. Optional fields stay visible when
 * they already have a selected value, or become visible after all dependency
 * fields are selected and the filtered preset options contain values for them.
 *
 * @param selection - Current preset facet values from the sidebar, keyed by
 *   compose-layout preset field key.
 * @param options - Filtered option map for the current selection, used to hide
 *   optional fields that have no valid catalog values yet.
 * @returns Catalog field metadata for the preset search controls that the
 *   widget should render, in catalog order.
 * @noThrows Filters static preset field metadata and reads plain selection and
 *   option maps; absent option keys produce empty option lists rather than
 *   exceptions.
 * @example
 * ```ts
 * const selection = { instruction: 'mma' };
 * const options = composeLayoutPresetOptions(selection);
 *
 * const fields = visiblePresetFields(selection, options);
 *
 * expect(fields.some((field) => field.required)).toBe(true);
 * expect(fields.every((field) => field.required
 *   || Boolean(selection[field.key])
 *   || field.dependsOn.every((key) => Boolean(selection[key])))).toBe(true);
 * ```
 */
function visiblePresetFields(
    selection: ComposeLayoutPresetSelection,
    options: ComposeLayoutPresetOptions,
): ComposeLayoutPresetField[] {
    // required fields always show.  optional fields appear once their dependency
    // path is active and the filtered catalog has values for them.
    return composeLayoutPresetFields().filter((field) => field.required
        || Boolean(selection[field.key])
        || (field.dependsOn.every((key) => Boolean(selection[key])) && presetFieldOptions(options, field.key).length > 0));
}

/**
 * Builds a stable snapshot of the preset selector fields that are currently rendered in a widget container.
 *
 * @param root - Preset widget root whose descendants may include elements with `data-preset-field` attributes.
 * @returns Comma-separated `data-preset-field` values in DOM order; callers compare this string with the catalog's visible field ids to decide whether the widget must be re-rendered.
 * @noThrows Uses `querySelectorAll` on a provided `HTMLElement` and reads optional dataset values, so missing attributes contribute an empty segment instead of throwing.
 * @example
 * const root = document.createElement('div');
 * root.innerHTML = `
 *   <label data-preset-field="family"></label>
 *   <label data-preset-field="instruction"></label>
 * `;
 *
 * renderedPresetFieldIds(root); // "family,instruction"
 */
function renderedPresetFieldIds(root: HTMLElement): string {
    return Array.from(root.querySelectorAll<HTMLElement>('[data-preset-field]'))
        .map((field) => field.dataset.presetField ?? '')
        .join(',');
}

/**
 * Renders the linear-layout preset chooser, wires its search fields and action buttons, and installs the capture listener that closes preset dropdowns when the user clicks elsewhere.
 *
 * @param ctx - Linear-layout UI context containing the preset widget element, mutable `linearLayoutState.presetSelection`, widget-title renderer, layout-apply callback dependencies, and editor re-render callback.
 * @returns Nothing; the function replaces `ctx.linearLayoutPresetWidget.innerHTML`, normalizes the stored preset selection, and registers DOM event handlers for preset inputs, Load Preset, Clear Preset, and outside clicks.
 * @noThrows Has no intentional validation throw path: unavailable buttons are handled with optional chaining, unmatched presets leave Load Preset disabled, and click handlers return early when no matching preset exists.
 * @example
 * const ctx = createLinearLayoutUiContextFixture({
 *   presetSelection: { family: 'mma' },
 * });
 *
 * renderLinearLayoutPresetWidget(ctx);
 *
 * ctx.linearLayoutPresetWidget.querySelector('#linear-layout-clear-preset')
 *   ?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
 *
 * renderedPresetFieldIds(ctx.linearLayoutPresetWidget); // e.g. "family,instruction"
 * ctx.state.linearLayoutState.presetSelection; // normalized empty preset selection after Clear Preset
 */
export function renderLinearLayoutPresetWidget(ctx: LinearLayoutUiContext): void {
    clearPresetOutsideClickHandler?.();
    const presetSelection = cloneComposeLayoutPresetSelection(ctx.state.linearLayoutState.presetSelection);
    ctx.state.linearLayoutState.presetSelection = presetSelection;
    const presetOptions = composeLayoutPresetOptions(presetSelection);
    const preset = composeLayoutPresetForSelection(presetSelection);
    ctx.linearLayoutPresetWidget.innerHTML = `
      ${ctx.widgetTitle('linear-layout-preset', 'Choose a preset layout family and load its saved spec into Layout Specs.')}
        <div class="widget-body">
        ${linearLayoutPresetHelpHtml()}
        <div class="preset-stack">
          ${visiblePresetFields(presetSelection, presetOptions).map((field) => presetSearchField(
            field,
            presetSelection[field.key] ?? '',
            presetFieldOptions(presetOptions, field.key),
            invalidPresetFieldOptions(field.key, presetFieldOptions(presetOptions, field.key)),
            presetSelection,
        )).join('')}
        </div>
        <div class="button-row linear-layout-action-row">
          <button class="primary-button" id="linear-layout-load-preset" type="button" ${preset ? '' : 'disabled'} title="Overwrite the editor with the selected preset and render it.">Load Preset</button>
          <button class="secondary-button" id="linear-layout-clear-preset" type="button" title="Clear the preset search fields without changing the loaded layout spec.">Clear Preset</button>
        </div>
        <div class="widget-copy preset-summary" id="linear-layout-preset-summary">${preset
            ? `Selected preset: <span class="inline-code">${escapeInfo(preset.title)}</span>`
            : 'No preset matches the current selection yet.'}</div>
      </div>
    `;
    const loadPreset = ctx.linearLayoutPresetWidget.querySelector<HTMLButtonElement>('#linear-layout-load-preset');
    const clearPreset = ctx.linearLayoutPresetWidget.querySelector<HTMLButtonElement>('#linear-layout-clear-preset');
    composeLayoutPresetFields().forEach((field) => {
        bindPresetInput(ctx, ctx.linearLayoutPresetWidget.querySelector<HTMLInputElement>(`#${field.id}`), field.key);
    });
    bindPresetOptions(ctx);
    /**
 * Closes any open preset option dropdown when a captured pointerdown starts outside the preset widget.
 *
 * @param event - Captured `pointerdown` event whose `target` is checked against `ctx.linearLayoutPresetWidget` before dropdown visibility is cleared.
 * @returns Nothing; outside clicks mutate the preset widget's dropdown visibility, while clicks inside the widget or events without a `Node` target are ignored.
 * @noThrows Guards non-`Node` targets and inside-widget targets before calling the dropdown visibility helper, so ordinary pointer events do not produce an expected error path.
 * @example
 * const outside = document.createElement('button');
 * document.body.append(outside);
 * setPresetDropdownVisibility(ctx.linearLayoutPresetWidget, 'linear-layout-preset-family');
 *
 * outside.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
 *
 * ctx.linearLayoutPresetWidget.querySelector('[data-preset-options][data-open="true"]'); // null
 */
    const outsideClickHandler = (event: PointerEvent) => {
        const target = event.target;
        if (!(target instanceof Node)) return;
        if (ctx.linearLayoutPresetWidget.contains(target)) return;
        setPresetDropdownVisibility(ctx.linearLayoutPresetWidget, null);
    };
    document.addEventListener('pointerdown', outsideClickHandler, true);
    clearPresetOutsideClickHandler = () => {
        // the widget can be re-rendered often while typing, so remove the old
        // capture listener before installing a replacement to avoid duplicate closes.
        document.removeEventListener('pointerdown', outsideClickHandler, true);
    };
    loadPreset?.addEventListener('click', async () => {
        const nextPreset = composeLayoutPresetForSelection(ctx.state.linearLayoutState.presetSelection);
        if (!nextPreset) return;
        ctx.state.linearLayoutState.specsText = nextPreset.state.specsText;
        ctx.state.linearLayoutState.operationText = nextPreset.state.operationText;
        ctx.state.linearLayoutState.inputName = nextPreset.state.inputName;
        await applyLinearLayoutSpec(ctx);
        ctx.renderLinearLayoutEditorWidgets();
    });
    clearPreset?.addEventListener('click', () => {
        ctx.state.linearLayoutState.presetSelection = normalizeComposeLayoutPresetSelection(undefined);
        syncPresetControls(ctx, null);
    });
}
