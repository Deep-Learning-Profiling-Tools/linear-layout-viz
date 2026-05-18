import { escapeInfo, infoButton, labelWithInfo } from '@tensor-viz/viewer-demo/extension-api';
import { autoColorLayoutState } from '../linear-layout.js';
import {
    cloneLinearLayoutCellTextState,
    type LinearLayoutChannel,
    type LinearLayoutUiContext,
} from '../linear-layout-state.js';
import { applyLinearLayoutCellText } from '../linear-layout-viewer-sync.js';
import { applyLinearLayoutSpec } from './linear-layout-widget-actions.js';
import {
    LINEAR_LAYOUT_CHANNELS,
    activeLinearLayoutTab,
    linearLayoutPropagationLabels,
    normalizeCellTextState,
} from './linear-layout-widget-shared.js';

/**
 * Builds the static usage-guide markup shown above the linear-layout color mapper.
 *
 * @returns HTML for the collapsible guide that explains dragging propagated axes onto H/S/L channels, clearing or swapping channel chips, and applying Propagate Outputs changes with Recolor Layout.
 * @noThrows The helper returns a fixed template literal and has no inputs, DOM access, parsing, or asynchronous work.
 * @example
 * const html = linearLayoutColorHelpHtml();
 *
 * console.assert(html.includes('<details class="usage-guide">'));
 * console.assert(html.includes('Drag a propagated axis'));
 * console.assert(html.includes('Recolor Layout'));
 */
function linearLayoutColorHelpHtml(): string {
    return `
      <details class="usage-guide">
        <summary>How do I use this?</summary>
        <div class="usage-guide-body">
          <div class="usage-guide-step">
            <span>Drag a propagated axis from <strong>Available Axes</strong> onto H, S, or L to control that channel.</span>
          </div>
          <div class="usage-guide-step">
            <span>Drag an assigned chip back to the pool to clear it, or drag between channels to swap assignments.</span>
          </div>
          <div class="usage-guide-step">
            <span>Toggle <strong>Propagate Outputs</strong> to switch between input-driven and output-driven labels/colors, then click <strong>Recolor Layout</strong> to apply the new mapping.</span>
          </div>
        </div>
      </details>
    `;
}

/**
 * Chooses the explanatory copy for the Propagate Outputs control based on
 * whether the active linear-layout mapping is injective.
 *
 * @param injective - `true` when each output coordinate maps back to a single input coordinate; `false` when the layout can have multi-input or ghost-layer behavior.
 * @returns Help text for the toolbar or info button that tells users how turning Propagate Outputs on changes the coordinate space used for colors and cell labels.
 * @noThrows The helper only branches on the supplied boolean and returns one of two fixed strings.
 * @example
 * console.assert(
 *   linearLayoutPropagateOutputsInfo(true).includes('flow backward'),
 * );
 * console.assert(
 *   linearLayoutPropagateOutputsInfo(false).includes('multi-input behavior'),
 * );
 */
export function linearLayoutPropagateOutputsInfo(injective: boolean): string {
    return injective
        ? 'When off, colors and cell text come from the input space and flow forward. When on, they come from the final output space and flow backward.'
        : 'When off, non-injective layouts keep the current popup, ghost-layer, and multi-input behavior. When on, colors and cell text come from the final output space.';
}

/**
 * Flips the Propagate Outputs setting, rebuilds the H/S/L color mapping for the
 * new propagation space, synchronizes cell-text labels for the active tab, then
 * reapplies the linear-layout spec and refreshes the editor widgets.
 *
 * @param ctx - Linear-layout UI context containing the current `linearLayoutState` spec and operation text, per-tab cell-text state, active-tab lookup data, and the apply/render callbacks used by the sidebar.
 * @returns A promise that resolves after the silent spec reapply completes and the linear-layout editor widgets have been rendered again.
 * @noThrows This wrapper performs no explicit validation and throws no errors itself; failures can only surface from the parsing, application, or rendering helpers it awaits or calls.
 * @example
 * const ctx = makeLinearLayoutUiContext({
 *   propagateOutputs: false,
 *   specsText: 'layout = identity',
 *   operationText: 'out = input',
 * });
 *
 * await toggleLinearLayoutPropagateOutputs(ctx);
 *
 * console.assert(ctx.state.linearLayoutState.propagateOutputs === true);
 * console.assert(ctx.applyLinearLayoutSpecCalls[0].silent === true);
 * console.assert(ctx.applyLinearLayoutSpecCalls[0].preserveTensorViews === true);
 * console.assert(ctx.renderLinearLayoutEditorWidgetsCalls === 1);
 */
export async function toggleLinearLayoutPropagateOutputs(ctx: LinearLayoutUiContext): Promise<void> {
    ctx.state.linearLayoutState.propagateOutputs = !ctx.state.linearLayoutState.propagateOutputs;
    // propagation changes the coordinate space that H/S/L channels reference,
    // so recolor from the current spec instead of trying to reinterpret ranges.
    const autoColor = autoColorLayoutState(
        ctx.state.linearLayoutState.specsText,
        ctx.state.linearLayoutState.operationText,
        ctx.state.linearLayoutState.propagateOutputs,
    );
    ctx.state.linearLayoutState.mapping = autoColor.mapping;
    ctx.state.linearLayoutState.ranges = autoColor.ranges;
    ctx.state.linearLayoutCellTextState = normalizeCellTextState(
        ctx.state.linearLayoutCellTextState,
        linearLayoutPropagationLabels(ctx).labels,
    );
    const tab = activeLinearLayoutTab(ctx);
    if (tab) ctx.state.linearLayoutCellTextStates.set(tab.id, cloneLinearLayoutCellTextState(ctx.state.linearLayoutCellTextState));
    await applyLinearLayoutSpec(ctx, { silent: true, preserveTensorViews: true });
    ctx.renderLinearLayoutEditorWidgets();
}

/**
 * Rebuilds the linear-layout color sidebar controls for propagated axis labels, H/S/L channel assignments, numeric color ranges, and the Propagate Outputs toggle.
 *
 * The render replaces the widget body, marks already-assigned axes as unavailable in the drag source pool, binds checkbox/drag/drop/range/recolor handlers, and restores focus for an input that survived the repaint.
 *
 * @param ctx - Linear-layout UI context containing the color widget element, widget title renderer, active linear-layout state, per-tab cell-text state, and viewer repaint helpers.
 * @returns Void; callers observe the refreshed `ctx.linearLayoutColorWidget` DOM and subsequent user events mutating `ctx.state.linearLayoutState` or `ctx.state.linearLayoutCellTextState`.
 * @noThrows For a valid mounted linear-layout UI context, the render uses optional DOM lookups for controls that may be absent and treats missing active tabs as a no-op for tab persistence.
 * @example
 * ```ts
 * renderLinearLayoutColorWidget(ctx);
 *
 * expect(ctx.linearLayoutColorWidget.querySelector('#linear-layout-recolor')?.textContent).toBe('Recolor Layout');
 * expect(ctx.linearLayoutColorWidget.querySelector('[data-channel="H"] [data-axis="i"]')).not.toBeNull();
 * expect(ctx.linearLayoutColorWidget.querySelector('.mapping-pool')?.textContent).not.toContain('i');
 * ```
 */
export function renderLinearLayoutColorWidget(ctx: LinearLayoutUiContext): void {
    const activeElement = document.activeElement;
    const focusedInput = activeElement instanceof HTMLInputElement && ctx.linearLayoutColorWidget.contains(activeElement)
        ? { id: activeElement.id, start: activeElement.selectionStart, end: activeElement.selectionEnd }
        : null;
    const channelLabels: Record<LinearLayoutChannel, string> = { H: 'Hue', S: 'Sat', L: 'Light' };
    const { labels, injective } = linearLayoutPropagationLabels(ctx);
    // assigned labels disappear from the source pool; dragging a chip back to
    // the pool clears that color channel without needing a separate reset button.
    const assignedLabels = new Set(
        LINEAR_LAYOUT_CHANNELS
            .map((channel) => ctx.state.linearLayoutState.mapping[channel])
            .filter((label): label is string => label !== 'none' && labels.includes(label)),
    );
    const availableLabels = labels.filter((label) => !assignedLabels.has(label));
    ctx.linearLayoutColorWidget.innerHTML = `
      ${ctx.widgetTitle('linear-layout-color', 'Configure propagated cell labels, H/S/L color mapping, and whether labels/colors follow inputs forward or outputs backward.')}
      <div class="widget-body">
        ${linearLayoutColorHelpHtml()}
        <div class="field">
          <label class="checklist-row" for="linear-layout-propagate-outputs">
            <span class="label-row"><span class="meta-label">Propagate Outputs</span>${infoButton(linearLayoutPropagateOutputsInfo(injective))}</span>
            <input id="linear-layout-propagate-outputs" type="checkbox" ${ctx.state.linearLayoutState.propagateOutputs ? 'checked' : ''} />
          </label>
        </div>
        <div class="field">
          <div class="label-row"><span class="meta-label">Cell Text</span>${infoButton('Choose which propagated axes are drawn as per-cell labels. The available labels follow the current Propagate Outputs mode.')}</div>
          <div class="checklist-field">
            ${labels.map((label) => `
              <label class="checklist-row" for="cell-text-${label}">
                <span>${label}</span>
                <input id="cell-text-${label}" type="checkbox" ${ctx.state.linearLayoutCellTextState[label] ? 'checked' : ''} />
              </label>
            `).join('')}
          </div>
        </div>
        <div class="field">
          ${labelWithInfo('Available Axes', 'Drag one propagated axis onto H, S, or L. Drag a colored axis back here to clear that channel.')}
          <div class="mapping-pool mapping-drop-zone" data-pool="true">
            ${availableLabels.map((label) => `<button class="mapping-chip" type="button" draggable="true" data-axis="${label}">${label}</button>`).join('')}
            ${availableLabels.length === 0 ? '<span class="mapping-empty">all axes assigned</span>' : ''}
          </div>
        </div>
        ${LINEAR_LAYOUT_CHANNELS.map((channel) => {
            const assignedAxis = ctx.state.linearLayoutState.mapping[channel];
            const assigned = assignedAxis !== 'none' && labels.includes(assignedAxis);
            return `
          <div class="inline-row mapping-row">
            <span class="range-label">${channelLabels[channel]}</span>
            <div class="mapping-drop-zone" data-channel="${channel}">
              ${assigned
        ? `<button class="mapping-chip mapping-chip-assigned" type="button" draggable="true" data-channel="${channel}" data-axis="${assignedAxis}">${assignedAxis}</button>`
        : '<span class="mapping-empty">none</span>'}
            </div>
            <input id="linear-layout-${channel.toLowerCase()}-min" type="number" step="0.01" value="${escapeInfo(ctx.state.linearLayoutState.ranges[channel][0])}" />
            <span class="range-separator${assigned ? '' : ' range-separator-unused'}">to</span>
            <input
              id="linear-layout-${channel.toLowerCase()}-max"
              class="${assigned ? '' : 'unused-range-input'}"
              type="${assigned ? 'number' : 'text'}"
              ${assigned ? 'step="0.01"' : ''}
              value="${assigned ? escapeInfo(ctx.state.linearLayoutState.ranges[channel][1]) : 'N/A'}"
              ${assigned ? '' : 'readonly aria-readonly="true" title="This upper bound is unused while no axis is mapped to this color channel."'}
            />
          </div>
        `;
        }).join('')}
        <div class="button-row">
          <button class="primary-button" id="linear-layout-recolor" type="button" title="Apply the current H/S/L axis assignments and numeric ranges to recolor the layout.">Recolor Layout</button>
        </div>
      </div>
    `;

    ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>('#linear-layout-propagate-outputs')?.addEventListener('change', async () => {
        // checkbox state is read from the DOM because the click may come from a
        // label activation rather than from a direct input event target.
        const checked = ctx.linearLayoutColorWidget
            .querySelector<HTMLInputElement>('#linear-layout-propagate-outputs')?.checked ?? false;
        if (checked === ctx.state.linearLayoutState.propagateOutputs) return;
        await toggleLinearLayoutPropagateOutputs(ctx);
    });
    /**
 * Copies the checked state of the rendered per-axis Cell Text checkboxes into the active tab's linear-layout label-visibility settings, then reapplies cell labels to the viewer.
 *
 * @returns Void; callers observe `ctx.state.linearLayoutCellTextState`, the active tab's saved cell-text state, and the rendered cell labels reflecting the checked boxes.
 * @noThrows Checkbox lookups are optional and default missing axis controls to `false`, while a missing active tab only skips tab-local persistence.
 * @example
 * ```ts
 * ctx.linearLayoutColorWidget.innerHTML = '<input id="cell-text-i" type="checkbox" checked><input id="cell-text-j" type="checkbox">';
 * labels = ['i', 'j'];
 *
 * syncCellText();
 *
 * expect(ctx.state.linearLayoutCellTextState).toEqual({ i: true, j: false });
 * expect(ctx.state.linearLayoutCellTextStates.get(activeTab.id)).toEqual({ i: true, j: false });
 * expect(applyLinearLayoutCellText).toHaveBeenCalledWith(ctx);
 * ```
 */
    const syncCellText = (): void => {
        ctx.state.linearLayoutCellTextState = Object.fromEntries(labels.map((label) => [
            label,
            ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>(`#cell-text-${CSS.escape(label)}`)?.checked ?? false,
        ]));
        const tab = activeLinearLayoutTab(ctx);
        if (tab) ctx.state.linearLayoutCellTextStates.set(tab.id, cloneLinearLayoutCellTextState(ctx.state.linearLayoutCellTextState));
        applyLinearLayoutCellText(ctx);
    };
    labels.forEach((label) => {
        // each checkbox writes to the persisted tab state immediately so a
        // later spec re-apply keeps the user's current label visibility.
        ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>(`#cell-text-${CSS.escape(label)}`)?.addEventListener('change', syncCellText);
    });

    /**
 * Serializes a color-mapping drag payload into the drag event so drop zones can distinguish pooled axes from already-assigned color channels.
 *
 * @param event - `dragstart` event from a mapping chip; when `event.dataTransfer` exists, it receives both the private linear-layout MIME payload and a `text/plain` fallback.
 * @param payload - String-valued drag description, such as `{ kind: 'axis', axis: 'i' }` from the available-axis pool or `{ kind: 'channel', channel: 'H', axis: 'i' }` from an assigned H/S/L chip.
 * @returns Void; callers observe the JSON payload stored on `event.dataTransfer` and `effectAllowed` set to `move`.
 * @noThrows Linear-layout callers pass plain string records that `JSON.stringify` can serialize, and the helper skips all DataTransfer writes when the browser did not provide a transfer object.
 * @example
 * ```ts
 * const transfer = new DataTransfer();
 * const event = new DragEvent('dragstart', { dataTransfer: transfer });
 *
 * writeDragPayload(event, { kind: 'channel', channel: 'H', axis: 'i' });
 *
 * expect(transfer.getData('application/x-linear-layout-mapping')).toBe('{"kind":"channel","channel":"H","axis":"i"}');
 * expect(transfer.getData('text/plain')).toBe('{"kind":"channel","channel":"H","axis":"i"}');
 * expect(transfer.effectAllowed).toBe('move');
 * ```
 */
    const writeDragPayload = (event: DragEvent, payload: Record<string, string>): void => {
        event.dataTransfer?.setData('application/x-linear-layout-mapping', JSON.stringify(payload));
        event.dataTransfer?.setData('text/plain', JSON.stringify(payload));
        if (event.dataTransfer) event.dataTransfer.effectAllowed = 'move';
    };
    /**
 * Reads a linear-layout color-mapping drag payload from the private MIME slot, falling back to `text/plain`, and ignores drags that are empty or not valid JSON.
 *
 * @param event - Drop or dragover `DragEvent` whose `dataTransfer` may contain a serialized mapping-chip payload.
 * @returns Parsed string record for a dragged axis/channel chip, or `null` when the transfer has no payload or contains malformed JSON from an unrelated drag source.
 * @noThrows Malformed JSON is caught and converted to `null`, and missing `dataTransfer` is handled with optional access.
 * @example
 * ```ts
 * const transfer = new DataTransfer();
 * transfer.setData('application/x-linear-layout-mapping', '{"kind":"axis","axis":"j"}');
 *
 * expect(readDragPayload(new DragEvent('drop', { dataTransfer: transfer }))).toEqual({ kind: 'axis', axis: 'j' });
 *
 * transfer.setData('application/x-linear-layout-mapping', 'not json');
 * expect(readDragPayload(new DragEvent('drop', { dataTransfer: transfer }))).toBeNull();
 * ```
 */
    const readDragPayload = (event: DragEvent): Record<string, string> | null => {
        const raw = event.dataTransfer?.getData('application/x-linear-layout-mapping') || event.dataTransfer?.getData('text/plain');
        if (!raw) return null;
        try {
            return JSON.parse(raw) as Record<string, string>;
        } catch {
            return null;
        }
    };
    ctx.linearLayoutColorWidget.querySelectorAll<HTMLElement>('[draggable="true"]').forEach((element) => {
        element.addEventListener('dragstart', (event) => {
            // chips dragged from a channel carry both source channel and axis so
            // drops can distinguish swaps from pool-to-channel assignments.
            const channel = element.dataset.channel;
            const axis = element.dataset.axis;
            if (axis) writeDragPayload(event, channel ? { kind: 'channel', channel, axis } : { kind: 'axis', axis });
        });
    });
    ctx.linearLayoutColorWidget.querySelectorAll<HTMLElement>('.mapping-drop-zone').forEach((element) => {
        element.addEventListener('dragover', (event) => {
            // this widget accepts every internal chip drag; the payload is
            // validated again on drop before mutating mapping state.
            event.preventDefault();
            element.classList.add('drag-over');
        });
        element.addEventListener('dragleave', () => {
            element.classList.remove('drag-over');
        });
        element.addEventListener('drop', (event) => {
            event.preventDefault();
            element.classList.remove('drag-over');
            const payload = readDragPayload(event);
            const targetChannel = element.dataset.channel as LinearLayoutChannel | undefined;
            if (!payload) return;
            if (element.dataset.pool === 'true') {
                if (payload.kind === 'channel' && payload.channel) {
                    ctx.state.linearLayoutState.mapping[payload.channel as LinearLayoutChannel] = 'none';
                    renderLinearLayoutColorWidget(ctx);
                }
                return;
            }
            if (!targetChannel) return;
            if (payload.kind === 'channel') {
                const sourceChannel = payload.channel as LinearLayoutChannel;
                if (!sourceChannel || sourceChannel === targetChannel) return;
                const sourceAxis = ctx.state.linearLayoutState.mapping[sourceChannel];
                ctx.state.linearLayoutState.mapping[sourceChannel] = ctx.state.linearLayoutState.mapping[targetChannel];
                ctx.state.linearLayoutState.mapping[targetChannel] = sourceAxis;
            } else if (payload.axis) {
                LINEAR_LAYOUT_CHANNELS.forEach((channel) => {
                    if (ctx.state.linearLayoutState.mapping[channel] === payload.axis) ctx.state.linearLayoutState.mapping[channel] = 'none';
                });
                ctx.state.linearLayoutState.mapping[targetChannel] = payload.axis;
            }
            renderLinearLayoutColorWidget(ctx);
        });
    });
    ([
        ['H', 'h'],
        ['S', 's'],
        ['L', 'l'],
    ] as const).forEach(([channel, key]) => {
        ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>(`#linear-layout-${key}-min`)?.addEventListener('input', (event) => {
            ctx.state.linearLayoutState.ranges[channel][0] = (event.currentTarget as HTMLInputElement).value;
        });
        ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>(`#linear-layout-${key}-max`)?.addEventListener('input', (event) => {
            ctx.state.linearLayoutState.ranges[channel][1] = (event.currentTarget as HTMLInputElement).value;
        });
    });
    ctx.linearLayoutColorWidget.querySelector<HTMLButtonElement>('#linear-layout-recolor')?.addEventListener('click', async () => {
        await applyLinearLayoutSpec(ctx, { silent: true, preserveTensorViews: true });
        ctx.renderLinearLayoutEditorWidgets();
    });
    if (focusedInput) {
        const nextInput = ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>(`#${focusedInput.id}`);
        nextInput?.focus();
        if (nextInput && focusedInput.start !== null && focusedInput.end !== null) {
            nextInput.setSelectionRange(focusedInput.start, focusedInput.end);
        }
    }
}
