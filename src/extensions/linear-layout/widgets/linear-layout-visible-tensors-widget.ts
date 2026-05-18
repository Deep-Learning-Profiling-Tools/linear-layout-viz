import { escapeInfo } from '@tensor-viz/viewer-demo/extension-api';
import { composeLayoutMetaForTab, type LinearLayoutUiContext } from '../linear-layout-state.js';
import { applyLinearLayoutSpec } from './linear-layout-widget-actions.js';
import { VISIBLE_TENSORS_ERROR, activeLinearLayoutTab } from './linear-layout-widget-shared.js';

/**
 * Renders the checklist that lets users hide or show tensors from the active linear-layout render chain.
 *
 * @param ctx - Linear-layout UI context containing the active tab lookup state, the visible-tensors widget element, `linearLayoutState.visibleTensors`, any visible-tensor error notice, and the apply/render callbacks used after a checkbox changes.
 * @returns Nothing. The function hides and clears the widget when the active tab has no compose-layout metadata; otherwise it populates one checkbox per rendered tensor and binds each checkbox to update `visibleTensors` and reapply the layout.
 * @noThrows The synchronous render path only reads active-tab metadata, writes escaped checkbox markup, and attaches optional change listeners; the asynchronous rebuild triggered by a later checkbox change is not part of the initial render call.
 * @example
 * const ctx = makeLinearLayoutUiContextWithTensors([
 *   { id: 'input', title: 'Input' },
 *   { id: 'Tile2x1', title: 'Tile2x1' },
 * ]);
 *
 * renderLinearLayoutVisibleTensorsWidget(ctx);
 *
 * console.assert(!ctx.linearLayoutVisibleTensorsWidget.classList.contains('hidden'));
 * console.assert(ctx.linearLayoutVisibleTensorsWidget.querySelectorAll('input[type="checkbox"]').length === 2);
 * console.assert(ctx.linearLayoutVisibleTensorsWidget.textContent?.includes('Tile2x1'));
 */
export function renderLinearLayoutVisibleTensorsWidget(ctx: LinearLayoutUiContext): void {
    const tab = activeLinearLayoutTab(ctx);
    const meta = tab ? composeLayoutMetaForTab(tab) : null;
    if (!meta) {
        ctx.linearLayoutVisibleTensorsWidget.classList.add('hidden');
        ctx.linearLayoutVisibleTensorsWidget.innerHTML = '';
        return;
    }
    ctx.linearLayoutVisibleTensorsWidget.classList.remove('hidden');
    // surface apply errors in this widget because toggles are the only control
    // that can hide the tensor whose spec currently fails to rebuild.
    const status = ctx.state.linearLayoutNotice?.text === VISIBLE_TENSORS_ERROR
        ? `<div class="error-box">${escapeInfo(ctx.state.linearLayoutNotice.text)}</div>`
        : '';
    ctx.linearLayoutVisibleTensorsWidget.innerHTML = `
      ${ctx.widgetTitle('linear-layout-visible-tensors', 'Toggle which tensors in the render chain stay visible for the current tab.')}
      <div class="widget-body">
        <div class="checklist-field">
          ${meta.tensors.map((tensor) => `
            <label class="checklist-row" for="linear-layout-visible-${tensor.id}">
              <span>${escapeInfo(tensor.title)}</span>
              <input id="linear-layout-visible-${tensor.id}" type="checkbox" ${ctx.state.linearLayoutState.visibleTensors[tensor.id] !== false ? 'checked' : ''} />
            </label>
          `).join('')}
        </div>
        ${status}
      </div>
    `;
    meta.tensors.forEach((tensor) => {
        // visibility edits rebuild the current layout while preserving any
        // per-tensor view strings the user has already typed.
        ctx.linearLayoutVisibleTensorsWidget.querySelector<HTMLInputElement>(`#linear-layout-visible-${CSS.escape(tensor.id)}`)?.addEventListener('change', async (event) => {
            const target = event.currentTarget as HTMLInputElement;
            ctx.state.linearLayoutState.visibleTensors[tensor.id] = target.checked;
            await applyLinearLayoutSpec(ctx, { silent: true, preserveTensorViews: true });
            ctx.renderLinearLayoutEditorWidgets();
        });
    });
}
