import type { LinearLayoutUiContext } from '../linear-layout-state.js';
import { renderLinearLayoutColorWidget } from './linear-layout-color-widget.js';
import { renderLinearLayoutWidget } from './linear-layout-specs-widget.js';
import { renderLinearLayoutVisibleTensorsWidget } from './linear-layout-visible-tensors-widget.js';

/**
 * Refreshes the linear-layout sidebar after a spec, visibility, or color change by rerendering the specs, visible-tensor, and color-mapping widgets from the shared UI context.
 *
 * @param ctx - Linear-layout UI context containing the current extension state plus the sidebar DOM containers consumed by the individual widget renderers.
 * @returns Nothing. The sidebar DOM owned by the linear-layout widgets is replaced or rebound by the delegated render functions.
 * @noThrows This wrapper only delegates to the three widget renderers and contains no validation or branching that raises its own error.
 * @example
 * const ctx = createLinearLayoutUiContextWithSidebar();
 * renderLinearLayoutEditorWidgets(ctx);
 * expect(ctx.linearLayoutWidget.innerHTML).toContain('Layout Specs');
 * expect(ctx.linearLayoutVisibleTensorsWidget.innerHTML).toContain('Visible tensors');
 * expect(ctx.linearLayoutColorWidget.innerHTML).toContain('Color');
 */
export function renderLinearLayoutEditorWidgets(ctx: LinearLayoutUiContext): void {
    renderLinearLayoutWidget(ctx);
    renderLinearLayoutVisibleTensorsWidget(ctx);
    renderLinearLayoutColorWidget(ctx);
}
