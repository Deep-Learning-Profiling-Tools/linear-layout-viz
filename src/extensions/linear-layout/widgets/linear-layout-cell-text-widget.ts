import type { LinearLayoutUiContext } from '../linear-layout-state.js';

/**
 * Clears the placeholder cell-text sidebar widget and keeps it hidden until the
 * linear-layout cell-text UI is implemented.
 *
 * @param ctx - Linear-layout UI context whose `cellTextWidget` element is the sidebar container registered for the cell-text widget.
 * @returns Nothing; callers observe the widget element with empty markup and the `hidden` CSS class applied.
 * @noThrows The renderer performs only deterministic DOM assignments on the provided element and does not parse user text, query missing selectors, or call asynchronous helpers.
 * @example
 * const cellTextWidget = document.createElement('div');
 * cellTextWidget.innerHTML = '<button>Old cell text control</button>';
 *
 * renderCellTextWidget({ cellTextWidget } as LinearLayoutUiContext);
 *
 * console.assert(cellTextWidget.innerHTML === '');
 * console.assert(cellTextWidget.classList.contains('hidden'));
 */
export function renderCellTextWidget(ctx: LinearLayoutUiContext): void {
    ctx.cellTextWidget.innerHTML = '';
    ctx.cellTextWidget.classList.add('hidden');
}
