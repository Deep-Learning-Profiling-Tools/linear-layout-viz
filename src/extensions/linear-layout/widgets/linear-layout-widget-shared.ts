import type { LoadedBundleDocument } from '@tensor-viz/viewer-core';
import { buildComposeRuntime, propagationLabels } from '../linear-layout.js';
import {
    composeLayoutMetaForTab,
    isLinearLayoutTab,
    type LinearLayoutChannel,
    type LinearLayoutUiContext,
} from '../linear-layout-state.js';

export const LINEAR_LAYOUT_CHANNELS: LinearLayoutChannel[] = ['H', 'S', 'L'];
export const VISIBLE_TENSORS_ERROR = 'At least one tensor in the render chain must stay visible.';

/**
 * Returns the active viewer tab only when it is a rendered linear-layout document.
 *
 * Sidebar widgets use this guard before reading compose-layout metadata so ordinary tensor tabs or an empty session do not look like editable linear-layout output.
 *
 * @param ctx - Linear-layout widget context whose `getActiveTab()` method supplies the currently selected session document.
 * @returns The active `LoadedBundleDocument` when it contains linear-layout metadata; otherwise `null` for non-linear-layout tabs or no active tab.
 * @noThrows The helper only reads the active tab and applies the linear-layout type guard; it does not parse specs or mutate viewer state.
 * @example
 * const tab = activeLinearLayoutTab(ctx);
 *
 * expect(tab?.id).toBe('mma-layout');
 *
 * @example
 * ctx.getActiveTab = () => plainTensorTab;
 *
 * expect(activeLinearLayoutTab(ctx)).toBeNull();
 */
export function activeLinearLayoutTab(ctx: LinearLayoutUiContext): LoadedBundleDocument | null {
    const tab = ctx.getActiveTab();
    return tab && isLinearLayoutTab(tab) ? tab : null;
}

/**
 * Resolves the label set that color-channel and cell-text widgets should offer for the current linear-layout edit.
 *
 * The helper prefers labels from a freshly built compose runtime so unsaved spec edits are reflected immediately. If the edit cannot be parsed or evaluated, it falls back to metadata embedded in the active linear-layout tab.
 *
 * @param ctx - Linear-layout widget context containing the editable layout state, the propagate-outputs flag, and optional active-tab metadata used for fallback labels.
 * @returns Object with `labels`, the root-input or final-output labels to display in widget controls, and `injective`, the compose runtime or saved tab injectivity flag used to decide whether mappings are one-to-one.
 * @noThrows Runtime construction is wrapped in a catch block; invalid in-progress edits return saved tab metadata or an empty label list instead of propagating parser/evaluation errors.
 * @example
 * const result = linearLayoutPropagationLabels(ctx);
 *
 * expect(result).toEqual({ labels: ['lane', 'row', 'col'], injective: true });
 *
 * @example
 * ctx.state.linearLayoutState.specsText = 'unfinished edit';
 * ctx.getActiveTab = () => savedLinearLayoutTabWithMeta;
 *
 * expect(linearLayoutPropagationLabels(ctx)).toEqual({
 *   labels: savedLinearLayoutTabWithMeta.linearLayout.finalOutputLabels,
 *   injective: savedLinearLayoutTabWithMeta.linearLayout.injective,
 * });
 */
export function linearLayoutPropagationLabels(ctx: LinearLayoutUiContext): { labels: string[]; injective: boolean } {
    const tab = activeLinearLayoutTab(ctx);
    const meta = tab ? composeLayoutMetaForTab(tab) : null;
    try {
        const runtime = buildComposeRuntime(ctx.state.linearLayoutState);
        return { labels: propagationLabels(runtime, ctx.state.linearLayoutState.propagateOutputs)[0], injective: runtime.injective };
    } catch {
        return {
            labels: ctx.state.linearLayoutState.propagateOutputs
                ? meta?.finalOutputLabels.slice() ?? []
                : meta?.rootInputLabels.slice() ?? [],
            injective: meta?.injective ?? true,
        };
    }
}

/**
 * Builds the cell-text visibility map for the currently propagated linear-layout labels.
 * Labels that already exist in the saved map keep their boolean setting, and newly
 * propagated labels default to visible.
 *
 * @param state - Previously saved cell-text visibility flags keyed by propagation label.
 * @param labels - Propagation labels that should be represented in the returned map, in display order.
 * @returns A new record containing exactly the provided labels, with missing entries initialized to `true`.
 * @noThrows Uses only array mapping and object construction over caller-provided strings, so there is no expected validation or browser API failure path.
 * @example
 * const state = { lane: false, stale: false };
 * const labels = ['lane', 'warp'];
 *
 * normalizeCellTextState(state, labels);
 * // => { lane: false, warp: true }
 */
export function normalizeCellTextState(state: Record<string, boolean>, labels: string[]): Record<string, boolean> {
    return Object.fromEntries(labels.map((label) => [label, state[label] ?? true]));
}

/**
 * Checks whether every linear-layout color channel still points at an available
 * propagation label, allowing the sentinel `none` value for unmapped channels.
 *
 * @param mapping - Color-channel selections keyed by each `LinearLayoutChannel`.
 * @param labels - Propagation labels that are valid for the current compose-layout runtime.
 * @returns `true` when every channel is `none` or one of the supplied labels; otherwise `false` so callers can rebuild the mapping.
 * @noThrows Performs set membership checks against in-memory strings and does not call parsing, DOM, or browser clipboard APIs.
 * @example
 * const mapping = { red: 'lane', green: 'none', blue: 'warp' } as Record<LinearLayoutChannel, string>;
 *
 * mappingMatchesLabels(mapping, ['lane', 'warp']);
 * // => true
 *
 * mappingMatchesLabels(mapping, ['lane']);
 * // => false
 */
export function mappingMatchesLabels(mapping: Record<LinearLayoutChannel, string>, labels: string[]): boolean {
    const allowed = new Set(labels);
    return LINEAR_LAYOUT_CHANNELS.every((channel) => mapping[channel] === 'none' || allowed.has(mapping[channel]));
}

/**
 * Resizes a sidebar textarea to fit its current content by resetting the inline
 * height and then applying the element's measured scroll height.
 *
 * @param textarea - Rendered textarea from a widget input whose content should determine its CSS height.
 * @returns Nothing; the textarea's `style.height` is updated in place.
 * @noThrows Reads `scrollHeight` and writes inline style properties on an existing textarea, with no parsing or external browser API call expected to fail.
 * @example
 * const textarea = document.createElement('textarea');
 * Object.defineProperty(textarea, 'scrollHeight', { value: 84 });
 *
 * autosizeTextarea(textarea);
 *
 * textarea.style.height;
 * // => '84px'
 */
export function autosizeTextarea(textarea: HTMLTextAreaElement): void {
    textarea.style.height = '0';
    textarea.style.height = `${textarea.scrollHeight}px`;
}

/**
 * Copies generated linear-layout text, such as Python initialization code, to the
 * user's clipboard using the async Clipboard API when available and a hidden
 * textarea fallback otherwise.
 *
 * @param text - Clipboard payload to copy exactly as displayed or generated by the widget.
 * @returns A promise that resolves after the browser clipboard operation or fallback copy attempt completes.
 * @throws Propagates browser clipboard failures, such as `navigator.clipboard.writeText` rejecting because clipboard permission is denied or the page is not allowed to write to the clipboard.
 * @example
 * navigator.clipboard = {
 *   writeText: async (value: string) => {
 *     copied = value;
 *   },
 * } as Clipboard;
 *
 * await copyText('layout = make_layout()');
 * copied;
 * // => 'layout = make_layout()'
 *
 * @example
 * navigator.clipboard = {
 *   writeText: async () => {
 *     throw new DOMException('Write permission denied', 'NotAllowedError');
 *   },
 * } as Clipboard;
 *
 * await expect(copyText('layout = make_layout()')).rejects.toThrow('Write permission denied');
 */
export async function copyText(text: string): Promise<void> {
    if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
        return;
    }
    const input = document.createElement('textarea');
    input.value = text;
    input.style.position = 'fixed';
    input.style.opacity = '0';
    document.body.appendChild(input);
    input.select();
    document.execCommand('copy');
    document.body.removeChild(input);
}

/**
 * Waits for the linear-layout viewer panel to stop changing size, then asks the viewer to resize its canvas and refit the tensor view.
 *
 * @param ctx - Linear-layout widget context containing the viewport DOM element to measure and the viewer instance that should be resized/refit after the sidebar has loaded its initial tab.
 * @returns Promise that resolves after document fonts, when available, have settled, two consecutive animation frames report the same viewport dimensions, and `ctx.viewer.resize()` and `ctx.viewer.refitView()` have been called.
 * @noThrows Font readiness rejections are ignored so a failed web-font load does not abort initial layout settlement; callers should still provide a valid viewport and viewer because their property access and methods are used directly.
 * @example
 * const calls: string[] = [];
 * const ctx = {
 *   viewport: { clientWidth: 640, clientHeight: 480 },
 *   viewer: {
 *     resize: () => calls.push('resize'),
 *     refitView: () => calls.push('refitView'),
 *   },
 * } as LinearLayoutUiContext;
 *
 * await settleInitialLayout(ctx);
 *
 * expect(calls).toEqual(['resize', 'refitView']);
 */
export async function settleInitialLayout(ctx: LinearLayoutUiContext): Promise<void> {
    if ('fonts' in document) {
        try {
            await (document as Document & { fonts: { ready: Promise<unknown> } }).fonts.ready;
        } catch {
            // ignore font-settlement errors
        }
    }
    let stableFrames = 0;
    let previousSize = '';
    while (stableFrames < 2) {
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        const nextSize = `${ctx.viewport.clientWidth}x${ctx.viewport.clientHeight}`;
        stableFrames = nextSize === previousSize ? stableFrames + 1 : 0;
        previousSize = nextSize;
    }
    ctx.viewer.resize();
    ctx.viewer.refitView();
}
