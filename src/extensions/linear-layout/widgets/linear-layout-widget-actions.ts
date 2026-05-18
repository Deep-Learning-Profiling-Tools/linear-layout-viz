import {
    autoColorLayoutState,
    buildComposeRuntime,
    createComposeLayoutDocument,
    propagationLabels,
} from '../linear-layout.js';
import {
    cloneLinearLayoutCellTextState,
    cloneLinearLayoutMultiInputState,
    cloneLinearLayoutState,
    composeLayoutMetaForTab,
    refreshLinearLayoutMatrixPreview,
    storeLinearLayoutState,
    type LinearLayoutUiContext,
} from '../linear-layout-state.js';
import { preservedLinearLayoutTensorViews } from '../linear-layout-viewer-sync.js';
import {
    activeLinearLayoutTab,
    mappingMatchesLabels,
    normalizeCellTextState,
} from './linear-layout-widget-shared.js';

/**
 * Builds the edited compose-layout spec into a viewer tab and updates the sidebar feedback state.
 *
 * Recomputes propagation labels, repairs the color mapping when the label space changed, refreshes the
 * matrix preview, persists the linear-layout state, and loads the rendered document into the session tabs.
 * Failures are reported through `ctx.state.linearLayoutNotice` instead of being rethrown.
 *
 * @param ctx - Linear-layout widget context containing the editable spec state, viewer snapshot, session-tab APIs, and notice state to update.
 * @param options - Apply-flow switches: `replaceTabs` replaces the session tab list with the rendered document, `silent` suppresses the success notice, and `preserveTensorViews` carries existing tensor-view expressions into the rebuilt document.
 * @returns `true` after the rendered layout document is stored and loaded; `false` when parsing, runtime construction, document creation, persistence, or tab loading fails and the error message has been copied into `ctx.state.linearLayoutNotice`.
 * @noThrows Expected apply errors are caught so editor event handlers can await this helper and render the error notice without wrapping it in their own try/catch.
 * @example
 * const applied = await applyLinearLayoutSpec(ctx, { silent: true, preserveTensorViews: true });
 *
 * expect(applied).toBe(true);
 * expect(ctx.state.linearLayoutNotice).toBeNull();
 * expect(ctx.loadTab).toHaveBeenCalledWith(ctx.getActiveTabId());
 *
 * @example
 * ctx.state.linearLayoutState.specsText = 'not valid compose-layout syntax';
 *
 * const applied = await applyLinearLayoutSpec(ctx);
 *
 * expect(applied).toBe(false);
 * expect(ctx.state.linearLayoutNotice).toMatchObject({ tone: 'error' });
 */
export async function applyLinearLayoutSpec(
    ctx: LinearLayoutUiContext,
    options: { replaceTabs?: boolean; silent?: boolean; preserveTensorViews?: boolean } = {},
): Promise<boolean> {
    try {
        const activeTab = activeLinearLayoutTab(ctx);
        const activeMeta = activeTab ? composeLayoutMetaForTab(activeTab) : null;
        // recolor only when the evaluated label space changed or the current
        // mapping references labels that no longer exist after an edit.
        const layoutChanged = !activeMeta
            || activeMeta.specsText !== ctx.state.linearLayoutState.specsText
            || activeMeta.operationText !== ctx.state.linearLayoutState.operationText;
        const runtime = buildComposeRuntime(ctx.state.linearLayoutState);
        if (layoutChanged || !mappingMatchesLabels(
            ctx.state.linearLayoutState.mapping,
            propagationLabels(runtime, ctx.state.linearLayoutState.propagateOutputs)[0],
        )) {
            const autoColor = autoColorLayoutState(
                ctx.state.linearLayoutState.specsText,
                ctx.state.linearLayoutState.operationText,
                ctx.state.linearLayoutState.propagateOutputs,
            );
            ctx.state.linearLayoutState.mapping = autoColor.mapping;
            ctx.state.linearLayoutState.ranges = autoColor.ranges;
        }
        refreshLinearLayoutMatrixPreview(ctx);
        const document = createComposeLayoutDocument(
            ctx.state.linearLayoutState,
            ctx.viewer.getSnapshot(),
            undefined,
            options.preserveTensorViews ? preservedLinearLayoutTensorViews(ctx) : undefined,
        );
        ctx.state.linearLayoutCellTextState = normalizeCellTextState(
            ctx.state.linearLayoutCellTextState,
            propagationLabels(runtime, ctx.state.linearLayoutState.propagateOutputs)[0],
        );
        storeLinearLayoutState(ctx.state.linearLayoutState);
        await upsertLinearLayoutTab(ctx, document, options.replaceTabs);
        const activeTitle = ctx.getSessionTabs().find((tab) => tab.id === ctx.getActiveTabId())?.title ?? document.title;
        ctx.state.linearLayoutNotice = options.silent ? null : { tone: 'success', text: `Rendered ${activeTitle}.` };
        return true;
    } catch (error) {
        ctx.state.linearLayoutNotice = { tone: 'error', text: error instanceof Error ? error.message : String(error) };
        return false;
    }
}

/**
 * Saves the current sidebar state for the target tab, inserts or replaces the rendered layout document in the session, and loads that tab.
 *
 * When a tab is already active, the new document keeps that tab's id and title so applying a spec updates the visible tab in place. With no active tab, the document's own id and title become the target.
 *
 * @param ctx - Linear-layout widget context that provides the active tab id, session tab list setters, per-tab linear-layout state caches, and `loadTab` callback.
 * @param document - Newly created compose-layout viewer document to insert into the session; its id and title are used only when there is no active tab to update.
 * @param replaceTabs - When `true`, discard all existing session tabs and keep only the target document; otherwise append a missing target tab or replace the matching tab in place.
 * @returns Promise that resolves after the session tabs and tab-local linear-layout caches have been updated and `ctx.loadTab(targetId)` has completed.
 * @noThrows Updates tab state through the widget context and returns the `loadTab` promise; context callbacks own any asynchronous failure handling.
 * @example
 * await upsertLinearLayoutTab(ctx, document, false);
 *
 * expect(ctx.setSessionTabs).toHaveBeenCalledWith([
 *   expect.objectContaining({ id: ctx.getActiveTabId(), title: ctx.getActiveTab()?.title }),
 * ]);
 * expect(ctx.state.linearLayoutStates.has(ctx.getActiveTabId())).toBe(true);
 * expect(ctx.loadTab).toHaveBeenCalledWith(ctx.getActiveTabId());
 *
 * @example
 * await upsertLinearLayoutTab(ctxWithNoExistingTabs, document, true);
 *
 * expect(ctxWithNoExistingTabs.setSessionTabs).toHaveBeenCalledWith([document]);
 * expect(ctxWithNoExistingTabs.loadTab).toHaveBeenCalledWith(document.id);
 */
async function upsertLinearLayoutTab(
    ctx: LinearLayoutUiContext,
    document: ReturnType<typeof createComposeLayoutDocument>,
    replaceTabs = false,
): Promise<void> {
    const activeTab = ctx.getSessionTabs().find((tab) => tab.id === ctx.getActiveTabId());
    const targetId = activeTab?.id ?? document.id;
    const targetTitle = activeTab?.title ?? document.title;
    const nextDocument = { ...document, id: targetId, title: targetTitle };
    // tab-local widget state is saved before loadTab swaps viewer data; without
    // this, applying a layout would reset cell text, multi-input, and selection caches.
    ctx.state.linearLayoutStates.set(targetId, cloneLinearLayoutState(ctx.state.linearLayoutState));
    ctx.state.linearLayoutCellTextStates.set(targetId, cloneLinearLayoutCellTextState(ctx.state.linearLayoutCellTextState));
    ctx.state.linearLayoutMultiInputStates.set(targetId, cloneLinearLayoutMultiInputState(ctx.state.linearLayoutMultiInputState));
    ctx.state.linearLayoutSelectionMaps.delete(targetId);
    if (replaceTabs || ctx.getSessionTabs().length === 0) {
        ctx.setSessionTabs([nextDocument]);
    } else {
        const index = ctx.getSessionTabs().findIndex((tab) => tab.id === targetId);
        ctx.setSessionTabs(index === -1
            ? [...ctx.getSessionTabs(), nextDocument]
            : ctx.getSessionTabs().map((tab, tabIndex) => tabIndex === index ? nextDocument : tab));
    }
    await ctx.loadTab(targetId);
}
