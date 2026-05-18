import { escapeInfo, labelWithInfo } from '@tensor-viz/viewer-demo/extension-api';
import {
    bakedComposeLayoutExamples,
    buildComposeRuntime,
    createComposeLayoutDocument,
} from '../linear-layout.js';
import {
    cloneLinearLayoutMultiInputState,
    cloneLinearLayoutState,
    composeLayoutMetaForTab,
    defaultLinearLayoutCellTextState,
    defaultLinearLayoutMultiInputState,
    snapshotTensorViews,
    type LinearLayoutUiContext,
} from '../linear-layout-state.js';
import { applyLinearLayoutSpec } from './linear-layout-widget-actions.js';
import {
    VISIBLE_TENSORS_ERROR,
    autosizeTextarea,
    copyText,
    settleInitialLayout,
} from './linear-layout-widget-shared.js';

// this widget owns the editable linear-layout notation.
// parser/model files own semantic validation; this file only moves text between
// form controls, examples, tabs, and apply/copy actions.
// keeping the examples close to the UI matters because contributors use these
// strings as executable documentation while debugging new preset families.
// render actions preserve tensor views by default because tensor-view edits are
// often made after the layout spec itself is already correct.
// textarea sizing is repeated after content changes so long notation examples
// do not leave stale scroll heights in collapsed/expanded widgets.
// notices are split between this widget and Visible Tensors because rebuilding
// hidden tensors has a different failure recovery path than parser failures.

/**
 * Produces the collapsible help panel shown beside the Layouts textarea, including compose-layout signature rules and single-, multi-, and non-surjective layout examples.
 *
 * @returns HTML string for a `<details class="usage-guide">` block that the specs widget injects above the layout specification textarea.
 * @noThrows Returns a static template literal and performs no DOM access, parsing, interpolation from user input, or validation.
 * @example
 * const html = linearLayoutSpecsHelpHtml();
 *
 * html.includes('<summary>How do I use this?</summary>'); // true
 * html.includes('Tile2x1: [T,W] -&gt; [Y,X]'); // true
 */
function linearLayoutSpecsHelpHtml(): string {
    return `
      <details class="usage-guide">
        <summary>How do I use this?</summary>
        <div class="usage-guide-body">
          <div class="usage-guide-step">
            <span>Add one or more named layout blocks in the form <strong>name: [inputs] -> [outputs]</strong>.</span>
          </div>
          <div class="usage-guide-step">
            <span>Each block must include exactly one labeled basis row per input, such as <strong>T: [[1,0],[0,1]]</strong>.</span>
          </div>
          <div class="usage-guide-step">
            <span>Separate multiple layout definitions with a blank line.</span>
          </div>
          <div class="usage-guide-step">
            <span>Input/output labels and basis-row labels must look like <strong>T</strong>, <strong>A0</strong>, or <strong>B12</strong>: one leading letter followed by optional digits only.</span>
          </div>
          <div class="usage-guide-step">
            <span>Layout names may use letters, digits, and underscores, but must start with a letter or underscore.</span>
          </div>
          <div class="usage-guide-column">
            <div class="usage-guide-subtitle">Single Layout</div>
            <div class="usage-guide-example">
              <code>Tile2x1: [T,W] -&gt; [Y,X]</code>
              <code>T: [[0,1],[0,2]]</code>
              <code>W: [[1,0]]</code>
            </div>
            <div class="usage-guide-example">
              <code>Swizzle: [Y,X] -&gt; [Y,X]</code>
              <code>Y: [[1,1],[2,2]]</code>
              <code>X: [[0,1],[0,2]]</code>
            </div>
            <div class="usage-guide-example">
              <code>Block: [T,W,R] -&gt; [A,B]</code>
              <code>T: [[2,0],[4,0],[0,1],[0,2],[0,4]]</code>
              <code>W: [[0,8],[0,16]]</code>
              <code>R: [[1,0]]</code>
            </div>
            <div class="usage-guide-subtitle">Multiple Layouts</div>
            <div class="usage-guide-example">
              <code>Tile2x1: [T,W] -&gt; [Y,X]</code>
              <code>T: [[0,1],[0,2]]</code>
              <code>W: [[1,0]]</code>
              <code></code>
              <code>Swizzle: [Y,X] -&gt; [Y,X]</code>
              <code>Y: [[1,1],[2,2]]</code>
              <code>X: [[0,1],[0,2]]</code>
              <code></code>
              <code>GetT: [Y1,X1] -&gt; [T]</code>
              <code>Y1: [[1],[2],[16]]</code>
              <code>X1: [[4],[8]]</code>
              <code></code>
              <code>GetW: [Y2,X2] -&gt; [W]</code>
              <code>Y2: [[1]]</code>
              <code>X2: [[2]]</code>
              <code></code>
              <code>GetR: [X3] -&gt; [R]</code>
              <code>X3: [[1]]</code>
            </div>
            <div class="usage-guide-subtitle">Non-Surjective Layout</div>
            <div class="usage-guide-example">
              <code>Sparse_Block: [T,W,R] -&gt; [A,B]</code>
              <code>T: [[4,0],[8,0],[0,1],[0,2],[0,4]]</code>
              <code>W: [[0,8],[0,16]]</code>
              <code>R: [[1,0]]</code>
            </div>
          </div>
        </div>
      </details>
    `;
}

/**
 * Builds the static disclosure block that teaches users how to write a Layout Operation expression.
 *
 * @returns HTML for the Layout Operation help panel, including supported operators, precedence, reference layout specs, and copyable example expressions.
 * @noThrows The block is assembled from a fixed template string and does not read viewer state, query the DOM, or call helpers that can reject.
 * @example
 * const html = linearLayoutOperationHelpHtml();
 * console.assert(html.includes('Supported operators'));
 * console.assert(html.includes('Swizzle(Tile2x1)'));
 * console.assert(html.includes('inv(GetT * GetW * GetR)(inv(Block))'));
 */
function linearLayoutOperationHelpHtml(): string {
    return `
      <details class="usage-guide">
        <summary>How do I use this?</summary>
        <div class="usage-guide-body">
          <div class="usage-guide-step">
            <span>Use the names you defined above to choose which layout chain to render.</span>
          </div>
          <div class="usage-guide-step">
            <span>Supported operators are parentheses, inverse like <strong>inv(A)</strong>, composition like <strong>A(B)</strong>, and products like <strong>A * B</strong>.</span>
          </div>
          <div class="usage-guide-step">
            <span>Precedence is <strong>parentheses</strong>, then <strong>inv(...)</strong>, then <strong>composition A(B)</strong>, then <strong>product A * B</strong>.</span>
          </div>
          <div class="usage-guide-column">
            <div class="usage-guide-subtitle">Reference Specs</div>
            <div class="usage-guide-step">
              <span>Copy this entire block into the <strong>Layouts</strong> textbox above.</span>
            </div>
            <div class="usage-guide-example">
              <code>Tile2x1: [T,W] -&gt; [Y,X]</code>
              <code>T: [[0,1],[0,2]]</code>
              <code>W: [[1,0]]</code>
              <code></code>
              <code>Swizzle: [Y,X] -&gt; [S,B]</code>
              <code>Y: [[1,1],[2,2]]</code>
              <code>X: [[0,1],[0,2]]</code>
              <code></code>
              <code>GetT: [Y1,X1] -&gt; [T]</code>
              <code>Y1: [[1],[2],[16]]</code>
              <code>X1: [[4],[8]]</code>
              <code></code>
              <code>GetW: [Y2,X2] -&gt; [W]</code>
              <code>Y2: [[1]]</code>
              <code>X2: [[2]]</code>
              <code></code>
              <code>GetR: [X3] -&gt; [R]</code>
              <code>X3: [[1]]</code>
              <code></code>
              <code>Block: [T,W,R] -&gt; [A,B]</code>
              <code>T: [[2,0],[4,0],[0,1],[0,2],[0,4]]</code>
              <code>W: [[0,8],[0,16]]</code>
              <code>R: [[1,0]]</code>
              <code></code>
              <code>Sparse_Block: [T,W,R] -&gt; [A,B]</code>
              <code>T: [[4,0],[8,0],[0,1],[0,2],[0,4]]</code>
              <code>W: [[0,8],[0,16]]</code>
              <code>R: [[1,0]]</code>
            </div>
            <div class="usage-guide-subtitle">Examples</div>
            <div class="usage-guide-step">
              <span>Copy any one pale-box example below into <strong>Layout Operation</strong>.</span>
            </div>
            <div class="usage-guide-subtitle">Single Layout</div>
            <div class="usage-guide-example"><code>Tile2x1</code></div>
            <div class="usage-guide-example"><code>Swizzle</code></div>
            <div class="usage-guide-example"><code>Sparse_Block</code></div>
            <div class="usage-guide-subtitle">Composition</div>
            <div class="usage-guide-example"><code>Swizzle(Tile2x1)</code></div>
            <div class="usage-guide-subtitle">Inverse</div>
            <div class="usage-guide-example"><code>inv(GetT)</code></div>
            <div class="usage-guide-example"><code>inv(Swizzle(Tile2x1))</code></div>
            <div class="usage-guide-example"><code>inv(Swizzle)(Tile2x1)</code></div>
            <div class="usage-guide-subtitle">Product</div>
            <div class="usage-guide-example"><code>GetT * GetW * GetR</code></div>
            <div class="usage-guide-example"><code>Swizzle * Swizzle</code></div>
            <div class="usage-guide-subtitle">Needs Parentheses</div>
            <div class="usage-guide-example"><code>(Swizzle * Swizzle)(Tile2x1)</code></div>
            <div class="usage-guide-subtitle">Composition + Inverse + Product</div>
            <div class="usage-guide-example"><code>Sparse_Block(inv(inv(GetT)) * GetW * GetR)</code></div>
            <div class="usage-guide-example"><code>inv(GetT * GetW * GetR)(inv(Block))</code></div>
          </div>
        </div>
      </details>
    `;
}

/**
 * Renders the Layout Specs sidebar controls and wires them to the active linear-layout editing state.
 *
 * @param ctx - Linear-layout UI context containing the specs widget element, the editable specs/operation/input-name state, matrix-preview state, notice text, render callbacks, and clipboard/apply dependencies used by the button handlers.
 * @returns Nothing. The function replaces `ctx.linearLayoutWidget.innerHTML`, autosizes the specs and operation textareas, and attaches input/click handlers that keep `ctx.state.linearLayoutState` synchronized with the form.
 * @noThrows The render pass only interpolates escaped state into DOM markup, queries the nodes it just created, and attaches optional listeners; failures from async apply/copy button actions are handled inside their event callbacks instead of being thrown by the initial render call.
 * @example
 * const ctx = makeLinearLayoutUiContext({
 *   specsText: 'Tile2x1: [T,W] -> [Y,X]\nT: [[0,1],[0,2]]\nW: [[1,0]]',
 *   operationText: 'Tile2x1',
 *   inputName: 'Input',
 * });
 *
 * renderLinearLayoutWidget(ctx);
 *
 * console.assert(ctx.linearLayoutWidget.querySelector('#linear-layout-specs')?.textContent?.includes('Tile2x1'));
 * console.assert(ctx.linearLayoutWidget.querySelector('#linear-layout-operation') instanceof HTMLTextAreaElement);
 * console.assert(ctx.linearLayoutWidget.querySelector('#linear-layout-apply')?.textContent === 'Render Layout');
 */
export function renderLinearLayoutWidget(ctx: LinearLayoutUiContext): void {
    const showLocalStatus = ctx.state.linearLayoutNotice?.text !== VISIBLE_TENSORS_ERROR;
    const statusClass = ctx.state.linearLayoutNotice?.tone === 'success' ? 'success-box' : 'error-box';
    // visible-tensor errors are rendered next to the toggles that caused them;
    // parser/apply/copy messages stay here by the text fields they refer to.
    const status = showLocalStatus && ctx.state.linearLayoutNotice ? `<div class="${statusClass}">${escapeInfo(ctx.state.linearLayoutNotice.text)}</div>` : '';
    const matrixBlock = !ctx.state.showLinearLayoutMatrix
        ? ''
        : `<div class="mono-block linear-layout-matrix-preview">${ctx.state.linearLayoutMatrixPreview}</div>`;
    ctx.linearLayoutWidget.innerHTML = `
      ${ctx.widgetTitle('linear-layout', 'Define one or more named injective layouts, then use Layout Operation to build the rendered tensor chain.').replace('<h2>Linear Layout Specifications</h2>', '<h2>Layout Specs</h2>')}
      <div class="widget-body">
        <p class="widget-copy">Each specification starts with <span class="inline-code">name: [inputs] -> [outputs]</span>, followed by exactly one labeled basis row per input such as <span class="inline-code">T: [[1,0],[0,1]]</span>. Separate specifications with blank lines. Layout Operation supports names, <span class="inline-code">inv(...)</span>, <span class="inline-code">*</span>, and parentheses.</p>
        <div class="field">
          ${labelWithInfo('Layouts', 'Enter one or more specification blocks. Each block has one signature line plus one labeled basis row per input label.', 'linear-layout-specs')}
          ${linearLayoutSpecsHelpHtml()}
          <textarea id="linear-layout-specs" class="compact-textarea" rows="8" spellcheck="false">${escapeInfo(ctx.state.linearLayoutState.specsText)}</textarea>
        </div>
        <div class="field">
          ${labelWithInfo('Layout Operation', 'Enter the layout expression to visualize, such as A(B), inv(A), A * B, or parenthesized combinations.', 'linear-layout-operation')}
          ${linearLayoutOperationHelpHtml()}
          <textarea id="linear-layout-operation" class="compact-textarea" rows="2" spellcheck="false">${escapeInfo(ctx.state.linearLayoutState.operationText)}</textarea>
        </div>
        <div class="field">
          ${labelWithInfo('Input Tensor Name', 'Sets the display name of the root tensor at the start of the rendered chain.', 'linear-layout-input-name')}
          <input id="linear-layout-input-name" type="text" value="${escapeInfo(ctx.state.linearLayoutState.inputName)}" />
        </div>
        <div class="button-row linear-layout-action-row">
          <button class="primary-button" id="linear-layout-apply" type="button" title="Build the current layout chain and render the resulting tensors.">Render Layout</button>
          <button class="secondary-button" id="linear-layout-matrix" type="button" title="${ctx.state.showLinearLayoutMatrix ? 'Hide the matrix blocks for the current layout chain.' : 'Show the matrix blocks for the current layout chain.'}">${ctx.state.showLinearLayoutMatrix ? 'Hide Matrix' : 'Show Matrix'}</button>
          <button class="secondary-button" id="linear-layout-copy" type="button" title="Copy Python initialization code for the current layout definitions and operation.">Copy Init Code</button>
        </div>
        ${status}
        ${matrixBlock}
      </div>
    `;

    const specsInput = ctx.linearLayoutWidget.querySelector<HTMLTextAreaElement>('#linear-layout-specs');
    const operationInput = ctx.linearLayoutWidget.querySelector<HTMLTextAreaElement>('#linear-layout-operation');
    const inputNameInput = ctx.linearLayoutWidget.querySelector<HTMLInputElement>('#linear-layout-input-name');
    const apply = ctx.linearLayoutWidget.querySelector<HTMLButtonElement>('#linear-layout-apply');
    const copy = ctx.linearLayoutWidget.querySelector<HTMLButtonElement>('#linear-layout-copy');
    const matrix = ctx.linearLayoutWidget.querySelector<HTMLButtonElement>('#linear-layout-matrix');
    // initial autosize runs after innerHTML so browser layout has real textareas
    // to measure; doing this before render would read stale nodes.
    if (specsInput) autosizeTextarea(specsInput);
    if (operationInput) autosizeTextarea(operationInput);
    // save text on every keystroke so snapshot/export actions cannot race a
    // focused textarea that has not emitted blur yet.
    specsInput?.addEventListener('input', () => {
        ctx.state.linearLayoutState.specsText = specsInput.value;
        autosizeTextarea(specsInput);
    });
    operationInput?.addEventListener('input', () => {
        ctx.state.linearLayoutState.operationText = operationInput.value;
        autosizeTextarea(operationInput);
    });
    inputNameInput?.addEventListener('input', () => {
        ctx.state.linearLayoutState.inputName = inputNameInput.value;
    });
    apply?.addEventListener('click', async () => {
        // editor widgets depend on rebuilt compose metadata, so refresh them
        // only after the async apply step has updated the active document.
        await applyLinearLayoutSpec(ctx);
        ctx.renderLinearLayoutEditorWidgets();
    });
    copy?.addEventListener('click', async () => {
        try {
            await copyText(buildComposeRuntime(ctx.state.linearLayoutState).pythonCode);
            ctx.state.linearLayoutNotice = { tone: 'success', text: 'Copied Python init code.' };
        } catch (error) {
            ctx.state.linearLayoutNotice = { tone: 'error', text: error instanceof Error ? error.message : String(error) };
        }
        renderLinearLayoutWidget(ctx);
    });
    matrix?.addEventListener('click', () => {
        // matrix preview is derived text, not persisted document state.
        ctx.state.showLinearLayoutMatrix = !ctx.state.showLinearLayoutMatrix;
        renderLinearLayoutWidget(ctx);
    });
}

/**
 * Replaces the session with the built-in compose-layout examples and opens the first fallback tab.
 *
 * @param ctx - Linear-layout UI context with session-tab setters/loaders plus the per-tab linear-layout, cell-text, multi-input, tensor-view, and selection-map state stores that must be reset for baked examples.
 * @returns A promise that resolves to `true` after at least one baked example is installed, loaded, silently applied, and re-rendered; resolves to `false` when no baked examples are available or no initial tab can be selected.
 * @throws Propagates rejections from loading the first tab, settling its initial layout, or silently applying the baked layout specification.
 * @example
 * const ctx = makeLinearLayoutUiContext();
 *
 * await expect(loadBakedLinearLayoutTabs(ctx)).resolves.toBe(true);
 * console.assert(ctx.getSessionTabs().length > 0);
 * console.assert(ctx.getSessionTabs()[0]?.id === 'tab-1');
 * console.assert(ctx.renderLinearLayoutEditorWidgetsCalls === 1);
 */
export async function loadBakedLinearLayoutTabs(ctx: LinearLayoutUiContext): Promise<boolean> {
    const examples = bakedComposeLayoutExamples();
    if (examples.length === 0) return false;
    ctx.state.linearLayoutStates.clear();
    ctx.state.linearLayoutCellTextStates.clear();
    ctx.state.linearLayoutMultiInputStates.clear();
    ctx.state.linearLayoutTensorViewsStates.clear();
    ctx.setSessionTabs(examples.map(({ state, title }, index) => {
        const document = createComposeLayoutDocument(state, undefined, title);
        const id = `tab-${index + 1}`;
        const meta = composeLayoutMetaForTab(document);
        ctx.state.linearLayoutStates.set(id, cloneLinearLayoutState(state));
        ctx.state.linearLayoutCellTextStates.set(id, defaultLinearLayoutCellTextState(meta?.rootInputLabels ?? []));
        ctx.state.linearLayoutMultiInputStates.set(id, defaultLinearLayoutMultiInputState());
        ctx.state.linearLayoutTensorViewsStates.set(id, snapshotTensorViews(document.manifest.viewer));
        return { ...document, id, title };
    }));
    ctx.state.linearLayoutSelectionMaps.clear();
    const initialTabId = ctx.getSessionTabs()[0]?.id ?? null;
    if (!initialTabId) return false;
    await ctx.loadTab(initialTabId);
    await settleInitialLayout(ctx);
    await applyLinearLayoutSpec(ctx, { silent: true, preserveTensorViews: true });
    ctx.renderLinearLayoutEditorWidgets();
    return true;
}
