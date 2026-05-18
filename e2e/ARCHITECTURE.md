The `e2e/` directory contains browser-level checks for the packaged LL-viz app.

`viewer-smoke.spec.ts` starts the root Vite app through Playwright, verifies that
the tensor-viz shell boots, confirms the linear-layout extension widgets are
visible, checks that the viewport paints nonblank tensor content, and opens the
command palette. This catches the failures unit tests miss: missing package
exports, broken extension registration, CSS/asset mistakes, and renderer startup
problems.

Keep e2e tests small. Parser, preset, and mapping behavior should be covered in
`src/extensions/linear-layout/linear-layout.test.ts`; Playwright should only
protect workflows that require a real browser.
