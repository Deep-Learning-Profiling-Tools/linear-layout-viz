import { expect, test } from '@playwright/test';
import { PNG } from 'pngjs';

test('viewer demo boots, paints tensors, and exposes core controls', async ({ page }) => {
    const browserErrors: string[] = [];
    const failedResponses: string[] = [];
    const optionalStaticSession = /\/api\/session\.json$/;
    const headlessShaderValidation = /^THREE\.THREE\.WebGLProgram: Shader Error/;

    page.on('pageerror', (error) => {
        browserErrors.push(error.message);
    });
    page.on('console', (message) => {
        // headless chromium can emit three.js shader validation errors even when the 2d canvas paints
        if (message.type() === 'error' && !headlessShaderValidation.test(message.text())) {
            browserErrors.push(message.text());
        }
    });
    page.on('response', (response) => {
        // static demos intentionally probe for a live session manifest before falling back to baked tabs
        if (response.status() >= 400 && !optionalStaticSession.test(response.url())) {
            failedResponses.push(`${response.status()} ${response.url()}`);
        }
    });

    await page.goto('/');

    // startup
    await expect(page.locator('.ribbon')).toBeVisible();
    await expect(page.locator('#viewport')).toBeVisible();
    expect(await page.locator('.tab-button').count()).toBeGreaterThan(0);

    // extension widgets
    await expect(page.locator('#linear-layout-preset-widget')).toBeVisible();
    await expect(page.locator('#linear-layout-widget')).toBeVisible();
    await expect(page.locator('#tensor-view-widget')).toBeVisible();
    await expect(page.getByText('Preset', { exact: true })).toBeVisible();
    await expect(page.getByText('Layout Specs', { exact: true })).toBeVisible();

    // viewport paint
    await page.waitForFunction(() => (
        Array.from(document.querySelectorAll<HTMLCanvasElement>('#viewport canvas'))
            .some((canvas) => canvas.width > 0 && canvas.height > 0)
    ));
    const viewportImage = PNG.sync.read(await page.locator('#viewport').screenshot());
    const colors = new Set<string>();
    for (let index = 0; index < viewportImage.data.length; index += 16) {
        const alpha = viewportImage.data[index + 3];
        if (alpha === 0) continue;
        colors.add(`${viewportImage.data[index]},${viewportImage.data[index + 1]},${viewportImage.data[index + 2]}`);
        if (colors.size > 16) break;
    }
    expect(colors.size).toBeGreaterThan(16);

    // command palette
    await page.keyboard.press('?');
    await expect(page.locator('#command-palette')).toBeVisible();
    await page.locator('#command-palette-input').fill('display');
    await expect(page.getByText('Display as 2D', { exact: true })).toBeVisible();

    expect(browserErrors).toEqual([]);
    expect(failedResponses).toEqual([]);
});
