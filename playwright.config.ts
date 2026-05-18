import { defineConfig } from '@playwright/test';

export default defineConfig({
    testDir: './e2e',
    webServer: {
        command: 'npm run dev -- --host 127.0.0.1',
        url: 'http://127.0.0.1:5173',
        reuseExistingServer: !process.env.CI,
        timeout: 120_000,
    },
    use: {
        baseURL: 'http://127.0.0.1:5173',
    },
});
