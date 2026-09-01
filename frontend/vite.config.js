/// <reference types="vitest" />
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
// The Vite dev server proxies /api to the FastAPI backend during development.
export default defineConfig(function (_a) {
    var mode = _a.mode;
    var env = loadEnv(mode, ".");
    var proxyTarget = env.VITE_API_PROXY || "http://localhost:8000";
    return {
        plugins: [react()],
        server: {
            host: true,
            port: 5173,
            proxy: {
                "/api": {
                    target: proxyTarget,
                    changeOrigin: true,
                },
                "/health": {
                    target: proxyTarget,
                    changeOrigin: true,
                },
            },
        },
        test: {
            environment: "jsdom",
            globals: true,
            setupFiles: "./src/test/setup.ts",
        },
    };
});
