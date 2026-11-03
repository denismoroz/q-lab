import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Both the dev server (`npm run dev`) and the preview server that serves the built page
// (`npm run preview`, what launchd runs: ops/console.sh, docs/CONSOLE.md) proxy /api to the FastAPI
// backend, so the frontend never needs to know the API's host and CORS stays out of the picture.
// QLAB_API points a second console (another port, another API) somewhere else, for trying a change.
const proxy = {
  '/api': {
    target: process.env.QLAB_API ?? 'http://127.0.0.1:8890',
    changeOrigin: true,
  },
}

export default defineConfig({
  plugins: [react()],
  server: { port: 5890, strictPort: true, proxy },
  preview: { port: 5890, strictPort: true, proxy },
})
