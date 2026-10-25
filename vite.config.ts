import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteSingleFile } from 'vite-plugin-singlefile'

// One self-contained dist/index.html that opens by double-click, no server needed.
export default defineConfig({ plugins: [react(), viteSingleFile()], base: './' })
