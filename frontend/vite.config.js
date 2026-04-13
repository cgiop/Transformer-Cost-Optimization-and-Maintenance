import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/upload_data': 'http://127.0.0.1:8000',
      '/train': 'http://127.0.0.1:8000',
      '/predict_raw': 'http://127.0.0.1:8000'
    }
  }
})
