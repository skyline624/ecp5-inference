import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  server: {
    host: '127.0.0.1',   // IPv4 fixe (sinon Windows résout localhost -> ::1, curl KO)
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
});