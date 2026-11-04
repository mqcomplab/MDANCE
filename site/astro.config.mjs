import { defineConfig } from 'astro/config';

// GitHub Pages: https://mqcomplab.github.io/MDANCE
// If you move the site to its own domain or a user/org page, change `site` and remove `base`.
export default defineConfig({
  site: 'https://mqcomplab.github.io',
  base: '/MDANCE',
  compressHTML: false,
});
