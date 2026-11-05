# MDANCE website (Astro)

The MDANCE landing page as an Astro project. It builds to the same page as the final `index.html`.

## Run it locally

```bash
npm install
npm run dev       # http://localhost:4321/MDANCE/
npm run build     # writes the site to dist/
npm run preview   # serves dist/ to check the build
```

Needs Node 18.17 or newer.

## Where things are

- `src/pages/index.astro`: the whole page. Edit text and markup here.
  - Styles sit in one `<style is:global>` block near the top; Astro bundles them into a CSS file on build.
  - Scripts are `<script is:inline>`, so they run exactly as written.
  - Curly braces in page text are written as `&#123;` and `&#125;`, because Astro reads `{ }` as code.
- `public/.nojekyll`: tells GitHub Pages to serve the `_astro` folder.
- `astro.config.mjs`: set for https://mqcomplab.github.io/MDANCE. For a different address, change `site` and `base`.

## Publish with GitHub Pages

The workflow `.github/workflows/deploy-site.yml` (at the root of the MDANCE repo) builds this folder and publishes it.

1. Once, in the repo: Settings → Pages → Source: **GitHub Actions**.
2. Push to `main`. Every push that changes `site/` redeploys the site.
