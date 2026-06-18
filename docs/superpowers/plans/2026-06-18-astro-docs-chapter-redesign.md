# Astro Docs Chapter Redesign Implementation Plan

**Goal:** Replace the current external-link Docs entry with a bilingual, chaptered, Markdown-backed documentation system inside the Astro site.

**Architecture:** Refactor repo-level Markdown into mirrored `docs/en` and `docs/zh` chapter trees, load those chapters from the Astro app, generate internal docs overview pages plus dynamic chapter routes, and keep per-chapter locale switching stable.

**Tech Stack:** Astro, TypeScript, Markdown content collections or loader-backed content, Vitest, Playwright, client-side Mermaid rendering

---

## Files To Add

- `docs/en/index.md`
- `docs/en/project-overview.md`
- `docs/en/tokenizer.md`
- `docs/en/transformer-core.md`
- `docs/en/training-loop.md`
- `docs/en/flash-attention.md`
- `docs/en/distributed-training.md`
- `docs/en/data-pipeline.md`
- `docs/en/sft-gsm8k.md`
- `docs/en/rlft-gsm8k.md`
- `docs/zh/index.md`
- `docs/zh/project-overview.md`
- `docs/zh/tokenizer.md`
- `docs/zh/transformer-core.md`
- `docs/zh/training-loop.md`
- `docs/zh/flash-attention.md`
- `docs/zh/distributed-training.md`
- `docs/zh/data-pipeline.md`
- `docs/zh/sft-gsm8k.md`
- `docs/zh/rlft-gsm8k.md`
- `site/src/content.config.ts`
- `site/src/components/DocsChapterNav.astro`
- `site/src/components/MermaidInit.astro`
- `site/src/layouts/DocsLayout.astro`
- `site/src/lib/docs.ts`
- `site/src/pages/docs/[slug].astro`
- `site/src/pages/zh/docs/[slug].astro`
- `site/tests/docs.spec.ts`

## Files To Modify

- `site/package.json`
- `site/package-lock.json`
- `site/src/lib/locale.ts`
- `site/src/content/site.ts`
- `site/src/pages/docs/index.astro`
- `site/src/pages/zh/docs/index.astro`
- `site/src/styles/global.css`
- `site/tests/routes.spec.ts`
- `site/tests/content.spec.ts`
- `site/tests/e2e/site.spec.ts`

## Files To Retire From Published Docs Surface

- `docs/1.md`
- `docs/2.md`
- `docs/3.md`
- `docs/4.md`
- `docs/5-sft.md`
- `docs/qwen25-math-gsm8k-rl-finetune.md`
- `docs/technical_article3.md`

These remain as migration inputs during implementation but should no longer be the primary published docs shape.

---

## Task 1: Create the New Markdown Chapter Tree

- [ ] Create mirrored English and Chinese chapter files under `docs/en` and `docs/zh`.
- [ ] Add frontmatter for title, summary, slug, locale, group, order, and source file metadata.
- [ ] Refactor the existing technical docs into:
  - flash attention
  - distributed training
  - data pipeline
  - SFT
  - RLFT
- [ ] Author new focused chapters for:
  - project overview
  - tokenizer
  - transformer core
  - training loop
- [ ] Add overview markdown files for both locales.

## Task 2: Wire Astro to Load Repo-Level Markdown

- [ ] Add a content configuration that can load Markdown from repo-level `docs/en` and `docs/zh`.
- [ ] Add a docs helper module for:
  - locale-aware chapter lookup
  - grouping
  - previous/next chapter resolution
  - slug-to-peer-locale lookup
- [ ] Confirm that the site can render docs sourced outside `site/`.

## Task 3: Build the Docs Overview and Chapter Pages

- [ ] Replace the current external-link docs overview pages with grouped internal chapter cards.
- [ ] Add dynamic English and Chinese chapter routes.
- [ ] Add a long-form docs layout with:
  - chapter header
  - source-file metadata
  - related chapter links
  - previous and next navigation
- [ ] Add client-side Mermaid initialization for Markdown code blocks tagged as `mermaid`.

## Task 4: Extend Locale Switching for Chapter Routes

- [ ] Update locale path utilities to support `/docs/:slug` <-> `/zh/docs/:slug`.
- [ ] Ensure the global language switch works on docs chapter pages.
- [ ] Keep the existing static route behavior unchanged for the rest of the site.

## Task 5: Add Verification Coverage

- [ ] Add unit tests for docs metadata and locale parity.
- [ ] Extend route tests to cover dynamic docs route mapping.
- [ ] Extend content tests to verify chapter groups and bilingual parity.
- [ ] Extend Playwright smoke tests to cover:
  - docs overview rendering
  - chapter page rendering
  - chapter locale switch
  - Mermaid block presence

## Task 6: Run Local Verification

- [ ] Run `npm --prefix site run test`
- [ ] Run `npm --prefix site run build`
- [ ] Run `npm --prefix site run test:e2e`
- [ ] Restart or verify the local preview server and confirm the docs routes load:
  - `/lm/docs`
  - `/lm/docs/transformer-core`
  - `/lm/zh/docs`
  - `/lm/zh/docs/transformer-core`

## Acceptance Checks

- [ ] `Docs` is now an internal chaptered docs experience.
- [ ] All planned chapters exist in both locales.
- [ ] The site still defaults to English.
- [ ] Docs chapters render from Markdown rather than hardcoded page copy.
- [ ] Chapter-to-chapter locale switching works.
- [ ] Existing deep technical material survives the migration.
- [ ] Duplicate distributed-training article content is consolidated.
