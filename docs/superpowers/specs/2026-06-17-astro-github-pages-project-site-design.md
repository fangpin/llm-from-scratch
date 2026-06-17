# Astro GitHub Pages Project Site Design

## Summary

Build a bilingual project site for the `LLM from Scratch` repository and publish it with GitHub Pages. The site should default to English, provide a clean Chinese mirror under `/zh/`, and present the repository as a high-quality technical project rather than a generic personal portfolio.

The site's primary audiences are:

1. GitHub visitors deciding whether to star, read, or clone the repository.
2. People evaluating the project as a strong representative work.

The design direction is a dark, futuristic, motion-rich project site with strong first-screen impact, while still keeping technical content legible and grounded in the repository's actual implementation and results.

## Goals

1. Create a visually distinctive multi-page homepage experience for GitHub Pages.
2. Make `LLM from Scratch` the primary first-viewport signal.
3. Support English by default and Chinese as a first-class mirror, not a machine-translated overlay.
4. Showcase the repository's strongest proof points:
   - decoder-only Transformer implemented from scratch in PyTorch
   - BPE tokenizer
   - RoPE, RMSNorm, SwiGLU
   - Flash Attention 2
   - distributed training
   - SFT and RLFT examples on Qwen2.5-Math-1.5B and gsm8k
5. Provide durable routes for architecture, fine-tuning results, benchmarks, and docs entry points.
6. Set up GitHub Pages deployment so the site can be published automatically from the repository.

## Non-Goals

1. Do not turn the site into a personal portfolio homepage.
2. Do not auto-ingest every existing markdown file into a full documentation portal in the first pass.
3. Do not add a blog, roadmap, or unrelated marketing content.
4. Do not rewrite the Python training code or existing docs outside of what is necessary to support the site.

## Project Constraints

1. The repository is primarily a Python project, not an existing frontend app.
2. The repo already contains usable project content in `README.md`, `README_cn.md`, `BENCHMARK.md`, `docs/*.md`, and `img/`.
3. The site must be static and compatible with GitHub Pages.
4. English must be the default experience.
5. Chinese must live at stable mirrored routes such as `/zh/architecture`.

## Recommended Technical Approach

Use `Astro` as a self-contained static site inside this repository.

Why Astro fits this project:

1. It is a strong fit for multi-page static sites on GitHub Pages.
2. It supports a content-heavy project site better than a React app-first setup.
3. It can render fast static pages while still allowing selective motion and client-side enhancements.
4. It supports a clear bilingual route structure without runtime translation hacks.
5. It leaves room for later markdown-backed docs integration.

## Site Structure

The first implementation should create these top-level English routes:

- `/`
- `/architecture`
- `/sft-rlft`
- `/benchmarks`
- `/docs`

The first implementation should create these mirrored Chinese routes:

- `/zh/`
- `/zh/architecture`
- `/zh/sft-rlft`
- `/zh/benchmarks`
- `/zh/docs`

## Page Responsibilities

### Home

Purpose: create impact quickly and explain why the repository matters.

Content responsibilities:

1. Hero with `LLM from Scratch` as the main headline.
2. Supporting line describing a from-scratch decoder-only Transformer in PyTorch.
3. Primary actions for GitHub and Docs exploration.
4. High-signal capability tags such as `RoPE`, `RMSNorm`, `SwiGLU`, `Flash Attention 2`, `DDP`, `SFT`, and `RLFT`.
5. A proof band showing high-value metrics and implementation scope.
6. Strong visual entry points into Architecture, SFT and RLFT, Benchmarks, and Docs.

### Architecture

Purpose: explain what was implemented and how the pieces fit together.

Content responsibilities:

1. Tokenizer section.
2. Transformer core section.
3. Training loop and checkpointing section.
4. Parallel and distributed training section.
5. Kernel optimization section.
6. Data processing pipeline section.

This page should emphasize clarity through visual grouping, sequence explanation, and code-adjacent descriptions rather than dense prose.

### SFT and RLFT

Purpose: showcase the strongest applied-results story in the repository.

Content responsibilities:

1. Qwen2.5-Math-1.5B + gsm8k context.
2. Baseline and improved metrics.
3. Format-compliance improvement.
4. Fine-tuning workflow summary.
5. Links into the relevant docs and scripts.

This page should feel like a proof page, not a generic "alignment" overview.

### Benchmarks

Purpose: show concrete training and performance evidence.

Content responsibilities:

1. Existing loss and learning-rate charts from `img/loss.png` and `img/lr.png`.
2. Benchmark summary cards and implementation-performance talking points.
3. Flash Attention 2 context and why it matters in this project.

### Docs

Purpose: provide a curated entry point into project learning material.

Content responsibilities:

1. Quickstart path.
2. Training path.
3. Tokenizer path.
4. Data processing path.
5. Fine-tuning path.

The first version should present structured doc entry cards and short summaries. It should not attempt to fully restyle every markdown document into a site-native page in the initial implementation.

## Visual Direction

The visual direction is "dark experimental lab" with strong motion and technical polish.

### Tone

1. Dark graphite or near-black base.
2. High-contrast cyan and acid-lime accents.
3. Small warm accents for signal separation where useful.
4. Avoid the common purple-on-white AI landing page look.

### Composition

1. Full-bleed hero, not a card-based landing page.
2. Dense but controlled information layout after the hero.
3. Strong contrast between atmosphere and legible technical content.
4. Use section bands and asymmetric composition rather than repeated floating cards.

### Motion

1. Hero background should feel like an "attention field" or "token stream" visualization.
2. Use motion on load, on scroll, and on hover, but keep it purposeful.
3. Prioritize high-impact transitions over many tiny effects.
4. Keep mobile motion lighter to preserve readability and performance.

### Typography

1. Headline typography should feel sharp and technical, not playful.
2. Body typography should stay highly readable.
3. Type scale should be aggressive in the hero and restrained in content sections.

## Interaction Design

### Navigation

1. Sticky or floating top navigation.
2. Project name on the left.
3. Route links for main sections.
4. Language switcher visible in the global navigation.
5. GitHub link visible at all times.

### Language Switching

1. English is the default route tree.
2. Chinese is the mirrored `/zh/` route tree.
3. Every page should expose a direct switch to its peer route in the other language.
4. The language switch should be explicit and immediate, not hidden in a settings panel.

### Calls to Action

Primary CTAs:

1. `View on GitHub`
2. `Explore Docs`

Secondary CTAs can include:

1. `See Architecture`
2. `View Benchmarks`

## Content Strategy

The site should reuse and refine repository content rather than inventing new claims.

Content sources for the first version:

1. `README.md`
2. `README_cn.md`
3. `BENCHMARK.md`
4. `docs/qwen25-math-gsm8k-rl-finetune.md`
5. other existing docs that map cleanly into the curated Docs page
6. `img/loss.png`
7. `img/lr.png`

Content should be selectively rewritten for web presentation where necessary, especially on the Home and Architecture pages.

## Information Architecture and Data Flow

### Content Organization

Use a small localized content layer instead of hardcoding all copy directly into page components.

Recommended structure:

1. shared layout and section components
2. per-locale content dictionaries for navigation, hero copy, labels, and page-specific copy
3. page modules that assemble content and visuals from those dictionaries

This keeps bilingual parity manageable and avoids mixed-language component logic.

### Asset Flow

1. Reuse repository charts from `img/`.
2. Add any new decorative assets under the site's public asset directory.
3. Keep decorative visuals lightweight enough for GitHub Pages delivery.

## Implementation Outline

Create a dedicated site directory in the repo, preferably `site/`, containing:

1. Astro config
2. site package manifest
3. page routes
4. shared components
5. styles
6. localized content definitions
7. public assets

Expected implementation responsibilities:

1. Initialize Astro in `site/`.
2. Create the bilingual route structure.
3. Build the five English pages and five Chinese pages.
4. Create a shared layout system and section components.
5. Implement a motion-rich hero and high-signal content sections.
6. Wire GitHub Pages deployment through GitHub Actions.
7. Document local development and deployment commands.

## GitHub Pages Deployment

Deployment should be automated through a repository workflow.

The deployment work should include:

1. Static site build configuration for Astro.
2. Correct base-path handling for GitHub Pages.
3. Workflow file under `.github/workflows/` to build and publish the site.
4. A short repo note explaining how Pages is expected to be enabled.

The site should be deployable without requiring a custom server.

## Error Handling

1. If a localized page peer is missing, the build should fail rather than silently omitting language parity.
2. If required content assets like charts are missing, the affected page should degrade clearly or the build should surface the issue.
3. If GitHub Pages base path changes, configuration should be isolated so the fix is local and obvious.

## Testing and Verification Strategy

Verification should include:

1. local Astro build success
2. local dev-server smoke test
3. desktop and mobile layout checks
4. bilingual route checks
5. language-switcher link correctness
6. GitHub Pages workflow syntax validation if practical

Visual verification should focus on:

1. hero readability
2. section spacing
3. non-overlapping text
4. stable navigation and language switch behavior
5. image rendering for benchmark charts

## Acceptance Criteria

The first release is successful when all of the following are true:

1. The repo contains a self-contained Astro site for GitHub Pages.
2. The site defaults to English and supports mirrored Chinese pages.
3. The site includes `Home`, `Architecture`, `SFT & RLFT`, `Benchmarks`, and `Docs`.
4. The homepage strongly presents `LLM from Scratch` as the first-screen focus.
5. The visual design is distinctly dark, futuristic, and motion-aware without sacrificing readability.
6. Existing project proof points and charts are incorporated into the experience.
7. GitHub Pages deployment is wired through repository automation.

## Risks and Mitigations

### Risk: visual ambition overwhelms content

Mitigation:

Keep the hero expressive but make technical proof points visible within the first scroll.

### Risk: bilingual maintenance becomes messy

Mitigation:

Use mirrored route structure and centralized localized content.

### Risk: static Pages path issues

Mitigation:

Handle base-path concerns explicitly in Astro config and verify with a production-style local build.

### Risk: docs scope expands too quickly

Mitigation:

Keep the initial Docs page curated and route-focused rather than attempting full markdown site conversion.

## Open Decisions Resolved in This Spec

1. Audience: GitHub visitors and project-showcase viewers.
2. Visual direction: dark futuristic and motion-rich.
3. Primary first-screen headline: `LLM from Scratch`.
4. Personal branding presence: minimal, limited to repo-level attribution or footer links if needed.
5. Site shape: multi-page project site.
6. Main page skeleton: `Home / Architecture / SFT & RLFT / Benchmarks / Docs`.
7. Framework: Astro.
8. Locale default: English.
9. Chinese route strategy: mirrored `/zh/` routes.

## Implementation Boundary for the Next Phase

The next phase should implement the static site and deployment setup described here. It should stay tightly scoped to site creation, localized content structure, and GitHub Pages publishing. It should not expand into unrelated repo refactors or a full documentation migration.
