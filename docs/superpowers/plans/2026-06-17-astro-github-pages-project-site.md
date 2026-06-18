# Astro GitHub Pages Project Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a bilingual Astro project site for `LLM from Scratch` with a dark futuristic visual style, mirrored English and Chinese routes, and GitHub Pages deployment.

**Architecture:** Add a self-contained `site/` Astro app inside the Python repository. Keep content and UI composition localized through shared data modules, mirror the route tree under `/zh/`, and ship a static export through a new GitHub Actions Pages workflow.

**Tech Stack:** Astro, TypeScript, CSS, GitHub Pages Actions, Vitest, Playwright

---

## File Structure

### New Files

- `site/package.json` - frontend scripts and dependencies
- `site/package-lock.json` - lockfile required for `npm ci` in GitHub Actions
- `site/astro.config.mjs` - Astro configuration including GitHub Pages base path
- `site/tsconfig.json` - TypeScript config for Astro
- `site/public/favicon.svg` - site favicon
- `site/public/images/loss.png` - copied benchmark chart for the site build
- `site/public/images/lr.png` - copied benchmark chart for the site build
- `site/src/layouts/BaseLayout.astro` - shared page shell, meta tags, nav, footer
- `site/src/styles/global.css` - global tokens, layout, effects, responsive rules
- `site/src/components/HeroBackground.astro` - animated token-stream hero backdrop
- `site/src/components/SiteHeader.astro` - global navigation and locale switcher
- `site/src/components/SiteFooter.astro` - footer links and repo attribution
- `site/src/components/HeroSection.astro` - homepage hero
- `site/src/components/ProofBand.astro` - metrics strip and project proof points
- `site/src/components/FeaturePortalGrid.astro` - strong entry links to core sections
- `site/src/components/SectionIntro.astro` - reusable band heading block
- `site/src/components/ArchitectureFlow.astro` - architecture visualization section
- `site/src/components/MetricCards.astro` - benchmark and result cards
- `site/src/content/site.ts` - localized content dictionaries and route metadata
- `site/src/lib/locale.ts` - locale helpers and mirrored-route utilities
- `site/src/pages/index.astro` - English home
- `site/src/pages/architecture.astro` - English architecture page
- `site/src/pages/sft-rlft.astro` - English fine-tuning page
- `site/src/pages/benchmarks.astro` - English benchmarks page
- `site/src/pages/docs/index.astro` - English docs entry page
- `site/src/pages/zh/index.astro` - Chinese home
- `site/src/pages/zh/architecture.astro` - Chinese architecture page
- `site/src/pages/zh/sft-rlft.astro` - Chinese fine-tuning page
- `site/src/pages/zh/benchmarks.astro` - Chinese benchmarks page
- `site/src/pages/zh/docs/index.astro` - Chinese docs entry page
- `site/tests/routes.spec.ts` - Vitest checks for locale structure and mirrored routes
- `site/tests/content.spec.ts` - Vitest checks for required localized content fields
- `site/playwright.config.ts` - Playwright config for local verification
- `site/tests/e2e/site.spec.ts` - browser checks for desktop and mobile navigation/layout
- `.github/workflows/deploy-site.yml` - GitHub Pages build and deploy workflow
- `docs/site.md` - local run, build, and deployment notes for the new site

### Existing Files To Modify

- `.gitignore` - ignore `site/node_modules`, `site/dist`, Playwright artifacts, and Astro cache
- `README.md` - add a short section pointing to the project site and local dev command
- `README_cn.md` - add the same note in Chinese

## Task 1: Set Up an Isolated Workspace

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Check whether the current checkout is already an isolated worktree**

Run:

```bash
GIT_DIR=$(cd "$(git rev-parse --git-dir)" 2>/dev/null && pwd -P)
GIT_COMMON=$(cd "$(git rev-parse --git-common-dir)" 2>/dev/null && pwd -P)
SUPERPROJECT=$(git rev-parse --show-superproject-working-tree 2>/dev/null)
if [ "$GIT_DIR" = "$GIT_COMMON" ] && [ -z "$SUPERPROJECT" ]; then
  printf "normal-checkout\n%s\n" "$(git branch --show-current)"
fi
```

Expected:

```text
normal-checkout
master
```

- [ ] **Step 2: Verify that `.worktrees/` is ignored before creating a local worktree**

Run:

```bash
git check-ignore -q .worktrees || printf "not-ignored\n"
```

Expected:

```text
not-ignored
```

because the current `.gitignore` does not include `.worktrees`.

- [ ] **Step 3: Add `.worktrees/` to `.gitignore`**

Update `.gitignore` by appending:

```gitignore
.worktrees
```

- [ ] **Step 4: Verify `.worktrees/` is now ignored**

Run:

```bash
git check-ignore -q .worktrees && printf "ignored-worktrees\n"
```

Expected:

```text
ignored-worktrees
```

- [ ] **Step 5: Commit the ignore-rule change**

Run:

```bash
git add .gitignore
git commit -m "chore: ignore local worktrees"
git log -1 --pretty=%s
```

Expected:

```text
chore: ignore local worktrees
```

- [ ] **Step 6: Create the feature worktree**

Run:

```bash
git worktree add .worktrees/pin-astro-pages-site -b pin/astro-pages-site
git -C .worktrees/pin-astro-pages-site branch --show-current
```

Expected:

```text
pin/astro-pages-site
```

- [ ] **Step 7: Move into the isolated workspace and verify branch**

Run:

```bash
cd .worktrees/pin-astro-pages-site
pwd
git branch --show-current
```

Expected:

```text
/Users/bytedance/repos/llm/.worktrees/pin-astro-pages-site
pin/astro-pages-site
```

- [ ] **Step 8: Commit**

Run:

```bash
git status --short
```

Expected:

```text
```

No additional commit is needed here because the worktree is created from the latest committed state.

## Task 2: Bootstrap the Astro App and Verify a Clean Baseline

**Files:**
- Create: `site/package.json`
- Create: `site/astro.config.mjs`
- Create: `site/tsconfig.json`
- Create: `site/package-lock.json`
- Create: `site/public/favicon.svg`
- Modify: `.gitignore`
- Test: `site/package.json`

- [ ] **Step 1: Write the failing build expectation by adding the site scripts before the app exists**

Create `site/package.json` with:

```json
{
  "name": "llm-project-site",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "astro dev",
    "build": "astro build",
    "preview": "astro preview",
    "test": "vitest run",
    "test:e2e": "playwright test"
  },
  "dependencies": {
    "astro": "^5.10.0"
  },
  "devDependencies": {
    "@astrojs/check": "^0.9.0",
    "@playwright/test": "^1.53.0",
    "typescript": "^5.8.0",
    "vitest": "^3.2.0"
  }
}
```

- [ ] **Step 2: Run the build to verify it fails before Astro config and pages exist**

Run:

```bash
npm --prefix site install
test -f site/astro.config.mjs && test -f site/src/pages/index.astro && printf "unexpected-scaffold\n" || printf "missing-astro-scaffold\n"
```

Expected:

```text
missing-astro-scaffold
```

- [ ] **Step 3: Add the minimal Astro scaffold**

Create `site/astro.config.mjs`:

```javascript
import { defineConfig } from "astro/config";

const repo = "lm";

export default defineConfig({
  site: "https://fangpin.github.io",
  base: `/${repo}`,
  output: "static",
});
```

Create `site/tsconfig.json`:

```json
{
  "extends": "astro/tsconfigs/strict",
  "compilerOptions": {
    "baseUrl": "."
  }
}
```

Create `site/public/favicon.svg`:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="LLM">
  <rect width="64" height="64" rx="10" fill="#07101a"/>
  <path d="M14 16h8v24h12v8H14V16zm24 0h8v24h4v8H38V16z" fill="#7ceeff"/>
  <circle cx="48" cy="22" r="6" fill="#d8ff72"/>
</svg>
```

Append to `.gitignore`:

```gitignore
site/node_modules
site/dist
site/.astro
site/test-results
site/playwright-report
```

Create `site/src/pages/index.astro`:

```astro
---
---

<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>LLM from Scratch</title>
  </head>
  <body>
    <h1>LLM from Scratch</h1>
  </body>
</html>
```

- [ ] **Step 4: Run the build to verify it now passes with the minimal scaffold**

Run:

```bash
npm --prefix site run build
test -f site/dist/index.html && printf "built-index\n"
```

Expected:

```text
built-index
```

- [ ] **Step 5: Commit**

Run:

```bash
git add .gitignore site/package.json site/package-lock.json site/astro.config.mjs site/tsconfig.json site/public/favicon.svg site/src/pages/index.astro
git commit -m "feat: bootstrap astro project site"
git log -1 --pretty=%s
```

Expected:

```text
feat: bootstrap astro project site
```

## Task 3: Add Locale Metadata and Route-Parity Tests

**Files:**
- Create: `site/src/content/site.ts`
- Create: `site/src/lib/locale.ts`
- Create: `site/tests/routes.spec.ts`
- Create: `site/tests/content.spec.ts`
- Modify: `site/package.json`
- Test: `site/tests/routes.spec.ts`
- Test: `site/tests/content.spec.ts`

- [ ] **Step 1: Write the failing route-parity test**

Create `site/tests/routes.spec.ts`:

```ts
import { describe, expect, it } from "vitest";
import { localeRoutes, toBasePath } from "../src/lib/locale";

describe("locale route parity", () => {
  it("mirrors every english route with a chinese route", () => {
    expect(localeRoutes.en).toEqual([
      "/",
      "/architecture",
      "/sft-rlft",
      "/benchmarks",
      "/docs",
    ]);
    expect(localeRoutes.zh).toEqual([
      "/zh/",
      "/zh/architecture",
      "/zh/sft-rlft",
      "/zh/benchmarks",
      "/zh/docs",
    ]);
  });

  it("prefixes internal routes with the deployment base path", () => {
    expect(toBasePath("/", "/lm/")).toBe("/lm/");
    expect(toBasePath("/docs", "/lm/")).toBe("/lm/docs");
  });
});
```

Create `site/tests/content.spec.ts`:

```ts
import { describe, expect, it } from "vitest";
import { siteContent } from "../src/content/site";

describe("localized content", () => {
  it("has both english and chinese content for required pages", () => {
    expect(Object.keys(siteContent)).toEqual(["en", "zh"]);
    expect(siteContent.en.pages.home.title).toBeTruthy();
    expect(siteContent.zh.pages.home.title).toBeTruthy();
    expect(siteContent.en.navigation.items).toHaveLength(4);
    expect(siteContent.zh.navigation.items).toHaveLength(4);
  });
});
```

Update `site/package.json` scripts:

```json
{
  "scripts": {
    "dev": "astro dev",
    "build": "astro build",
    "preview": "astro preview",
    "test": "vitest run",
    "test:e2e": "playwright test"
  }
}
```

- [ ] **Step 2: Run the unit tests to verify they fail because the locale modules do not exist**

Run:

```bash
npm --prefix site run test && printf "unexpected-pass\n" || printf "locale-red\n"
```

Expected:

```text
locale-red
```

- [ ] **Step 3: Add the minimal locale metadata implementation**

Create `site/src/lib/locale.ts`:

```ts
export const localeRoutes = {
  en: ["/", "/architecture", "/sft-rlft", "/benchmarks", "/docs"],
  zh: ["/zh/", "/zh/architecture", "/zh/sft-rlft", "/zh/benchmarks", "/zh/docs"],
} as const;

export type Locale = keyof typeof localeRoutes;

export function getAlternatePath(locale: Locale, index: number) {
  return localeRoutes[locale][index];
}

export function toBasePath(path: string, base = "/") {
  if (/^https?:\/\//.test(path)) return path;
  const normalizedBase = base === "/" ? "" : base.replace(/\/$/, "");
  return path === "/" ? `${normalizedBase}/` || "/" : `${normalizedBase}${path}`;
}
```

Create `site/src/content/site.ts`:

```ts
export const siteContent = {
  en: {
    navigation: {
      items: [
        { label: "Architecture", href: "/architecture" },
        { label: "SFT & RLFT", href: "/sft-rlft" },
        { label: "Benchmarks", href: "/benchmarks" },
        { label: "Docs", href: "/docs" },
      ],
    },
    pages: {
      home: { title: "LLM from Scratch" },
    },
  },
  zh: {
    navigation: {
      items: [
        { label: "架构", href: "/zh/architecture" },
        { label: "SFT 与 RLFT", href: "/zh/sft-rlft" },
        { label: "基准", href: "/zh/benchmarks" },
        { label: "文档", href: "/zh/docs" },
      ],
    },
    pages: {
      home: { title: "从零开始的 LLM" },
    },
  },
} as const;
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
npm --prefix site run test
printf "locale-tests-green\n"
```

Expected:

```text
locale-tests-green
```

- [ ] **Step 5: Commit**

Run:

```bash
git add site/package.json site/src/content/site.ts site/src/lib/locale.ts site/tests/routes.spec.ts site/tests/content.spec.ts
git commit -m "test: add locale parity coverage"
git log -1 --pretty=%s
```

Expected:

```text
test: add locale parity coverage
```

## Task 4: Build the Shared Layout and Navigation Shell

**Files:**
- Create: `site/src/layouts/BaseLayout.astro`
- Create: `site/src/components/SiteHeader.astro`
- Create: `site/src/components/SiteFooter.astro`
- Create: `site/src/styles/global.css`
- Modify: `site/src/content/site.ts`
- Modify: `site/src/lib/locale.ts`
- Test: `site/tests/routes.spec.ts`

- [ ] **Step 1: Write the failing navigation-content test for locale switching**

Extend `site/tests/content.spec.ts` with:

```ts
it("defines language-switch labels and github/docs calls to action", () => {
  expect(siteContent.en.navigation.githubLabel).toBe("GitHub");
  expect(siteContent.zh.navigation.githubLabel).toBe("GitHub");
  expect(siteContent.en.navigation.localeSwitchLabel).toBe("中文");
  expect(siteContent.zh.navigation.localeSwitchLabel).toBe("EN");
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
npm --prefix site run test && printf "unexpected-pass\n" || printf "shell-red\n"
```

Expected:

```text
shell-red
```

- [ ] **Step 3: Implement the shared shell and required navigation fields**

Update `site/src/content/site.ts` so the navigation block becomes:

```ts
export const siteContent = {
  en: {
    navigation: {
      items: [
        { label: "Architecture", href: "/architecture" },
        { label: "SFT & RLFT", href: "/sft-rlft" },
        { label: "Benchmarks", href: "/benchmarks" },
        { label: "Docs", href: "/docs" },
      ],
      githubLabel: "GitHub",
      docsLabel: "Docs",
      localeSwitchLabel: "中文",
    },
    pages: {
      home: { title: "LLM from Scratch" },
    },
  },
  zh: {
    navigation: {
      items: [
        { label: "架构", href: "/zh/architecture" },
        { label: "SFT 与 RLFT", href: "/zh/sft-rlft" },
        { label: "基准", href: "/zh/benchmarks" },
        { label: "文档", href: "/zh/docs" },
      ],
      githubLabel: "GitHub",
      docsLabel: "文档",
      localeSwitchLabel: "EN",
    },
    pages: {
      home: { title: "从零开始的 LLM" },
    },
  },
} as const;
```

Update `site/src/lib/locale.ts`:

```ts
export function getAlternateLocale(locale: Locale): Locale {
  return locale === "en" ? "zh" : "en";
}

export function getAlternatePathFor(currentPath: string) {
  const routePairs = new Map([
    ["/", "/zh/"],
    ["/architecture", "/zh/architecture"],
    ["/sft-rlft", "/zh/sft-rlft"],
    ["/benchmarks", "/zh/benchmarks"],
    ["/docs", "/zh/docs"],
  ]);

  for (const [en, zh] of routePairs.entries()) {
    if (currentPath === en) return zh;
    if (currentPath === zh) return en;
  }

  throw new Error(`Missing alternate route for ${currentPath}`);
}
```

Create `site/src/styles/global.css`:

```css
:root {
  --bg: #06090f;
  --bg-elevated: rgba(12, 18, 31, 0.82);
  --panel: rgba(12, 20, 35, 0.7);
  --panel-strong: rgba(16, 26, 45, 0.92);
  --text: #f5f7fb;
  --muted: #97a8c4;
  --line: rgba(128, 180, 255, 0.18);
  --cyan: #7ceeff;
  --lime: #d8ff72;
  --orange: #ff9a62;
  --shadow: 0 20px 80px rgba(0, 0, 0, 0.45);
  --radius: 8px;
  --container: min(1180px, calc(100vw - 40px));
}

* {
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  margin: 0;
  min-width: 320px;
  background:
    radial-gradient(circle at top left, rgba(124, 238, 255, 0.12), transparent 30%),
    radial-gradient(circle at 80% 10%, rgba(216, 255, 114, 0.08), transparent 22%),
    linear-gradient(180deg, #07101a 0%, #04070d 100%);
  color: var(--text);
  font-family: "Segoe UI", "Helvetica Neue", sans-serif;
}

a {
  color: inherit;
  text-decoration: none;
}

img {
  display: block;
  max-width: 100%;
}

main {
  overflow: clip;
}

.container {
  width: var(--container);
  margin: 0 auto;
}

.site-header {
  position: sticky;
  top: 0;
  z-index: 20;
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 24px;
  align-items: center;
  width: var(--container);
  margin: 0 auto;
  padding: 18px 0;
  backdrop-filter: blur(18px);
}

.site-header nav {
  display: flex;
  gap: 18px;
  justify-content: center;
  flex-wrap: wrap;
}

.site-header .brand {
  font-size: 0.95rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.site-header .actions {
  display: flex;
  gap: 12px;
  align-items: center;
}

.button,
.site-header .actions a,
.portal-card,
.metric-card,
.docs-card {
  border: 1px solid var(--line);
  border-radius: var(--radius);
}

.button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  min-height: 46px;
  padding: 0 18px;
  background: var(--panel);
  transition: transform 180ms ease, border-color 180ms ease, background 180ms ease;
}

.button.primary {
  background: linear-gradient(135deg, rgba(124, 238, 255, 0.18), rgba(216, 255, 114, 0.14));
  border-color: rgba(216, 255, 114, 0.5);
}

.button:hover,
.portal-card:hover,
.metric-card:hover,
.docs-card:hover {
  transform: translateY(-2px);
  border-color: rgba(124, 238, 255, 0.45);
}

.section {
  padding: 56px 0;
}

.section-intro {
  width: min(760px, 100%);
  margin-bottom: 28px;
}

.section-intro .eyebrow {
  color: var(--lime);
  font-size: 0.85rem;
  text-transform: uppercase;
}

.section-intro h1,
.section-intro h2 {
  margin: 8px 0 12px;
  font-size: clamp(2rem, 6vw, 4.8rem);
  line-height: 0.98;
}

.section-intro p {
  margin: 0;
  color: var(--muted);
  font-size: 1.02rem;
  line-height: 1.6;
}

.site-footer {
  width: var(--container);
  margin: 0 auto;
  padding: 48px 0 72px;
  display: flex;
  justify-content: space-between;
  gap: 16px;
  color: var(--muted);
}

@media (max-width: 900px) {
  .site-header {
    grid-template-columns: 1fr;
    justify-items: start;
  }

  .site-header nav,
  .site-header .actions,
  .site-footer {
    justify-content: flex-start;
    flex-wrap: wrap;
  }
}
```

Create `site/src/components/SiteHeader.astro`:

```astro
---
import type { Locale } from "../lib/locale";
import { getAlternatePathFor } from "../lib/locale";
import { siteContent } from "../content/site";

interface Props {
  locale: Locale;
  currentPath: string;
}

const { locale, currentPath } = Astro.props;
const copy = siteContent[locale];
const alternateHref = getAlternatePathFor(currentPath);
---

<header class="site-header">
  <a class="brand" href={locale === "en" ? "/" : "/zh/"}>LLM from Scratch</a>
  <nav>
    {copy.navigation.items.map((item) => (
      <a href={item.href}>{item.label}</a>
    ))}
  </nav>
  <div class="actions">
    <a href={alternateHref}>{copy.navigation.localeSwitchLabel}</a>
    <a href="https://github.com/fangpin/lm" target="_blank" rel="noreferrer">{copy.navigation.githubLabel}</a>
  </div>
</header>
```

Create `site/src/components/SiteFooter.astro`:

```astro
---
import type { Locale } from "../lib/locale";

interface Props {
  locale: Locale;
}

const { locale } = Astro.props;
---

<footer class="site-footer">
  <p>{locale === "en" ? "Built from the repository itself." : "内容直接来自项目仓库。"}</p>
  <a href="https://github.com/fangpin/lm" target="_blank" rel="noreferrer">GitHub</a>
</footer>
```

Create `site/src/layouts/BaseLayout.astro`:

```astro
---
import SiteFooter from "../components/SiteFooter.astro";
import SiteHeader from "../components/SiteHeader.astro";
import "../styles/global.css";
import type { Locale } from "../lib/locale";

interface Props {
  locale: Locale;
  currentPath: string;
  title: string;
  description: string;
}

const { locale, currentPath, title, description } = Astro.props;
---

<!doctype html>
<html lang={locale === "en" ? "en" : "zh-CN"}>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <meta name="description" content={description} />
  </head>
  <body>
    <SiteHeader locale={locale} currentPath={currentPath} />
    <main>
      <slot />
    </main>
    <SiteFooter locale={locale} />
  </body>
</html>
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
npm --prefix site run test
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit**

Run:

```bash
git add site/src/layouts/BaseLayout.astro site/src/components/SiteHeader.astro site/src/components/SiteFooter.astro site/src/styles/global.css site/src/content/site.ts site/src/lib/locale.ts site/tests/content.spec.ts
git commit -m "feat: add bilingual site shell"
git log -1 --pretty=%s
```

Expected:

```text
feat: add bilingual site shell
```

## Task 5: Build the Homepage Experience in English and Chinese

**Files:**
- Create: `site/src/components/HeroBackground.astro`
- Create: `site/src/components/HeroSection.astro`
- Create: `site/src/components/ProofBand.astro`
- Create: `site/src/components/FeaturePortalGrid.astro`
- Modify: `site/src/content/site.ts`
- Modify: `site/src/pages/index.astro`
- Create: `site/src/pages/zh/index.astro`
- Test: `site/tests/content.spec.ts`

- [ ] **Step 1: Write the failing homepage-content test**

Extend `site/tests/content.spec.ts` with:

```ts
it("provides hero actions and proof metrics for both locales", () => {
  expect(siteContent.en.pages.home.actions).toHaveLength(2);
  expect(siteContent.zh.pages.home.actions).toHaveLength(2);
  expect(siteContent.en.pages.home.proofMetrics).toHaveLength(3);
  expect(siteContent.zh.pages.home.proofMetrics).toHaveLength(3);
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
npm --prefix site run test && printf "unexpected-pass\n" || printf "home-red\n"
```

Expected:

```text
home-red
```

- [ ] **Step 3: Implement the homepage sections**

Update `site/src/content/site.ts` so the `pages.home` blocks become:

```ts
pages: {
  home: {
    title: "LLM from Scratch",
    description: "A bilingual GitHub Pages project site for a from-scratch decoder-only transformer in PyTorch.",
    eyebrow: "Decoder-only Transformer • PyTorch • From Scratch",
    subtitle: "Build, train, benchmark, and fine-tune a modern language model from first principles.",
    actions: [
      { label: "View on GitHub", href: "https://github.com/fangpin/lm", variant: "primary" },
      { label: "Explore Docs", href: "/docs", variant: "secondary" }
    ],
    proofMetrics: [
      { value: "1.56% -> 62.9%", label: "Zero-shot accuracy on gsm8k" },
      { value: "18.9% -> 100%", label: "Format compliance" },
      { value: "Tokenizer / Training / Parallel / Kernel", label: "Implemented stack" }
    ],
    portals: [
      { title: "Architecture", href: "/architecture", summary: "Tokenizer, transformer blocks, training loop, kernels, and distributed training." },
      { title: "SFT & RLFT", href: "/sft-rlft", summary: "Qwen2.5-Math-1.5B fine-tuning story with concrete results on gsm8k." },
      { title: "Benchmarks", href: "/benchmarks", summary: "Loss curves, learning-rate schedule, and performance context." },
      { title: "Docs", href: "/docs", summary: "Curated entry points into quickstart, tokenizer, training, and fine-tuning docs." }
    ],
    capabilityTags: ["RoPE", "RMSNorm", "SwiGLU", "Flash Attention 2", "DDP", "SFT", "RLFT"]
  }
}
```

and for Chinese:

```ts
pages: {
  home: {
    title: "从零开始的 LLM",
    description: "一个为从零实现的 PyTorch decoder-only Transformer 打造的双语 GitHub Pages 项目站。",
    eyebrow: "Decoder-only Transformer • PyTorch • 从零实现",
    subtitle: "从分词器、训练、基准到微调，完整展示这个语言模型项目的核心实现。",
    actions: [
      { label: "查看 GitHub", href: "https://github.com/fangpin/lm", variant: "primary" },
      { label: "浏览文档", href: "/zh/docs", variant: "secondary" }
    ],
    proofMetrics: [
      { value: "1.56% -> 62.9%", label: "gsm8k 零样本准确率" },
      { value: "18.9% -> 100%", label: "输出格式遵循率" },
      { value: "Tokenizer / Training / Parallel / Kernel", label: "实现覆盖范围" }
    ],
    portals: [
      { title: "架构", href: "/zh/architecture", summary: "Tokenizer、Transformer blocks、训练循环、Kernel 与分布式训练。" },
      { title: "SFT 与 RLFT", href: "/zh/sft-rlft", summary: "Qwen2.5-Math-1.5B 在 gsm8k 上的微调与强化学习结果。" },
      { title: "基准", href: "/zh/benchmarks", summary: "Loss 曲线、学习率调度和性能说明。" },
      { title: "文档", href: "/zh/docs", summary: "快速开始、Tokenizer、训练与微调文档入口。" }
    ],
    capabilityTags: ["RoPE", "RMSNorm", "SwiGLU", "Flash Attention 2", "DDP", "SFT", "RLFT"]
  }
}
```

Create `site/src/components/HeroBackground.astro`:

```astro
<div class="hero-background" aria-hidden="true">
  <div class="hero-grid"></div>
  <div class="hero-orbit hero-orbit-cyan"></div>
  <div class="hero-orbit hero-orbit-lime"></div>
  <div class="hero-stream">
    <span></span><span></span><span></span><span></span><span></span><span></span>
  </div>
</div>
```

Append these rules to `site/src/styles/global.css`:

```css
.hero {
  position: relative;
  padding: 120px 0 64px;
}

.hero-background {
  position: absolute;
  inset: 0;
  pointer-events: none;
  overflow: hidden;
}

.hero-grid {
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(124, 238, 255, 0.08) 1px, transparent 1px),
    linear-gradient(90deg, rgba(124, 238, 255, 0.08) 1px, transparent 1px);
  background-size: 80px 80px;
  mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.95), transparent 88%);
}

.hero-orbit,
.hero-stream span {
  position: absolute;
  border-radius: 999px;
  filter: blur(0.5px);
}

.hero-orbit {
  border: 1px solid rgba(124, 238, 255, 0.18);
  animation: drift 12s linear infinite;
}

.hero-orbit-cyan {
  width: 420px;
  height: 420px;
  right: -120px;
  top: -40px;
}

.hero-orbit-lime {
  width: 280px;
  height: 280px;
  left: -60px;
  bottom: 20px;
  border-color: rgba(216, 255, 114, 0.22);
  animation-duration: 9s;
}

.hero-stream span {
  width: 10px;
  height: 10px;
  background: var(--cyan);
  box-shadow: 0 0 20px rgba(124, 238, 255, 0.5);
  animation: stream 7s linear infinite;
}

.hero-stream span:nth-child(1) { left: 14%; top: 32%; animation-delay: 0s; }
.hero-stream span:nth-child(2) { left: 26%; top: 16%; animation-delay: 1s; }
.hero-stream span:nth-child(3) { left: 44%; top: 48%; animation-delay: 2s; }
.hero-stream span:nth-child(4) { left: 63%; top: 24%; animation-delay: 3s; }
.hero-stream span:nth-child(5) { left: 76%; top: 58%; animation-delay: 4s; }
.hero-stream span:nth-child(6) { left: 86%; top: 18%; animation-delay: 5s; }

.hero-shell {
  position: relative;
  width: var(--container);
  margin: 0 auto;
  display: grid;
  gap: 28px;
}

.hero-copy {
  width: min(780px, 100%);
}

.hero-copy .eyebrow {
  color: var(--lime);
  font-size: 0.9rem;
  text-transform: uppercase;
}

.hero-copy h1 {
  margin: 10px 0 14px;
  font-size: clamp(3.2rem, 11vw, 7.6rem);
  line-height: 0.92;
}

.hero-copy p {
  margin: 0;
  max-width: 56ch;
  color: var(--muted);
  font-size: 1.08rem;
  line-height: 1.7;
}

.hero-actions,
.capability-tags,
.portal-grid,
.metric-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
}

.capability-tag {
  padding: 8px 12px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: rgba(12, 19, 31, 0.65);
  color: var(--muted);
  font-size: 0.9rem;
}

.proof-band {
  width: var(--container);
  margin: 0 auto;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
}

.portal-grid {
  width: var(--container);
  margin: 0 auto;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.portal-card,
.metric-card,
.docs-card {
  background: linear-gradient(180deg, rgba(15, 24, 40, 0.95), rgba(8, 12, 20, 0.88));
  box-shadow: var(--shadow);
  padding: 22px;
}

.portal-card h3,
.metric-card strong,
.docs-card h3 {
  display: block;
  margin-bottom: 10px;
}

@keyframes drift {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@keyframes stream {
  0% { transform: translate3d(-20px, 10px, 0) scale(0.8); opacity: 0; }
  20% { opacity: 1; }
  100% { transform: translate3d(120px, -80px, 0) scale(1.15); opacity: 0; }
}

@media (max-width: 900px) {
  .proof-band,
  .portal-grid {
    grid-template-columns: 1fr;
  }

  .hero {
    padding-top: 96px;
  }
}
```

Create `site/src/components/HeroSection.astro`:

```astro
---
import HeroBackground from "./HeroBackground.astro";

interface Props {
  locale: "en" | "zh";
  page: {
    title: string;
    eyebrow: string;
    subtitle: string;
    actions: { label: string; href: string; variant: string }[];
    capabilityTags: string[];
  };
}

const { page } = Astro.props;
---

<section class="hero">
  <HeroBackground />
  <div class="hero-shell">
    <div class="hero-copy">
      <div class="eyebrow">{page.eyebrow}</div>
      <h1>{page.title}</h1>
      <p>{page.subtitle}</p>
    </div>
    <div class="hero-actions">
      {page.actions.map((action) => (
        <a class={`button ${action.variant}`} href={action.href}>{action.label}</a>
      ))}
    </div>
    <div class="capability-tags">
      {page.capabilityTags.map((tag) => (
        <span class="capability-tag">{tag}</span>
      ))}
    </div>
  </div>
</section>
```

Create `site/src/components/ProofBand.astro`:

```astro
---
interface Props {
  metrics: { value: string; label: string }[];
}

const { metrics } = Astro.props;
---

<section class="section">
  <div class="proof-band">
    {metrics.map((metric) => (
      <article class="metric-card">
        <strong>{metric.value}</strong>
        <span>{metric.label}</span>
      </article>
    ))}
  </div>
</section>
```

Create `site/src/components/FeaturePortalGrid.astro`:

```astro
---
interface Props {
  portals: { title: string; href: string; summary: string }[];
}

const { portals } = Astro.props;
---

<section class="section">
  <div class="portal-grid">
    {portals.map((portal) => (
      <a class="portal-card" href={portal.href}>
        <h3>{portal.title}</h3>
        <p>{portal.summary}</p>
      </a>
    ))}
  </div>
</section>
```

Replace `site/src/pages/index.astro` with:

```astro
---
import BaseLayout from "../layouts/BaseLayout.astro";
import FeaturePortalGrid from "../components/FeaturePortalGrid.astro";
import HeroSection from "../components/HeroSection.astro";
import ProofBand from "../components/ProofBand.astro";
import { siteContent } from "../content/site";

const copy = siteContent.en.pages.home;
---

<BaseLayout locale="en" currentPath="/" title={copy.title} description={copy.description}>
  <HeroSection locale="en" page={copy} />
  <ProofBand metrics={copy.proofMetrics} />
  <FeaturePortalGrid portals={copy.portals} />
</BaseLayout>
```

Create `site/src/pages/zh/index.astro`:

```astro
---
import BaseLayout from "../../layouts/BaseLayout.astro";
import FeaturePortalGrid from "../../components/FeaturePortalGrid.astro";
import HeroSection from "../../components/HeroSection.astro";
import ProofBand from "../../components/ProofBand.astro";
import { siteContent } from "../../content/site";

const copy = siteContent.zh.pages.home;
---

<BaseLayout locale="zh" currentPath="/zh/" title={copy.title} description={copy.description}>
  <HeroSection locale="zh" page={copy} />
  <ProofBand metrics={copy.proofMetrics} />
  <FeaturePortalGrid portals={copy.portals} />
</BaseLayout>
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
npm --prefix site run test
```

Expected:

```text
4 passed
```

- [ ] **Step 5: Commit**

Run:

```bash
git add site/src/components/HeroBackground.astro site/src/components/HeroSection.astro site/src/components/ProofBand.astro site/src/components/FeaturePortalGrid.astro site/src/content/site.ts site/src/pages/index.astro site/src/pages/zh/index.astro site/tests/content.spec.ts
git commit -m "feat: add bilingual homepage experience"
git log -1 --pretty=%s
```

Expected:

```text
feat: add bilingual homepage experience
```

## Task 6: Add the Remaining Content Pages

**Files:**
- Create: `site/src/components/SectionIntro.astro`
- Create: `site/src/components/ArchitectureFlow.astro`
- Create: `site/src/components/MetricCards.astro`
- Modify: `site/src/content/site.ts`
- Create: `site/src/pages/architecture.astro`
- Create: `site/src/pages/sft-rlft.astro`
- Create: `site/src/pages/benchmarks.astro`
- Create: `site/src/pages/docs/index.astro`
- Create: `site/src/pages/zh/architecture.astro`
- Create: `site/src/pages/zh/sft-rlft.astro`
- Create: `site/src/pages/zh/benchmarks.astro`
- Create: `site/src/pages/zh/docs/index.astro`
- Create: `site/public/images/loss.png`
- Create: `site/public/images/lr.png`
- Test: `site/tests/content.spec.ts`

- [ ] **Step 1: Write the failing page-data test**

Extend `site/tests/content.spec.ts` with:

```ts
it("defines all secondary pages in both locales", () => {
  const expectedPages = ["home", "architecture", "sftRlft", "benchmarks", "docs"];
  expect(Object.keys(siteContent.en.pages)).toEqual(expectedPages);
  expect(Object.keys(siteContent.zh.pages)).toEqual(expectedPages);
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
npm --prefix site run test && printf "unexpected-pass\n" || printf "page-red\n"
```

Expected:

```text
page-red
```

- [ ] **Step 3: Implement localized page content and page files**

Update `site/src/content/site.ts` so both locales contain these page blocks:

```ts
architecture: {
  title: "Architecture",
  description: "See how tokenizer, transformer blocks, training, kernels, and distributed execution fit together.",
  sections: [
    { title: "Tokenizer", summary: "A from-scratch BPE tokenizer that learns merges and special tokens from raw text." },
    { title: "Transformer Core", summary: "Decoder-only blocks with RMSNorm, RoPE, SwiGLU, custom attention, and custom loss." },
    { title: "Training Loop", summary: "Checkpointing, validation, generation, cosine scheduling, and optimizer utilities." },
    { title: "Parallel + Kernel", summary: "Custom DDP, sharded optimizer, and Triton Flash Attention 2 support." },
    { title: "Data Pipeline", summary: "HTML extraction, filtering, deduplication, masking, and quality tooling." }
  ]
},
sftRlft: {
  title: "SFT & RLFT",
  description: "The strongest proof page in the repo: Qwen2.5-Math-1.5B on gsm8k with measured gains.",
  metrics: [
    { value: "1.56%", label: "Zero-shot baseline" },
    { value: "62.9%", label: "After SFT" },
    { value: "100%", label: "Format compliance" }
  ],
  steps: [
    "Load gsm8k train and test sets.",
    "Evaluate Qwen2.5-Math-1.5B zero-shot behavior.",
    "Run supervised fine-tuning for answer-format alignment.",
    "Run reinforcement fine-tuning to improve reward-shaped reasoning behavior."
  ]
},
benchmarks: {
  title: "Benchmarks",
  description: "Training curves and implementation notes that ground the project in measurable behavior.",
  cards: [
    { value: "Flash Attention 2", label: "Triton kernel path included" },
    { value: "Loss Curve", label: "TinyStories training convergence" },
    { value: "LR Schedule", label: "Cosine warmup and decay" }
  ],
  charts: [
    { src: "/images/loss.png", alt: "Loss curve" },
    { src: "/images/lr.png", alt: "Learning rate schedule" }
  ]
},
docs: {
  title: "Docs",
  description: "Curated entry points for learning the repository without forcing a full docs migration in v1.",
  cards: [
    { title: "Quickstart", href: "https://github.com/fangpin/lm#usage", summary: "Data preparation, tokenizer training, model training, and generation." },
    { title: "Training", href: "https://github.com/fangpin/lm#training", summary: "Loss curve, learning-rate schedule, and example outputs." },
    { title: "Tokenizer", href: "https://github.com/fangpin/lm#tokenizer-llmbpe_tokenizerpy", summary: "How the BPE tokenizer is implemented and trained." },
    { title: "Data Processing", href: "https://github.com/fangpin/lm#data-processing-data_processing", summary: "Cleaning, filtering, deduplication, and harmful-content detection." },
    { title: "Fine-tuning", href: "https://github.com/fangpin/lm/blob/master/docs/qwen25-math-gsm8k-rl-finetune.md", summary: "Detailed notes for the Qwen2.5-Math-1.5B gsm8k workflow." }
  ]
}
```

Use these Chinese peers:

```ts
architecture: {
  title: "架构",
  description: "从 tokenizer、transformer blocks 到训练、kernel 与分布式执行，整体说明这个项目是如何组成的。",
  sections: [
    { title: "Tokenizer", summary: "从零实现的 BPE tokenizer，可从原始文本中学习 merge 规则和特殊 token。" },
    { title: "Transformer Core", summary: "带有 RMSNorm、RoPE、SwiGLU、自定义 attention 与 loss 的 decoder-only blocks。" },
    { title: "Training Loop", summary: "包含 checkpoint、validation、generation、cosine scheduler 与优化器工具。" },
    { title: "Parallel + Kernel", summary: "自定义 DDP、sharded optimizer，以及 Triton Flash Attention 2 支持。" },
    { title: "Data Pipeline", summary: "HTML 抽取、过滤、去重、PII masking 和质量筛选工具。" }
  ]
},
sftRlft: {
  title: "SFT 与 RLFT",
  description: "仓库里最强的结果展示页：Qwen2.5-Math-1.5B 在 gsm8k 上的微调与强化学习结果。",
  metrics: [
    { value: "1.56%", label: "零样本基线" },
    { value: "62.9%", label: "SFT 后准确率" },
    { value: "100%", label: "格式遵循率" }
  ],
  steps: [
    "准备 gsm8k 训练集与测试集。",
    "评估 Qwen2.5-Math-1.5B 的零样本表现。",
    "运行监督微调以稳定答案格式。",
    "运行强化学习微调以提升奖励驱动的推理表现。"
  ]
},
benchmarks: {
  title: "基准",
  description: "用训练曲线和实现说明把这个项目落到可观测的性能与训练行为上。",
  cards: [
    { value: "Flash Attention 2", label: "包含 Triton kernel 路径" },
    { value: "Loss Curve", label: "TinyStories 训练收敛过程" },
    { value: "LR Schedule", label: "Cosine warmup 与 decay" }
  ],
  charts: [
    { src: "/images/loss.png", alt: "损失曲线" },
    { src: "/images/lr.png", alt: "学习率调度" }
  ]
},
docs: {
  title: "文档",
  description: "第一版只做精选入口，不强行把整个仓库的 markdown 一次性迁成站内文档。",
  cards: [
    { title: "快速开始", href: "https://github.com/fangpin/lm#usage", summary: "准备数据、训练 tokenizer、训练模型和生成文本。" },
    { title: "训练", href: "https://github.com/fangpin/lm#training", summary: "Loss 曲线、学习率计划与示例输出。" },
    { title: "Tokenizer", href: "https://github.com/fangpin/lm#tokenizer-llmbpe_tokenizerpy", summary: "BPE tokenizer 的实现与训练方式。" },
    { title: "数据处理", href: "https://github.com/fangpin/lm#data-processing-data_processing", summary: "清洗、过滤、去重和有害内容检测。" },
    { title: "微调", href: "https://github.com/fangpin/lm/blob/master/docs/qwen25-math-gsm8k-rl-finetune.md", summary: "Qwen2.5-Math-1.5B 在 gsm8k 上的详细实验记录。" }
  ]
}
```

Create `site/src/components/SectionIntro.astro`:

```astro
---
interface Props {
  eyebrow: string;
  title: string;
  description: string;
}

const { eyebrow, title, description } = Astro.props;
---

<div class="container section-intro">
  <div class="eyebrow">{eyebrow}</div>
  <h1>{title}</h1>
  <p>{description}</p>
</div>
```

Create `site/src/components/ArchitectureFlow.astro`:

```astro
---
interface Props {
  sections: { title: string; summary: string }[];
}

const { sections } = Astro.props;
---

<section class="section">
  <div class="container portal-grid">
    {sections.map((section, index) => (
      <article class="portal-card">
        <strong>{String(index + 1).padStart(2, "0")}</strong>
        <h3>{section.title}</h3>
        <p>{section.summary}</p>
      </article>
    ))}
  </div>
</section>
```

Create `site/src/components/MetricCards.astro`:

```astro
---
interface Props {
  cards: { value: string; label: string }[];
}

const { cards } = Astro.props;
---

<section class="section">
  <div class="container metric-grid">
    {cards.map((card) => (
      <article class="metric-card">
        <strong>{card.value}</strong>
        <span>{card.label}</span>
      </article>
    ))}
  </div>
</section>
```

Create `site/src/pages/architecture.astro`:

```astro
---
import ArchitectureFlow from "../components/ArchitectureFlow.astro";
import BaseLayout from "../layouts/BaseLayout.astro";
import SectionIntro from "../components/SectionIntro.astro";
import { siteContent } from "../content/site";

const copy = siteContent.en.pages.architecture;
---

<BaseLayout locale="en" currentPath="/architecture" title={copy.title} description={copy.description}>
  <section class="section">
    <SectionIntro eyebrow="Implemented System" title={copy.title} description={copy.description} />
    <ArchitectureFlow sections={copy.sections} />
  </section>
</BaseLayout>
```

Create `site/src/pages/sft-rlft.astro`:

```astro
---
import BaseLayout from "../layouts/BaseLayout.astro";
import MetricCards from "../components/MetricCards.astro";
import SectionIntro from "../components/SectionIntro.astro";
import { siteContent } from "../content/site";

const copy = siteContent.en.pages.sftRlft;
---

<BaseLayout locale="en" currentPath="/sft-rlft" title={copy.title} description={copy.description}>
  <section class="section">
    <SectionIntro eyebrow="Applied Results" title={copy.title} description={copy.description} />
    <MetricCards cards={copy.metrics} />
    <div class="container">
      <ol>
        {copy.steps.map((step) => <li>{step}</li>)}
      </ol>
    </div>
  </section>
</BaseLayout>
```

Create `site/src/pages/benchmarks.astro`:

```astro
---
import BaseLayout from "../layouts/BaseLayout.astro";
import MetricCards from "../components/MetricCards.astro";
import SectionIntro from "../components/SectionIntro.astro";
import { siteContent } from "../content/site";

const copy = siteContent.en.pages.benchmarks;
---

<BaseLayout locale="en" currentPath="/benchmarks" title={copy.title} description={copy.description}>
  <section class="section">
    <SectionIntro eyebrow="Measured Behavior" title={copy.title} description={copy.description} />
    <MetricCards cards={copy.cards} />
    <div class="container portal-grid">
      {copy.charts.map((chart) => (
        <article class="portal-card">
          <img src={chart.src} alt={chart.alt} loading="lazy" />
        </article>
      ))}
    </div>
  </section>
</BaseLayout>
```

Create `site/src/pages/docs/index.astro`:

```astro
---
import BaseLayout from "../../layouts/BaseLayout.astro";
import SectionIntro from "../../components/SectionIntro.astro";
import { siteContent } from "../../content/site";

const copy = siteContent.en.pages.docs;
---

<BaseLayout locale="en" currentPath="/docs" title={copy.title} description={copy.description}>
  <section class="section">
    <SectionIntro eyebrow="Curated Entry Points" title={copy.title} description={copy.description} />
    <div class="container portal-grid">
      {copy.cards.map((card) => (
        <a class="docs-card" href={card.href} target="_blank" rel="noreferrer">
          <h3>{card.title}</h3>
          <p>{card.summary}</p>
        </a>
      ))}
    </div>
  </section>
</BaseLayout>
```

Create `site/src/pages/zh/architecture.astro`:

```astro
---
import ArchitectureFlow from "../../components/ArchitectureFlow.astro";
import BaseLayout from "../../layouts/BaseLayout.astro";
import SectionIntro from "../../components/SectionIntro.astro";
import { siteContent } from "../../content/site";

const copy = siteContent.zh.pages.architecture;
---

<BaseLayout locale="zh" currentPath="/zh/architecture" title={copy.title} description={copy.description}>
  <section class="section">
    <SectionIntro eyebrow="实现结构" title={copy.title} description={copy.description} />
    <ArchitectureFlow sections={copy.sections} />
  </section>
</BaseLayout>
```

Create `site/src/pages/zh/sft-rlft.astro`:

```astro
---
import BaseLayout from "../../layouts/BaseLayout.astro";
import MetricCards from "../../components/MetricCards.astro";
import SectionIntro from "../../components/SectionIntro.astro";
import { siteContent } from "../../content/site";

const copy = siteContent.zh.pages.sftRlft;
---

<BaseLayout locale="zh" currentPath="/zh/sft-rlft" title={copy.title} description={copy.description}>
  <section class="section">
    <SectionIntro eyebrow="结果展示" title={copy.title} description={copy.description} />
    <MetricCards cards={copy.metrics} />
    <div class="container">
      <ol>
        {copy.steps.map((step) => <li>{step}</li>)}
      </ol>
    </div>
  </section>
</BaseLayout>
```

Create `site/src/pages/zh/benchmarks.astro`:

```astro
---
import BaseLayout from "../../layouts/BaseLayout.astro";
import MetricCards from "../../components/MetricCards.astro";
import SectionIntro from "../../components/SectionIntro.astro";
import { siteContent } from "../../content/site";

const copy = siteContent.zh.pages.benchmarks;
---

<BaseLayout locale="zh" currentPath="/zh/benchmarks" title={copy.title} description={copy.description}>
  <section class="section">
    <SectionIntro eyebrow="训练与性能" title={copy.title} description={copy.description} />
    <MetricCards cards={copy.cards} />
    <div class="container portal-grid">
      {copy.charts.map((chart) => (
        <article class="portal-card">
          <img src={chart.src} alt={chart.alt} loading="lazy" />
        </article>
      ))}
    </div>
  </section>
</BaseLayout>
```

Create `site/src/pages/zh/docs/index.astro`:

```astro
---
import BaseLayout from "../../../layouts/BaseLayout.astro";
import SectionIntro from "../../../components/SectionIntro.astro";
import { siteContent } from "../../../content/site";

const copy = siteContent.zh.pages.docs;
---

<BaseLayout locale="zh" currentPath="/zh/docs" title={copy.title} description={copy.description}>
  <section class="section">
    <SectionIntro eyebrow="精选入口" title={copy.title} description={copy.description} />
    <div class="container portal-grid">
      {copy.cards.map((card) => (
        <a class="docs-card" href={card.href} target="_blank" rel="noreferrer">
          <h3>{card.title}</h3>
          <p>{card.summary}</p>
        </a>
      ))}
    </div>
  </section>
</BaseLayout>
```

Copy the existing repo charts:

```bash
cp img/loss.png site/public/images/loss.png
cp img/lr.png site/public/images/lr.png
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
npm --prefix site run test
```

Expected:

```text
5 passed
```

- [ ] **Step 5: Commit**

Run:

```bash
git add site/src/components/SectionIntro.astro site/src/components/ArchitectureFlow.astro site/src/components/MetricCards.astro site/src/content/site.ts site/src/pages/architecture.astro site/src/pages/sft-rlft.astro site/src/pages/benchmarks.astro site/src/pages/docs/index.astro site/src/pages/zh/architecture.astro site/src/pages/zh/sft-rlft.astro site/src/pages/zh/benchmarks.astro site/src/pages/zh/docs/index.astro site/public/images/loss.png site/public/images/lr.png site/tests/content.spec.ts
git commit -m "feat: add project site pages"
git log -1 --pretty=%s
```

Expected:

```text
feat: add project site pages
```

## Task 7: Add GitHub Pages Deployment and Project Documentation

**Files:**
- Create: `.github/workflows/deploy-site.yml`
- Create: `docs/site.md`
- Modify: `README.md`
- Modify: `README_cn.md`
- Test: `.github/workflows/deploy-site.yml`

- [ ] **Step 1: Write the failing workflow-validation check**

Run:

```bash
test -f .github/workflows/deploy-site.yml || printf "missing-workflow\n"
```

Expected:

```text
missing-workflow
```

- [ ] **Step 2: Create the deployment workflow and docs**

Create `.github/workflows/deploy-site.yml`:

```yaml
name: Deploy Project Site

on:
  push:
    branches: ["master"]
    paths:
      - "site/**"
      - ".github/workflows/deploy-site.yml"
      - "docs/site.md"
      - "README.md"
      - "README_cn.md"
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
          cache: npm
          cache-dependency-path: site/package-lock.json
      - name: Install dependencies
        run: npm ci
        working-directory: site
      - name: Build site
        run: npm run build
        working-directory: site
      - name: Configure Pages
        uses: actions/configure-pages@v5
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: site/dist

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

Create `docs/site.md` describing:

```md
# Project Site

## Local development

```bash
npm --prefix site install
npm --prefix site run dev
```

## Production build

```bash
npm --prefix site run build
```

## GitHub Pages

Enable Pages in the repository settings and select GitHub Actions as the source.
```

Append to `README.md`:

```md
## Project Site

The repository includes a bilingual Astro project site for GitHub Pages in `site/`.

```bash
npm --prefix site install
npm --prefix site run dev
```
```

Append the Chinese peer text to `README_cn.md`.

- [ ] **Step 3: Run a syntax check for the workflow and docs presence**

Run:

```bash
test -f .github/workflows/deploy-site.yml
test -f docs/site.md
rg -n "Project Site|项目主页" README.md README_cn.md docs/site.md
printf "workflow-docs-green\n"
```

Expected:

```text
workflow-docs-green
```

- [ ] **Step 4: Commit**

Run:

```bash
git add .github/workflows/deploy-site.yml docs/site.md README.md README_cn.md
git commit -m "chore: add pages deployment workflow"
git log -1 --pretty=%s
```

Expected:

```text
chore: add pages deployment workflow
```

## Task 8: Add End-to-End Verification and Run the Full Verification Suite

**Files:**
- Create: `site/playwright.config.ts`
- Create: `site/tests/e2e/site.spec.ts`
- Test: `site/tests/e2e/site.spec.ts`

- [ ] **Step 1: Write the failing end-to-end test**

Create `site/playwright.config.ts`:

```ts
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  use: {
    baseURL: "http://127.0.0.1:4321",
  },
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 4321",
    port: 4321,
    cwd: ".",
    reuseExistingServer: false,
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile", use: { ...devices["iPhone 13"] } },
  ],
});
```

Create `site/tests/e2e/site.spec.ts`:

```ts
import { expect, test } from "@playwright/test";

test("english home renders headline and language switch", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "LLM from Scratch" })).toBeVisible();
  await expect(page.getByRole("link", { name: "中文" })).toBeVisible();
});

test("mobile chinese docs route renders without overlapping primary navigation", async ({ page }) => {
  await page.goto("/zh/docs");
  await expect(page.getByRole("heading", { name: /文档|Docs/ })).toBeVisible();
  const header = page.locator(".site-header");
  await expect(header).toBeVisible();
});
```

- [ ] **Step 2: Run the end-to-end tests to verify they fail before the remaining pages are complete**

Run:

```bash
npm --prefix site run test:e2e && printf "unexpected-pass\n" || printf "e2e-red\n"
```

Expected:

```text
e2e-red
```

because the locale pages, final navigation, or headings are not fully implemented yet.

- [ ] **Step 3: Finish any missing page semantics and run the full suite**

Ensure the final page files expose:

```astro
<h1>LLM from Scratch</h1>
```

on the English home page and a visible docs heading on `/zh/docs`.

Run:

```bash
npm --prefix site run test
npm --prefix site run build
npm --prefix site run test:e2e
printf "verification-green\n"
```

Expected:

```text
verification-green
```

for unit tests, successful Astro build output, and Playwright passing on desktop and mobile.

- [ ] **Step 4: Commit**

Run:

```bash
git add site/playwright.config.ts site/tests/e2e/site.spec.ts
git commit -m "test: add site verification coverage"
git log -1 --pretty=%s
```

Expected:

```text
test: add site verification coverage
```

## Task 9: Final Review and Delivery

**Files:**
- Modify: any files required to fix verification gaps discovered in Task 8

- [ ] **Step 1: Review the implementation against the spec**

Run:

```bash
git diff --stat master...HEAD
printf "final-diff-reviewed\n"
```

Expected:

```text
final-diff-reviewed
```

and no unrelated Python training files should appear in the diff.

- [ ] **Step 2: Inspect the working tree for unexpected changes**

Run:

```bash
git status --short
```

Expected:

```text
```

or only intentional generated verification artifacts already ignored by `.gitignore`.

- [ ] **Step 3: Commit any last fixes**

Run:

```bash
git add -A
git commit -m "fix: polish project site delivery"
git log -1 --pretty=%s
```

Expected:

```text
fix: polish project site delivery
```

Only do this if Task 8 produced real code changes.
