export const localeRoutes = {
  en: ["/", "/architecture", "/sft-rlft", "/benchmarks", "/docs"],
  zh: ["/zh/", "/zh/architecture", "/zh/sft-rlft", "/zh/benchmarks", "/zh/docs"],
} as const;

export type Locale = keyof typeof localeRoutes;

export function getAlternatePath(locale: Locale, index: number) {
  return localeRoutes[locale][index];
}

export function getAlternateLocale(locale: Locale): Locale {
  return locale === "en" ? "zh" : "en";
}

export function getAlternatePathFor(currentPath: string) {
  if (currentPath === "/docs") return "/zh/docs";
  if (currentPath === "/zh/docs") return "/docs";
  if (currentPath.startsWith("/docs/")) return `/zh${currentPath}`;
  if (currentPath.startsWith("/zh/docs/")) return currentPath.replace("/zh", "");

  const routePairs = new Map<string, string>([
    ["/", "/zh/"],
    ["/architecture", "/zh/architecture"],
    ["/sft-rlft", "/zh/sft-rlft"],
    ["/benchmarks", "/zh/benchmarks"],
  ]);

  for (const [en, zh] of routePairs.entries()) {
    if (currentPath === en) return zh;
    if (currentPath === zh) return en;
  }

  throw new Error(`Missing alternate route for ${currentPath}`);
}

export function toBasePath(path: string, base = "/") {
  if (/^https?:\/\//.test(path)) return path;
  const normalizedBase = base === "/" ? "" : base.replace(/\/$/, "");
  return path === "/" ? `${normalizedBase}/` || "/" : `${normalizedBase}${path}`;
}

export const SITE_BASE_PATH = "/llm-from-scratch/";
