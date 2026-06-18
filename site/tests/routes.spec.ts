import { describe, expect, it } from "vitest";
import { getAlternatePathFor, localeRoutes, toBasePath } from "../src/lib/locale";

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
    expect(toBasePath("/", "/llm-from-scratch/")).toBe("/llm-from-scratch/");
    expect(toBasePath("/docs", "/llm-from-scratch/")).toBe("/llm-from-scratch/docs");
  });

  it("maps mirrored routes in both directions", () => {
    expect(getAlternatePathFor("/docs")).toBe("/zh/docs");
    expect(getAlternatePathFor("/zh/docs")).toBe("/docs");
    expect(getAlternatePathFor("/docs/transformer-core")).toBe("/zh/docs/transformer-core");
    expect(getAlternatePathFor("/zh/docs/transformer-core")).toBe("/docs/transformer-core");
  });
});
