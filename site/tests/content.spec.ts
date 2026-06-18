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

  it("defines language-switch labels and github/docs calls to action", () => {
    expect(siteContent.en.navigation.githubLabel).toBe("GitHub");
    expect(siteContent.zh.navigation.githubLabel).toBe("GitHub");
    expect(siteContent.en.navigation.localeSwitchLabel).toBe("中文");
    expect(siteContent.zh.navigation.localeSwitchLabel).toBe("EN");
  });

  it("provides hero actions and proof metrics for both locales", () => {
    expect(siteContent.en.pages.home.actions).toHaveLength(2);
    expect(siteContent.zh.pages.home.actions).toHaveLength(2);
    expect(siteContent.en.pages.home.proofMetrics).toHaveLength(3);
    expect(siteContent.zh.pages.home.proofMetrics).toHaveLength(3);
  });

  it("defines all secondary pages in both locales", () => {
    const expectedPages = ["home", "architecture", "sftRlft", "benchmarks", "docs"];
    expect(Object.keys(siteContent.en.pages)).toEqual(expectedPages);
    expect(Object.keys(siteContent.zh.pages)).toEqual(expectedPages);
  });

  it("provides localized docs overview copy and group labels", () => {
    expect(siteContent.en.pages.docs.systemMap).toHaveLength(5);
    expect(siteContent.zh.pages.docs.systemMap).toHaveLength(5);
    expect(siteContent.en.pages.docs.groups["core-stack"].title).toBe("Core Stack");
    expect(siteContent.zh.pages.docs.groups["alignment-workflows"].title).toBe("对齐流程");
  });
});
