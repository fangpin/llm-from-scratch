import { describe, expect, it } from "vitest";
import {
  docGroupOrder,
  getDoc,
  getDocOverview,
  getDocs,
  getDocsByGroup,
  getRelatedDocs,
} from "../src/lib/docs";

describe("chaptered docs", () => {
  it("loads nine mirrored chapters in each locale", () => {
    const english = getDocs("en");
    const chinese = getDocs("zh");

    expect(english).toHaveLength(9);
    expect(chinese).toHaveLength(9);
    expect(english.map((entry) => entry.slug)).toEqual(chinese.map((entry) => entry.slug));
  });

  it("loads overview markdown for both locales", () => {
    expect(getDocOverview("en")?.slug).toBe("index");
    expect(getDocOverview("zh")?.slug).toBe("index");
  });

  it("groups chapters into the three overview bands", () => {
    const groups = getDocsByGroup("en");
    expect(groups.map((group) => group.group)).toEqual(docGroupOrder);
    expect(groups[0].entries).toHaveLength(4);
    expect(groups[1].entries).toHaveLength(3);
    expect(groups[2].entries).toHaveLength(2);
  });

  it("exposes source files and related chapters for the transformer core chapter", () => {
    const entry = getDoc("en", "transformer-core");
    expect(entry?.sourceFiles).toContain("llm/transformer.py");
    expect(getRelatedDocs("en", "transformer-core").length).toBeGreaterThan(0);
  });
});
