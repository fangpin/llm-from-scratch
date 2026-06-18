import { expect, test } from "@playwright/test";

test("english home renders headline and language switch", async ({ page }) => {
  await page.goto("/lm/");
  await expect(page.getByRole("heading", { name: "LLM from Scratch" })).toBeVisible();
  await expect(page.getByRole("link", { name: "中文" })).toBeVisible();
});

test("english docs overview renders grouped internal chapter cards", async ({ page }) => {
  await page.goto("/lm/docs");
  await expect(page.getByRole("heading", { name: "Docs" })).toBeVisible();
  await expect(page.getByRole("link", { name: /Transformer Core/i })).toBeVisible();
  await expect(page.getByRole("link", { name: /Distributed Training/i })).toBeVisible();
});

test("docs chapter locale switch keeps the same chapter slug", async ({ page }) => {
  await page.goto("/lm/docs/distributed-training");
  await expect(page.getByRole("heading", { name: "Distributed Training and Sharded Optimizer" })).toBeVisible();
  await page.getByRole("link", { name: "中文" }).click();
  await expect(page).toHaveURL(/\/lm\/zh\/docs\/distributed-training$/);
  await expect(page.getByRole("heading", { name: "分布式训练与 Sharded Optimizer" })).toBeVisible();
});

test("distributed training chapter renders mermaid output", async ({ page }) => {
  await page.goto("/lm/docs/distributed-training");
  await expect(page.locator(".mermaid svg").first()).toBeVisible();
});

test("mobile chinese docs route renders overview and chapter links", async ({ page }) => {
  await page.goto("/lm/zh/docs");
  await expect(page.getByRole("heading", { name: "文档" })).toBeVisible();
  await expect(page.locator(".site-header")).toBeVisible();
  await expect(page.getByRole("link", { name: /Transformer 核心/ })).toBeVisible();
});
