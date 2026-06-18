import { defineConfig } from "astro/config";
import remarkGfm from "remark-gfm";

const repo = "lm";

export default defineConfig({
  site: "https://fangpin.github.io",
  base: `/${repo}`,
  output: "static",
  markdown: {
    remarkPlugins: [remarkGfm],
  },
  vite: {
    server: {
      fs: {
        allow: [".."],
      },
    },
  },
});
