import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "vitest";

const here = dirname(fileURLToPath(import.meta.url));
const lockfilePath = resolve(here, "..", "package-lock.json");

describe("package lock", () => {
  it("does not pin dependencies to a private registry", () => {
    const lockfile = JSON.parse(readFileSync(lockfilePath, "utf8"));
    const offenders = Object.entries(lockfile.packages)
      .filter(([, pkg]) => typeof pkg?.resolved === "string" && pkg.resolved.includes("bnpm.byted.org"))
      .slice(0, 5)
      .map(([pkgPath, pkg]) => `${pkgPath}: ${pkg.resolved}`);

    if (offenders.length > 0) {
      throw new Error(
        `package-lock.json contains private registry URLs:\n${offenders.join("\n")}`,
      );
    }
  });
});
