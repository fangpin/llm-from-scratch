#!/usr/bin/env bash
# Vendor this documentation template into a repository's root directory.
#
# Run from the target repo's root:
#   curl -sSL https://raw.githubusercontent.com/fangpin/doc_template/master/install.sh | bash
#
# From a local clone (installs elsewhere or wherever you point it):
#   ./install.sh /path/to/target-repo
#
# Pin a ref (tag/commit) instead of master:  REF=v1.0.0 (env var)
# Install into a different directory:      pass it as the first argument
#
# Idempotent: template-owned files are overwritten, everything else is left alone.

set -euo pipefail

REF="${REF:-master}"
OWNER_REPO="${OWNER_REPO:-fangpin/doc_template}"  # source repo for curl mode
TARGET="${1:-$PWD}"

# Files that belong to the template (paths relative to repo root). Sync-generated
# content (docs/source/chapters, assets, index.md, project.json) is intentionally
# NOT vendored -- each target repo generates its own via `make -f doc.mk docs`.
ITEMS=(doc.mk install.sh requirements-docs.txt doc_scripts docs/source/conf.py .github/workflows/docs.yml)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)"

if [[ -n "${BASH_SOURCE[0]:-}" && -f "$SCRIPT_DIR/doc.mk" ]]; then
    SRC="$SCRIPT_DIR"
else
    # curl|bash mode: fetch the tarball of the requested ref
    SRC="$(mktemp -d)"
    trap 'rm -rf "$SRC"' EXIT
    curl -fsSL "https://codeload.github.com/$OWNER_REPO/tar.gz/refs/heads/$REF" \
        | tar -xz --strip-components=1 -C "$SRC"
fi

if [[ -f "$TARGET/doc.mk" ]]; then
    echo "note: doc.mk already exists in target; template files will be updated in place"
fi

for item in "${ITEMS[@]}"; do
    mkdir -p "$TARGET/$(dirname "$item")"
    cp -R "$SRC/$item" "$TARGET/$(dirname "$item")/"
    echo "  vendored: $item"
done

# Make sure generated artifacts don't get committed in the target repo.
for entry in ".venv/" "docs/_build/" "dist/platform-posts/"; do
    touch "$TARGET/.gitignore"
    grep -qxF "$entry" "$TARGET/.gitignore" || echo "$entry" >> "$TARGET/.gitignore"
done

cat <<EOF

Done. Next steps in the target repo:
  1. Install toolchain:      make -f doc.mk docs-install
  2. Build the doc site:     make -f doc.mk docs DOC="<feishu-doc-url>"
  3. Enable GitHub Pages:    Settings -> Pages -> Source -> GitHub Actions
  4. Commit the vendored files (docs/source included) and push.
EOF
