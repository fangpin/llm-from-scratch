#!/usr/bin/env python3
"""Export chapter markdown for third-party publishing platforms (WeChat, Zhihu, ...).

Rewrites image references to absolute GitHub Pages URLs of the built site
(<site>/_images/...) so hotlinked images keep rendering outside the repo.
Mapping is derived from the built HTML, so Sphinx's duplicate-name renaming
in _images/ stays transparent to us.

Usage:
    python doc_scripts/export_platform_posts.py [--site-url URL]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "docs" / "source"
BUILD_DIR = ROOT / "docs" / "_build" / "html"
PROJECT_JSON = ROOT / "docs" / "project.json"
OUTPUT_DIR = ROOT / "dist" / "platform-posts"
REPO_DOCS_PREFIX = "docs/source"  # chapters live in docs/source/chapters/

IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(\s+\"[^\"]*\")?\)")
HTML_IMG_RE = re.compile(r'<img[^>]*\bsrc="(?:\.\./)*_images/([^"]+)"')
CONTENTS_RE = re.compile(r"```{contents}[^\n]*\n(?:---\n.*?\n)*---\n```\n\n?", re.S)


def github_slug() -> str | None:
    if slug := os.environ.get("GITHUB_REPOSITORY"):
        return slug
    try:
        url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    m = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
    return m.group(1) if m else None


def default_site_url(meta: dict) -> str | None:
    if meta.get("site_url"):
        return meta["site_url"]
    slug = meta.get("github_repo") or github_slug()
    if not slug:
        return None
    owner, repo = slug.split("/", 1)
    if repo.lower() == f"{owner.lower()}.github.io":
        return f"https://{owner}.github.io/"
    return f"https://{owner}.github.io/{repo}/"


def built_image_names(slug: str) -> list[str]:
    html_path = BUILD_DIR / "chapters" / f"{slug}.html"
    if not html_path.exists():
        return []
    return HTML_IMG_RE.findall(html_path.read_text(encoding="utf-8"))


def rewrite_chapter(chapter: Path, site_url: str, raw_base: str | None) -> tuple[str, int, int]:
    text = chapter.read_text(encoding="utf-8")
    text = CONTENTS_RE.sub("", text)  # MyST-only directive, meaningless off-Sphinx
    images = list(IMAGE_RE.finditer(text))
    built = [] if raw_base else built_image_names(chapter.stem)
    rewritten = 0

    def sub_factory():
        idx = -1

        def sub(m: re.Match) -> str:
            nonlocal idx, rewritten
            url = m.group(2)
            if url.startswith(("http://", "https://")):
                return m.group(0)  # external image stays untouched
            if raw_base is not None:
                # ../assets/images/... relative to docs/source/chapters/
                local = f"{REPO_DOCS_PREFIX}/{url.removeprefix('../')}"
                rewritten += 1
                return f'![{m.group(1)}]({raw_base}/{quote(local)}{m.group(3) or ""})'
            idx += 1
            if idx >= len(built):
                return m.group(0)
            rewritten += 1
            return f'![{m.group(1)}]({site_url}_images/{quote(built[idx])}{m.group(3) or ""})'

        return sub

    text = IMAGE_RE.sub(sub_factory(), text)
    return text, len(images), rewritten


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-url", help="Base URL of the deployed GitHub Pages site")
    parser.add_argument("--image-base", choices=["pages", "raw"], default="pages",
                        help="pages: hotlink the built site's _images (needs Pages deployed); "
                             "raw: hotlink raw.githubusercontent.com repo files (needs push only)")
    args = parser.parse_args()

    meta = json.loads(PROJECT_JSON.read_text(encoding="utf-8")) if PROJECT_JSON.exists() else {}

    site_url = ""
    raw_base = None
    if args.image_base == "raw":
        slug = meta.get("github_repo") or github_slug()
        if not slug:
            sys.exit("error: unknown GitHub repo. Set github_repo in docs/project.json "
                     "or configure a git origin remote.")
        branch = meta.get("github_branch") or os.environ.get("GITHUB_REF_NAME", "main")
        raw_base = f"https://raw.githubusercontent.com/{slug}/{branch}"
    else:
        site_url = (args.site_url or default_site_url(meta) or "").rstrip("/") + "/"
        if site_url == "/":
            sys.exit("error: unknown site URL. Pass --site-url, set github_repo/site_url in "
                     "docs/project.json, or configure a git origin remote.")
        if not BUILD_DIR.exists():
            sys.exit("error: docs/_build/html not found. Run `make html` first.")

    chapters = sorted((SOURCE_DIR / "chapters").glob("*.md"))
    if not chapters:
        sys.exit("error: no chapters found. Run `make sync` first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for chapter in chapters:
        text, total, rewritten = rewrite_chapter(chapter, site_url, raw_base)
        if total and rewritten < total:
            print(f"  warning: {chapter.name}: only {rewritten}/{total} images rewritten "
                  "(built HTML missing?)", file=sys.stderr)
        out = OUTPUT_DIR / chapter.name
        out.write_text(text, encoding="utf-8")
        print(f"  post: dist/platform-posts/{chapter.name} ({rewritten} image(s))")


if __name__ == "__main__":
    main()
