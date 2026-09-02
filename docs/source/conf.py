import json
import os
import re
import subprocess
from pathlib import Path

meta_path = Path(__file__).resolve().parent.parent / "project.json"
meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

project = meta.get("title", "Docs")
author = ""

extensions = ["myst_parser"]
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "tasklist",
]
myst_heading_anchors = 3
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_title = project
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}


def _github_slug() -> str | None:
    """owner/repo from GITHUB_REPOSITORY (CI) or the local git origin remote."""
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


def _default_branch() -> str:
    """Remote default branch (e.g. master) via origin/HEAD, else main."""
    try:
        ref = subprocess.check_output(
            ["git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "main"
    return ref.split("/", 1)[1] if "/" in ref else "main"


# Renders an "Edit on GitHub" link in the page header (instead of keeping only
# "View page source"). Set "github_repo": "owner/repo" in docs/project.json to
# override auto-detection.
html_show_sourcelink = False
github_slug = meta.get("github_repo") or _github_slug()
if github_slug:
    github_user, github_repo = github_slug.split("/", 1)
    html_context = {
        "display_github": True,
        "github_user": github_user,
        "github_repo": github_repo,
        "github_version": meta.get("github_branch") or os.environ.get("GITHUB_REF_NAME") or _default_branch(),
        "conf_py_path": "/docs/source/",
    }
