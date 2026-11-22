"""Render Lark callout blocks as MyST admonitions during Sphinx builds."""

import re

from myst_parser.parsers.parse_html import tokenize_html


FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
OPEN_RE = re.compile(r"^ {0,3}<callout(?:\s[^>]*)?>\s*$")
CLOSE_RE = re.compile(r"^ {0,3}</callout>\s*$")
LIST_RE = re.compile(r"^(?:[-+*]|\d+[.)])\s+")
BOLD_LABEL_RE = re.compile(r"^((?:(?:[-+*]|\d+[.)])\s+)?\*\*[^*\n]+\*\*)(?=\S)")


def render_callouts(text: str) -> str:
    output = []
    fence = ""
    in_comment = in_callout = False
    for line in text.splitlines(keepends=True):
        marker = FENCE_RE.match(line)
        if fence:
            if (
                marker
                and marker[1][0] == fence[0]
                and len(marker[1]) >= len(fence)
                and not line[marker.end():].strip()
            ):
                fence = ""
            output.append(line)
            continue
        if in_comment or line.lstrip().startswith("<!--"):
            in_comment = "-->" not in line
            output.append(line)
            continue
        if marker:
            fence = marker[1]
            output.append(line)
            continue
        if OPEN_RE.fullmatch(line):
            attrs = tokenize_html(line.strip() + "</callout>")[0].attrs
            title = attrs.get("emoji") or "Note"
            output.append(f"\n:::{{admonition}} {title}\n:class: note\n\n")
            in_callout = True
        elif in_callout and CLOSE_RE.fullmatch(line):
            output.append("\n:::\n\n")
            in_callout = False
        else:
            if in_callout:
                # Lark emits each list item/paragraph on its own unindented line.
                if (
                    output
                    and LIST_RE.match(output[-1])
                    and line[:1].strip()
                    and not LIST_RE.match(line)
                ):
                    output.append("\n")
                # CommonMark needs a boundary after punctuation-ending bold labels.
                line = BOLD_LABEL_RE.sub(r"\1 ", line)
            output.append(line)
    return "".join(output)


def on_source_read(app, docname, source):
    source[0] = render_callouts(source[0])


def setup(app):
    app.connect("source-read", on_source_read)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
