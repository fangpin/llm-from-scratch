import unittest

from doc_scripts.lark_callouts import render_callouts


class LarkCalloutsTest(unittest.TestCase):
    def test_callout_preserves_markdown_and_separates_trailing_paragraph(self):
        text = (
            '<callout emoji="tip">\n'
            "**Route:**Introduction.\n"
            "- **First:**One.\n"
            "- **Second:**Two.\n"
            "Read [more](chapter.md).\n"
            "</callout>\n"
        )
        rendered = render_callouts(text)
        self.assertNotIn("<callout", rendered)
        self.assertIn(":::{admonition} tip\n:class: note\n", rendered)
        self.assertIn("**Route:** Introduction.", rendered)
        self.assertIn("- **Second:** Two.\n\nRead [more](chapter.md).", rendered)

    def test_code_examples_and_comments_are_unchanged(self):
        examples = (
            '````markdown\n```xml\n<callout emoji="tip">\n'
            "**Route:**Code.\n</callout>\n```\n````\n\n"
            '~~~xml\n<callout>\n**Route:**Code.\n</callout>\n~~~\n\n'
            '    <callout>\n    **Route:**Code.\n    </callout>\n\n'
            '`<callout>` and `**Route:**Code`.\n\n'
            '<!--\n<callout>\n**Route:**Comment.\n</callout>\n-->\n'
        )
        self.assertEqual(render_callouts(examples), examples)

    def test_callout_keeps_inline_and_fenced_code_literal(self):
        text = (
            "<callout>\n"
            "`**Route:**Code` stays literal.\n\n"
            "```xml\n<callout>\n**Route:**Code.\n</callout>\n```\n"
            "</callout>\n"
        )
        rendered = render_callouts(text)
        self.assertIn("`**Route:**Code` stays literal.", rendered)
        self.assertIn("```xml\n<callout>\n**Route:**Code.\n</callout>\n```", rendered)
        self.assertIn(":::{admonition} Note", rendered)

    def test_non_callout_markdown_is_unchanged(self):
        text = "**Route:**Outside.\n\n<div>**Route:**HTML.</div>\n"
        self.assertEqual(render_callouts(text), text)


if __name__ == "__main__":
    unittest.main()
