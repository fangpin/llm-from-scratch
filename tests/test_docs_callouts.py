import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]


class DocsCalloutsTest(unittest.TestCase):
    def test_sphinx_renders_callouts_without_changing_source(self):
        markdown = (
            "# Introduction\n\n"
            '<callout emoji="tip">\n'
            "**\u9605\u8bfb\u8def\u7ebf\uff1a**Start here.\n"
            "- **Part one:**First.\n"
            "- **Part two:**Second.\n"
            "Read [more](https://example.com/book/) with `code` and $E = mc^2$.\n"
            "</callout>\n\n"
            "```text\n<callout>\n**Literal:**Example.\n</callout>\n```\n"
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "docs" / "source"
            source.mkdir(parents=True)
            shutil.copyfile(ROOT / "docs/source/conf.py", source / "conf.py")
            shutil.copytree(ROOT / "docs/source/_templates", source / "_templates")
            shutil.copytree(ROOT / "doc_scripts", root / "doc_scripts")
            (source / "index.md").write_text(markdown, encoding="utf-8")
            build = root / "build"
            result = subprocess.run(
                [sys.executable, "-m", "sphinx", "-b", "html", "-W", "--keep-going",
                 str(source), str(build)],
                cwd=root,
                env={**os.environ, "GITHUB_REPOSITORY": "example/book",
                     "GITHUB_REF_NAME": "master"},
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            html = (build / "index.html").read_text(encoding="utf-8")
            self.assertNotIn("<callout", html)
            self.assertIn('<div class="note admonition">', html)
            self.assertIn("<strong>\u9605\u8bfb\u8def\u7ebf\uff1a</strong>", html)
            self.assertIn("<strong>Part one:</strong>", html)
            self.assertIn('<ul class="simple">', html)
            self.assertIn("</ul>\n<p>Read ", html)
            self.assertIn('href="https://example.com/book/"', html)
            self.assertIn('<span class="pre">code</span>', html)
            self.assertIn('class="math notranslate nohighlight"', html)
            self.assertIn("&lt;callout&gt;", html)
            self.assertIn("**Literal:**Example.", html)
            self.assertEqual((source / "index.md").read_text(encoding="utf-8"), markdown)


if __name__ == "__main__":
    unittest.main()
