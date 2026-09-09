import importlib.util
import json
import subprocess
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "doc_scripts" / "sync_lark_doc.py"
spec = importlib.util.spec_from_file_location("sync_lark_doc", SCRIPT_PATH)
sync_lark_doc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(sync_lark_doc)

PNG_BYTES = b"\x89PNG\r\n\x1a\npng data"


class FakeHttpResponse:
    def __init__(self, data: bytes, content_type: str):
        self._data = data
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._data


def test_localize_images_does_not_save_non_image_http_response(tmp_path, monkeypatch):
    monkeypatch.setattr(sync_lark_doc, "IMAGES_DIR", tmp_path)
    monkeypatch.setattr(
        sync_lark_doc.urllib.request,
        "urlopen",
        lambda _req, timeout: FakeHttpResponse(b"<!doctype html><title>login</title>", "text/html"),
    )

    content = "![diagram](https://example.com/protected-image)"

    assert sync_lark_doc.localize_images(content, "chapter") == content
    assert list(tmp_path.rglob("*")) == []


def test_localize_images_uses_lark_cli_for_feishu_file_urls(tmp_path, monkeypatch):
    monkeypatch.setattr(sync_lark_doc, "IMAGES_DIR", tmp_path)

    def fail_urlopen(_req, timeout):
        raise AssertionError("Feishu file URLs must be downloaded with lark-cli auth")

    monkeypatch.setattr(sync_lark_doc.urllib.request, "urlopen", fail_urlopen)

    def fake_run(cmd, capture_output, text, env, cwd):
        assert cmd[:3] == ["lark-cli", "docs", "+media-preview"]
        assert cmd[cmd.index("--token") + 1] == "XUbWbt0NPowmqyxci38lestsgAb"
        output_base = cmd[cmd.index("--output") + 1]
        saved_path = Path(cwd) / f"{output_base}.png"
        saved_path.write_bytes(PNG_BYTES)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "data": {
                        "content_type": "image/png",
                        "saved_path": str(saved_path),
                    },
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(sync_lark_doc.subprocess, "run", fake_run)

    localized = sync_lark_doc.localize_images(
        "![bf16](https://feishu.cn/file/XUbWbt0NPowmqyxci38lestsgAb)",
        "02-llm",
    )

    assert localized == "![bf16](../assets/images/02-llm/image-01.png)"
    assert (tmp_path / "02-llm" / "image-01.png").read_bytes() == PNG_BYTES
