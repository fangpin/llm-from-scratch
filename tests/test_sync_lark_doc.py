import io
import urllib.request
from pathlib import Path

from doc_scripts import sync_lark_doc
from doc_scripts.sync_lark_doc import extension_for, localize_images

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
LARK_LOGIN_PAGE = b'<!doctype html>\n<html lang="zh-CN"><head><title></title></head></html>'


class _FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, content_type: str):
        super().__init__(data)
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


def _fake_urlopen(data: bytes, content_type: str):
    def urlopen(request, timeout=None):
        return _FakeResponse(data, content_type)

    return urlopen


def test_extension_for_detects_images_by_magic_bytes_and_content_type():
    assert extension_for(PNG_BYTES, "application/octet-stream") == ".png"
    assert extension_for(b"\xff\xd8\xff\xe0", None) == ".jpg"
    assert extension_for(b"GIF89a...", None) == ".gif"
    assert extension_for(b"<svg xmlns='http://www.w3.org/2000/svg'/>", "image/svg+xml") == ".svg"


def test_extension_for_rejects_lark_login_page():
    assert extension_for(LARK_LOGIN_PAGE, "text/html; charset=utf-8") is None


def test_localize_images_downloads_real_image(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sync_lark_doc, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(PNG_BYTES, "image/png"))

    content, failures = localize_images("![alt](https://feishu.example/img/token)\n", "02-llm")

    assert failures == 0
    assert content == "![alt](../assets/images/02-llm/image-01.png)\n"
    assert (tmp_path / "images" / "02-llm" / "image-01.png").read_bytes() == PNG_BYTES


def test_localize_images_keeps_remote_url_when_response_is_not_an_image(tmp_path: Path, monkeypatch):
    """An unauthenticated Feishu fetch must not land in the tree as a corrupt .png."""
    monkeypatch.setattr(sync_lark_doc, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(LARK_LOGIN_PAGE, "text/html"))

    original = "![alt](https://feishu.example/img/token)\n"
    content, failures = localize_images(original, "02-llm")

    assert failures == 1
    assert content == original
    assert not (tmp_path / "images").exists()
