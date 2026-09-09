import io
import urllib.request
from pathlib import Path

from doc_scripts import sync_lark_doc
from doc_scripts.sync_lark_doc import extension_for, lark_media_token, localize_images

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


def _unreachable_urlopen(request, timeout=None):
    raise AssertionError("Feishu media must be downloaded through lark-cli, not anonymously")


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


def test_lark_media_token_extracts_file_token_from_tenant_urls():
    assert (
        lark_media_token("https://bytedance.sg.larkoffice.com/file/XUbWbt0NPowmqyxci38lestsgAb")
        == "XUbWbt0NPowmqyxci38lestsgAb"
    )
    assert (
        lark_media_token("https://open.feishu.cn/open-apis/drive/v1/medias/boxcnrHpsg1QDqXAAAyachoCbQe/download")
        == "boxcnrHpsg1QDqXAAAyachoCbQe"
    )
    assert (
        lark_media_token("https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/all"
                         "/?file_token=boxcnrHpsg1QDqXAAAyachoCbQe")
        == "boxcnrHpsg1QDqXAAAyachoCbQe"
    )


def test_lark_media_token_ignores_public_image_hosts():
    assert lark_media_token("https://example.com/file/some-public-image.png") is None
    assert lark_media_token("https://notfeishu.cn.example.com/file/XUbWbt0NPowmqyxci38lestsgAb") is None


def test_localize_images_downloads_feishu_media_through_lark_cli(tmp_path: Path, monkeypatch):
    """Feishu URLs must not be fetched anonymously; only lark-cli carries the login."""
    monkeypatch.setattr(sync_lark_doc, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(urllib.request, "urlopen", _unreachable_urlopen)
    calls = []

    def fake_run_lark_cli(argv, check=True):
        calls.append(argv)
        output = Path(argv[argv.index("--output") + 1])
        output.with_suffix(".png").write_bytes(PNG_BYTES)
        return {"ok": True}

    monkeypatch.setattr(sync_lark_doc, "run_lark_cli", fake_run_lark_cli)

    content, failures = localize_images(
        "![alt](https://bytedance.sg.larkoffice.com/file/XUbWbt0NPowmqyxci38lestsgAb)\n", "02-llm"
    )

    assert failures == 0
    assert content == "![alt](../assets/images/02-llm/image-01.png)\n"
    assert (tmp_path / "images" / "02-llm" / "image-01.png").read_bytes() == PNG_BYTES
    assert calls[0][:2] == ["docs", "+media-download"]
    assert "XUbWbt0NPowmqyxci38lestsgAb" in calls[0]


def test_localize_images_falls_back_to_media_preview(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sync_lark_doc, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(urllib.request, "urlopen", _unreachable_urlopen)
    commands = []

    def fake_run_lark_cli(argv, check=True):
        commands.append(argv[1])
        if argv[1] == "+media-download":
            return None  # permission_denied, as the docs skill describes
        Path(argv[argv.index("--output") + 1]).with_suffix(".jpg").write_bytes(b"\xff\xd8\xff\xe0")
        return {"ok": True}

    monkeypatch.setattr(sync_lark_doc, "run_lark_cli", fake_run_lark_cli)

    content, failures = localize_images(
        "![alt](https://bytedance.sg.larkoffice.com/file/XUbWbt0NPowmqyxci38lestsgAb)\n", "02-llm"
    )

    assert commands == ["+media-download", "+media-preview"]
    assert failures == 0
    assert content == "![alt](../assets/images/02-llm/image-01.jpg)\n"


def test_localize_images_reports_failure_when_lark_cli_cannot_download(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sync_lark_doc, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr(urllib.request, "urlopen", _unreachable_urlopen)
    monkeypatch.setattr(sync_lark_doc, "run_lark_cli", lambda argv, check=True: None)

    original = "![alt](https://bytedance.sg.larkoffice.com/file/XUbWbt0NPowmqyxci38lestsgAb)\n"
    content, failures = localize_images(original, "02-llm")

    assert failures == 1
    assert content == original
    assert not (tmp_path / "images").exists()
