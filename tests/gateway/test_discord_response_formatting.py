from typing import Any, Optional
from urllib.parse import quote

from gateway.config import Platform, PlatformConfig
from gateway.platforms import base as base_mod
from gateway.platforms.base import BasePlatformAdapter, SendResult


class _StubAdapter(BasePlatformAdapter):
    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        return SendResult(success=True)

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        return {}


def _adapter(platform=Platform.DISCORD):
    return _StubAdapter(PlatformConfig(enabled=True), platform)


def test_discord_markdown_path_becomes_local_obsidian_opener(tmp_path):
    report = tmp_path / "weekly-report.md"
    report.write_text("# Report\n", encoding="utf-8")

    rendered = _adapter()._format_response_for_platform(f"Report: {report}")

    assert str(report) not in rendered
    assert rendered == "Report: http://127.0.0.1:8765/open?path=" + quote(str(report), safe="")


def test_discord_report_path_is_mirrored_into_obsidian_vault_before_linking(tmp_path, monkeypatch):
    report_root = tmp_path / "reports"
    report_root.mkdir()
    report = report_root / "weekly-report.md"
    report.write_text("# Report\n", encoding="utf-8")
    realpath = base_mod.os.path.realpath
    copied = []

    def fake_realpath(path):
        if path == "/Users/royce/Documents/hermes/.hermes/reports":
            return str(report_root)
        return realpath(path)

    def fake_copy2(src, dst):
        copied.append((src, dst))
        return dst

    monkeypatch.setattr(base_mod.os.path, "realpath", fake_realpath)
    monkeypatch.setattr(base_mod.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(base_mod.shutil, "copy2", fake_copy2)

    rendered = _adapter()._format_response_for_platform(f"Report: {report}")

    mirrored = "/Users/royce/Documents/Obsidian/Hermes/Hermes Reports/weekly-report.md"
    assert copied == [(str(report), mirrored)]
    assert rendered == "Report: http://127.0.0.1:8765/open?path=" + quote(mirrored, safe="")


def test_discord_markdown_path_inside_code_is_left_alone(tmp_path):
    report = tmp_path / "weekly-report.md"
    report.write_text("# Report\n", encoding="utf-8")

    rendered = _adapter()._format_response_for_platform(f"`{report}`")

    assert rendered == f"`{report}`"


def test_discord_pipe_table_becomes_fixed_width_text_block():
    rendered = _adapter()._format_response_for_platform(
        "Findings:\n"
        "| Repo | Status |\n"
        "| --- | --- |\n"
        "| apiscout | ok |\n"
        "| starterpick | needs review |\n"
        "Done."
    )

    assert "| --- |" not in rendered
    assert "```text" in rendered
    assert "Repo         Status" in rendered
    assert "starterpick  needs review" in rendered


def test_non_discord_platform_is_not_rewritten(tmp_path):
    report = tmp_path / "weekly-report.md"
    report.write_text("# Report\n", encoding="utf-8")
    content = f"| A | B |\n| --- | --- |\n| 1 | 2 |\n{report}"

    rendered = _adapter(Platform.TELEGRAM)._format_response_for_platform(content)

    assert rendered == content
