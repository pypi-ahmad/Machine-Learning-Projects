"""CLI integration tests using typer.testing.CliRunner + respx mocking."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx
from typer.testing import CliRunner

from check_website_connectivity.cli import app

runner = CliRunner()


def _invoke(*args: str):
    return runner.invoke(app, list(args))


# ======================================================================
# Basic invocation
# ======================================================================


class TestNoArgs:
    def test_no_args_exits_with_error(self) -> None:
        result = _invoke()
        # Single-command Typer app runs the command; no URLs → exit 2
        assert result.exit_code == 2
        assert "no urls" in result.output.lower() or "error" in result.output.lower()


class TestInvalidInput:
    def test_invalid_url_scheme(self) -> None:
        result = _invoke("ftp://files.example.com")
        assert result.exit_code == 2
        assert "invalid" in result.output.lower() or "error" in result.output.lower()

    def test_file_not_found(self) -> None:
        result = _invoke("--file", "/nonexistent/urls.txt")
        assert result.exit_code == 2

    def test_no_urls_from_empty_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.txt"
        empty.write_text("", encoding="utf-8")
        result = _invoke("--file", str(empty))
        assert result.exit_code == 2


# ======================================================================
# Single-URL check
# ======================================================================


class TestSingleUrl:
    @respx.mock
    def test_reachable(self) -> None:
        respx.get("https://example.com/").mock(return_value=httpx.Response(200))
        result = _invoke("https://example.com/")
        assert result.exit_code == 0
        assert "reachable" in result.output.lower()

    @respx.mock
    def test_unreachable(self) -> None:
        respx.get("https://down.example.com/").mock(return_value=httpx.Response(500))
        result = _invoke("https://down.example.com/")
        assert result.exit_code == 1
        assert "unreachable" in result.output.lower()

    @respx.mock
    def test_adds_scheme(self) -> None:
        respx.get("https://example.com/").mock(return_value=httpx.Response(200))
        result = _invoke("example.com")
        # It should normalise to https://example.com
        assert result.exit_code == 0


# ======================================================================
# File input
# ======================================================================


class TestFileInput:
    @respx.mock
    def test_txt_file(self, tmp_path: Path) -> None:
        urls_file = tmp_path / "urls.txt"
        urls_file.write_text("https://a.com\nhttps://b.com\n", encoding="utf-8")
        respx.get("https://a.com/").mock(return_value=httpx.Response(200))
        respx.get("https://b.com/").mock(return_value=httpx.Response(200))
        # respx may match with or without trailing slash; mock both
        respx.get("https://a.com").mock(return_value=httpx.Response(200))
        respx.get("https://b.com").mock(return_value=httpx.Response(200))
        result = _invoke("--file", str(urls_file))
        assert result.exit_code == 0
        assert "https://a.com" in result.output
        assert "https://b.com" in result.output

    @respx.mock
    def test_csv_file(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "sites.csv"
        csv_file.write_text(
            "Website,Status\nhttps://a.com,working\nhttps://b.com,working\n",
            encoding="utf-8",
        )
        respx.get("https://a.com").mock(return_value=httpx.Response(200))
        respx.get("https://b.com").mock(return_value=httpx.Response(200))
        respx.get("https://a.com/").mock(return_value=httpx.Response(200))
        respx.get("https://b.com/").mock(return_value=httpx.Response(200))
        result = _invoke("--file", str(csv_file))
        assert result.exit_code == 0

    @respx.mock
    def test_mixed_file_and_args(self, tmp_path: Path) -> None:
        urls_file = tmp_path / "urls.txt"
        urls_file.write_text("https://a.com\n", encoding="utf-8")
        respx.get("https://a.com").mock(return_value=httpx.Response(200))
        respx.get("https://a.com/").mock(return_value=httpx.Response(200))
        respx.get("https://b.com").mock(return_value=httpx.Response(200))
        respx.get("https://b.com/").mock(return_value=httpx.Response(200))
        result = _invoke("--file", str(urls_file), "https://b.com")
        assert result.exit_code == 0
        assert "https://a.com" in result.output
        assert "https://b.com" in result.output


# ======================================================================
# Output formats
# ======================================================================


class TestJsonOutput:
    @respx.mock
    def test_json_flag(self) -> None:
        respx.get("https://example.com/").mock(return_value=httpx.Response(200))
        respx.get("https://example.com").mock(return_value=httpx.Response(200))
        result = _invoke("--json", "https://example.com")
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert data[0]["status"] == "reachable"


class TestCsvOutput:
    @respx.mock
    def test_csv_flag(self, tmp_path: Path) -> None:
        out = tmp_path / "results.csv"
        respx.get("https://example.com/").mock(return_value=httpx.Response(200))
        respx.get("https://example.com").mock(return_value=httpx.Response(200))
        result = _invoke("--csv", str(out), "https://example.com")
        assert result.exit_code == 0
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "Website" in content
        assert "working" in content


# ======================================================================
# Verbose
# ======================================================================


class TestVerbose:
    @respx.mock
    def test_verbose_flag(self) -> None:
        respx.get("https://example.com/").mock(return_value=httpx.Response(200))
        respx.get("https://example.com").mock(return_value=httpx.Response(200))
        result = _invoke("--verbose", "https://example.com")
        assert result.exit_code == 0


# ======================================================================
# Exit codes
# ======================================================================


class TestExitCodes:
    @respx.mock
    def test_all_reachable_exit_0(self) -> None:
        respx.get("https://a.com").mock(return_value=httpx.Response(200))
        respx.get("https://a.com/").mock(return_value=httpx.Response(200))
        result = _invoke("https://a.com")
        assert result.exit_code == 0

    @respx.mock
    def test_any_unreachable_exit_1(self) -> None:
        respx.get("https://a.com").mock(return_value=httpx.Response(200))
        respx.get("https://a.com/").mock(return_value=httpx.Response(200))
        respx.get("https://b.com").mock(return_value=httpx.Response(500))
        respx.get("https://b.com/").mock(return_value=httpx.Response(500))
        result = _invoke("https://a.com", "https://b.com")
        assert result.exit_code == 1

    def test_invalid_input_exit_2(self) -> None:
        result = _invoke("ftp://x.com")
        assert result.exit_code == 2
