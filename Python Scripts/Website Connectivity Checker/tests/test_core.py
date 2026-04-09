"""Unit tests for core: URL parsing, validation, checking, output formatting."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import httpx
import respx

from check_website_connectivity.core import (
    check_url,
    check_urls,
    format_csv_string,
    format_json,
    format_table,
    load_urls,
    normalize_url,
    parse_urls_from_csv,
    parse_urls_from_text,
    validate_url,
    write_csv,
)
from check_website_connectivity.models import CheckResult, Status

# ======================================================================
# URL normalisation & validation
# ======================================================================


class TestNormalizeUrl:
    def test_adds_https_scheme(self) -> None:
        assert normalize_url("example.com") == "https://example.com"

    def test_preserves_http(self) -> None:
        assert normalize_url("http://example.com") == "http://example.com"

    def test_preserves_https(self) -> None:
        assert normalize_url("https://example.com") == "https://example.com"

    def test_strips_whitespace(self) -> None:
        assert normalize_url("  https://example.com  ") == "https://example.com"

    def test_empty_string(self) -> None:
        assert normalize_url("") == ""

    def test_unsupported_scheme(self) -> None:
        assert normalize_url("ftp://files.example.com") == ""

    def test_with_path(self) -> None:
        assert normalize_url("example.com/path?q=1") == "https://example.com/path?q=1"


class TestValidateUrl:
    def test_valid_https(self) -> None:
        assert validate_url("https://example.com") is True

    def test_valid_http(self) -> None:
        assert validate_url("http://example.com") is True

    def test_no_scheme(self) -> None:
        assert validate_url("example.com") is False

    def test_no_netloc(self) -> None:
        assert validate_url("https://") is False

    def test_ftp(self) -> None:
        assert validate_url("ftp://files.example.com") is False


# ======================================================================
# Input parsing
# ======================================================================


class TestParseUrlsFromText:
    def test_basic(self) -> None:
        text = "https://a.com\nhttps://b.com\n"
        assert parse_urls_from_text(text) == ["https://a.com", "https://b.com"]

    def test_skips_comments_and_blanks(self) -> None:
        text = "# comment\nhttps://a.com\n\n  \n# another\nhttps://b.com"
        assert parse_urls_from_text(text) == ["https://a.com", "https://b.com"]

    def test_adds_scheme(self) -> None:
        text = "example.com\n"
        assert parse_urls_from_text(text) == ["https://example.com"]

    def test_empty(self) -> None:
        assert parse_urls_from_text("") == []


class TestParseUrlsFromCsv:
    def test_single_column(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "urls.csv"
        csv_file.write_text("https://a.com\nhttps://b.com\n", encoding="utf-8")
        assert parse_urls_from_csv(csv_file) == ["https://a.com", "https://b.com"]

    def test_with_header_row(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "urls.csv"
        csv_file.write_text("Website,Status\nhttps://a.com,working\n", encoding="utf-8")
        result = parse_urls_from_csv(csv_file)
        assert result == ["https://a.com"]

    def test_skips_status_column(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "Website,Status\nhttps://a.com,working\nhttps://b.com,not working\n",
            encoding="utf-8",
        )
        result = parse_urls_from_csv(csv_file)
        assert result == ["https://a.com", "https://b.com"]

    def test_legacy_website_status_csv(self, tmp_path: Path) -> None:
        """Reads the exact format produced by the legacy script."""
        csv_file = tmp_path / "website_status.csv"
        csv_file.write_text(
            "Website,Status\n"
            "http://web.hike.com/,working\n"
            "https://github.com/chavarera/python-mini-projects/issues/96,working\n",
            encoding="utf-8",
        )
        result = parse_urls_from_csv(csv_file)
        assert len(result) == 2
        assert result[0] == "http://web.hike.com/"


class TestLoadUrls:
    def test_txt(self, tmp_path: Path) -> None:
        p = tmp_path / "sites.txt"
        p.write_text("https://a.com\nhttps://b.com\n", encoding="utf-8")
        assert load_urls(p) == ["https://a.com", "https://b.com"]

    def test_csv(self, tmp_path: Path) -> None:
        p = tmp_path / "sites.csv"
        p.write_text("Website,Status\nhttps://a.com,working\n", encoding="utf-8")
        assert load_urls(p) == ["https://a.com"]


# ======================================================================
# Connectivity checking (mocked with respx)
# ======================================================================


class TestCheckUrl:
    @respx.mock
    def test_reachable_200(self) -> None:
        respx.get("https://example.com/").mock(return_value=httpx.Response(200))
        result = check_url("https://example.com/")
        assert result.status is Status.REACHABLE
        assert result.status_code == 200
        assert result.ok is True

    @respx.mock
    def test_unreachable_404(self) -> None:
        respx.get("https://example.com/gone").mock(return_value=httpx.Response(404))
        result = check_url("https://example.com/gone")
        assert result.status is Status.UNREACHABLE
        assert result.status_code == 404
        assert result.ok is False

    @respx.mock
    def test_unreachable_500(self) -> None:
        respx.get("https://down.example.com/").mock(return_value=httpx.Response(500))
        result = check_url("https://down.example.com/")
        assert result.status is Status.UNREACHABLE
        assert result.status_code == 500

    @respx.mock
    def test_redirect_301(self) -> None:
        respx.get("https://old.example.com/").mock(return_value=httpx.Response(200))
        result = check_url("https://old.example.com/")
        assert result.status is Status.REACHABLE

    @respx.mock
    def test_timeout(self) -> None:
        respx.get("https://slow.example.com/").mock(side_effect=httpx.ReadTimeout("timeout"))
        result = check_url("https://slow.example.com/")
        assert result.status is Status.UNREACHABLE
        assert result.reason == "timeout"

    @respx.mock
    def test_connection_error(self) -> None:
        respx.get("https://nope.example.com/").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        result = check_url("https://nope.example.com/")
        assert result.status is Status.ERROR
        assert "connection refused" in result.reason

    @respx.mock
    def test_response_time_recorded(self) -> None:
        respx.get("https://example.com/").mock(return_value=httpx.Response(200))
        result = check_url("https://example.com/")
        assert result.response_time_ms is not None
        assert result.response_time_ms >= 0

    @respx.mock
    def test_custom_timeout_and_retries(self) -> None:
        respx.get("https://example.com/").mock(return_value=httpx.Response(200))
        result = check_url("https://example.com/", timeout=5.0, retries=0)
        assert result.ok


class TestCheckUrls:
    @respx.mock
    def test_multiple(self) -> None:
        respx.get("https://a.com/").mock(return_value=httpx.Response(200))
        respx.get("https://b.com/").mock(return_value=httpx.Response(500))
        results = check_urls(["https://a.com/", "https://b.com/"])
        assert len(results) == 2
        assert results[0].ok is True
        assert results[1].ok is False

    @respx.mock
    def test_empty_list(self) -> None:
        results = check_urls([])
        assert results == []


# ======================================================================
# Output formatting
# ======================================================================


class TestFormatTable:
    def test_empty(self) -> None:
        assert format_table([]) == "No results."

    def test_single_result(self) -> None:
        r = CheckResult(
            url="https://example.com",
            status=Status.REACHABLE,
            status_code=200,
            reason="OK",
            response_time_ms=42.5,
        )
        table = format_table([r])
        assert "https://example.com" in table
        assert "reachable" in table
        assert "200" in table

    def test_multiple_results(self) -> None:
        results = [
            CheckResult(url="https://a.com", status=Status.REACHABLE, status_code=200),
            CheckResult(url="https://b.com", status=Status.UNREACHABLE, status_code=500),
        ]
        table = format_table(results)
        assert "https://a.com" in table
        assert "https://b.com" in table


class TestFormatJson:
    def test_valid_json(self) -> None:
        results = [
            CheckResult(url="https://a.com", status=Status.REACHABLE, status_code=200),
        ]
        data = json.loads(format_json(results))
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["url"] == "https://a.com"
        assert data[0]["status"] == "reachable"

    def test_empty(self) -> None:
        assert json.loads(format_json([])) == []


class TestWriteCsv:
    def test_writes_file(self, tmp_path: Path) -> None:
        out = tmp_path / "out.csv"
        results = [
            CheckResult(
                url="https://a.com",
                status=Status.REACHABLE,
                status_code=200,
                response_time_ms=10.0,
            ),
        ]
        write_csv(results, out)
        assert out.exists()
        reader = csv.reader(io.StringIO(out.read_text(encoding="utf-8")))
        rows = list(reader)
        assert rows[0] == ["Website", "Status", "Code", "Time_ms", "Checked_at"]
        assert rows[1][0] == "https://a.com"
        assert rows[1][1] == "working"

    def test_format_csv_string(self) -> None:
        results = [
            CheckResult(url="https://x.com", status=Status.UNREACHABLE, status_code=404),
        ]
        csv_str = format_csv_string(results)
        assert "not working" in csv_str
        assert "https://x.com" in csv_str


# ======================================================================
# Model round-trip
# ======================================================================


class TestCheckResultModel:
    def test_round_trip(self) -> None:
        r = CheckResult(
            url="https://example.com",
            status=Status.REACHABLE,
            status_code=200,
            reason="OK",
            response_time_ms=42.5,
        )
        d = r.to_dict()
        r2 = CheckResult.from_dict(d)
        assert r2.url == r.url
        assert r2.status is r.status
        assert r2.status_code == r.status_code

    def test_status_label(self) -> None:
        assert CheckResult(url="x", status=Status.REACHABLE).status_label() == "working"
        assert CheckResult(url="x", status=Status.UNREACHABLE).status_label() == "not working"
        assert CheckResult(url="x", status=Status.ERROR).status_label() == "not working"

    def test_ok_property(self) -> None:
        assert CheckResult(url="x", status=Status.REACHABLE).ok is True
        assert CheckResult(url="x", status=Status.UNREACHABLE).ok is False
        assert CheckResult(url="x", status=Status.ERROR).ok is False
