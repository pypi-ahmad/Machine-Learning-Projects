"""CLI integration tests via typer.testing.CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from convert_xml_to_json.cli import app

runner = CliRunner()

SIMPLE_XML = "<root><name>Alice</name><age>30</age></root>"

CATALOG_XML = """\
<?xml version="1.0"?>
<catalog>
  <book id="bk101">
    <author>Gambardella, Matthew</author>
    <title>XML Developer's Guide</title>
  </book>
</catalog>
"""


def _xml_file(tmp_path: Path, content: str = SIMPLE_XML, name: str = "test.xml") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ======================================================================
# Basic conversion to stdout
# ======================================================================


class TestStdout:
    def test_compact_by_default(self, tmp_path: Path) -> None:
        f = _xml_file(tmp_path)
        result = runner.invoke(app, [str(f)])
        assert result.exit_code == 0
        # compact means no leading spaces
        data = json.loads(result.output)
        assert data["root"]["name"] == "Alice"

    def test_pretty_flag(self, tmp_path: Path) -> None:
        f = _xml_file(tmp_path)
        result = runner.invoke(app, [str(f), "--pretty"])
        assert result.exit_code == 0
        assert "\n" in result.output
        assert '  "root"' in result.output

    def test_indent_flag(self, tmp_path: Path) -> None:
        f = _xml_file(tmp_path)
        result = runner.invoke(app, [str(f), "--indent", "4"])
        assert result.exit_code == 0
        assert '    "root"' in result.output

    def test_indent_overrides_pretty(self, tmp_path: Path) -> None:
        f = _xml_file(tmp_path)
        result = runner.invoke(app, [str(f), "--pretty", "--indent", "4"])
        assert result.exit_code == 0
        assert '    "root"' in result.output

    def test_sort_keys(self, tmp_path: Path) -> None:
        f = _xml_file(tmp_path)
        result = runner.invoke(app, [str(f), "--pretty", "--sort-keys"])
        assert result.exit_code == 0
        assert result.output.index('"age"') < result.output.index('"name"')


# ======================================================================
# Output to file
# ======================================================================


class TestFileOutput:
    def test_out_flag(self, tmp_path: Path) -> None:
        f = _xml_file(tmp_path)
        out = tmp_path / "result.json"
        result = runner.invoke(app, [str(f), "--out", str(out), "--pretty"])
        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["root"]["name"] == "Alice"
        assert "Written to" in result.output

    def test_out_creates_parent_dirs(self, tmp_path: Path) -> None:
        f = _xml_file(tmp_path)
        out = tmp_path / "sub" / "out.json"
        result = runner.invoke(app, [str(f), "--out", str(out)])
        assert result.exit_code == 0
        assert out.exists()


# ======================================================================
# Error handling
# ======================================================================


class TestErrors:
    def test_missing_file(self) -> None:
        result = runner.invoke(app, ["/nonexistent/file.xml"])
        assert result.exit_code == 2

    def test_invalid_xml(self, tmp_path: Path) -> None:
        f = _xml_file(tmp_path, content="<<< not xml >>>")
        result = runner.invoke(app, [str(f)])
        assert result.exit_code == 2
        assert "error" in result.output.lower()

    def test_no_args(self) -> None:
        result = runner.invoke(app, [])
        assert result.exit_code == 2


# ======================================================================
# Verbose flag
# ======================================================================


class TestVerbose:
    def test_verbose_does_not_crash(self, tmp_path: Path) -> None:
        f = _xml_file(tmp_path)
        result = runner.invoke(app, [str(f), "--verbose"])
        assert result.exit_code == 0


# ======================================================================
# Encoding
# ======================================================================


class TestEncoding:
    def test_latin1(self, tmp_path: Path) -> None:
        content = "<root><val>\u00e9</val></root>"
        p = tmp_path / "latin.xml"
        p.write_bytes(content.encode("latin-1"))
        result = runner.invoke(app, [str(p), "--encoding", "latin-1", "--pretty"])
        assert result.exit_code == 0
        assert "\u00e9" in result.output


# ======================================================================
# Catalog sample (matches legacy behaviour)
# ======================================================================


class TestLegacyCompat:
    def test_catalog_attributes(self, tmp_path: Path) -> None:
        f = _xml_file(tmp_path, content=CATALOG_XML, name="catalog.xml")
        result = runner.invoke(app, [str(f), "--pretty"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        book = data["catalog"]["book"]
        assert book["@id"] == "bk101"
        assert book["author"] == "Gambardella, Matthew"
