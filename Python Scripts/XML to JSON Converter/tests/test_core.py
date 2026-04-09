"""Unit tests for core conversion logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from convert_xml_to_json.core import (
    convert_xml_to_json,
    dict_to_json,
    read_xml_file,
    write_json_file,
    xml_to_dict,
)

# ---------------------------------------------------------------------------
# Sample XML fixtures
# ---------------------------------------------------------------------------

SIMPLE_XML = "<root><name>Alice</name><age>30</age></root>"

CATALOG_XML = """\
<?xml version="1.0"?>
<catalog>
  <book id="bk101">
    <author>Gambardella, Matthew</author>
    <title>XML Developer's Guide</title>
    <price>44.95</price>
  </book>
  <book id="bk102">
    <author>Ralls, Kim</author>
    <title>Midnight Rain</title>
    <price>5.95</price>
  </book>
</catalog>
"""

EMPTY_ROOT_XML = "<root />"

ATTRIBUTES_XML = '<item key="k1" value="v1" />'

NESTED_XML = """\
<a>
  <b>
    <c>deep</c>
  </b>
</a>
"""

UNICODE_XML = "<msg>Hello \u00e9\u00e0\u00fc \u4e16\u754c</msg>"

CDATA_XML = "<note><body><![CDATA[Some <raw> & unescaped text]]></body></note>"


# ======================================================================
# xml_to_dict
# ======================================================================


class TestXmlToDict:
    def test_simple(self) -> None:
        result = xml_to_dict(SIMPLE_XML)
        assert result["root"]["name"] == "Alice"
        assert result["root"]["age"] == "30"

    def test_catalog_with_list(self) -> None:
        result = xml_to_dict(CATALOG_XML)
        books = result["catalog"]["book"]
        assert isinstance(books, list)
        assert len(books) == 2
        assert books[0]["@id"] == "bk101"

    def test_empty_root(self) -> None:
        result = xml_to_dict(EMPTY_ROOT_XML)
        assert result["root"] is None

    def test_attributes(self) -> None:
        result = xml_to_dict(ATTRIBUTES_XML)
        assert result["item"]["@key"] == "k1"
        assert result["item"]["@value"] == "v1"

    def test_nested(self) -> None:
        result = xml_to_dict(NESTED_XML)
        assert result["a"]["b"]["c"] == "deep"

    def test_unicode(self) -> None:
        result = xml_to_dict(UNICODE_XML)
        assert "\u00e9" in result["msg"]
        assert "\u4e16" in result["msg"]

    def test_cdata(self) -> None:
        result = xml_to_dict(CDATA_XML)
        assert "<raw>" in result["note"]["body"]

    def test_bytes_input(self) -> None:
        xml_bytes = SIMPLE_XML.encode("utf-8")
        result = xml_to_dict(xml_bytes)
        assert result["root"]["name"] == "Alice"

    def test_invalid_xml_raises(self) -> None:
        with pytest.raises(ValueError, match="Failed to parse XML"):
            xml_to_dict("<unclosed>")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            xml_to_dict("")


# ======================================================================
# dict_to_json
# ======================================================================


class TestDictToJson:
    def test_compact(self) -> None:
        result = dict_to_json({"a": 1}, indent=None)
        assert result == '{"a": 1}'

    def test_pretty(self) -> None:
        result = dict_to_json({"a": 1}, indent=2)
        assert "\n" in result
        assert '  "a": 1' in result

    def test_sort_keys(self) -> None:
        result = dict_to_json({"b": 2, "a": 1}, sort_keys=True, indent=None)
        assert result.index('"a"') < result.index('"b"')

    def test_unicode_preserved(self) -> None:
        result = dict_to_json({"msg": "\u00e9"}, indent=None)
        assert "\u00e9" in result
        assert "\\u00e9" not in result  # ensure_ascii=False


# ======================================================================
# convert_xml_to_json (end-to-end)
# ======================================================================


class TestConvertXmlToJson:
    def test_simple(self) -> None:
        result = convert_xml_to_json(SIMPLE_XML, indent=2)
        assert '"name": "Alice"' in result

    def test_compact(self) -> None:
        result = convert_xml_to_json(SIMPLE_XML, indent=None)
        assert "\n" not in result

    def test_catalog(self) -> None:
        result = convert_xml_to_json(CATALOG_XML, indent=2)
        assert '"@id": "bk101"' in result
        assert '"@id": "bk102"' in result

    def test_custom_indent(self) -> None:
        result = convert_xml_to_json(SIMPLE_XML, indent=4)
        assert '    "root"' in result

    def test_sort_keys(self) -> None:
        result = convert_xml_to_json(SIMPLE_XML, indent=None, sort_keys=True)
        assert result.index('"age"') < result.index('"name"')

    def test_invalid_xml(self) -> None:
        with pytest.raises(ValueError):
            convert_xml_to_json("not xml at all <<<")


# ======================================================================
# File helpers
# ======================================================================


class TestReadXmlFile:
    def test_reads_file(self, tmp_path: Path) -> None:
        p = tmp_path / "test.xml"
        p.write_text(SIMPLE_XML, encoding="utf-8")
        text = read_xml_file(p)
        assert "<root>" in text

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="File not found"):
            read_xml_file(tmp_path / "nope.xml")

    def test_not_a_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Not a file"):
            read_xml_file(tmp_path)

    def test_custom_encoding(self, tmp_path: Path) -> None:
        p = tmp_path / "latin.xml"
        content = "<root><val>\u00e9</val></root>"
        p.write_bytes(content.encode("latin-1"))
        text = read_xml_file(p, encoding="latin-1")
        assert "\u00e9" in text


class TestWriteJsonFile:
    def test_writes_file(self, tmp_path: Path) -> None:
        p = tmp_path / "out.json"
        write_json_file(p, '{"a": 1}')
        assert p.exists()
        assert '"a": 1' in p.read_text(encoding="utf-8")

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "sub" / "dir" / "out.json"
        write_json_file(p, "{}")
        assert p.exists()

    def test_trailing_newline(self, tmp_path: Path) -> None:
        p = tmp_path / "out.json"
        write_json_file(p, '{"x": 1}')
        raw = p.read_text(encoding="utf-8")
        assert raw.endswith("\n")
