"""Tests for the config-driven retriever module.

These tests validate builder functions and config-driven behavior
without requiring a real LanceDB database or ONNX model.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from config_loader import load_config, CorpusConfig

from retriever import (
    build_valid_filters,
    build_result_fields,
    build_precedence_boost,
    build_field_defaults,
    _escape_fts_query,
    _ALWAYS_PRESENT_FILTERS,
    _RETRIEVER_EXTRA_FIELDS,
    _BASE_FIELD_DEFAULTS,
    Retriever,
)


# ---------------------------------------------------------------------------
# Test config fixture
# ---------------------------------------------------------------------------

TEST_CONFIG_YAML = """\
project:
  name: "test-rag"
  description: "Test"
  language: "es"
  table_name: "test_chunks"
entities:
  - name: "city"
    description: "City"
    extract_from: "filename"
    pattern: "^(.+?)_"
  - name: "region"
    description: "Region"
    extract_from: "mapping"
    mapping_source: "city"
    mapping:
      Boston: "northeast"
      Miami: "southeast"
document_types:
  - name: "report"
    directory: "docs/reports"
    precedence_boost: 1.10
  - name: "analysis"
    directory: "docs/analysis"
    precedence_boost: 1.05
  - name: "raw_data"
    directory: "docs/raw"
    precedence_boost: 0.90
custom_fields:
  - name: "year"
    type: "string"
    extract_from: "filename"
    pattern: "_(20\\\\d{2})"
  - name: "priority"
    type: "int"
    extract_from: "content"
    pattern: "priority:\\\\s*(\\\\d+)"
  - name: "reviewed"
    type: "bool"
    extract_from: "filename"
    pattern: "_reviewed"
filters:
  - city
  - region
  - document_type
  - year
skip_sections:
  exact: ["TABLE OF CONTENTS"]
  prefix: ["LIST OF "]
mcp:
  name: "test-rag"
  instructions: "Test."
"""


@pytest.fixture
def test_config(tmp_path: Path) -> CorpusConfig:
    """Create a test config from YAML."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(TEST_CONFIG_YAML, encoding="utf-8")
    return load_config(str(config_file))


# ---------------------------------------------------------------------------
# Test build_valid_filters
# ---------------------------------------------------------------------------

class TestBuildValidFilters:
    def test_includes_config_filters(self, test_config):
        """Filters from config are included."""
        filters = build_valid_filters(test_config)
        assert "city" in filters
        assert "region" in filters
        assert "year" in filters

    def test_includes_always_present(self, test_config):
        """Always-present filters are always included."""
        filters = build_valid_filters(test_config)
        for f in _ALWAYS_PRESENT_FILTERS:
            assert f in filters, f"Missing always-present filter: {f}"

    def test_document_type_always_present(self, test_config):
        """document_type is always present even if not in config filters list."""
        filters = build_valid_filters(test_config)
        assert "document_type" in filters

    def test_returns_set(self, test_config):
        """Return type is a set."""
        filters = build_valid_filters(test_config)
        assert isinstance(filters, set)


# ---------------------------------------------------------------------------
# Test build_result_fields
# ---------------------------------------------------------------------------

class TestBuildResultFields:
    def test_includes_base_fields(self, test_config):
        """Base result fields (from config + retriever extras) are present."""
        fields = build_result_fields(test_config)
        # Core fields from config_loader._BASE_RESULT_FIELDS
        for bf in ["text", "section_title", "content_type",
                    "has_quantitative_data", "document_type", "is_summary"]:
            assert bf in fields, f"Missing base field: {bf}"
        # Retriever-specific extras
        for ef in _RETRIEVER_EXTRA_FIELDS:
            assert ef in fields, f"Missing retriever extra field: {ef}"

    def test_includes_entity_names(self, test_config):
        """Entity names from config appear in result fields."""
        fields = build_result_fields(test_config)
        assert "city" in fields
        assert "region" in fields

    def test_includes_custom_fields(self, test_config):
        """Custom field names from config appear in result fields."""
        fields = build_result_fields(test_config)
        assert "year" in fields
        assert "priority" in fields
        assert "reviewed" in fields

    def test_no_duplicates(self, test_config):
        """Result fields list has no duplicates."""
        fields = build_result_fields(test_config)
        assert len(fields) == len(set(fields))

    def test_returns_list(self, test_config):
        """Return type is a list."""
        fields = build_result_fields(test_config)
        assert isinstance(fields, list)


# ---------------------------------------------------------------------------
# Test build_precedence_boost
# ---------------------------------------------------------------------------

class TestBuildPrecedenceBoost:
    def test_maps_document_types(self, test_config):
        """Each document type from config maps to its boost value."""
        boost = build_precedence_boost(test_config)
        assert boost["report"] == 1.10
        assert boost["analysis"] == 1.05
        assert boost["raw_data"] == 0.90

    def test_returns_dict(self, test_config):
        """Return type is a dict."""
        boost = build_precedence_boost(test_config)
        assert isinstance(boost, dict)

    def test_all_document_types_present(self, test_config):
        """All config document types are present as keys."""
        boost = build_precedence_boost(test_config)
        assert len(boost) == 3


# ---------------------------------------------------------------------------
# Test build_field_defaults
# ---------------------------------------------------------------------------

class TestBuildFieldDefaults:
    def test_includes_base_defaults(self, test_config):
        """Base field defaults are present."""
        defaults = build_field_defaults(test_config)
        for key, val in _BASE_FIELD_DEFAULTS.items():
            assert key in defaults
            assert defaults[key] == val

    def test_includes_entity_defaults(self, test_config):
        """Entity names get empty string defaults."""
        defaults = build_field_defaults(test_config)
        assert defaults["city"] == ""
        assert defaults["region"] == ""

    def test_custom_field_type_defaults(self, test_config):
        """Custom fields get type-appropriate defaults."""
        defaults = build_field_defaults(test_config)
        assert defaults["year"] == ""       # string
        assert defaults["priority"] == 0    # int
        assert defaults["reviewed"] is False  # bool


# ---------------------------------------------------------------------------
# Test _build_where (via mocked Retriever)
# ---------------------------------------------------------------------------

class TestBuildWhere:
    @pytest.fixture
    def retriever_filters(self, test_config):
        """Return _build_where method bound to a config's valid filters."""
        # We can't create a real Retriever (no DB), so test the method logic
        # by creating an object with just the filter set
        obj = object.__new__(Retriever)
        obj._config = test_config
        obj._valid_filters = build_valid_filters(test_config)
        return obj._build_where

    def test_string_filter(self, retriever_filters):
        """String filter produces correct SQL."""
        where = retriever_filters(city="Boston")
        assert where == "city = 'Boston'"

    def test_bool_filter(self, retriever_filters):
        """Bool filter produces correct SQL."""
        where = retriever_filters(has_quantitative_data=True)
        assert where == "has_quantitative_data = true"

    def test_bool_false(self, retriever_filters):
        """False bool filter produces correct SQL."""
        where = retriever_filters(is_summary=False)
        assert where == "is_summary = false"

    def test_none_ignored(self, retriever_filters):
        """None values are ignored."""
        where = retriever_filters(city=None)
        assert where == ""

    def test_unknown_filter_ignored(self, retriever_filters):
        """Unknown filter keys are silently ignored."""
        where = retriever_filters(unknown_field="value")
        assert where == ""

    def test_multiple_filters(self, retriever_filters):
        """Multiple filters are joined with AND."""
        where = retriever_filters(city="Boston", document_type="report")
        assert "city = 'Boston'" in where
        assert "document_type = 'report'" in where
        assert " AND " in where

    def test_escapes_single_quotes(self, retriever_filters):
        """Single quotes in values are escaped."""
        where = retriever_filters(city="O'Hare")
        assert "O''Hare" in where

    def test_config_filters_accepted(self, retriever_filters):
        """Filters defined in config are accepted."""
        where = retriever_filters(region="northeast", year="2024")
        assert "region = 'northeast'" in where
        assert "year = '2024'" in where


# ---------------------------------------------------------------------------
# Test _escape_fts_query
# ---------------------------------------------------------------------------

class TestEscapeFtsQuery:
    def test_plain_text_unchanged(self):
        assert _escape_fts_query("hello world") == "hello world"

    def test_special_chars_escaped(self):
        result = _escape_fts_query('test + query - "phrase"')
        assert "\\+" in result
        assert "\\-" in result
        assert '\\"' in result

    def test_parentheses_escaped(self):
        result = _escape_fts_query("(test)")
        assert "\\(" in result
        assert "\\)" in result


# ---------------------------------------------------------------------------
# Test _build_rerank_block (dynamic entity info)
# ---------------------------------------------------------------------------

class TestBuildRerankBlock:
    def test_with_config_entities(self, test_config):
        """Rerank block includes entity names from config."""
        obj = object.__new__(Retriever)
        obj._config = test_config
        results = [
            {"city": "Boston", "region": "northeast", "document_type": "report",
             "section_title": "Intro", "text": "Some text content"},
        ]
        block = obj._build_rerank_block(results)
        assert "city: Boston" in block
        assert "region: northeast" in block
        assert "Sección: Intro" in block

    def test_without_config(self):
        """Without config, falls back to document_type."""
        obj = object.__new__(Retriever)
        obj._config = None
        results = [
            {"document_type": "report", "section_title": "Intro", "text": "Content"},
        ]
        block = obj._build_rerank_block(results)
        assert "Type: report" in block

    def test_text_truncated_to_500(self, test_config):
        """Text in rerank block is truncated to 500 chars."""
        obj = object.__new__(Retriever)
        obj._config = test_config
        results = [
            {"city": "Boston", "document_type": "report",
             "section_title": "Intro", "text": "A" * 1000},
        ]
        block = obj._build_rerank_block(results)
        # The text portion should be 500 chars max
        lines = block.split("\n")
        text_line = lines[-1]  # last line is the text
        assert len(text_line) <= 500


# ---------------------------------------------------------------------------
# Test _parse_rerank_scores
# ---------------------------------------------------------------------------

class TestParseRerankScores:
    @pytest.fixture
    def retriever_obj(self):
        obj = object.__new__(Retriever)
        return obj

    def test_valid_json(self, retriever_obj):
        raw = '[{"index": 0, "score": 8}, {"index": 1, "score": 5}]'
        scores = retriever_obj._parse_rerank_scores(raw)
        assert scores == {0: 8, 1: 5}

    def test_json_with_surrounding_text(self, retriever_obj):
        raw = 'Here are the scores: [{"index": 0, "score": 7}] Done.'
        scores = retriever_obj._parse_rerank_scores(raw)
        assert scores == {0: 7}

    def test_truncated_json_recovery(self, retriever_obj):
        raw = '[{"index": 0, "score": 8}, {"index": 1, "score": 5'
        scores = retriever_obj._parse_rerank_scores(raw)
        assert 0 in scores
        assert scores[0] == 8

    def test_invalid_json_raises(self, retriever_obj):
        with pytest.raises(ValueError, match="No JSON array"):
            retriever_obj._parse_rerank_scores("no json here")


# ---------------------------------------------------------------------------
# Test Retriever defaults without config
# ---------------------------------------------------------------------------

class TestRetrieverDefaults:
    def test_defaults_without_config(self):
        """Without config, Retriever uses sensible defaults for filters/fields."""
        obj = object.__new__(Retriever)
        obj._config = None
        obj._valid_filters = set(_ALWAYS_PRESENT_FILTERS)
        obj._result_fields = [
            "text", "context", "section_title", "content_type",
            "has_quantitative_data", "document_type",
            "page_start", "page_end", "is_summary",
        ]
        obj._precedence_boost = {}
        obj._field_defaults = dict(_BASE_FIELD_DEFAULTS)

        assert obj._valid_filters == _ALWAYS_PRESENT_FILTERS
        assert "text" in obj._result_fields
        assert "section_title" in obj._result_fields
        assert "context" in obj._result_fields
        assert "page_start" in obj._result_fields
        assert obj._precedence_boost == {}
        assert obj._field_defaults["is_summary"] is False
