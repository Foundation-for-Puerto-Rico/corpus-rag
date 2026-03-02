"""Tests for config_loader module."""

import pytest

from config_loader import ConfigError, load_config


MINIMAL_CONFIG = """\
project:
  name: "test-rag"
  description: "Test corpus"
  language: "es"
  table_name: "test_chunks"
entities:
  - name: "category"
    description: "Document category"
    extract_from: "filename"
    pattern: "^(.+?)_"
document_types:
  - name: "report"
    directory: "docs/reports"
    precedence_boost: 1.10
custom_fields:
  - name: "year"
    type: "string"
    extract_from: "filename"
    pattern: "_(20\\\\d{2})"
filters:
  - category
  - document_type
  - content_type
skip_sections:
  exact: ["TABLE OF CONTENTS"]
  prefix: ["LIST OF "]
mcp:
  name: "test-rag"
  instructions: "Test RAG server."
"""

MAPPING_CONFIG = """\
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
  - name: "plan"
    directory: "docs/plans"
    precedence_boost: 1.20
filters:
  - city
  - region
  - document_type
skip_sections:
  exact: []
  prefix: []
mcp:
  name: "test-rag"
  instructions: "Test."
"""

MISSING_NAME_CONFIG = """\
project:
  description: "No name"
  table_name: "chunks"
"""

MISSING_TABLE_CONFIG = """\
project:
  name: "test"
"""


def _write_config(tmp_path, content: str) -> str:
    """Write YAML content to a temp file and return the path."""
    p = tmp_path / "corpus_config.yaml"
    p.write_text(content, encoding="utf-8")
    return str(p)


class TestLoadMinimalConfig:
    """Test loading a minimal valid config."""

    def test_load_project(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MINIMAL_CONFIG))
        assert cfg.project.name == "test-rag"
        assert cfg.project.description == "Test corpus"
        assert cfg.project.language == "es"
        assert cfg.project.table_name == "test_chunks"

    def test_entities_parsed(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MINIMAL_CONFIG))
        assert len(cfg.entities) == 1
        e = cfg.entities[0]
        assert e.name == "category"
        assert e.extract_from == "filename"
        assert e.pattern == "^(.+?)_"

    def test_document_types_parsed(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MINIMAL_CONFIG))
        assert len(cfg.document_types) == 1
        dt = cfg.document_types[0]
        assert dt.name == "report"
        assert dt.directory == "docs/reports"
        assert dt.precedence_boost == pytest.approx(1.10)

    def test_filters_parsed(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MINIMAL_CONFIG))
        assert "category" in cfg.filters
        assert "document_type" in cfg.filters
        assert "content_type" in cfg.filters

    def test_skip_sections(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MINIMAL_CONFIG))
        assert cfg.skip_sections.exact == ["TABLE OF CONTENTS"]
        assert cfg.skip_sections.prefix == ["LIST OF "]

    def test_mcp(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MINIMAL_CONFIG))
        assert cfg.mcp.name == "test-rag"
        assert cfg.mcp.instructions == "Test RAG server."

    def test_custom_fields_parsed(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MINIMAL_CONFIG))
        assert len(cfg.custom_fields) == 1
        cf = cfg.custom_fields[0]
        assert cf.name == "year"
        assert cf.type == "string"
        assert cf.extract_from == "filename"


class TestPrecedenceBoostMap:
    def test_returns_mapping(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MINIMAL_CONFIG))
        m = cfg.precedence_boost_map()
        assert m == {"report": pytest.approx(1.10)}

    def test_multiple_types(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MAPPING_CONFIG))
        m = cfg.precedence_boost_map()
        assert m == {"plan": pytest.approx(1.20)}


class TestDocsDir:
    def test_returns_mapping(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MINIMAL_CONFIG))
        assert cfg.docs_dirs() == {"report": "docs/reports"}


class TestShouldSkipSection:
    def test_exact_match_case_insensitive(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MINIMAL_CONFIG))
        assert cfg.should_skip_section("Table of Contents") is True
        assert cfg.should_skip_section("TABLE OF CONTENTS") is True
        assert cfg.should_skip_section("table of contents") is True

    def test_prefix_match(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MINIMAL_CONFIG))
        assert cfg.should_skip_section("List of Figures") is True
        assert cfg.should_skip_section("LIST OF TABLES") is True

    def test_no_match(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MINIMAL_CONFIG))
        assert cfg.should_skip_section("Introduction") is False
        assert cfg.should_skip_section("Some other section") is False


class TestResultFields:
    def test_includes_base_entity_custom(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MINIMAL_CONFIG))
        fields = cfg.result_fields()
        # Base fields
        assert "text" in fields
        assert "section_title" in fields
        assert "score" in fields
        assert "document_type" in fields
        # Entity
        assert "category" in fields
        # Custom field
        assert "year" in fields


class TestValidFilters:
    def test_includes_config_and_always_present(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MINIMAL_CONFIG))
        vf = cfg.valid_filters()
        # From config
        assert "category" in vf
        # Always present
        assert "content_type" in vf
        assert "has_quantitative_data" in vf
        assert "document_type" in vf
        assert "is_summary" in vf


class TestEntityMapping:
    def test_mapping_entity_resolves(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MAPPING_CONFIG))
        region_entity = cfg.get_entity("region")
        assert region_entity is not None
        assert region_entity.extract_from == "mapping"
        assert region_entity.mapping_source == "city"
        assert region_entity.mapping["Boston"] == "northeast"


class TestExtractEntities:
    def test_filename_pattern(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MINIMAL_CONFIG))
        result = cfg.extract_entities(filename="Science_report_2024.md")
        assert result["category"] == "Science"

    def test_mapping_chain(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MAPPING_CONFIG))
        result = cfg.extract_entities(filename="Boston_plan.md")
        assert result["city"] == "Boston"
        assert result["region"] == "northeast"

    def test_mapping_unknown_value(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MAPPING_CONFIG))
        result = cfg.extract_entities(filename="Chicago_plan.md")
        assert result["city"] == "Chicago"
        assert "region" not in result  # Chicago not in mapping

    def test_no_match(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MINIMAL_CONFIG))
        result = cfg.extract_entities(filename="nodash.md")
        assert result == {}


class TestExtractCustomFields:
    def test_filename_pattern(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MINIMAL_CONFIG))
        result = cfg.extract_custom_fields(filename="Report_2024_final.md")
        assert result["year"] == "2024"

    def test_no_match(self, tmp_path):
        cfg = load_config(_write_config(tmp_path, MINIMAL_CONFIG))
        result = cfg.extract_custom_fields(filename="report.md")
        assert result == {}


class TestConfigErrors:
    def test_missing_project_name(self, tmp_path):
        with pytest.raises(ConfigError, match="project.name"):
            load_config(_write_config(tmp_path, MISSING_NAME_CONFIG))

    def test_missing_table_name(self, tmp_path):
        with pytest.raises(ConfigError, match="project.table_name"):
            load_config(_write_config(tmp_path, MISSING_TABLE_CONFIG))

    def test_file_not_found(self):
        with pytest.raises(ConfigError, match="not found"):
            load_config("/nonexistent/path.yaml")

    def test_missing_project_key(self, tmp_path):
        with pytest.raises(ConfigError, match="project"):
            load_config(_write_config(tmp_path, "filters:\n  - foo\n"))
