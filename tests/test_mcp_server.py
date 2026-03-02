"""Tests for the config-driven MCP server (non-DB parts).

Since the mcp_server module performs heavy imports at module level (mcp, lancedb,
etc.), we mock those dependencies before importing the module under test.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import (
    CorpusConfig,
    DocumentTypeConfig,
    EntityConfig,
    McpConfig,
    ProjectConfig,
)


def _make_config(
    instructions: str = "",
    entities: list | None = None,
    document_types: list | None = None,
    filters: list | None = None,
) -> CorpusConfig:
    """Build a minimal CorpusConfig for testing."""
    _default_entities = [
        EntityConfig(name="city", description="City name"),
        EntityConfig(name="state", description="State name"),
    ]
    _default_doc_types = [
        DocumentTypeConfig(name="report", precedence_boost=1.2),
        DocumentTypeConfig(name="memo", precedence_boost=1.0),
    ]
    return CorpusConfig(
        project=ProjectConfig(
            name="TestCorpus",
            description="A test corpus for unit tests.",
            language="en",
            table_name="test_chunks",
        ),
        entities=_default_entities if entities is None else entities,
        document_types=_default_doc_types if document_types is None else document_types,
        filters=["city", "state"] if filters is None else filters,
        mcp=McpConfig(
            name="test-rag",
            instructions=instructions,
        ),
    )


@pytest.fixture(autouse=True)
def _mock_heavy_deps(monkeypatch, tmp_path):
    """Mock mcp, lancedb, and config loading so mcp_server can be imported
    without real dependencies."""
    # Create a fake mcp module hierarchy
    mock_mcp = MagicMock()
    mock_fastmcp_class = MagicMock()
    mock_fastmcp_instance = MagicMock()
    mock_fastmcp_class.return_value = mock_fastmcp_instance
    # Make the tool decorator a passthrough
    mock_fastmcp_instance.tool.return_value = lambda fn: fn
    mock_mcp.server.fastmcp.FastMCP = mock_fastmcp_class

    monkeypatch.setitem(sys.modules, "mcp", mock_mcp)
    monkeypatch.setitem(sys.modules, "mcp.server", mock_mcp.server)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", mock_mcp.server.fastmcp)

    # Mock lancedb
    mock_lancedb = MagicMock()
    monkeypatch.setitem(sys.modules, "lancedb", mock_lancedb)

    # Mock dotenv
    mock_dotenv = MagicMock()
    monkeypatch.setitem(sys.modules, "dotenv", mock_dotenv)

    # Write a minimal config YAML
    config_yaml = tmp_path / "corpus_config.yaml"
    config_yaml.write_text(
        "project:\n"
        "  name: TestCorpus\n"
        "  description: A test corpus\n"
        "  table_name: test_chunks\n"
        "entities:\n"
        "  - name: city\n"
        "    description: City name\n"
        "  - name: state\n"
        "    description: State name\n"
        "document_types:\n"
        "  - name: report\n"
        "    precedence_boost: 1.2\n"
        "  - name: memo\n"
        "    precedence_boost: 1.0\n"
        "filters:\n"
        "  - city\n"
        "  - state\n"
        "mcp:\n"
        "  name: test-rag\n"
        "  instructions: ''\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CORPUS_CONFIG_PATH", str(config_yaml))

    # Force reload of mcp_server to pick up mocked modules
    if "mcp_server" in sys.modules:
        del sys.modules["mcp_server"]

    yield

    # Clean up
    if "mcp_server" in sys.modules:
        del sys.modules["mcp_server"]


# ---------------------------------------------------------------------------
# build_instructions tests
# ---------------------------------------------------------------------------


class TestBuildInstructions:
    """Tests for build_instructions placeholder interpolation."""

    def test_project_name_interpolated(self):
        from mcp_server import build_instructions

        cfg = _make_config(instructions="Welcome to {project_name}.")
        result = build_instructions(cfg)
        assert result == "Welcome to TestCorpus."

    def test_project_description_interpolated(self):
        from mcp_server import build_instructions

        cfg = _make_config(instructions="About: {project_description}")
        result = build_instructions(cfg)
        assert result == "About: A test corpus for unit tests."

    def test_entity_names_interpolated(self):
        from mcp_server import build_instructions

        cfg = _make_config(instructions="Entities: {entity_names}")
        result = build_instructions(cfg)
        assert result == "Entities: city, state"

    def test_entity_count_interpolated(self):
        from mcp_server import build_instructions

        cfg = _make_config(instructions="Count: {entity_count}")
        result = build_instructions(cfg)
        assert result == "Count: 2"

    def test_filter_list_interpolated(self):
        from mcp_server import build_instructions

        cfg = _make_config(instructions="Filters: {filter_list}")
        result = build_instructions(cfg)
        assert "city" in result
        assert "state" in result
        assert "content_type" in result
        assert "document_type" in result

    def test_document_types_interpolated(self):
        from mcp_server import build_instructions

        cfg = _make_config(instructions="Types:\n{document_types}")
        result = build_instructions(cfg)
        assert "- report" in result
        assert "- memo" in result

    def test_multiple_placeholders(self):
        from mcp_server import build_instructions

        cfg = _make_config(
            instructions="{project_name} has {entity_count} entities: {entity_names}"
        )
        result = build_instructions(cfg)
        assert result == "TestCorpus has 2 entities: city, state"

    def test_empty_instructions_uses_fallback(self):
        from mcp_server import build_instructions

        cfg = _make_config(instructions="")
        result = build_instructions(cfg)
        assert "TestCorpus" in result
        assert "report" in result

    def test_no_entities_shows_none(self):
        from mcp_server import build_instructions

        cfg = _make_config(instructions="Entities: {entity_names}", entities=[])
        result = build_instructions(cfg)
        assert result == "Entities: (none)"

    def test_no_document_types_shows_none(self):
        from mcp_server import build_instructions

        cfg = _make_config(
            instructions="Types:\n{document_types}",
            document_types=[],
        )
        result = build_instructions(cfg)
        assert "(none)" in result


# ---------------------------------------------------------------------------
# Search tool tests
# ---------------------------------------------------------------------------


class TestSearchEmptyQuery:
    """Test that search returns empty list for empty queries."""

    def test_search_empty_string(self):
        import mcp_server

        result = mcp_server.search(query="")
        assert result == []

    def test_search_whitespace_only(self):
        import mcp_server

        result = mcp_server.search(query="   ")
        assert result == []

    def test_search_does_not_call_retriever(self):
        import mcp_server

        # The retriever should never be called for empty queries
        mcp_server.retriever.hybrid_search = MagicMock()
        mcp_server.retriever.semantic_search = MagicMock()
        mcp_server.retriever.fts_search = MagicMock()

        mcp_server.search(query="")

        mcp_server.retriever.hybrid_search.assert_not_called()
        mcp_server.retriever.semantic_search.assert_not_called()
        mcp_server.retriever.fts_search.assert_not_called()


class TestSearchBatchEmptyQuery:
    """Test that search_batch returns empty dict for empty queries."""

    def test_search_batch_empty_string(self):
        import mcp_server

        result = mcp_server.search_batch(
            query="", entity_name="city", entity_values=["Boston"]
        )
        assert result == {}

    def test_search_batch_whitespace(self):
        import mcp_server

        result = mcp_server.search_batch(
            query="  ", entity_name="city", entity_values=["Boston"]
        )
        assert result == {}


class TestSearchFilterValidation:
    """Test that search validates filter keys against config."""

    def test_invalid_filter_rejected(self):
        import mcp_server

        result = mcp_server.search(query="test", filters={"nonexistent_field": "value"})
        assert len(result) == 1
        assert "_error" in result[0]
        assert "nonexistent_field" in result[0]["_error"]

    def test_valid_filter_accepted(self):
        import mcp_server

        mcp_server.retriever.hybrid_search = MagicMock(return_value=[])
        mock_table = MagicMock()
        mock_table.schema.names = []
        mcp_server.retriever._table = mock_table

        result = mcp_server.search(query="test", filters={"city": "Boston"})
        assert result == []
        mcp_server.retriever.hybrid_search.assert_called_once()
