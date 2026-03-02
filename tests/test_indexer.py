"""Tests for the config-driven indexer module."""

import pytest
import textwrap
from pathlib import Path

from config_loader import load_config, CorpusConfig

# We import after ensuring the config_loader is available
from indexer import (
    estimate_tokens,
    detect_content_type,
    has_quantitative,
    should_skip,
    clean_markdown,
    parse_document,
    is_flat_doc,
    Section,
    Chunk,
    chunk_sections,
    process_document,
    postprocess_chunks,
    _extract_page_range,
    _split_paragraphs,
    _build_context_line,
    MIN_CHUNK_TOKENS,
    MAX_SECTION_TOKENS,
    TOKENS_PER_WORD,
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
filters:
  - city
  - region
  - document_type
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


@pytest.fixture
def sample_md_file(tmp_path: Path) -> Path:
    """Create a sample markdown file for testing."""
    content = textwrap.dedent("""\
        ## Introduction

        This is the introduction section with enough content to be a valid chunk.
        It contains several sentences about the city of Boston and its surrounding areas.
        The population is approximately 685,000 residents living in the metropolitan area.
        There are many neighborhoods and districts that make up the urban landscape.
        Public transportation includes the MBTA subway and bus system.
        The city is known for its historical significance and educational institutions.

        ## Demographics

        ### Population

        The total population is 685,000 personas according to the latest census.
        The city has seen steady growth over the past decade with 15% increase.
        Many families have moved to the surrounding suburbs for more affordable housing.
        The median household income is $75,000 with significant variation by neighborhood.

        ### Housing

        There are approximately 280,000 viviendas in the metro area.
        Housing costs have increased by 25% in the last five years.
        New construction projects are underway in several neighborhoods.
        The average rent for a two-bedroom apartment is $2,500 per month.

        ## TABLE OF CONTENTS

        This should be skipped entirely by the chunker.

        ## LIST OF FIGURES

        Figure 1. Map of the area.
        Figure 2. Population chart.
    """)
    md_file = tmp_path / "Boston_Report.md"
    md_file.write_text(content, encoding="utf-8")
    return md_file


# ---------------------------------------------------------------------------
# Test estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_single_word(self):
        assert estimate_tokens("hello") == int(1 * TOKENS_PER_WORD)

    def test_multiple_words(self):
        text = "one two three four five"
        expected = int(5 * TOKENS_PER_WORD)
        assert estimate_tokens(text) == expected

    def test_spanish_text(self):
        text = "La comunidad tiene aproximadamente quinientos residentes"
        words = text.split()
        expected = int(len(words) * TOKENS_PER_WORD)
        assert estimate_tokens(text) == expected


# ---------------------------------------------------------------------------
# Test detect_content_type
# ---------------------------------------------------------------------------

class TestDetectContentType:
    def test_prose(self):
        text = "This is a simple paragraph.\nAnother sentence here.\nAnd one more."
        assert detect_content_type(text) == "prose"

    def test_table(self):
        text = "| Col1 | Col2 |\n| --- | --- |\n| val1 | val2 |\n| val3 | val4 |\n| val5 | val6 |"
        assert detect_content_type(text) == "table"

    def test_list(self):
        text = "- item one\n- item two\n- item three\n- item four\n- item five"
        assert detect_content_type(text) == "list"

    def test_mixed(self):
        text = "A regular paragraph.\n| Col | Val |\nAnother line.\nMore text."
        assert detect_content_type(text) == "mixed"

    def test_empty(self):
        assert detect_content_type("") == "prose"

    def test_numbered_list(self):
        text = "1. first\n2. second\n3. third\n4. fourth\n5. fifth"
        assert detect_content_type(text) == "list"


# ---------------------------------------------------------------------------
# Test has_quantitative
# ---------------------------------------------------------------------------

class TestHasQuantitative:
    def test_percentage(self):
        assert has_quantitative("The rate is 15%") is True

    def test_dollar_amount(self):
        assert has_quantitative("Cost is $5,000") is True

    def test_population_count(self):
        assert has_quantitative("There are 500 personas in the area") is True

    def test_no_quantitative(self):
        assert has_quantitative("A simple text with no numbers") is False

    def test_large_number(self):
        assert has_quantitative("Revenue of 1,000,000") is True


# ---------------------------------------------------------------------------
# Test parse_document
# ---------------------------------------------------------------------------

class TestParseDocument:
    def test_hierarchical_parsing(self):
        text = textwrap.dedent("""\
            ## Section One

            Content for section one.

            ### Subsection A

            Content for subsection A.

            ## Section Two

            Content for section two.
        """)
        sections = parse_document(text)
        assert len(sections) == 2
        assert sections[0].title == "Section One"
        assert sections[0].level == 2
        assert len(sections[0].subsections) == 1
        assert sections[0].subsections[0].title == "Subsection A"
        assert sections[1].title == "Section Two"

    def test_flat_parsing(self):
        # Only # headers (flat PDF conversion)
        text = "# Title\n\n" + "Word " * 50 + "\n\n# Another\n\n" + "More text. " * 50
        assert is_flat_doc(text) is True
        sections = parse_document(text)
        assert len(sections) >= 1

    def test_is_flat_doc(self):
        assert is_flat_doc("# Only h1 headers\nsome text") is True
        assert is_flat_doc("## Has h2 header\nsome text") is False

    def test_empty_document(self):
        sections = parse_document("")
        assert sections == []


# ---------------------------------------------------------------------------
# Test should_skip with config
# ---------------------------------------------------------------------------

class TestShouldSkip:
    def test_exact_match(self, test_config):
        assert should_skip("TABLE OF CONTENTS", test_config) is True

    def test_exact_match_case_insensitive(self, test_config):
        assert should_skip("table of contents", test_config) is True

    def test_prefix_match(self, test_config):
        assert should_skip("LIST OF FIGURES", test_config) is True

    def test_prefix_match_case_insensitive(self, test_config):
        assert should_skip("list of tables", test_config) is True

    def test_no_match(self, test_config):
        assert should_skip("Introduction", test_config) is False

    def test_delegates_to_config(self, test_config):
        """Verify should_skip delegates to config.should_skip_section."""
        # This tests the exact same behavior, proving delegation works
        assert should_skip("TABLE OF CONTENTS", test_config) == test_config.should_skip_section("TABLE OF CONTENTS")
        assert should_skip("Introduction", test_config) == test_config.should_skip_section("Introduction")


# ---------------------------------------------------------------------------
# Test chunk_sections
# ---------------------------------------------------------------------------

class TestChunkSections:
    def test_produces_chunks_with_entity_values(self, test_config):
        """chunk_sections fills entity_values on each chunk."""
        sections = [
            Section(
                title="Overview",
                level=2,
                content="Word " * 100,  # ~130 tokens, above MIN_CHUNK_TOKENS
            ),
        ]
        entity_values = {"city": "Boston", "region": "northeast"}
        chunks = chunk_sections(sections, entity_values, "test.md", test_config)

        assert len(chunks) >= 1
        for c in chunks:
            assert c.entity_values == {"city": "Boston", "region": "northeast"}
            assert "city: Boston" in c.text
            assert "region: northeast" in c.text

    def test_skips_configured_sections(self, test_config):
        """Sections matching skip config are excluded."""
        sections = [
            Section(title="TABLE OF CONTENTS", level=2, content="Word " * 100),
            Section(title="Introduction", level=2, content="Word " * 100),
        ]
        chunks = chunk_sections(sections, {}, "test.md", test_config)
        titles = [c.section_title for c in chunks]
        assert "TABLE OF CONTENTS" not in titles
        assert "Introduction" in titles

    def test_skips_small_sections(self, test_config):
        """Sections below MIN_CHUNK_TOKENS are skipped."""
        sections = [
            Section(title="Tiny", level=2, content="Short."),
        ]
        chunks = chunk_sections(sections, {}, "test.md", test_config)
        assert len(chunks) == 0

    def test_context_line_in_chunks(self, test_config):
        """Entity values appear as context line prefix in chunk text."""
        sections = [
            Section(title="Data", level=2, content="Word " * 100),
        ]
        entity_values = {"city": "Miami"}
        chunks = chunk_sections(sections, entity_values, "test.md", test_config)
        assert chunks[0].text.startswith("city: Miami\n\n")

    def test_empty_entity_values_no_context_line(self, test_config):
        """With no entity values, no context line prefix."""
        sections = [
            Section(title="Data", level=2, content="Word " * 100),
        ]
        chunks = chunk_sections(sections, {}, "test.md", test_config)
        assert chunks[0].text.startswith("## Data\n\n")


# ---------------------------------------------------------------------------
# Test process_document with config
# ---------------------------------------------------------------------------

class TestProcessDocument:
    def test_entity_extraction(self, test_config, sample_md_file):
        """process_document extracts entities from filename via config."""
        chunks = process_document(str(sample_md_file), test_config, document_type="report")

        assert len(chunks) > 0
        # The filename is "Boston_Report.md", pattern "^(.+?)_" should capture "Boston"
        for c in chunks:
            assert c.entity_values.get("city") == "Boston"
            # "Boston" maps to "northeast" via the mapping entity
            assert c.entity_values.get("region") == "northeast"

    def test_document_type_assigned(self, test_config, sample_md_file):
        """process_document sets document_type on all chunks."""
        chunks = process_document(str(sample_md_file), test_config, document_type="report")
        for c in chunks:
            assert c.document_type == "report"

    def test_unknown_entity_graceful(self, test_config, tmp_path):
        """Unknown entity values result in empty dict entries (graceful fallback)."""
        filler = " ".join(["word"] * 120)  # ~156 tokens, well above MIN_CHUNK_TOKENS
        content = f"## Section\n\n{filler}\n"
        # Filename "Unknown_Thing.md" - "Unknown" won't be in region mapping
        md_file = tmp_path / "Unknown_Thing.md"
        md_file.write_text(content, encoding="utf-8")

        chunks = process_document(str(md_file), test_config, document_type="report")
        assert len(chunks) > 0
        # "Unknown" extracted as city, but no mapping to region
        for c in chunks:
            assert c.entity_values.get("city") == "Unknown"
            assert "region" not in c.entity_values  # mapping failed, so not present

    def test_skips_toc_section(self, test_config, sample_md_file):
        """TABLE OF CONTENTS section is skipped per config."""
        chunks = process_document(str(sample_md_file), test_config, document_type="report")
        section_titles = [c.section_title for c in chunks]
        assert "TABLE OF CONTENTS" not in section_titles

    def test_skips_list_prefix_section(self, test_config, sample_md_file):
        """LIST OF FIGURES section is skipped per config prefix rule."""
        chunks = process_document(str(sample_md_file), test_config, document_type="report")
        section_titles = [c.section_title for c in chunks]
        # "LIST OF FIGURES" should match the "LIST OF " prefix skip rule
        assert not any("LIST OF" in t for t in section_titles)


# ---------------------------------------------------------------------------
# Test build_context_line
# ---------------------------------------------------------------------------

class TestBuildContextLine:
    def test_with_values(self):
        line = _build_context_line({"city": "Boston", "region": "northeast"})
        assert "city: Boston" in line
        assert "region: northeast" in line
        assert line.endswith("\n\n")

    def test_empty_values(self):
        assert _build_context_line({}) == ""

    def test_skips_empty_string_values(self):
        line = _build_context_line({"city": "Boston", "region": ""})
        assert "region" not in line
        assert "city: Boston" in line


# ---------------------------------------------------------------------------
# Test _extract_page_range
# ---------------------------------------------------------------------------

class TestExtractPageRange:
    def test_with_markers(self):
        text = "some text <!-- Page 5 --> more <!-- Page 8 --> end"
        assert _extract_page_range(text) == (5, 8)

    def test_single_marker(self):
        text = "text <!-- Page 12 --> more"
        assert _extract_page_range(text) == (12, 12)

    def test_no_markers(self):
        assert _extract_page_range("no markers here") == (-1, -1)


# ---------------------------------------------------------------------------
# Test postprocess_chunks
# ---------------------------------------------------------------------------

class TestPostprocessChunks:
    def test_reindexes_chunks(self):
        chunks = [
            Chunk(text="Word " * 200, entity_values={}, custom_values={},
                  section_title="A", section_level=2, content_type="prose",
                  has_quantitative_data=False, chunk_index=99, source_file="a.md"),
            Chunk(text="Word " * 200, entity_values={}, custom_values={},
                  section_title="B", section_level=2, content_type="prose",
                  has_quantitative_data=False, chunk_index=99, source_file="a.md"),
        ]
        result = postprocess_chunks(chunks)
        assert result[0].chunk_index == 0
        assert result[1].chunk_index == 1

    def test_empty_input(self):
        assert postprocess_chunks([]) == []

    def test_extracts_page_range(self):
        text = "<!-- Page 3 --> content here <!-- Page 5 -->"
        # Make it big enough not to be merged
        text += " word" * 200
        chunks = [
            Chunk(text=text, entity_values={}, custom_values={},
                  section_title="A", section_level=2, content_type="prose",
                  has_quantitative_data=False, chunk_index=0, source_file="a.md"),
        ]
        result = postprocess_chunks(chunks)
        assert result[0].page_start == 3
        assert result[0].page_end == 5


# ---------------------------------------------------------------------------
# Test clean_markdown
# ---------------------------------------------------------------------------

class TestCleanMarkdown:
    def test_strips_image_tags(self):
        text = "Before\n![alt text](image.png)\nAfter"
        result = clean_markdown(text)
        assert "![" not in result
        assert "Before" in result
        assert "After" in result

    def test_strips_page_numbers(self):
        text = "Content\n42\nMore content"
        result = clean_markdown(text)
        assert "\n42\n" not in result

    def test_strips_captions(self):
        text = "Content\nImagen 1. A photo description\nMore content"
        result = clean_markdown(text)
        assert "Imagen 1." not in result
