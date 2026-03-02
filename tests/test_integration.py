"""Integration test: config -> indexer -> chunk extraction pipeline."""
import pytest
from pathlib import Path


INTEGRATION_CONFIG = """\
project:
  name: "integration-test"
  description: "Integration test corpus"
  language: "en"
  table_name: "test_chunks"

entities:
  - name: "author"
    description: "Document author"
    extract_from: "filename"
    pattern: "^(.+?)_"
  - name: "department"
    description: "Department"
    extract_from: "mapping"
    mapping_source: "author"
    mapping:
      Alice: "Engineering"
      Bob: "Marketing"

document_types:
  - name: "memo"
    directory: "{docs_dir}"
    precedence_boost: 1.0

custom_fields:
  - name: "year"
    type: "string"
    extract_from: "filename"
    pattern: '_(20\\d{2})_'

filters:
  - author
  - department
  - document_type
  - year

skip_sections:
  exact: ["TABLE OF CONTENTS"]
  prefix: ["LIST OF "]

mcp:
  name: "integration-test"
  instructions: "Test server with {{entity_count}} authors."
"""

# Longer document content to exceed MIN_CHUNK_TOKENS (100 tokens ~ 77 words)
ALICE_DOC = """\
## Project Update

The engineering team has completed the alpha release of the new platform.
Performance benchmarks show a 2x improvement over the previous baseline
measurements. Next steps include beta testing with 50 users across three
different offices and collecting detailed feedback from stakeholders across
all departments. The team has also identified several areas for optimization
in the database layer that could yield an additional 30% improvement in
query response times. Load testing confirmed the system handles 10,000
concurrent connections without degradation. Memory usage remains stable at
under 2GB even under peak load conditions. The deployment pipeline has been
fully automated with zero-downtime releases now standard practice.

## Timeline

Alpha release was completed as of January 2024 and all milestones were met
on schedule. Beta testing is scheduled to begin in Q2 2024 with a target of
50 external users participating in the program. General availability is
targeted for Q3 2024 pending successful completion of the beta phase and
resolution of any critical issues identified during testing. The team has
allocated additional resources for Q2 to ensure the beta launch proceeds
smoothly. Documentation updates will be completed by the end of March and
training materials will be distributed to all support staff by mid-April.
The rollback plan has been tested and validated by the operations team.
"""

BOB_DOC = """\
## Marketing Campaign Results

The Q4 campaign reached 15,000 people across all digital channels including
social media, email, and paid search. Conversion rate improved to 3.2% from
the previous quarter's 2.1%, representing a significant improvement in
campaign effectiveness. Total revenue attributed to marketing efforts was
$450,000 for the quarter, exceeding the target by 12%. Email campaigns
performed particularly well with a 4.5% click-through rate. Social media
engagement increased by 35% compared to Q3. The paid search campaigns
delivered a return on ad spend of 3.8x. Customer acquisition cost decreased
to $42 per customer from $58 in the previous quarter. Overall brand awareness
metrics improved by 18% according to our quarterly survey of 2,500
respondents across all target demographics and geographic regions.

## Recommendations

Based on the strong Q4 results, we recommend increasing the budget allocation
for social media advertising by 25% in the upcoming fiscal year. The data
clearly shows that social channels deliver the highest engagement rates and
best return on investment. We should also expand to two new markets in the
southeast region where our competitor analysis shows significant untapped
demand. Additional investment in content marketing would support long-term
organic growth. The team recommends hiring two additional content creators
and one data analyst to support the expanded campaign scope. A/B testing
infrastructure should be upgraded to support more concurrent experiments.
"""


@pytest.fixture
def setup(tmp_path):
    """Create sample docs and config."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "Alice_memo_2024_Q1.md").write_text(ALICE_DOC)
    (docs_dir / "Bob_report_2024_annual.md").write_text(BOB_DOC)

    config_text = INTEGRATION_CONFIG.replace("{docs_dir}", str(docs_dir).replace("\\", "/"))
    config_file = tmp_path / "corpus_config.yaml"
    config_file.write_text(config_text)

    return config_file, docs_dir


def test_config_loads(setup):
    from config_loader import load_config
    config_file, _ = setup
    cfg = load_config(str(config_file))
    assert cfg.project.name == "integration-test"
    assert len(cfg.entities) == 2
    assert len(cfg.document_types) == 1


def test_entity_extraction(setup):
    from config_loader import load_config
    config_file, _ = setup
    cfg = load_config(str(config_file))
    entities = cfg.extract_entities("Alice_memo_2024_Q1")
    assert entities["author"] == "Alice"
    assert entities["department"] == "Engineering"


def test_process_document_pipeline(setup):
    from config_loader import load_config
    from indexer import process_document
    config_file, docs_dir = setup
    cfg = load_config(str(config_file))

    chunks = process_document(
        str(docs_dir / "Alice_memo_2024_Q1.md"), cfg, document_type="memo"
    )
    assert len(chunks) > 0
    assert chunks[0].entity_values["author"] == "Alice"
    assert chunks[0].entity_values["department"] == "Engineering"
    assert chunks[0].document_type == "memo"


def test_quantitative_detection(setup):
    from config_loader import load_config
    from indexer import process_document
    config_file, docs_dir = setup
    cfg = load_config(str(config_file))

    chunks = process_document(
        str(docs_dir / "Bob_report_2024_annual.md"), cfg, document_type="memo"
    )
    assert len(chunks) > 0
    # Bob's report has $450,000 and percentages
    quant_chunks = [c for c in chunks if c.has_quantitative_data]
    assert len(quant_chunks) > 0


def test_unknown_entity_graceful(setup):
    from config_loader import load_config
    from indexer import process_document
    config_file, docs_dir = setup
    cfg = load_config(str(config_file))

    # Create doc with unknown author — content must exceed MIN_CHUNK_TOKENS
    unknown = docs_dir / "Charlie_notes_2024.md"
    unknown.write_text(
        "## Notes\n\n"
        + "Some notes from Charlie about the project and its various components. " * 15
    )

    chunks = process_document(str(unknown), cfg, document_type="memo")
    assert len(chunks) > 0
    assert chunks[0].entity_values["author"] == "Charlie"
    assert chunks[0].entity_values.get("department", "") == ""  # no mapping for Charlie


def test_skip_sections(setup):
    from config_loader import load_config
    from indexer import process_document
    config_file, docs_dir = setup
    cfg = load_config(str(config_file))

    doc_with_toc = docs_dir / "Alice_toc_test_2024.md"
    doc_with_toc.write_text(
        "## TABLE OF CONTENTS\n\nPage 1\nPage 2\n\n"
        "## Real Content\n\n"
        + "This is the actual content that should be indexed into the system. " * 20
    )

    chunks = process_document(str(doc_with_toc), cfg, document_type="memo")
    section_titles = [c.section_title for c in chunks]
    assert "TABLE OF CONTENTS" not in section_titles


def test_config_result_fields(setup):
    from config_loader import load_config
    config_file, _ = setup
    cfg = load_config(str(config_file))
    fields = cfg.result_fields()
    assert "text" in fields
    assert "author" in fields
    assert "department" in fields
    assert "year" in fields


def test_config_valid_filters(setup):
    from config_loader import load_config
    config_file, _ = setup
    cfg = load_config(str(config_file))
    filters = cfg.valid_filters()
    assert "author" in filters
    assert "department" in filters
    assert "document_type" in filters
    assert "content_type" in filters  # always present
