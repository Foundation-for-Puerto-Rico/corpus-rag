"""LLM-powered corpus analyzer that generates initial corpus_config.yaml.

Scans directories for .md files, samples content, and uses Claude Haiku
to infer the config schema from filenames, content patterns, and directory
structure.

Usage:
    python analyze_corpus.py docs/reports/ docs/memos/
    python analyze_corpus.py docs/ --output my_config.yaml --max-files 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def collect_sample(
    directory: str, max_files: int = 5, max_words: int = 2000
) -> list[dict]:
    """Collect a sample of .md files from a directory.

    Args:
        directory: Path to scan for .md files.
        max_files: Maximum number of files to read.
        max_words: Maximum words to include per file.

    Returns:
        List of dicts with 'filename', 'directory', and 'content' keys.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []

    md_files = sorted(dir_path.glob("*.md"))[:max_files]

    samples = []
    for f in md_files:
        text = f.read_text(encoding="utf-8", errors="replace")
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words]) + "\n[... truncated]"
        samples.append(
            {
                "filename": f.name,
                "directory": dir_path.name,
                "content": text,
            }
        )

    return samples


CONFIG_SCHEMA_EXAMPLE = """\
# --- Full corpus_config.yaml schema ---

project:
  name: "my-project"              # Short identifier (used in table names, MCP server name)
  description: "Description"      # What this corpus contains
  language: "es"                  # Primary language: "en", "es", etc.
  table_name: "my_chunks"         # LanceDB table name

entities:
  # The PRIMARY entity that distinguishes documents (e.g., community, company, patient)
  - name: "community"             # Entity name (becomes a metadata field and filter)
    description: "Community name"
    extract_from: "filename"      # How to extract: filename | directory | content | mapping
    pattern: "^(.+?)_"            # Regex with capture group (group 1 is the value)

  # DERIVED entity resolved via mapping from another entity
  - name: "region"
    description: "Geographic region"
    extract_from: "mapping"       # Use mapping lookup
    mapping_source: "community"   # Which entity to look up
    mapping:                      # Map of source_value -> derived_value
      "Springfield": "midwest"
      "Portland": "northwest"

document_types:
  # Each type represents a category of documents (different directories, phases, etc.)
  - name: "final_report"          # Type identifier
    directory: "docs/reports"     # Source directory for this type
    precedence_boost: 1.20        # Higher = prioritized in search results (1.0 = neutral)
    deliverable_code: "8.1"       # Optional code/label for this deliverable

  - name: "draft"
    directory: "docs/drafts"
    precedence_boost: 1.0

custom_fields:
  # Additional metadata extracted per document
  - name: "year"
    type: "string"                # string | int | bool
    extract_from: "filename"      # filename | content
    pattern: "_(20\\d{2})"        # Regex with capture group

filters:
  # Which fields should be available as search filters
  # (content_type, has_quantitative_data, document_type, is_summary are always available)
  - community
  - region
  - document_type

skip_sections:
  exact:                          # Exact section titles to skip (case-insensitive)
    - "TABLE OF CONTENTS"
    - "COVER PAGE"
  prefix:                         # Section title prefixes to skip (case-insensitive)
    - "LIST OF "
    - "APPENDIX "

mcp:
  name: "my-project"              # MCP server display name
  instructions: |                 # Instructions shown to LLM users of the MCP server
    This server provides search over project documents.
    Use the search tool to find relevant information.
"""


def build_analysis_prompt(samples: list[dict], directories: list[str]) -> str:
    """Build the LLM prompt for corpus analysis.

    Args:
        samples: List of sample dicts from collect_sample().
        directories: List of directory paths being analyzed.

    Returns:
        The complete prompt string.
    """
    prompt = (
        "You are analyzing a document corpus to generate a configuration file.\n"
        "Based on the filenames, directory structure, and content samples below,\n"
        "generate a corpus_config.yaml file.\n\n"
        "Your task:\n"
        "1. Identify the PRIMARY entity that distinguishes documents "
        "(e.g., community, company, patient, project). Look at filenames for patterns.\n"
        "2. Look for naming patterns in filenames (underscores, hyphens, prefixes).\n"
        "3. Identify any derived entities that can be mapped from the primary entity.\n"
        "4. Detect if there are multiple document types (from different directories "
        "or filename patterns).\n"
        "5. Find boilerplate sections to skip (table of contents, cover pages, etc.).\n"
        "6. Write MCP instructions describing what this corpus contains and how to search it.\n"
        "7. Choose an appropriate table_name (lowercase, underscores, descriptive).\n\n"
        "Respond ONLY with valid YAML following this schema exactly "
        "(no explanations, no markdown fences, just YAML):\n\n"
        f"{CONFIG_SCHEMA_EXAMPLE}\n\n"
        "---\n\n"
        f"Directories being analyzed: {directories}\n\n"
    )

    for s in samples:
        prompt += f"--- File: {s['filename']} (directory: {s['directory']}) ---\n"
        prompt += s["content"][:3000] + "\n\n"

    return prompt


def analyze_with_llm(prompt: str) -> str:
    """Send the analysis prompt to Claude Haiku and return the response.

    Args:
        prompt: The complete analysis prompt.

    Returns:
        Raw LLM response text.

    Raises:
        SystemExit: If ANTHROPIC_API_KEY is not set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "Error: ANTHROPIC_API_KEY environment variable is not set.\n\n"
            "Set it with:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n\n"
            "Get your key at: https://console.anthropic.com/settings/keys"
        )
        sys.exit(1)

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-20250414",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text


def extract_yaml(response: str) -> str:
    """Extract YAML content from an LLM response, stripping markdown fences.

    Handles responses wrapped in ```yaml ... ```, ``` ... ```, or raw YAML.

    Args:
        response: Raw LLM response text.

    Returns:
        Clean YAML string.
    """
    text = response.strip()

    # Try ```yaml ... ``` first
    if text.startswith("```yaml"):
        text = text[len("```yaml") :].strip()
        if text.endswith("```"):
            text = text[: -len("```")].strip()
        return text

    # Try generic ``` ... ```
    if text.startswith("```"):
        text = text[len("```") :].strip()
        if text.endswith("```"):
            text = text[: -len("```")].strip()
        return text

    return text


def main() -> None:
    """CLI entry point for corpus analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze a document corpus and generate corpus_config.yaml"
    )
    parser.add_argument(
        "directories",
        nargs="+",
        help="One or more directories containing .md files to analyze",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="corpus_config.yaml",
        help="Output file path (default: corpus_config.yaml)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5,
        help="Maximum files to sample per directory (default: 5)",
    )

    args = parser.parse_args()

    # Collect samples from all directories
    all_samples: list[dict] = []
    for d in args.directories:
        samples = collect_sample(d, max_files=args.max_files)
        all_samples.extend(samples)

    if not all_samples:
        print(f"Error: No .md files found in: {', '.join(args.directories)}")
        sys.exit(1)

    print(
        f"Collected {len(all_samples)} sample files "
        f"from {len(args.directories)} director{'ies' if len(args.directories) > 1 else 'y'}"
    )

    # Build prompt and call LLM
    prompt = build_analysis_prompt(all_samples, args.directories)
    print("Analyzing corpus with Claude Haiku...")
    raw_response = analyze_with_llm(prompt)

    # Extract and validate YAML
    yaml_text = extract_yaml(raw_response)

    try:
        yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        print(f"Error: LLM response is not valid YAML: {e}")
        print("\nRaw response:\n")
        print(raw_response)
        sys.exit(1)

    # Write output
    output_path = Path(args.output)
    output_path.write_text(yaml_text, encoding="utf-8")
    print(f"Config written to: {output_path}")


if __name__ == "__main__":
    main()
