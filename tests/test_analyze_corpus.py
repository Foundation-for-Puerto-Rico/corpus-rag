"""Tests for analyze_corpus module (non-LLM parts)."""

import pytest

from analyze_corpus import build_analysis_prompt, collect_sample, extract_yaml


class TestCollectSample:
    def test_collect_sample_files(self, tmp_path):
        """Collects up to N .md files from a directory."""
        for i in range(7):
            (tmp_path / f"doc_{i}.md").write_text(f"Content {i}", encoding="utf-8")

        samples = collect_sample(str(tmp_path), max_files=5)
        assert len(samples) == 5
        assert all(s["filename"].endswith(".md") for s in samples)

    def test_collect_sample_reads_content(self, tmp_path):
        """Truncates content to max_words."""
        long_text = " ".join(f"word{i}" for i in range(500))
        (tmp_path / "long.md").write_text(long_text, encoding="utf-8")

        samples = collect_sample(str(tmp_path), max_files=5, max_words=100)
        assert len(samples) == 1
        # Should be truncated: 100 words + "[... truncated]"
        assert "[... truncated]" in samples[0]["content"]
        words_before_truncation = samples[0]["content"].split("[... truncated]")[0].split()
        assert len(words_before_truncation) == 100

    def test_collect_sample_empty_dir(self, tmp_path):
        """Returns empty list for directory with no .md files."""
        (tmp_path / "readme.txt").write_text("not markdown", encoding="utf-8")

        samples = collect_sample(str(tmp_path))
        assert samples == []

    def test_collect_sample_nonexistent_dir(self):
        """Returns empty list for nonexistent directory."""
        samples = collect_sample("/nonexistent/path/nowhere")
        assert samples == []

    def test_collect_sample_includes_directory(self, tmp_path):
        """Each sample includes the directory name."""
        (tmp_path / "test.md").write_text("hello", encoding="utf-8")
        samples = collect_sample(str(tmp_path))
        assert samples[0]["directory"] == tmp_path.name

    def test_collect_sample_short_content_not_truncated(self, tmp_path):
        """Short content is not truncated."""
        (tmp_path / "short.md").write_text("just a few words", encoding="utf-8")
        samples = collect_sample(str(tmp_path), max_words=2000)
        assert "[... truncated]" not in samples[0]["content"]
        assert samples[0]["content"] == "just a few words"


class TestBuildPrompt:
    def test_build_prompt(self):
        """Prompt includes filenames, content, and directories."""
        samples = [
            {"filename": "Report_2024.md", "directory": "reports", "content": "Sample content here"},
            {"filename": "Memo_2023.md", "directory": "memos", "content": "Another document"},
        ]
        directories = ["docs/reports", "docs/memos"]

        prompt = build_analysis_prompt(samples, directories)

        assert "Report_2024.md" in prompt
        assert "Memo_2023.md" in prompt
        assert "Sample content here" in prompt
        assert "Another document" in prompt
        assert "docs/reports" in prompt
        assert "docs/memos" in prompt

    def test_build_prompt_includes_schema(self):
        """Prompt includes the config schema example."""
        samples = [{"filename": "test.md", "directory": "docs", "content": "test"}]
        prompt = build_analysis_prompt(samples, ["docs"])

        assert "project:" in prompt
        assert "entities:" in prompt
        assert "document_types:" in prompt
        assert "skip_sections:" in prompt
        assert "mcp:" in prompt
        assert "table_name:" in prompt


class TestExtractYaml:
    def test_extract_yaml_with_fences(self):
        """Strips ```yaml ... ``` fences."""
        response = "```yaml\nproject:\n  name: test\n```"
        result = extract_yaml(response)
        assert result == "project:\n  name: test"

    def test_extract_yaml_without_fences(self):
        """Returns raw content if no fences present."""
        response = "project:\n  name: test"
        result = extract_yaml(response)
        assert result == "project:\n  name: test"

    def test_extract_yaml_with_generic_fences(self):
        """Strips generic ``` ... ``` fences."""
        response = "```\nproject:\n  name: test\n```"
        result = extract_yaml(response)
        assert result == "project:\n  name: test"

    def test_extract_yaml_with_whitespace(self):
        """Handles leading/trailing whitespace."""
        response = "\n  ```yaml\nproject:\n  name: test\n```  \n"
        result = extract_yaml(response)
        assert result == "project:\n  name: test"
