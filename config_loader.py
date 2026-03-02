"""Config loader for corpus-rag pipeline.

Reads a corpus_config.yaml file and provides dataclasses for all components.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ConfigError(Exception):
    """Raised when config is invalid."""


@dataclass
class EntityConfig:
    """Configuration for a metadata entity (e.g., community, municipality)."""

    name: str
    description: str = ""
    extract_from: str = "filename"  # filename | directory | content | mapping
    pattern: str | None = None
    mapping_source: str | None = None
    mapping: dict[str, str] = field(default_factory=dict)

    def extract(
        self,
        filename: str = "",
        directory: str = "",
        content: str = "",
        resolved_entities: dict[str, str] | None = None,
    ) -> str | None:
        """Extract entity value from the given source.

        Args:
            filename: The document filename (without path).
            directory: The parent directory name.
            content: The document text content.
            resolved_entities: Already-resolved entities (for mapping lookups).

        Returns:
            Extracted value or None if not matched.
        """
        if self.extract_from == "mapping":
            if not self.mapping_source or resolved_entities is None:
                return None
            source_val = resolved_entities.get(self.mapping_source)
            if source_val is None:
                return None
            return self.mapping.get(source_val)

        source_text = {
            "filename": filename,
            "directory": directory,
            "content": content,
        }.get(self.extract_from, "")

        if not self.pattern or not source_text:
            return None

        m = re.search(self.pattern, source_text)
        if m and m.lastindex and m.lastindex >= 1:
            return m.group(1)
        if m:
            return m.group(0)
        return None


@dataclass
class DocumentTypeConfig:
    """Configuration for a document type."""

    name: str
    directory: str = ""
    precedence_boost: float = 1.0
    deliverable_code: str | None = None


@dataclass
class CustomFieldConfig:
    """Configuration for a custom metadata field."""

    name: str
    type: str = "string"  # string | int | bool
    extract_from: str = "filename"  # filename | content
    pattern: str | None = None

    def extract(self, filename: str = "", content: str = "") -> str | None:
        """Extract custom field value from the given source.

        Args:
            filename: The document filename.
            content: The document text content.

        Returns:
            Extracted value or None if not matched.
        """
        source_text = {
            "filename": filename,
            "content": content,
        }.get(self.extract_from, "")

        if not self.pattern or not source_text:
            return None

        m = re.search(self.pattern, source_text)
        if m and m.lastindex and m.lastindex >= 1:
            return m.group(1)
        if m:
            return m.group(0)
        return None


@dataclass
class SkipSectionsConfig:
    """Sections to skip during chunking."""

    exact: list[str] = field(default_factory=list)
    prefix: list[str] = field(default_factory=list)


@dataclass
class McpConfig:
    """MCP server configuration."""

    name: str = ""
    instructions: str = ""


@dataclass
class ProjectConfig:
    """Top-level project metadata."""

    name: str = ""
    description: str = ""
    language: str = "en"
    table_name: str = ""


# Base fields always present in search results
_BASE_RESULT_FIELDS = [
    "text",
    "section_title",
    "content_type",
    "has_quantitative_data",
    "document_type",
    "is_summary",
    "score",
    "rank",
]

# Filters always available regardless of config
_ALWAYS_PRESENT_FILTERS = {
    "content_type",
    "has_quantitative_data",
    "document_type",
    "is_summary",
}


@dataclass
class CorpusConfig:
    """Root configuration for a corpus-rag project."""

    project: ProjectConfig
    entities: list[EntityConfig] = field(default_factory=list)
    document_types: list[DocumentTypeConfig] = field(default_factory=list)
    custom_fields: list[CustomFieldConfig] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    skip_sections: SkipSectionsConfig = field(default_factory=SkipSectionsConfig)
    mcp: McpConfig = field(default_factory=McpConfig)

    def get_entity(self, name: str) -> EntityConfig | None:
        """Get entity config by name."""
        for e in self.entities:
            if e.name == name:
                return e
        return None

    def precedence_boost_map(self) -> dict[str, float]:
        """Return mapping of document_type name to precedence boost."""
        return {dt.name: dt.precedence_boost for dt in self.document_types}

    def docs_dirs(self) -> dict[str, str]:
        """Return mapping of document_type name to directory path."""
        return {dt.name: dt.directory for dt in self.document_types if dt.directory}

    def should_skip_section(self, title: str) -> bool:
        """Check if a section title should be skipped.

        Checks exact match (case-insensitive) and prefix match.
        """
        title_lower = title.lower()
        for exact in self.skip_sections.exact:
            if title_lower == exact.lower():
                return True
        for prefix in self.skip_sections.prefix:
            if title_lower.startswith(prefix.lower()):
                return True
        return False

    def result_fields(self) -> list[str]:
        """Return list of fields in search results.

        Includes base fields + entity names + custom field names.
        """
        fields = list(_BASE_RESULT_FIELDS)
        for e in self.entities:
            if e.name not in fields:
                fields.append(e.name)
        for cf in self.custom_fields:
            if cf.name not in fields:
                fields.append(cf.name)
        return fields

    def valid_filters(self) -> set[str]:
        """Return set of valid filter names.

        Includes config-specified filters + always-present filters.
        """
        return set(self.filters) | _ALWAYS_PRESENT_FILTERS

    def extract_entities(
        self, filename: str = "", directory: str = "", content: str = ""
    ) -> dict[str, str]:
        """Extract all entity values from the given sources.

        Processes entities in order so mapping entities can reference
        already-resolved values.

        Returns:
            Dict of entity_name -> extracted_value (only non-None).
        """
        resolved: dict[str, str] = {}
        for entity in self.entities:
            val = entity.extract(
                filename=filename,
                directory=directory,
                content=content,
                resolved_entities=resolved,
            )
            if val is not None:
                resolved[entity.name] = val
        return resolved

    def extract_custom_fields(
        self, filename: str = "", content: str = ""
    ) -> dict[str, str]:
        """Extract all custom field values.

        Returns:
            Dict of field_name -> extracted_value (only non-None).
        """
        result: dict[str, str] = {}
        for cf in self.custom_fields:
            val = cf.extract(filename=filename, content=content)
            if val is not None:
                result[cf.name] = val
        return result


def load_config(path: str) -> CorpusConfig:
    """Load and validate a corpus config from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated CorpusConfig.

    Raises:
        ConfigError: If required fields are missing or config is invalid.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError("Config must be a YAML mapping")

    # Validate required top-level
    if "project" not in raw:
        raise ConfigError("Missing required field: project")

    proj_raw = raw["project"]
    if not isinstance(proj_raw, dict):
        raise ConfigError("project must be a mapping")

    if not proj_raw.get("name"):
        raise ConfigError("Missing required field: project.name")
    if not proj_raw.get("table_name"):
        raise ConfigError("Missing required field: project.table_name")

    project = ProjectConfig(
        name=proj_raw["name"],
        description=proj_raw.get("description", ""),
        language=proj_raw.get("language", "en"),
        table_name=proj_raw["table_name"],
    )

    # Parse entities
    entities: list[EntityConfig] = []
    for e_raw in raw.get("entities", []):
        entities.append(
            EntityConfig(
                name=e_raw["name"],
                description=e_raw.get("description", ""),
                extract_from=e_raw.get("extract_from", "filename"),
                pattern=e_raw.get("pattern"),
                mapping_source=e_raw.get("mapping_source"),
                mapping=e_raw.get("mapping", {}),
            )
        )

    # Parse document types
    document_types: list[DocumentTypeConfig] = []
    for dt_raw in raw.get("document_types", []):
        document_types.append(
            DocumentTypeConfig(
                name=dt_raw["name"],
                directory=dt_raw.get("directory", ""),
                precedence_boost=float(dt_raw.get("precedence_boost", 1.0)),
                deliverable_code=dt_raw.get("deliverable_code"),
            )
        )

    # Parse custom fields
    custom_fields: list[CustomFieldConfig] = []
    for cf_raw in raw.get("custom_fields", []):
        custom_fields.append(
            CustomFieldConfig(
                name=cf_raw["name"],
                type=cf_raw.get("type", "string"),
                extract_from=cf_raw.get("extract_from", "filename"),
                pattern=cf_raw.get("pattern"),
            )
        )

    # Parse filters
    filters: list[str] = raw.get("filters", [])

    # Parse skip_sections
    skip_raw = raw.get("skip_sections", {})
    skip_sections = SkipSectionsConfig(
        exact=skip_raw.get("exact", []),
        prefix=skip_raw.get("prefix", []),
    )

    # Parse mcp
    mcp_raw = raw.get("mcp", {})
    mcp = McpConfig(
        name=mcp_raw.get("name", ""),
        instructions=mcp_raw.get("instructions", ""),
    )

    return CorpusConfig(
        project=project,
        entities=entities,
        document_types=document_types,
        custom_fields=custom_fields,
        filters=filters,
        skip_sections=skip_sections,
        mcp=mcp,
    )
