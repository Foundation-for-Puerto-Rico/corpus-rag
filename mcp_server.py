#!/usr/bin/env python3
"""Config-driven MCP Server — Exposes corpus search tools to Claude Code
via the Model Context Protocol.

All domain-specific behavior (entity names, filters, document types,
instructions) comes from a corpus_config.yaml file.

Usage (stdio, launched by Claude Code):
    CORPUS_CONFIG_PATH=corpus_config.yaml python mcp_server.py
"""

import os

from mcp.server.fastmcp import FastMCP

from config_loader import CorpusConfig, load_config
from retriever import Retriever

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_PATH = os.environ.get("CORPUS_CONFIG_PATH", "corpus_config.yaml")
config = load_config(CONFIG_PATH)
retriever = Retriever(config=config)

# ---------------------------------------------------------------------------
# Dynamic instructions
# ---------------------------------------------------------------------------


def build_instructions(cfg: CorpusConfig) -> str:
    """Generate MCP server instructions from config, interpolating placeholders.

    Supported placeholders in config.mcp.instructions:
        {project_name}   - project.name
        {project_description} - project.description
        {entity_names}   - comma-separated entity names
        {entity_count}   - number of configured entities
        {filter_list}    - comma-separated valid filter names
        {document_types} - bullet list of document type names
    """
    entity_names = [e.name for e in cfg.entities]
    doc_type_names = [dt.name for dt in cfg.document_types]
    valid_filters = sorted(cfg.valid_filters())

    placeholders = {
        "project_name": cfg.project.name,
        "project_description": cfg.project.description,
        "entity_names": ", ".join(entity_names) if entity_names else "(none)",
        "entity_count": str(len(entity_names)),
        "filter_list": ", ".join(valid_filters),
        "document_types": "\n".join(f"- {dt}" for dt in doc_type_names) if doc_type_names else "(none)",
    }

    template = cfg.mcp.instructions
    if not template:
        # Fallback: generate minimal instructions
        template = (
            "RAG server for {project_name}. {project_description}\n\n"
            "Available filters: {filter_list}\n\n"
            "Document types:\n{document_types}"
        )

    result = template
    for key, val in placeholders.items():
        result = result.replace("{" + key + "}", val)
    return result


mcp = FastMCP(
    config.mcp.name or config.project.name,
    instructions=build_instructions(config),
)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_entities_cache: dict[str, list[dict]] = {}

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def search(
    query: str,
    mode: str = "hybrid",
    limit: int = 5,
    filters: dict | None = None,
    apply_precedence: bool = True,
    rerank: bool = False,
    include_summaries: bool = False,
) -> list[dict]:
    """Search documents in the corpus.

    Args:
        query: Search query.
        mode: Search mode -- "hybrid" (default, combines semantic + keywords),
              "semantic" (conceptual similarity), or "fts" (exact keywords).
        limit: Maximum results (default 5).
        filters: Optional dict of filters to apply. Valid filter keys
                 are determined by the corpus configuration. Invalid keys
                 are rejected with an error. Example: {"city": "Boston"}.
        apply_precedence: Apply document-type precedence boost (default True).
        rerank: Re-rank results with LLM for higher precision (default False).
        include_summaries: Include AI-generated summary chunks (default False).

    Returns:
        List of result dicts with text, metadata, score, and rank.
    """
    if not query or not query.strip():
        return []

    # Validate and build filter kwargs
    valid = config.valid_filters()
    filter_kwargs: dict = {}
    if filters:
        invalid_keys = set(filters.keys()) - valid
        if invalid_keys:
            return [{"_error": f"Invalid filter keys: {sorted(invalid_keys)}. Valid filters: {sorted(valid)}"}]
        filter_kwargs = {k: v for k, v in filters.items() if v is not None}

    if not include_summaries:
        # Exclude summaries by default
        try:
            if "is_summary" in retriever._table.schema.names:
                filter_kwargs["is_summary"] = False
        except Exception:
            pass

    fetch_limit = limit * 4 if rerank else limit

    if mode == "semantic":
        results = retriever.semantic_search(query, limit=fetch_limit, apply_precedence=apply_precedence, **filter_kwargs)
    elif mode == "fts":
        results = retriever.fts_search(query, limit=fetch_limit, apply_precedence=apply_precedence, **filter_kwargs)
    else:
        results = retriever.hybrid_search(query, limit=fetch_limit, apply_precedence=apply_precedence, **filter_kwargs)

    if rerank and results:
        results = retriever.rerank(query, results, top_k=limit)

    return results


@mcp.tool()
def search_batch(
    query: str,
    entity_name: str,
    entity_values: list[str],
    mode: str = "hybrid",
    limit_per_entity: int = 3,
    document_type: str | None = None,
    apply_precedence: bool = True,
) -> dict[str, list[dict]]:
    """Search a topic across multiple entity values and group results by entity.

    Use this tool for counting or exhaustive queries like:
    'How many [entities] mention X?' or 'Which [entities] do NOT include Y?'

    Args:
        query: Search query.
        entity_name: Entity to filter by (e.g., "city", "community").
        entity_values: List of entity values to search across.
                       Maximum 40 per call.
        mode: "hybrid" (default), "semantic", or "fts".
        limit_per_entity: Maximum chunks per entity value (default 3).
        document_type: Filter by document type (optional).
        apply_precedence: Apply document-type precedence boost.

    Returns:
        Dict mapping entity value to its list of result chunks.
        Entities with no results appear with an empty list.
    """
    if not query or not query.strip():
        return {}

    # Validate entity_name is a valid filter
    valid = config.valid_filters()
    if entity_name not in valid:
        return {"_error": [{"_error": f"'{entity_name}' is not a valid filter. Valid: {sorted(valid)}"}]}

    entity_values = entity_values[:40]

    results: dict[str, list[dict]] = {}
    base_filters: dict = {}
    if document_type is not None:
        base_filters["document_type"] = document_type

    for val in entity_values:
        ent_filters = {**base_filters, entity_name: val}
        if mode == "semantic":
            hits = retriever.semantic_search(
                query, limit=limit_per_entity,
                apply_precedence=apply_precedence, **ent_filters,
            )
        elif mode == "fts":
            hits = retriever.fts_search(
                query, limit=limit_per_entity,
                apply_precedence=apply_precedence, **ent_filters,
            )
        else:
            hits = retriever.hybrid_search(
                query, limit=limit_per_entity,
                apply_precedence=apply_precedence, **ent_filters,
            )
        results[val] = hits

    return results


@mcp.tool()
def list_entities(
    entity_name: str,
    filters: dict | None = None,
) -> list[dict]:
    """List unique values for a given entity with their associated metadata.

    Use this tool to discover valid entity values, count entities, and see
    what document types are available for each entity.

    Args:
        entity_name: The entity to list (e.g., "city", "community").
        filters: Optional dict of filters to narrow results.

    Returns:
        List of dicts sorted alphabetically by entity value, each containing
        the entity value, related entity values, and document_types available.
    """
    cache_key = entity_name + "|" + (str(sorted(filters.items())) if filters else "")
    if cache_key in _entities_cache:
        return _entities_cache[cache_key]

    # Determine columns to select: the target entity + all other entities + document_type
    entity_names = [e.name for e in config.entities]
    if entity_name not in entity_names and entity_name != "document_type":
        return [{"_error": f"'{entity_name}' is not a configured entity. Available: {entity_names}"}]

    cols = list(entity_names)
    if "document_type" not in cols:
        cols.append("document_type")

    df = retriever._table.to_pandas()
    available = [c for c in cols if c in df.columns]
    if entity_name not in available:
        return [{"_error": f"'{entity_name}' column not found in database."}]

    df = df[available]

    # Apply optional filters
    if filters:
        valid = config.valid_filters()
        for k, v in filters.items():
            if k in valid and k in df.columns and v is not None:
                df = df[df[k] == v]

    # Group by entity and aggregate
    group_cols = [c for c in available if c != "document_type"]
    if "document_type" in available:
        grouped = (
            df.groupby(group_cols)["document_type"]
            .apply(lambda x: sorted(x.unique().tolist()))
            .reset_index(name="document_types")
        )
    else:
        grouped = df.drop_duplicates(subset=group_cols)
        grouped["document_types"] = [[] for _ in range(len(grouped))]

    result = grouped.sort_values(entity_name).to_dict(orient="records")
    _entities_cache[cache_key] = result
    return result


@mcp.tool()
def get_sections(
    entity_name: str,
    entity_value: str,
) -> list[str]:
    """Get the section structure of documents matching an entity filter.

    Args:
        entity_name: Entity to filter by (e.g., "city", "community").
        entity_value: Value of the entity (e.g., "Boston", "El Tuque").

    Returns:
        List of section titles in order of appearance.
    """
    escaped = entity_value.replace("'", "''")
    try:
        results = (
            retriever._table.search()
            .where(f"{entity_name} = '{escaped}'")
            .select(["section_title", "chunk_index"])
            .limit(500)
            .to_pandas()
        )
    except Exception:
        return []

    if results.empty:
        return []

    sections = (
        results.drop_duplicates(subset="section_title")
        .sort_values("chunk_index")
    )
    return sections["section_title"].tolist()


@mcp.tool()
def list_document_types() -> list[dict]:
    """List indexed document types with chunk counts and metadata.

    Use this tool to discover what document types are available before
    filtering with search(filters={"document_type": "..."}).

    Returns:
        List of dicts with document_type and count, sorted by count descending.
        May include deliverable_code and process_phase if present in the data.
    """
    cols = ["document_type"]
    df_full = retriever._table.to_pandas()
    df_cols = set(df_full.columns)

    has_deliverable = "deliverable_code" in df_cols
    has_phase = "process_phase" in df_cols

    if has_deliverable:
        cols.append("deliverable_code")
    if has_phase:
        cols.append("process_phase")

    df = df_full[cols]
    counts = df.groupby(cols).size().reset_index(name="count")
    counts = counts.sort_values("count", ascending=False)

    results = []
    for _, row in counts.iterrows():
        entry: dict = {"document_type": row["document_type"], "count": int(row["count"])}
        if has_deliverable:
            entry["deliverable_code"] = row.get("deliverable_code", "")
        if has_phase:
            entry["process_phase"] = row.get("process_phase", "")
        results.append(entry)
    return results


@mcp.tool()
def verify_absence(
    keywords: list[str],
    entity_name: str,
    entity_value: str,
    limit: int = 3,
) -> dict:
    """Verify whether specific keywords appear in documents matching an entity filter using exact FTS.

    This tool is more reliable than semantic search for confirming absence
    because it matches exact tokens rather than semantic similarity.

    Recommended pattern for negation queries:
    1. search(query='topic') -- identify entities WITH the topic
    2. list_entities(entity_name='X') -- get complete list
    3. verify_absence(keywords=[...], entity_name='X', entity_value='Y')
       -- confirm absence via FTS

    Args:
        keywords: List of keywords to search for (maximum 10).
                  Include variants for better coverage.
        entity_name: Entity to filter by (e.g., "city", "community").
        entity_value: Value of the entity to check.
        limit: Maximum evidence chunks to return.

    Returns:
        Dict with found (bool), evidence (matching chunks),
        and keywords_matched (which keywords had hits).
    """
    return retriever.verify_absence(
        keywords=keywords[:10],
        limit=limit,
        **{entity_name: entity_value},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
