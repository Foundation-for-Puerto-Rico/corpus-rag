#!/usr/bin/env python3
"""Corpus RAG HTTP Server — Unified MCP SSE + REST API.

Reuses the MCP server and retriever from mcp_server.py, adding:
- REST endpoints via custom_route() for external clients
- MCP SSE transport at /sse for remote Claude Code
- API key auth for REST endpoints (MCP SSE is open)

Usage:
    python server.py                       # Runs on 0.0.0.0:8080
    CORPUS_API_KEY=secret python server.py # With API key auth
"""

import json
import os

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from mcp.server.transport_security import TransportSecuritySettings

# Import the already-configured MCP server and retriever from the MCP module.
# This avoids duplicating tool definitions or retriever initialization.
from mcp_server import mcp, retriever, config, list_entities, list_document_types, search_batch, verify_absence

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("CORPUS_API_KEY", "")

# Override FastMCP settings for SSE transport
mcp.settings.host = "0.0.0.0"
mcp.settings.port = int(os.environ.get("CORPUS_PORT", "8080"))

# Disable DNS rebinding protection for remote SSE access.
# The FastMCP default auto-enables it for localhost, but this server
# is accessed via external IP so we need to allow any host.
mcp.settings.transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=False,
)

# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------


def _check_api_key(request: Request) -> Response | None:
    """Return an error Response if API key is required and missing/wrong.

    Returns None if auth passes.
    """
    if not API_KEY:
        return None  # No key configured — allow all
    provided = request.headers.get("X-API-Key", "")
    if provided != API_KEY:
        return JSONResponse(
            {"error": "Invalid or missing API key"},
            status_code=401,
        )
    return None


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> Response:
    """Health check — always open, no auth."""
    return JSONResponse({
        "status": "ok",
        "service": "corpus-rag",
        "version": "0.1.0",
    })


@mcp.custom_route("/api/search", methods=["POST"])
async def api_search(request: Request) -> Response:
    """Search documents via REST API.

    Request body (JSON):
        query (str): Search query (required)
        mode (str): "hybrid" | "semantic" | "fts" (default: "hybrid")
        limit (int): Max results (default: 5)
        filters (dict): Optional key/value pairs matching corpus config filters
        apply_precedence (bool): Apply document-type precedence boost (default: True)
        rerank (bool): Re-rank results with LLM (default: False)
        include_summaries (bool): Include AI-generated summary chunks (default: False)

    Response:
        {"results": [...], "count": N}
    """
    auth_err = _check_api_key(request)
    if auth_err:
        return auth_err

    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    query = body.get("query", "").strip()
    if not query:
        return JSONResponse({"error": "Missing 'query' field"}, status_code=400)

    mode = body.get("mode", "hybrid")
    limit = body.get("limit", 5)

    # Validate and collect filters
    filters: dict = {}
    raw_filters = body.get("filters", {})
    if raw_filters:
        valid = config.valid_filters()
        invalid_keys = set(raw_filters.keys()) - valid
        if invalid_keys:
            return JSONResponse(
                {"error": f"Invalid filter keys: {sorted(invalid_keys)}. Valid filters: {sorted(valid)}"},
                status_code=400,
            )
        filters = {k: v for k, v in raw_filters.items() if v is not None}

    apply_precedence = body.get("apply_precedence", True)
    rerank = body.get("rerank", False)
    include_summaries = body.get("include_summaries", False)

    if not include_summaries:
        try:
            if "is_summary" in retriever._table.schema.names:
                filters["is_summary"] = False
        except Exception:
            pass

    fetch_limit = limit * 4 if rerank else limit

    if mode == "semantic":
        results = retriever.semantic_search(query, limit=fetch_limit, apply_precedence=apply_precedence, **filters)
    elif mode == "fts":
        results = retriever.fts_search(query, limit=fetch_limit, apply_precedence=apply_precedence, **filters)
    else:
        results = retriever.hybrid_search(query, limit=fetch_limit, apply_precedence=apply_precedence, **filters)

    if rerank and results:
        results = retriever.rerank(query, results, top_k=limit)

    return JSONResponse({"results": results, "count": len(results)})


@mcp.custom_route("/api/search_batch", methods=["POST"])
async def api_search_batch(request: Request) -> Response:
    """Search across multiple entity values, grouped by entity value.

    Request body (JSON):
        query (str): Search query (required)
        entity_name (str): Entity filter field (required)
        entity_values (list[str]): Values to search across (required)
        mode (str): "hybrid" | "semantic" | "fts" (default: "hybrid")
        limit_per_entity (int): Max chunks per entity value (default: 3)
        document_type (str): Optional document type filter
        apply_precedence (bool): Apply document-type precedence boost (default: True)

    Response:
        {"results": {entity_value: [...]}, "entities_searched": N}
    """
    auth_err = _check_api_key(request)
    if auth_err:
        return auth_err
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    query = body.get("query", "").strip()
    entity_name = body.get("entity_name", "")
    entity_values = body.get("entity_values", [])
    if not query or not entity_name or not entity_values:
        return JSONResponse(
            {"error": "Missing 'query', 'entity_name', or 'entity_values'"}, status_code=400,
        )

    results = search_batch(
        query=query,
        entity_name=entity_name,
        entity_values=entity_values,
        mode=body.get("mode", "hybrid"),
        limit_per_entity=body.get("limit_per_entity", 3),
        document_type=body.get("document_type"),
        apply_precedence=body.get("apply_precedence", True),
    )
    return JSONResponse({"results": results, "entities_searched": len(entity_values)})


@mcp.custom_route("/api/entities", methods=["GET"])
async def api_entities(request: Request) -> Response:
    """List unique values for an entity via REST API.

    Query params:
        entity_name (str): Entity to list (required)
        filters (str): Optional JSON-encoded filter dict
    """
    auth_err = _check_api_key(request)
    if auth_err:
        return auth_err

    entity_name = request.query_params.get("entity_name", "").strip()
    if not entity_name:
        return JSONResponse({"error": "Missing 'entity_name' query param"}, status_code=400)

    filters: dict | None = None
    raw_filters = request.query_params.get("filters")
    if raw_filters:
        try:
            filters = json.loads(raw_filters)
        except (json.JSONDecodeError, ValueError):
            return JSONResponse({"error": "Invalid JSON in 'filters' query param"}, status_code=400)

    data = list_entities(entity_name=entity_name, filters=filters)
    return JSONResponse({"entities": data, "count": len(data)})


@mcp.custom_route("/api/document_types", methods=["GET"])
async def api_document_types(request: Request) -> Response:
    """List all indexed document types with chunk counts via REST API."""
    auth_err = _check_api_key(request)
    if auth_err:
        return auth_err

    data = list_document_types()
    return JSONResponse({"document_types": data, "count": len(data)})


@mcp.custom_route("/api/verify_absence", methods=["POST"])
async def api_verify_absence(request: Request) -> Response:
    """Verify keyword absence in documents matching an entity filter via FTS.

    Request body (JSON):
        keywords (list[str]): Keywords to search for (required)
        entity_name (str): Entity field to filter by (required)
        entity_value (str): Entity value to check (required)
        limit (int): Max evidence chunks to return (default: 3)

    Response:
        {"found": bool, "evidence": [...], "keywords_matched": [...]}
    """
    auth_err = _check_api_key(request)
    if auth_err:
        return auth_err
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    keywords = body.get("keywords", [])
    entity_name = body.get("entity_name", "")
    entity_value = body.get("entity_value", "")
    if not keywords or not entity_name or not entity_value:
        return JSONResponse(
            {"error": "Missing 'keywords', 'entity_name', or 'entity_value'"}, status_code=400,
        )

    result = verify_absence(
        keywords=keywords,
        entity_name=entity_name,
        entity_value=entity_value,
        limit=body.get("limit", 3),
    )
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    mcp.run(transport="sse")
