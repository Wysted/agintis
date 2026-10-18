"""Boot check: the lexical query configuration is the one `chunks.tsv` is generated with.

`tsv` is a generated column, so its text search configuration is fixed by the migration. A query
parsed with another configuration (other stemmer, other stop words, no `unaccent`) silently
matches little or nothing, with no error anywhere: this check turns that into a startup failure.
`verify_bm25_index` does the same for the optional BM25 index of `retrieval_mode: bm25_hybrid`.
"""

import re

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from agintis.modules.knowledge.infrastructure.postgres.tables import BM25_INDEX

# The generation expression as the catalog deparses it: `to_tsvector('cfg'::regconfig, text)`.
_GENERATION_CONFIG = re.compile(r"to_tsvector\('((?:[^']|'')+)'::regconfig\s*,")

_GENERATION_EXPRESSION = text(
    "SELECT pg_get_expr(d.adbin, d.adrelid) "
    "FROM pg_attrdef d "
    "JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum "
    "WHERE a.attrelid = to_regclass('chunks') AND a.attname = 'tsv'"
)


_BM25_INDEX_ON_CHUNKS = text(
    "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_textsearch'), "
    # Set by the library once loaded (shared_preload_libraries).
    "current_setting('pg_textsearch.library_version', true) IS NOT NULL, "
    "EXISTS (SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
    "JOIN pg_am am ON am.oid = c.relam "
    "WHERE i.indrelid = to_regclass('chunks') AND c.relname = :name AND am.amname = 'bm25')"
)


class TextSearchConfigError(RuntimeError):
    """The text search setup the retrieval needs is missing or does not match `chunks.tsv`."""


async def _regconfig_oid(conn: AsyncConnection, name: str) -> int | None:
    """OID of the text search configuration `name`, None when it does not exist."""
    try:
        async with conn.begin_nested():  # a failed cast must not poison the transaction
            oid = await conn.scalar(text("SELECT CAST(:name AS regconfig)::oid"), {"name": name})
    except DBAPIError:
        return None
    return int(oid) if oid is not None else None


async def verify_text_search_config(engine: AsyncEngine, config: str) -> None:
    """Raise `TextSearchConfigError` unless `config` resolves to the configuration of `tsv`."""
    async with engine.connect() as conn:
        wanted = await _regconfig_oid(conn, config)
        if wanted is None:
            raise TextSearchConfigError(f"Text search configuration {config!r} does not exist")
        expression = await conn.scalar(_GENERATION_EXPRESSION)
        if expression is None:
            raise TextSearchConfigError("Column chunks.tsv has no generation expression")
        match = _GENERATION_CONFIG.search(str(expression))
        if match is None:
            raise TextSearchConfigError(
                f"Cannot find the text search configuration in chunks.tsv: {expression}"
            )
        used = match.group(1).replace("''", "'")
        generated = await _regconfig_oid(conn, used)
        if generated != wanted:
            raise TextSearchConfigError(
                f"KNOWLEDGE_TEXT_SEARCH_CONFIG={config!r} differs from the configuration "
                f"chunks.tsv is generated with ({used!r}): lexical search would not match"
            )


async def verify_bm25_index(engine: AsyncEngine) -> None:
    """Raise `TextSearchConfigError` unless pg_textsearch and the BM25 index on chunks exist."""
    async with engine.connect() as conn:
        row = (await conn.execute(_BM25_INDEX_ON_CHUNKS, {"name": BM25_INDEX})).one()
    has_extension, is_loaded, has_index = (bool(value) for value in row)
    if not has_extension:
        raise TextSearchConfigError(
            "retrieval_mode bm25_hybrid needs the pg_textsearch extension, which this database "
            "lacks: use the project's Postgres image (docker/postgres; migration 0005 creates "
            "it, or lists the statements if it was skipped), or switch those profiles to hybrid"
        )
    if not is_loaded:
        raise TextSearchConfigError(
            "retrieval_mode bm25_hybrid needs pg_textsearch in shared_preload_libraries "
            "(the project's Postgres image sets it): restart the server with it"
        )
    if not has_index:
        raise TextSearchConfigError(
            f"retrieval_mode bm25_hybrid needs the BM25 index {BM25_INDEX} on chunks: see "
            "migration 0005 for the statement to run as a privileged role"
        )
