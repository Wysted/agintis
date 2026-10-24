"""Use case: ingest many documents, routing each to a profile.

Each source carries its `document_metadata`; a `ProfileSelector` (rules over that metadata)
picks the profile. A document nobody claims (or that two profiles claim equally) is reported
`unrouted` with the reason; a document whose own content is the problem (`DomainError`: scanned
PDF, unparseable, empty, invalid metadata) is reported as failed and the batch continues.
Infrastructure failures (database, embedding service) are NOT swallowed: they abort the batch
because every later document would fail too. Documents are processed one at a time (embedding is
the bottleneck; the ingest lock is per source anyway).

Outcomes are produced lazily (`stream`), so a long batch can report as it goes and the sources
can be generated on demand. The caller may also slip in outcomes it already knows (a row that
could not even be turned into a `SourceFile`): they are passed through in order.

Re-routing: a document is identified by (tenant, profile, source). When the selector now sends a
source to another profile than before, the old profile's READY version keeps being served next to
the new one. After a successful ingest the batch looks for those orphans, lists their profiles in
`BatchOutcome.rerouted_from` and, with `cleanup_rerouted`, deletes them (`removed_rerouted`).
Orphans are never touched when the new ingest failed, was unrouted or aborted.

Pruning: a source dropped upstream keeps its READY version forever. `prune` removes, after a
batch without failures, the READY documents of the given profiles whose source is not among the
batch's outcomes (a failed batch may simply have missed sources: nothing is removed then, and an
empty batch, with no document ingested, is no evidence either). Unrouted sources are protected: a
source that is now unrouted keeps its old READY version (neither prune nor cleanup touches it).
A source re-routed to another profile is protected under the old one too (the URI is in the
batch): `cleanup_rerouted` owns that orphan.

Revert: `revert` removes every version of an explicit list of sources (dry run unless `apply`),
to undo an ingest batch auditably.
"""

import logging
from collections.abc import AsyncIterator, Callable, Collection, Iterable
from dataclasses import dataclass
from typing import Literal

from agintis.modules.knowledge.application.dto import IngestStatus, SourceFile
from agintis.modules.knowledge.application.ingest_document import IngestDocument
from agintis.modules.knowledge.application.ports import ProfileSelector
from agintis.modules.knowledge.domain import Document
from agintis.shared_kernel.errors import DomainError
from agintis.shared_kernel.ids import TenantId

logger = logging.getLogger(__name__)

# Source URIs per lookup query. Must stay <= platform.db.criteria.MAX_IN_VALUES (256), which the
# application layer cannot import (SQLAlchemy); a unit test pins the relation.
REVERT_LOOKUP_SIZE = 200

type BatchStatus = IngestStatus | Literal["unrouted", "failed"]


@dataclass(frozen=True)
class BatchOutcome:
    uri: str
    status: BatchStatus
    profile: str | None = None
    chunk_count: int = 0
    reason: str | None = None
    # Profiles still holding a READY version of this source (re-routing orphans).
    rerouted_from: tuple[str, ...] = ()
    removed_rerouted: bool = False
    title_refreshed: bool = False


type BatchItem = SourceFile | BatchOutcome


class IngestBatch:
    def __init__(self, ingest: IngestDocument, selector: ProfileSelector) -> None:
        self._ingest = ingest
        self._selector = selector

    async def execute(
        self,
        tenant_id: TenantId,
        sources: Iterable[BatchItem],
        *,
        cleanup_rerouted: bool = False,
    ) -> list[BatchOutcome]:
        return [
            outcome
            async for outcome in self.stream(tenant_id, sources, cleanup_rerouted=cleanup_rerouted)
        ]

    async def stream(
        self,
        tenant_id: TenantId,
        sources: Iterable[BatchItem],
        *,
        cleanup_rerouted: bool = False,
    ) -> AsyncIterator[BatchOutcome]:
        """One outcome per item, yielded as soon as it is known; `sources` is iterated lazily."""
        for item in sources:
            if isinstance(item, BatchOutcome):
                yield item
            else:
                yield await self._one(tenant_id, item, cleanup_rerouted)

    async def prune(
        self, tenant_id: TenantId, profiles: Collection[str], outcomes: Iterable[BatchOutcome]
    ) -> list[Document] | None:
        """Remove the stale READY documents of `profiles`; None when skipped (a failure).

        Also None when no outcome was indexed, replaced or skipped (an empty index would otherwise
        delete everything). The caller must pass the outcomes of the WHOLE batch (not a
        `--limit`ed slice).
        """
        outcomes = list(outcomes)
        if any(o.status == "failed" for o in outcomes):
            logger.warning("Prune skipped: the batch had failures")
            return None
        if not any(o.status in ("indexed", "replaced", "skipped") for o in outcomes):
            logger.warning("Prune skipped: no document was ingested (empty index?)")
            return None
        seen = {o.uri for o in outcomes}
        removed: list[Document] = []
        for profile in sorted(profiles):
            for document in await self._ingest.find_stale(tenant_id, profile, seen):
                if await self._ingest.remove_rerouted(tenant_id, document):
                    removed.append(document)
                    logger.info(
                        "Pruned %s (%s) of profile %s", document.id, document.source_uri, profile
                    )
        return removed

    async def revert(
        self,
        tenant_id: TenantId,
        source_uris: Collection[str],
        *,
        apply: bool,
        on_removed: Callable[[Document], None] | None = None,
    ) -> list[Document]:
        """Documents of `source_uris` (every profile and status) of the tenant; deleted if `apply`.

        Without `apply` nothing changes (dry run). Chunks go with their document. Each deletion
        takes the source lock and is logged; `on_removed` is called right after it (a caller keeps
        what was deleted if a later one raises). A document already gone is left out of the result.
        """
        found: list[Document] = []
        uris = sorted(set(source_uris))
        for start in range(0, len(uris), REVERT_LOOKUP_SIZE):
            found += await self._ingest.find_by_sources(
                tenant_id, uris[start : start + REVERT_LOOKUP_SIZE]
            )
        if not apply:
            return found
        removed: list[Document] = []
        for document in found:
            if await self._ingest.remove_rerouted(tenant_id, document):
                removed.append(document)
                logger.info(
                    "Reverted %s (%s) of profile %s, tenant %s",
                    document.id,
                    document.source_uri,
                    document.profile,
                    tenant_id,
                )
                if on_removed is not None:
                    on_removed(document)
        return removed

    async def _one(
        self, tenant_id: TenantId, source: SourceFile, cleanup_rerouted: bool
    ) -> BatchOutcome:
        selection = self._selector.select(source.document_metadata)
        if selection.profile is None:
            return BatchOutcome(source.uri, "unrouted", reason=selection.reason)
        profile = selection.profile
        try:
            result = await self._ingest.execute(tenant_id, profile, source)
        except DomainError as exc:
            logger.warning("Ingest of %s with %s failed: %s", source.uri, profile.name, exc)
            return BatchOutcome(
                source.uri, "failed", profile.name, reason=f"{type(exc).__name__}: {exc}"
            )
        orphans = await self._ingest.find_rerouted(tenant_id, source.uri, profile.name)
        removed = False
        if orphans:
            names = tuple(sorted({orphan.profile for orphan in orphans}))
            if cleanup_rerouted:
                for orphan in orphans:
                    await self._ingest.remove_rerouted(tenant_id, orphan)
                removed = True
                logger.info("Removed %s re-routed version(s) of %s", names, source.uri)
            else:
                logger.warning(
                    "%s is also READY under %s (re-routed); pass cleanup to remove it",
                    source.uri,
                    names,
                )
            return BatchOutcome(
                source.uri,
                result.status,
                profile.name,
                result.chunk_count,
                rerouted_from=names,
                removed_rerouted=removed,
                title_refreshed=result.title_refreshed,
            )
        return BatchOutcome(
            source.uri,
            result.status,
            profile.name,
            result.chunk_count,
            title_refreshed=result.title_refreshed,
        )
