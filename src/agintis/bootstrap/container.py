"""Composition root: the only place that wires adapters, ports, nodes and graphs."""

import asyncio
import logging
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from types import TracebackType
from typing import Protocol, Self, runtime_checkable

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from agintis.agent_kit.runtime import TracedRunner
from agintis.bootstrap.adapters import (
    AgentsPendingQuestions,
    ChainedErrorClassifier,
    CheckpointerStateStore,
    ConversationsThreadAccess,
    RegistryAgentDirectory,
)
from agintis.bootstrap.checkpointer import build_checkpointer
from agintis.bootstrap.llm import EnvLookup, build_chat_models
from agintis.bootstrap.observability import (
    TRACING_SHUTDOWN_GRACE_SECONDS,
    build_tracing_from_settings,
    shutdown_tracing,
)
from agintis.bootstrap.settings import Settings
from agintis.bootstrap.tenancy import build_tenant_resolver
from agintis.modules.agents.direct_chat import direct_chat_runner
from agintis.modules.agents.rag_chat import RagChatDeps, load_agent_specs, rag_chat_runners
from agintis.modules.agents_catalog.api import (
    AgentRegistry,
    AgentRunner,
    AgentsCatalogServices,
    RunAgent,
)
from agintis.modules.conversations.api import (
    AcquireRunLease,
    CreateThread,
    DeleteThread,
    GetThread,
    ListThreads,
    NoThreadState,
    ReleaseRunLease,
    ThreadsServices,
    ThreadStateStore,
)
from agintis.modules.conversations.infrastructure.postgres import PgThreadRepository
from agintis.modules.knowledge.api import (
    DEFAULT_PROFILE,
    Chunker,
    Cleaner,
    IngestBatch,
    IngestDocument,
    IngestLock,
    MetadataResolver,
    Parser,
    ProbeProfiles,
    ProfileRouter,
    RagProfileRegistry,
    Reranker,
    RetrievalStrategy,
    RetrieveForTopic,
    load_profiles_dir,
    validate_profiles,
)
from agintis.modules.knowledge.domain import (
    BM25_HYBRID_RETRIEVAL_MODE,
    ChunkRepository,
    DocumentRepository,
)
from agintis.modules.knowledge.infrastructure.chunkers import HfTokenCounter
from agintis.modules.knowledge.infrastructure.errors import RetrievalErrorClassifier
from agintis.modules.knowledge.infrastructure.parsers import (
    OCR_INDEX_CARD_PARSER_KEY,
    require_ocr_extra,
)
from agintis.modules.knowledge.infrastructure.postgres import (
    MetadataIndexSync,
    PgChunkRepository,
    PgDocumentRepository,
    PgIngestLock,
    TextSearchConfigError,
    expected_metadata_indexes,
    metadata_index_problems,
    verify_pg_textsearch,
    verify_text_search_config,
)
from agintis.modules.knowledge.infrastructure.registry import (
    build_chunkers,
    build_cleaners,
    build_extractors,
    build_parsers,
    build_rerankers,
    build_retrieval_modes,
    build_transforms,
    chunker_param_schemas,
    cleaner_ordering,
    cleaner_param_schemas,
    extractor_param_schemas,
    transform_param_schemas,
)
from agintis.modules.knowledge.infrastructure.tei import TeiEmbeddings, verify_served_model
from agintis.platform.checkpointer import Checkpointer, CheckpointerSetupError
from agintis.platform.clock import SystemClock
from agintis.platform.db import (
    DatabaseCheckError,
    check_database,
    create_engine,
    create_session_factory,
    create_unpooled_engine,
)
from agintis.platform.http.tenant import TenantResolver
from agintis.platform.llm import (
    ChatModels,
    LlmError,
    ProviderErrorClassifier,
)
from agintis.platform.observability import Tracing, TracingError

# HNSW graph repair/vacuum fixes landed in pgvector 0.8.3 and 0.8.4; iterative scans need 0.8.0.
MIN_VECTOR_VERSION = (0, 8, 4)

# Alembic head this build was written for. Bump together with every new migration (a unit test
# compares it with the script directory); the startup check refuses any other schema revision.
SCHEMA_HEAD_REVISION = "0006"

logger = logging.getLogger(__name__)


@runtime_checkable
class AsyncCloseable(Protocol):
    async def aclose(self) -> None: ...


class StartupError(RuntimeError):
    """A boot-time check failed; the app must not start."""


@dataclass(frozen=True)
class Container:
    """Composition root. Adapters, ports and graphs get wired here.

    `build` is synchronous and does no I/O (clients and the tokenizer load lazily).
    Entering the container (`async with`) runs the startup checks; leaving it closes every
    adapter that owns resources, in reverse order, through one `AsyncExitStack`.
    """

    settings: Settings
    embeddings: TeiEmbeddings
    token_counter: HfTokenCounter
    parsers: Mapping[str, Parser]
    chunkers: Mapping[str, Chunker]
    cleaners: Mapping[str, Cleaner]
    rerankers: Mapping[str, Reranker]
    retrieval_modes: Mapping[str, RetrievalStrategy]
    profiles: RagProfileRegistry
    profile_router: ProfileRouter
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    documents: DocumentRepository
    chunks: ChunkRepository
    ingest_lock: IngestLock
    ingest_document: IngestDocument
    ingest_batch: IngestBatch
    retrieve_for_topic: RetrieveForTopic
    probe_profiles: ProbeProfiles
    index_sync: MetadataIndexSync
    chat_models: ChatModels
    # None unless built with `with_tenant_resolver=True` (the HTTP path); CLIs leave it off.
    tenant_resolver: TenantResolver | None
    agents_catalog: AgentsCatalogServices
    threads: ThreadsServices
    # None when CHECKPOINTER_ENABLED=false (threads off: no pool, no boot check).
    checkpointer: Checkpointer | None
    tracing: Tracing
    exit_stack: AsyncExitStack = field(default_factory=AsyncExitStack, repr=False, compare=False)

    @classmethod
    def build(
        cls,
        settings: Settings,
        *,
        with_tenant_resolver: bool = False,
        for_ingest: bool = False,
        environ: EnvLookup | None = None,
    ) -> Self:
        tracing = build_tracing_from_settings(settings)
        emb, rer, par = settings.embeddings, settings.reranker, settings.parser
        embeddings = TeiEmbeddings(
            emb.url,
            batch_size=emb.batch_size,
            timeout_seconds=emb.timeout_seconds,
            deadline_seconds=emb.deadline_seconds,
            query_prefix=emb.query_prefix,
            document_prefix=emb.document_prefix,
            expected_dim=emb.expected_dim,
            api_key=emb.api_key.get_secret_value() if emb.api_key else None,
        )
        token_counter = HfTokenCounter(emb.tokenizer_source)
        parsers = build_parsers(
            max_bytes=par.max_bytes,
            max_pages=par.max_pages,
            max_uncompressed_bytes=par.max_uncompressed_bytes,
            max_compression_ratio=par.max_compression_ratio,
        )
        chunkers = build_chunkers(token_counter)
        cleaners = build_cleaners()
        extractors = build_extractors()
        transforms = build_transforms()
        clock = SystemClock()
        rerankers = build_rerankers(
            tei_url=rer.url,
            timeout_seconds=rer.timeout_seconds,
            deadline_seconds=rer.deadline_seconds,
            api_key=rer.api_key.get_secret_value() if rer.api_key else None,
        )
        knowledge = settings.knowledge
        retrieval_modes = build_retrieval_modes(
            branch_depth=knowledge.hybrid_branch_depth, rrf_k=knowledge.hybrid_rrf_k
        )
        # Only local config files are read here (no network); a bad profile aborts the boot.
        profiles_dir = settings.knowledge.profiles_dir
        profiles = RagProfileRegistry(
            [DEFAULT_PROFILE, *(load_profiles_dir(profiles_dir) if profiles_dir else [])]
        )
        validate_profiles(
            profiles,
            parsers,
            chunkers,
            rerankers,
            retrieval_mode_keys=retrieval_modes.keys(),
            chunker_schemas=chunker_param_schemas(),
            chunker_min_tokens={key: c.min_chunk_max_tokens for key, c in chunkers.items()},
            cleaner_schemas=cleaner_param_schemas(),
            cleaner_ordering=cleaner_ordering(),
            extractors=extractors,
            extractor_schemas=extractor_param_schemas(),
            transform_schemas=transform_param_schemas(),
        )
        # Ingest path only (the app never parses): a missing `ocr` extra fails here, not at the
        # first scanned PDF.
        if for_ingest and any(p.parser_key == OCR_INDEX_CARD_PARSER_KEY for p in profiles.all()):
            require_ocr_extra()
        profile_router = ProfileRouter(profiles.all())
        # Fail fast on `indexed` declarations no single index can serve (ConflictingIndexError).
        expected_metadata_indexes(f for p in profiles.all() for f in p.metadata_fields)

        db = settings.database
        engine = create_engine(
            settings.database_url,
            pool_size=db.pool_size,
            max_overflow=db.max_overflow,
            pool_timeout_seconds=db.pool_timeout_seconds,
            pool_recycle_seconds=db.pool_recycle_seconds,
            connect_timeout_seconds=db.connect_timeout_seconds,
            statement_timeout_ms=db.statement_timeout_ms,
            read_only=db.read_only,
            echo=db.echo,
        )
        # Advisory locks get their own never-pooled connections (see PgIngestLock).
        lock_engine = create_unpooled_engine(
            settings.database_url, connect_timeout_seconds=db.connect_timeout_seconds
        )
        session_factory = create_session_factory(engine)
        documents = PgDocumentRepository(session_factory)
        chunks = PgChunkRepository(
            session_factory,
            ef_search=db.hnsw_ef_search,
            max_scan_tuples=db.hnsw_max_scan_tuples,
            text_search_config=knowledge.text_search_config,
            search_timeout_ms=db.search_timeout_ms,
            bm25_indexes=any(
                p.retrieval_mode == BM25_HYBRID_RETRIEVAL_MODE for p in profiles.all()
            ),
        )
        ingest_lock = PgIngestLock(
            lock_engine,
            max_concurrent=db.max_ingest_locks,
            acquire_timeout_seconds=db.ingest_lock_timeout_seconds,
        )
        ingest_document = IngestDocument(
            documents,
            chunks,
            parsers,
            chunkers,
            embeddings,
            clock,
            ingest_lock,
            cleaners=cleaners,
            metadata=MetadataResolver(extractors, transforms, clock),
        )
        ingest_batch = IngestBatch(ingest_document, profile_router)
        retrieve_for_topic = RetrieveForTopic(
            documents,
            chunks,
            embeddings,
            rerankers=rerankers,
            rerank_candidates=rer.max_candidates,
            retrieval_modes=retrieval_modes,
            max_query_chars=knowledge.max_query_chars,
        )
        # Dense-only probe of several profiles with one embedding (evidence topic router).
        probe_profiles = ProbeProfiles(
            chunks,
            embeddings,
            max_query_chars=knowledge.max_query_chars,
            max_concurrency=knowledge.probe_concurrency,
        )
        # Index builds run on the never-pooled engine: no statement timeout, closed afterwards.
        index_sync = MetadataIndexSync(engine, lock_engine)

        # Local file only (no network); a bad file aborts the boot. A profile's `api_key_env` is
        # looked up when an agent binds it (the error names the profile and the agent).
        chat_models = build_chat_models(settings, environ)

        # HTTP path only: CLIs neither read the JWKS file nor log the auth boot warnings.
        tenant_resolver = build_tenant_resolver(settings) if with_tenant_resolver else None
        # The pool is created closed (no I/O); `start()` opens it. Disabled = threads are off.
        cp = settings.checkpointer
        checkpointer = build_checkpointer(settings) if cp.enabled else None
        saver = checkpointer.saver if checkpointer else None
        # Like profiles: only local config files are read; a bad spec or an unknown profile
        # aborts the boot. A duplicate agent id fails in the registry.
        agents_dir = settings.agents.dir
        rag_runners = rag_chat_runners(
            load_agent_specs(agents_dir) if agents_dir else (),
            profiles,
            RagChatDeps(chat_models, retrieve_for_topic, knowledge.max_query_chars, probe_profiles),
            saver,
        )
        runners: tuple[AgentRunner, ...] = (
            direct_chat_runner(chat_models, checkpointer=saver),
            *rag_runners,
        )
        if tracing.enabled:
            runners = tuple(TracedRunner(runner, tracing) for runner in runners)
        agents = AgentRegistry(runners)
        thread_repository = PgThreadRepository(session_factory)
        state_store: ThreadStateStore = (
            CheckpointerStateStore(checkpointer) if checkpointer else NoThreadState()
        )
        threads = ThreadsServices(
            create=CreateThread(
                thread_repository,
                RegistryAgentDirectory(agents),
                clock,
                threads_enabled=checkpointer is not None,
            ),
            get=GetThread(thread_repository, AgentsPendingQuestions(agents)),
            list=ListThreads(thread_repository),
            delete=DeleteThread(thread_repository, state_store),
            acquire_lease=AcquireRunLease(
                thread_repository, timedelta(seconds=settings.threads.lease_ttl_seconds)
            ),
            release_lease=ReleaseRunLease(thread_repository),
        )
        run_agent = RunAgent(
            agents,
            ChainedErrorClassifier(ProviderErrorClassifier(), RetrievalErrorClassifier()),
            threads=ConversationsThreadAccess(threads) if checkpointer else None,
            run_deadline_seconds=settings.threads.run_deadline_seconds,
        )
        agents_catalog = AgentsCatalogServices(
            registry=agents,
            run_agent=run_agent,
            max_message_chars=settings.http.max_message_chars,
        )

        stack = AsyncExitStack()
        # Pushed first = closed last: in-flight runs end (and their spans) before the final flush.
        stack.push_async_callback(
            shutdown_tracing,
            tracing,
            settings.observability.otlp_timeout_seconds + TRACING_SHUTDOWN_GRACE_SECONDS,
        )
        # Closed in reverse order: HTTP adapters first, then the lock engine, then the pool.
        stack.push_async_callback(engine.dispose)
        stack.push_async_callback(lock_engine.dispose)
        if checkpointer is not None:
            stack.push_async_callback(checkpointer.aclose)
        for adapter in (embeddings, *rerankers.values(), tenant_resolver):
            if adapter is None:
                continue
            if isinstance(adapter, AsyncCloseable):
                stack.push_async_callback(adapter.aclose)
        # Last pushed = first closed: in-flight thread releases still need the database.
        stack.push_async_callback(run_agent.aclose)
        return cls(
            settings=settings,
            embeddings=embeddings,
            token_counter=token_counter,
            parsers=parsers,
            chunkers=chunkers,
            cleaners=cleaners,
            rerankers=rerankers,
            retrieval_modes=retrieval_modes,
            profiles=profiles,
            profile_router=profile_router,
            engine=engine,
            session_factory=session_factory,
            documents=documents,
            chunks=chunks,
            ingest_lock=ingest_lock,
            ingest_document=ingest_document,
            ingest_batch=ingest_batch,
            retrieve_for_topic=retrieve_for_topic,
            probe_profiles=probe_profiles,
            index_sync=index_sync,
            chat_models=chat_models,
            tenant_resolver=tenant_resolver,
            agents_catalog=agents_catalog,
            threads=threads,
            checkpointer=checkpointer,
            tracing=tracing,
            exit_stack=stack,
        )

    def require_tenant_resolver(self) -> TenantResolver:
        if self.tenant_resolver is None:
            raise RuntimeError("Container built without a tenant resolver (HTTP path only)")
        return self.tenant_resolver

    async def start(self) -> None:
        """Open owned resources, then run the boot checks (skipped when startup_checks is false)."""
        try:
            self.tracing.start()
        except TracingError as exc:
            raise StartupError(str(exc)) from exc
        if self.checkpointer is not None:
            await self.checkpointer.open()  # lazy connections: no network wait here
        if not self.settings.startup_checks:
            return
        source = self.settings.embeddings.tokenizer_source
        try:
            await asyncio.to_thread(self.token_counter.warm_up)
        except Exception as exc:
            raise StartupError(f"Cannot load the tokenizer {source!r}: {exc}") from exc
        await verify_served_model(self.embeddings, self.settings.embeddings.model)
        try:
            version = await check_database(
                self.engine,
                min_vector_version=MIN_VECTOR_VERSION,
                expected_embedding_dim=self.settings.embeddings.expected_dim,
                expected_revision=SCHEMA_HEAD_REVISION,
                timeout_seconds=self.settings.database.startup_check_timeout_seconds,
            )
        except DatabaseCheckError as exc:
            raise StartupError(str(exc)) from exc
        logger.info("Database reachable, pgvector %s, schema %s", version, SCHEMA_HEAD_REVISION)
        try:
            await verify_text_search_config(self.engine, self.settings.knowledge.text_search_config)
        except TextSearchConfigError as exc:
            raise StartupError(str(exc)) from exc
        # Optional extension: checked only when a profile needs it.
        if any(p.retrieval_mode == BM25_HYBRID_RETRIEVAL_MODE for p in self.profiles.all()):
            await self.require_pg_textsearch()
        await self._warn_metadata_index_problems()
        if self.checkpointer is not None:
            try:
                await self.checkpointer.verify(self.settings.database.startup_check_timeout_seconds)
            except CheckpointerSetupError as exc:
                raise StartupError(str(exc)) from exc
        try:
            # Offline; no-op without LLM_PROVIDER. Builds every role (missing extra / API key
            # surface here as LlmError) and checks the analyzer's structured output.
            self.chat_models.verify()
        except LlmError as exc:
            raise StartupError(str(exc)) from exc

    async def require_pg_textsearch(self) -> None:
        """StartupError unless pg_textsearch is installed and loaded (`bm25_hybrid` needs it)."""
        try:
            await verify_pg_textsearch(self.engine)
        except TextSearchConfigError as exc:
            raise StartupError(str(exc)) from exc

    async def _warn_metadata_index_problems(self) -> None:
        """`MetadataField.indexed` creates nothing by itself: say so when the index is not right."""
        fields = [f for profile in self.profiles.all() for f in profile.metadata_fields]
        try:
            problems = await metadata_index_problems(self.engine, fields)
        except Exception:
            logger.warning("Could not verify metadata indexes", exc_info=True)
            return
        for entry in problems:
            logger.warning(
                "Profile metadata field expects index %s but it is %s: run "
                "`python -m agintis.cli.indexes --profiles DIR --apply` (add --prune "
                "--confirm-prune to rebuild stale, unmanaged or invalid ones; an `indexed` "
                "flag alone creates nothing)",
                entry.name,
                entry.status.value,
            )

    async def aclose(self) -> None:
        """Release owned resources (HTTP clients, database engines, the checkpointer pool)."""
        await self.exit_stack.aclose()

    async def __aenter__(self) -> Self:
        try:
            await self.start()
        except BaseException:
            await self.aclose()
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
