import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy.ext.asyncio import AsyncEngine
from tokenizers import Tokenizer, models, pre_tokenizers

from agintis.bootstrap.container import SCHEMA_HEAD_REVISION, Container, StartupError
from agintis.bootstrap.settings import Settings
from agintis.modules.knowledge.api import (
    ConflictingIndexError,
    IndexEntry,
    IndexStatus,
    InvalidProfileError,
    UnknownStrategyError,
)
from agintis.modules.knowledge.domain import (
    BM25_HYBRID_RETRIEVAL_MODE,
    MetadataField,
    RagProfile,
)
from agintis.modules.knowledge.infrastructure.parsers import OcrNotInstalledError
from agintis.modules.knowledge.infrastructure.postgres import (
    PgChunkRepository,
    PgDocumentRepository,
    PgIngestLock,
    TextSearchConfigError,
    metadata_index_name,
)
from agintis.modules.knowledge.infrastructure.tei import FallbackReranker
from agintis.platform.auth import AuthConfigError
from agintis.platform.db import DatabaseCheckError
from agintis.platform.llm import ChatModels, ChatRole, LlmConfig, LlmError
from tests.support.postgres import PROJECT_ROOT
from tests.support.settings import make_settings


@pytest.fixture(autouse=True)
def _no_text_search_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """The boot check needs a real database: tests that exercise it patch it themselves."""

    async def no_check(engine: object, config: str) -> None:
        return None

    monkeypatch.setattr("agintis.bootstrap.container.verify_text_search_config", no_check)


_REAL_VERIFY = ChatModels.verify


@pytest.fixture(autouse=True)
def _no_llm_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keeps these tests independent of the dev env's LLM_*: tests that need it patch it back."""
    monkeypatch.setattr("agintis.bootstrap.container.ChatModels.verify", lambda self: None)


def _tokenizer_file(directory: Path) -> Path:
    tokenizer = Tokenizer(models.WordLevel({"hola": 0, "[UNK]": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = directory / "tokenizer.json"
    tokenizer.save(str(path))
    return path


def test_build_wires_strategies_from_settings() -> None:
    container = Container.build(
        make_settings(
            embeddings={"batch_size": 4, "query_prefix": "query: "},
            reranker={"url": "http://reranker:80", "max_candidates": 3},
            parser={"max_pages": 5},
        )
    )
    assert set(container.parsers) == {"simple", "simple_index_card", "ocr_index_card"}
    assert set(container.chunkers) == {
        "structured",
        "whole_document",
        "pattern_hierarchy",
        "records",
    }
    assert set(container.rerankers) == {"none", "tei"}
    assert isinstance(container.rerankers["tei"], FallbackReranker)
    assert {p.name for p in container.profiles.all()} == {"default"}


def test_build_wires_the_spensiones_case_profiles_router_and_batch_ingest() -> None:
    profiles_dir = PROJECT_ROOT / "case" / "spensiones" / "profiles"
    container = Container.build(make_settings(knowledge={"profiles_dir": profiles_dir}))
    assert len(container.profiles.all()) == 8  # the default profile + the seven of the case
    selection = container.profile_router.select({"doc_type": "ncg"})
    assert selection.profile is not None
    assert selection.profile.name == "spensiones_ncg"
    assert container.profile_router.select({"doc_type": "oficio"}).profile is None
    assert container.ingest_batch is not None


def test_build_registers_the_index_card_parser() -> None:
    container = Container.build(make_settings())
    assert container.parsers["simple_index_card"] is not container.parsers["simple"]


def test_an_ingest_build_fails_when_a_profile_needs_the_missing_ocr_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "scans.yaml").write_text(
        "key: scans\ntop_k: 4\nchunk_max_tokens: 300\nparser_key: ocr_index_card\n"
        "chunker_key: structured\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "agintis.modules.knowledge.infrastructure.parsers.ocr.find_spec", lambda name: None
    )
    settings = make_settings(knowledge={"profiles_dir": tmp_path})
    with pytest.raises(OcrNotInstalledError, match="'ocr' extra"):
        Container.build(settings, for_ingest=True)
    # The app never parses: it boots on the same profiles without the extra.
    Container.build(settings)
    # Without such a profile the extra is never checked.
    Container.build(make_settings(), for_ingest=True)


def test_build_registers_the_whitelisted_cleaners() -> None:
    container = Container.build(make_settings())
    assert "strip_repeated_page_lines" in container.cleaners
    assert "html_main_content" in container.cleaners


def test_build_loads_yaml_profiles_next_to_the_default(tmp_path: Path) -> None:
    (tmp_path / "legal.yaml").write_text(
        "key: legal\ntop_k: 4\nchunk_max_tokens: 300\nparser_key: simple\n"
        "chunker_key: structured\nchunker_params: {cross_pages: true}\n"
        "cleaners:\n  - {key: normalize_whitespace}\n"
    )
    container = Container.build(make_settings(knowledge={"profiles_dir": tmp_path}))
    assert [p.name for p in container.profiles.all()] == ["default", "legal"]
    assert container.profiles.get("legal").cleaners[0].key == "normalize_whitespace"


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ("chunker_key: hybrid\n", UnknownStrategyError),
        ("chunker_key: structured\nchunker_params: {nope: 1}\n", InvalidProfileError),
        ("chunker_key: structured\ncleaners: [{key: nope}]\n", UnknownStrategyError),
        ("chunker_key: structured\nunknown_key: 1\n", InvalidProfileError),
    ],
)
def test_build_fails_fast_on_an_invalid_profile(
    tmp_path: Path, body: str, error: type[Exception]
) -> None:
    (tmp_path / "bad.yaml").write_text(
        "key: bad\ntop_k: 4\nchunk_max_tokens: 300\nparser_key: simple\n" + body
    )
    with pytest.raises(error):
        Container.build(make_settings(knowledge={"profiles_dir": tmp_path}))


def test_build_fails_when_chunk_max_tokens_is_below_the_chunkers_minimum(tmp_path: Path) -> None:
    (tmp_path / "tiny.yaml").write_text(
        "key: tiny\ntop_k: 4\nchunk_max_tokens: 20\nparser_key: simple\nchunker_key: structured\n"
    )
    with pytest.raises(InvalidProfileError, match="chunk_max_tokens 20 is below the minimum 24"):
        Container.build(make_settings(knowledge={"profiles_dir": tmp_path}))


def test_build_fails_when_a_template_reads_an_unknown_variable(tmp_path: Path) -> None:
    (tmp_path / "t.yaml").write_text(
        "key: t\ntop_k: 4\nchunk_max_tokens: 300\nparser_key: simple\n"
        "chunker_key: structured\nmetadata_fields:\n"
        "  - name: path\n    source: 'extractor:template'\n"
        "    params: {template: '{ghost}/x'}\n"
    )
    with pytest.raises(InvalidProfileError, match="reads \\['ghost'\\]"):
        Container.build(make_settings(knowledge={"profiles_dir": tmp_path}))


def test_build_fails_when_the_profiles_dir_is_missing(tmp_path: Path) -> None:
    with pytest.raises(InvalidProfileError, match="does not exist"):
        Container.build(make_settings(knowledge={"profiles_dir": tmp_path / "missing"}))


def test_build_without_reranker_url_registers_only_the_noop() -> None:
    assert set(Container.build(make_settings()).rerankers) == {"none"}


def test_build_does_no_io_and_hides_secrets() -> None:
    container = Container.build(make_settings(embeddings={"api_key": "super-secret"}))
    assert "super-secret" not in repr(container)
    assert "secret-pw" not in repr(container)


async def test_startup_checks_off_skips_the_tokenizer() -> None:
    settings = make_settings(embeddings={"tokenizer": "/does/not/exist.json"})
    async with Container.build(settings):
        pass


async def test_missing_tokenizer_fails_fast_at_boot(tmp_path: Path) -> None:
    settings = make_settings(
        startup_checks=True,
        embeddings={"tokenizer": str(tmp_path / "missing.json"), "url": "http://127.0.0.1:9"},
    )
    with pytest.raises(StartupError, match=r"missing\.json"):
        async with Container.build(settings):
            pytest.fail("the container must not start")  # pragma: no cover


async def test_loadable_tokenizer_boots_and_is_warm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def database_ok(engine: object, **kwargs: Any) -> str:
        return "0.8.7"

    monkeypatch.setattr("agintis.bootstrap.container.check_database", database_ok)
    settings = make_settings(
        startup_checks=True,
        embeddings={"tokenizer": str(_tokenizer_file(tmp_path)), "url": "http://127.0.0.1:9"},
    )
    async with Container.build(settings) as container:
        assert container.token_counter.count("hola hola") == 2


async def test_adapters_are_closed_on_exit_and_on_failed_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    closed: list[str] = []

    async def record_embeddings_close(self: object) -> None:
        closed.append("embeddings")

    monkeypatch.setattr(
        "agintis.modules.knowledge.infrastructure.tei.TeiEmbeddings.aclose",
        record_embeddings_close,
    )
    async with Container.build(make_settings()):
        assert closed == []
    assert closed == ["embeddings"]

    failing = make_settings(
        startup_checks=True, embeddings={"tokenizer": str(tmp_path / "no.json")}
    )
    with pytest.raises(StartupError):
        await Container.build(failing).__aenter__()
    assert closed == ["embeddings", "embeddings"]


async def test_reranker_client_is_closed_through_the_fallback_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[str] = []

    async def record(self: object) -> None:
        closed.append("reranker")

    monkeypatch.setattr("agintis.modules.knowledge.infrastructure.tei.TeiReranker.aclose", record)
    async with Container.build(make_settings(reranker={"url": "http://reranker:80"})):
        pass
    assert closed == ["reranker"]


async def test_build_wires_the_use_cases_over_postgres() -> None:
    container = Container.build(make_settings())
    assert isinstance(container.documents, PgDocumentRepository)
    assert isinstance(container.chunks, PgChunkRepository)
    assert isinstance(container.ingest_lock, PgIngestLock)
    assert container.engine.pool.size() == 3  # type: ignore[attr-defined]  # small default pool
    await container.aclose()


async def test_engines_are_disposed_on_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    disposed: list[str] = []

    async def record(self: AsyncEngine) -> None:
        disposed.append("engine")

    monkeypatch.setattr(AsyncEngine, "dispose", record)
    async with Container.build(make_settings()):
        assert disposed == []
    assert disposed == ["engine", "engine"]  # pooled engine + advisory-lock engine


async def test_unreachable_database_fails_fast_at_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def database_down(engine: object, **kwargs: Any) -> str:
        raise DatabaseCheckError("Database is not reachable: OperationalError")

    monkeypatch.setattr("agintis.bootstrap.container.check_database", database_down)
    settings = make_settings(
        startup_checks=True,
        embeddings={"tokenizer": str(_tokenizer_file(tmp_path)), "url": "http://127.0.0.1:9"},
    )
    with pytest.raises(StartupError, match="not reachable"):
        await Container.build(settings).__aenter__()


async def test_database_is_checked_against_the_minimum_pgvector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []

    async def database_ok(engine: object, **kwargs: Any) -> str:
        seen.append(kwargs)
        return "0.8.7"

    monkeypatch.setattr("agintis.bootstrap.container.check_database", database_ok)
    settings = make_settings(
        startup_checks=True,
        embeddings={"tokenizer": str(_tokenizer_file(tmp_path)), "url": "http://127.0.0.1:9"},
    )
    async with Container.build(settings):
        pass
    assert seen == [
        {
            "min_vector_version": (0, 8, 4),
            "expected_embedding_dim": 1024,  # EMBEDDINGS_EXPECTED_DIM
            "expected_revision": SCHEMA_HEAD_REVISION,
            "timeout_seconds": 15.0,
        }
    ]


def test_schema_head_revision_matches_the_alembic_scripts() -> None:
    """Bumping a migration without bumping the constant would block every boot."""
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    assert ScriptDirectory.from_config(config).get_heads() == [SCHEMA_HEAD_REVISION]


async def test_indexed_metadata_fields_with_a_wrong_index_are_warned_about(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def database_ok(engine: object, **kwargs: Any) -> str:
        return "0.8.7"

    async def problems(engine: object, fields: Iterable[MetadataField]) -> list[IndexEntry]:
        statuses = iter((IndexStatus.MISSING, IndexStatus.STALE, IndexStatus.UNMANAGED))
        return [IndexEntry(metadata_index_name(f), next(statuses)) for f in fields if f.indexed]

    monkeypatch.setattr("agintis.bootstrap.container.check_database", database_ok)
    monkeypatch.setattr("agintis.bootstrap.container.metadata_index_problems", problems)
    settings = make_settings(
        startup_checks=True,
        embeddings={"tokenizer": str(_tokenizer_file(tmp_path)), "url": "http://127.0.0.1:9"},
    )
    container = Container.build(settings)
    indexed = MetadataField("article", "str", filterable=True, indexed=True)
    container.profiles.register(
        RagProfile("law", 3, 100, "simple", "structured", metadata_fields=(indexed,))
    )
    with caplog.at_level(logging.WARNING, logger="agintis.bootstrap.container"):
        async with container:
            pass
    assert "ix_chunks_meta_article" in caplog.text
    assert "missing" in caplog.text
    assert "agintis.cli.indexes" in caplog.text


async def test_a_text_search_config_that_does_not_match_tsv_stops_the_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def database_ok(engine: object, **kwargs: Any) -> str:
        return "0.8.7"

    async def mismatch(engine: object, config: str) -> None:
        raise TextSearchConfigError(f"{config} differs from tsv")

    monkeypatch.setattr("agintis.bootstrap.container.check_database", database_ok)
    monkeypatch.setattr("agintis.bootstrap.container.verify_text_search_config", mismatch)
    settings = make_settings(
        startup_checks=True,
        embeddings={"tokenizer": str(_tokenizer_file(tmp_path)), "url": "http://127.0.0.1:9"},
        knowledge={"text_search_config": "simple"},
    )
    with pytest.raises(StartupError, match="simple differs from tsv"):
        async with Container.build(settings):
            pass


async def test_the_bm25_index_is_checked_only_when_a_profile_uses_bm25_hybrid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def database_ok(engine: object, **kwargs: Any) -> str:
        return "0.8.7"

    checked: list[object] = []

    async def missing(engine: object) -> None:
        checked.append(engine)
        raise TextSearchConfigError("needs the BM25 index ix_chunks_text_bm25")

    monkeypatch.setattr("agintis.bootstrap.container.check_database", database_ok)
    monkeypatch.setattr("agintis.bootstrap.container.verify_bm25_index", missing)
    settings = make_settings(
        startup_checks=True,
        embeddings={"tokenizer": str(_tokenizer_file(tmp_path)), "url": "http://127.0.0.1:9"},
    )
    async with Container.build(settings):
        pass
    assert checked == []  # no bm25_hybrid profile: the optional extension is never required

    container = Container.build(settings)
    container.profiles.register(
        RagProfile("bm", 3, 100, "simple", "structured", retrieval_mode=BM25_HYBRID_RETRIEVAL_MODE)
    )
    with pytest.raises(StartupError, match="BM25 index"):
        async with container:
            pass
    assert checked == [container.engine]


def test_build_fails_fast_on_incompatible_indexed_declarations(tmp_path: Path) -> None:
    base = "top_k: 4\nchunk_max_tokens: 300\nparser_key: simple\nchunker_key: structured\n"
    for key, kind in (("one", "int"), ("two", "str")):
        (tmp_path / f"{key}.yaml").write_text(
            f"key: {key}\n{base}metadata_fields:\n"
            f"  - {{name: number, type: {kind}, source: 'document:number',"
            " filterable: true, indexed: true}\n"
        )
    with pytest.raises(ConflictingIndexError, match="incompatible"):
        Container.build(make_settings(knowledge={"profiles_dir": tmp_path}))


def test_build_wires_the_query_cap_and_the_search_timeout() -> None:
    container = Container.build(
        make_settings(knowledge={"max_query_chars": 123}, database={"search_timeout_ms": 777})
    )
    assert container.retrieve_for_topic._max_query_chars == 123
    assert isinstance(container.chunks, PgChunkRepository)
    assert container.chunks._search_timeout == "777"


async def test_llm_boot_check_failure_aborts_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def database_ok(engine: object, **kwargs: Any) -> str:
        return "0.8.7"

    def llm_broken(self: object) -> None:
        raise LlmError("no analyzer structured output")

    monkeypatch.setattr("agintis.bootstrap.container.check_database", database_ok)
    monkeypatch.setattr("agintis.bootstrap.container.verify_served_model", _noop_verify)
    monkeypatch.setattr("agintis.bootstrap.container.ChatModels.verify", llm_broken)
    settings = make_settings(
        startup_checks=True, embeddings={"tokenizer": str(_tokenizer_file(tmp_path))}
    )
    with pytest.raises(StartupError, match="analyzer structured output"):
        async with Container.build(settings):
            pytest.fail("the container must not start")  # pragma: no cover


async def _noop_verify(embeddings: object, model: str) -> None:
    return None


def test_build_wires_chat_models_without_building_anything() -> None:
    disabled = Container.build(make_settings())
    assert not disabled.chat_models.enabled
    configured = Container.build(
        make_settings(llm={"provider": "openai", "model": "base", "api_key": "sk-dummy"})
    )
    assert configured.chat_models.enabled
    assert configured.chat_models.catalog._clients == {}


async def test_real_llm_verify_failure_that_is_not_an_llm_error_aborts_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Real `verify`: a provider that cannot be built (no key) must surface as StartupError."""

    def failing_builder(model: str, **kwargs: Any) -> object:
        raise RuntimeError("api_key must be set")

    async def database_ok(engine: object, **kwargs: Any) -> str:
        return "0.8.7"

    real_verify = _REAL_VERIFY
    monkeypatch.setattr("agintis.bootstrap.container.check_database", database_ok)
    monkeypatch.setattr("agintis.bootstrap.container.verify_served_model", _noop_verify)
    monkeypatch.setattr("agintis.bootstrap.container.ChatModels.verify", real_verify)
    settings = make_settings(
        startup_checks=True,
        embeddings={"tokenizer": str(_tokenizer_file(tmp_path))},
        llm={"provider": "openai"},
    )
    container = Container.build(settings)
    config = LlmConfig(provider="openai", models={role: "m" for role in ChatRole})
    broken = ChatModels(config, failing_builder)  # type: ignore[arg-type]
    object.__setattr__(container, "chat_models", broken)
    with pytest.raises(StartupError, match="api_key must be set"):
        async with container:
            pytest.fail("the container must not start")  # pragma: no cover


class TestTenantResolverOnlyOnTheHttpPath:
    def _jwt_settings(self, tmp_path: Path) -> Settings:
        return make_settings(
            auth={
                "tenant_resolver": "jwt",
                "issuer": "https://idp.example",
                "audience": "api",
                "jwks_file": tmp_path / "missing.json",  # building the resolver would fail
            }
        )

    def test_a_cli_style_container_neither_builds_nor_logs_it(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            container = Container.build(self._jwt_settings(tmp_path))
            header = Container.build(make_settings())
        assert container.tenant_resolver is None
        assert header.tenant_resolver is None
        assert not any("AUTH_" in r.getMessage() for r in caplog.records)
        with pytest.raises(RuntimeError):
            container.require_tenant_resolver()

    def test_the_http_path_builds_it(self, tmp_path: Path) -> None:
        with pytest.raises(AuthConfigError):
            Container.build(self._jwt_settings(tmp_path), with_tenant_resolver=True)
        container = Container.build(make_settings(), with_tenant_resolver=True)
        assert container.require_tenant_resolver() is container.tenant_resolver
