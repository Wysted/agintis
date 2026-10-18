import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from langchain_core.embeddings import DeterministicFakeEmbedding

from agintis.modules.knowledge.api import (
    IngestDocument,
    InvalidProfileError,
    MetadataResolver,
    RagProfile,
    RagProfileRegistry,
    SourceFile,
    UnknownStrategyError,
    load_profile,
    load_profiles_dir,
    parse_profile,
    validate_profiles,
)
from agintis.modules.knowledge.infrastructure.parsers import SimpleParser
from agintis.modules.knowledge.infrastructure.registry import (
    build_chunkers,
    build_cleaners,
    build_extractors,
    build_parsers,
    build_retrieval_modes,
    build_transforms,
    chunker_param_schemas,
    cleaner_ordering,
    cleaner_param_schemas,
    extractor_param_schemas,
    transform_param_schemas,
)
from agintis.modules.knowledge.profiles.loader import (
    MAX_PROFILE_FILE_BYTES,
    load_yaml_text,
)
from agintis.shared_kernel.ids import TenantId
from tests.fakes.clock import FixedClock
from tests.fakes.knowledge import (
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    NullIngestLock,
)
from tests.support.postgres import PROJECT_ROOT

CASE_PROFILES = PROJECT_ROOT / "case" / "spensiones" / "profiles"

VALID = """\
key: legal
top_k: 4
chunk_max_tokens: 400
parser_key: simple
chunker_key: structured
chunker_params:
  include_heading: true
  cross_pages: true
cleaners:
  - key: strip_lines
    params:
      patterns: ['^\\d+$']
  - key: normalize_whitespace
metadata_fields:
  - {name: article, type: int, source: 'document:article', filterable: true, indexed: true}
"""


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())

    def split(self, text: str, max_tokens: int) -> list[str]:
        words = text.split()
        return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]


def _validate(*profiles: RagProfile) -> None:
    validate_profiles(
        RagProfileRegistry(profiles),
        ["simple"],
        build_chunkers(WordCounter()),
        chunker_schemas=chunker_param_schemas(),
        cleaner_schemas=cleaner_param_schemas(),
        cleaner_ordering=cleaner_ordering(),
    )


def _raw(text: str = VALID, **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(text)
    data.update(overrides)
    return data


class TestLoadProfile:
    def test_builds_the_domain_profile(self, tmp_path: Path) -> None:
        (tmp_path / "legal.yaml").write_text(VALID)
        profile = load_profile(tmp_path / "legal.yaml")
        assert (profile.name, profile.top_k, profile.chunk_max_tokens) == ("legal", 4, 400)
        assert (profile.parser_key, profile.chunker_key, profile.reranker_key) == (
            "simple",
            "structured",
            "none",
        )
        assert dict(profile.chunker_params) == {"include_heading": True, "cross_pages": True}
        assert [(c.key, dict(c.params)) for c in profile.cleaners] == [
            ("strip_lines", {"patterns": ("^\\d+$",)}),
            ("normalize_whitespace", {}),
        ]
        field = profile.metadata_fields[0]
        assert (field.name, field.type, field.filterable, field.indexed, field.source) == (
            "article",
            "int",
            True,
            True,
            "document:article",
        )
        _validate(profile)

    def test_same_yaml_same_fingerprint_and_param_change_changes_it(self) -> None:
        first = parse_profile(_raw(), origin="a")
        again = parse_profile(yaml.safe_load(VALID), origin="b")
        changed = parse_profile(_raw(chunker_params={"include_heading": False}), origin="c")
        reordered = parse_profile(
            _raw(chunker_params={"cross_pages": True, "include_heading": True}), origin="d"
        )
        assert first.fingerprint == again.fingerprint == reordered.fingerprint
        assert first.fingerprint != changed.fingerprint

    def test_yaml_matching_the_default_profile_has_its_fingerprint(self) -> None:
        profile = parse_profile(
            {
                "key": "default2",
                "top_k": 5,
                "chunk_max_tokens": 512,
                "parser_key": "simple",
                "chunker_key": "structured",
                "chunker_params": {"include_heading": True},
            },
            origin="x",
        )
        from agintis.modules.knowledge.api import DEFAULT_PROFILE

        assert profile.fingerprint == DEFAULT_PROFILE.fingerprint

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"bogus": 1}, "bogus"),
            ({"routing": {"index_filter": {}}}, "routing"),
            ({"top_k": "5"}, "top_k"),
            ({"top_k": 0}, "top_k must be greater than 0"),
            ({"schema_version": 2}, "schema_version"),
            ({"key": None}, "key"),
            ({"default_filter": {"a": 1}}, "default_filter"),
            ({"cleaners": [{"name": "strip_lines"}]}, "cleaners.0"),
            ({"cleaners": [{"key": "strip_lines", "extra": 1}]}, "extra"),
            ({"cleaners": [{"key": ""}]}, "key must not be empty"),
            ({"chunker_params": {"": 1}}, "keys must be non-empty"),
            ({"chunker_params": {"a": object()}}, "unsupported value"),
            ({"metadata_fields": [{"name": "a", "type": "str_list"}]}, "metadata_fields.0.type"),
            (
                {"metadata_fields": [{"name": "a", "source": "nope:id"}]},
                "source must start with",
            ),
            (
                {"metadata_fields": [{"name": "a", "params": {"x": 1}, "source": "group:a"}]},
                "params but no extractor source",
            ),
            (
                {"metadata_fields": [{"name": "a", "transforms": [{"key": "value_map"}]}]},
                "transforms but no source",
            ),
            ({"status": "retired"}, "status"),
            ({"routing": [{"match": {}}]}, "routing.0.match"),
            ({"routing": [{"match": {"title": {"regex": "(a+)+"}}}]}, "nested quantifier"),
            ({"extractors": {"Bad Name": {"key": "regex_all"}}}, "not a valid name"),
            ({"metadata_fields": [{"name": "a", "unknown": 1}]}, "unknown"),
            ({"metadata_fields": [{"name": "a"}, {"name": "a"}]}, "unique"),
        ],
    )
    def test_invalid_documents_name_the_problem_and_the_origin(
        self, overrides: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(InvalidProfileError, match=message) as excinfo:
            parse_profile(_raw(**overrides), origin="legal.yaml")
        assert str(excinfo.value).startswith("legal.yaml: ")

    def test_missing_required_fields_are_reported(self) -> None:
        with pytest.raises(InvalidProfileError, match=r"top_k.*Field required"):
            parse_profile({"key": "x"}, origin="x.yaml")

    def test_not_a_mapping(self, tmp_path: Path) -> None:
        (tmp_path / "list.yaml").write_text("- a\n- b\n")
        with pytest.raises(InvalidProfileError, match="YAML mapping"):
            load_profile(tmp_path / "list.yaml")

    def test_yaml_syntax_error_and_unreadable_file(self, tmp_path: Path) -> None:
        (tmp_path / "bad.yaml").write_text("key: [unclosed")
        with pytest.raises(InvalidProfileError, match=r"bad\.yaml: cannot read"):
            load_profile(tmp_path / "bad.yaml")
        with pytest.raises(InvalidProfileError, match="cannot read"):
            load_profile(tmp_path / "missing.yaml")

    def test_yaml_cannot_instantiate_python_objects(self, tmp_path: Path) -> None:
        (tmp_path / "evil.yaml").write_text("key: !!python/object/apply:os.getcwd []\n")
        with pytest.raises(InvalidProfileError, match="cannot read"):
            load_profile(tmp_path / "evil.yaml")


class TestLoadProfilesDir:
    def test_loads_yaml_files_in_name_order_and_ignores_the_rest(self, tmp_path: Path) -> None:
        (tmp_path / "b.yml").write_text(VALID.replace("key: legal", "key: b"))
        (tmp_path / "a.yaml").write_text(VALID.replace("key: legal", "key: a"))
        (tmp_path / "notes.txt").write_text("not a profile")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.yaml").write_text(VALID)
        assert [p.name for p in load_profiles_dir(tmp_path)] == ["a", "b"]

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert load_profiles_dir(tmp_path) == []

    def test_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidProfileError, match="does not exist"):
            load_profiles_dir(tmp_path / "nope")

    def test_duplicate_profile_keys_across_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.yaml").write_text(VALID)
        (tmp_path / "b.yaml").write_text(VALID)
        with pytest.raises(InvalidProfileError, match=r"b\.yaml.*already defined in a\.yaml"):
            load_profiles_dir(tmp_path)

    def test_one_bad_file_aborts_the_load_naming_it(self, tmp_path: Path) -> None:
        (tmp_path / "a.yaml").write_text(VALID)
        (tmp_path / "b.yaml").write_text("key: b\n")
        with pytest.raises(InvalidProfileError, match=r"b\.yaml"):
            load_profiles_dir(tmp_path)


class TestYamlHardening:
    BASE = "key: p\ntop_k: 5\nchunk_max_tokens: 512\nparser_key: simple\nchunker_key: structured\n"

    def _load(self, tmp_path: Path, text: str) -> RagProfile:
        path = tmp_path / "p.yaml"
        path.write_text(text)
        return load_profile(path)

    def test_duplicate_top_level_keys_are_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidProfileError, match=r"duplicate key 'top_k'"):
            self._load(tmp_path, self.BASE + "top_k: 50\n")

    def test_duplicate_nested_keys_are_rejected(self, tmp_path: Path) -> None:
        text = self.BASE + "chunker_params:\n  include_heading: true\n  include_heading: false\n"
        with pytest.raises(InvalidProfileError, match="duplicate key 'include_heading'"):
            self._load(tmp_path, text)

    def test_the_same_key_in_different_mappings_is_fine(self, tmp_path: Path) -> None:
        text = (
            self.BASE
            + "cleaners:\n  - {key: normalize_whitespace}\n  - {key: normalize_whitespace}\n"
        )
        assert len(self._load(tmp_path, text).cleaners) == 2

    def test_anchors_and_aliases_are_rejected(self, tmp_path: Path) -> None:
        text = self.BASE + "chunker_params:\n  a: &x true\n  b: *x\n"
        with pytest.raises(InvalidProfileError, match="anchors and aliases"):
            self._load(tmp_path, text)

    def test_an_anchor_alone_is_rejected_too(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidProfileError, match="anchors and aliases"):
            self._load(tmp_path, self.BASE + "chunker_params:\n  a: &x true\n")

    def test_merge_keys_are_rejected(self, tmp_path: Path) -> None:
        text = self.BASE + "chunker_params:\n  <<: {include_heading: true}\n"
        with pytest.raises(InvalidProfileError):
            self._load(tmp_path, text)

    def test_an_alias_bomb_fails_fast_without_expanding(self, tmp_path: Path) -> None:
        lines = ["l0: &l0 [x, x, x, x, x, x, x, x, x, x]"]
        for level in range(1, 9):
            refs = ", ".join([f"*l{level - 1}"] * 10)
            lines.append(f"l{level}: &l{level} [{refs}]")
        text = self.BASE + "chunker_params:\n" + "\n".join(f"  {line}" for line in lines) + "\n"
        started = time.perf_counter()
        with pytest.raises(InvalidProfileError, match="anchors and aliases"):
            self._load(tmp_path, text)
        assert time.perf_counter() - started < 1

    def test_a_file_over_the_size_cap_is_rejected_without_parsing(self, tmp_path: Path) -> None:
        (tmp_path / "big.yaml").write_text("# " + "x" * MAX_PROFILE_FILE_BYTES + "\n" + self.BASE)
        with pytest.raises(InvalidProfileError, match=r"big\.yaml.*larger than"):
            load_profile(tmp_path / "big.yaml")

    def test_a_file_at_the_cap_is_read(self, tmp_path: Path) -> None:
        padding = "#" * (MAX_PROFILE_FILE_BYTES - len(self.BASE) - 1) + "\n"
        (tmp_path / "ok.yaml").write_text(self.BASE + padding)
        assert (tmp_path / "ok.yaml").stat().st_size <= MAX_PROFILE_FILE_BYTES
        assert load_profile(tmp_path / "ok.yaml").name == "p"

    def test_non_utf8_content_is_a_clean_error(self, tmp_path: Path) -> None:
        (tmp_path / "latin.yaml").write_bytes(b"key: \xf1\n")
        with pytest.raises(InvalidProfileError, match="cannot read"):
            load_profile(tmp_path / "latin.yaml")

    def test_absurd_nesting_is_a_clean_error_not_a_crash(self, tmp_path: Path) -> None:
        text = self.BASE + "chunker_params: " + "[" * 5000 + "]" * 5000 + "\n"
        with pytest.raises(InvalidProfileError, match=r"nested too deeply|cannot read"):
            self._load(tmp_path, text)

    def test_deeply_nested_parsed_data_is_a_clean_error(self) -> None:
        deep: Any = {}
        for _ in range(5000):
            deep = {"a": deep}
        with pytest.raises(InvalidProfileError, match="nested too deeply"):
            parse_profile(_raw(chunker_params=deep), origin="deep.yaml")

    def test_symlinks_leaving_the_directory_are_ignored(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        outside = tmp_path / "outside"
        inside = tmp_path / "profiles"
        outside.mkdir()
        inside.mkdir()
        (outside / "secret.yaml").write_text(VALID.replace("key: legal", "key: secret"))
        (inside / "real.yaml").write_text(VALID.replace("key: legal", "key: real"))
        (inside / "escape.yaml").symlink_to(outside / "secret.yaml")
        assert [p.name for p in load_profiles_dir(inside)] == ["real"]
        assert "escape.yaml" in caplog.text

    def test_symlinks_inside_the_directory_are_followed(self, tmp_path: Path) -> None:
        (tmp_path / "real.yaml").write_text(VALID.replace("key: legal", "key: real"))
        (tmp_path / "alias.yaml").symlink_to(tmp_path / "real.yaml")
        with pytest.raises(InvalidProfileError, match="already defined"):
            load_profiles_dir(tmp_path)

    def test_a_symlinked_directory_is_accepted_as_the_profiles_dir(self, tmp_path: Path) -> None:
        real = tmp_path / "real"
        real.mkdir()
        (real / "a.yaml").write_text(VALID)
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)
        assert [p.name for p in load_profiles_dir(link)] == ["legal"]

    def test_a_dangling_symlink_is_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "a.yaml").write_text(VALID)
        (tmp_path / "gone.yaml").symlink_to(tmp_path / "missing.yaml")
        assert [p.name for p in load_profiles_dir(tmp_path)] == ["legal"]

    def test_the_bundled_case_files_still_parse_with_the_strict_loader(self) -> None:
        for name in CASE_FILES:
            load_yaml_text((CASE_PROFILES / name).read_text(encoding="utf-8"), origin=name)


class TestCleanerOrdering:
    """W6: html_main_content rebuilds the text, so only read-only cleaners may precede it."""

    def _profile(self, *keys: str) -> RagProfile:
        params = {
            "strip_lines": {"patterns": ["x"]},
            "html_collect": {"fields": {"k": {"selector": "p"}}},
        }
        cleaners = [{"key": k, "params": params.get(k, {})} for k in keys]
        return parse_profile(_raw(cleaners=cleaners), origin="t")

    def test_main_content_first_is_valid(self) -> None:
        _validate(self._profile("html_main_content", "strip_lines", "normalize_whitespace"))

    def test_html_collect_may_precede_it(self) -> None:
        _validate(self._profile("html_collect", "html_main_content", "normalize_whitespace"))

    @pytest.mark.parametrize("before", ["strip_lines", "normalize_whitespace", "normalize_quotes"])
    def test_a_text_cleaner_before_it_is_rejected_with_its_position(self, before: str) -> None:
        profile = self._profile(before, "html_main_content")
        with pytest.raises(
            InvalidProfileError, match=rf"cleaner #2 'html_main_content'.*#1 '{before}'"
        ):
            _validate(profile)

    def test_a_text_cleaner_between_collect_and_main_content_is_rejected(self) -> None:
        profile = self._profile("html_collect", "strip_lines", "html_main_content")
        with pytest.raises(InvalidProfileError, match=r"#2 'strip_lines'"):
            _validate(profile)

    def test_the_rule_is_off_without_a_declared_ordering(self) -> None:
        profile = self._profile("strip_lines", "html_main_content")
        validate_profiles(
            RagProfileRegistry([profile]),
            ["simple"],
            ["structured"],
            chunker_schemas=chunker_param_schemas(),
            cleaner_schemas=cleaner_param_schemas(),
        )

    def test_the_ordering_comes_from_the_cleaners_themselves(self) -> None:
        ordering = cleaner_ordering()
        assert ordering.rebuilds_text == {"html_main_content"}
        assert ordering.read_only == {"html_collect"}


class TestValidateProfilesParams:
    def _profile(self, **overrides: Any) -> RagProfile:
        return parse_profile(_raw(**overrides), origin="t")

    def test_valid_profile_passes(self) -> None:
        _validate(self._profile())

    def test_default_profile_passes(self) -> None:
        from agintis.modules.knowledge.api import DEFAULT_PROFILE

        _validate(DEFAULT_PROFILE)

    def test_unknown_chunker_param_is_reported_with_profile_and_key(self) -> None:
        profile = self._profile(chunker_params={"include_heading": True, "overlap": 5})
        with pytest.raises(
            InvalidProfileError, match=r"profile 'legal'.*chunker 'structured'.*overlap"
        ):
            _validate(profile)

    def test_wrong_chunker_param_type(self) -> None:
        profile = self._profile(chunker_params={"cross_pages": "yes"})
        with pytest.raises(InvalidProfileError, match="cross_pages"):
            _validate(profile)

    def test_invalid_cleaner_params_report_the_position(self) -> None:
        profile = self._profile(
            cleaners=[
                {"key": "normalize_whitespace"},
                {"key": "strip_lines", "params": {"patterns": ["("]}},
            ]
        )
        with pytest.raises(
            InvalidProfileError, match=r"cleaner #2 'strip_lines'.*regular expression"
        ):
            _validate(profile)

    def test_unknown_cleaner_key_is_an_unregistered_strategy(self) -> None:
        profile = self._profile(cleaners=[{"key": "os.system"}])
        with pytest.raises(UnknownStrategyError, match=r"cleaner 'os\.system'"):
            _validate(profile)

    def test_unknown_chunker_key(self) -> None:
        profile = self._profile(chunker_key="hybrid", chunker_params={})
        with pytest.raises(UnknownStrategyError, match="chunker 'hybrid'"):
            _validate(profile)

    def test_a_registered_chunker_without_schema_accepts_no_params(self) -> None:
        profile = self._profile()
        registry = RagProfileRegistry([profile])
        with pytest.raises(InvalidProfileError, match="takes no parameters"):
            validate_profiles(
                registry,
                ["simple"],
                ["structured"],
                chunker_schemas={},
                cleaner_schemas=cleaner_param_schemas(),
            )

    def test_param_validation_is_skipped_without_schemas(self) -> None:
        profile = self._profile(chunker_params={"anything": 1}, cleaners=[])
        validate_profiles(RagProfileRegistry([profile]), ["simple"], ["structured"])

    def test_every_problem_is_listed(self) -> None:
        profiles = [
            self._profile(key="a", chunker_params={"x": 1}),
            self._profile(key="b", chunker_params={"y": 1}),
        ]
        with pytest.raises(InvalidProfileError) as excinfo:
            _validate(*profiles)
        assert "profile 'a'" in str(excinfo.value)
        assert "profile 'b'" in str(excinfo.value)


CASE_FILES = sorted(p.name for p in CASE_PROFILES.glob("*.yaml"))


def _case_validate(registry: RagProfileRegistry) -> None:
    """The case profiles against the REAL registries, the way the container validates them."""
    validate_profiles(
        registry,
        build_parsers(
            max_bytes=10**7,
            max_pages=100,
            max_uncompressed_bytes=10**7,
            max_compression_ratio=100.0,
        ),
        build_chunkers(WordCounter()),
        chunker_schemas=chunker_param_schemas(),
        cleaner_schemas=cleaner_param_schemas(),
        cleaner_ordering=cleaner_ordering(),
        extractors=build_extractors(),
        extractor_schemas=extractor_param_schemas(),
        transform_schemas=transform_param_schemas(),
        retrieval_mode_keys=build_retrieval_modes(branch_depth=50, rrf_k=60).keys(),
    )


class TestSpensionesCase:
    def test_the_case_files_are_found(self) -> None:
        assert "articulo_faq.yaml" in CASE_FILES
        assert len(CASE_FILES) == 7

    def test_the_directory_loads_without_any_adapter(self) -> None:
        profiles = load_profiles_dir(CASE_PROFILES)
        assert [p.name for p in profiles] == [
            "spensiones_articulo_faq",
            "spensiones_circular",
            "spensiones_compendio",
            "spensiones_jurisprudencia_indice",
            "spensiones_jurisprudencia_nota",
            "spensiones_ncg",
            "spensiones_resolucion",
        ]

    def test_every_profile_validates_against_the_real_registries(self) -> None:
        _case_validate(RagProfileRegistry(load_profiles_dir(CASE_PROFILES)))

    def test_canonical_shape_cleaners_sources_and_types(self) -> None:
        by_name = {p.name: p for p in load_profiles_dir(CASE_PROFILES)}
        compendio = by_name["spensiones_compendio"]
        assert [c.key for c in compendio.cleaners] == [
            "drop_toc_pages",
            "strip_running_headers",
            "dedupe_heading_echo",
            "normalize_whitespace",
        ]
        types = {f.name: f.type for f in compendio.metadata_fields}
        assert types["numeros"] == "list[str]"
        assert types["amended_by_ncg"] == "list[int]"
        sources = {f.name: f.source for f in compendio.metadata_fields}
        assert sources["titulo"] == "group:titulo"
        assert sources["libro"] == "document:number"
        assert sources["validity"] == "const:consolidated_current"
        assert sources["compendio_path"] == "extractor:template"

    def test_every_profile_retrieves_bm25_hybrid(self) -> None:
        """Adopted after the pre-registered measurement (see README)."""
        modes = {p.retrieval_mode for p in load_profiles_dir(CASE_PROFILES)}
        assert modes == {"bm25_hybrid"}

    def test_only_the_unvalidated_note_profile_is_a_draft(self) -> None:
        drafts = [p.name for p in load_profiles_dir(CASE_PROFILES) if p.status == "draft"]
        assert drafts == ["spensiones_jurisprudencia_nota"]

    def test_profile_fingerprints_are_stable_across_loads(self) -> None:
        first = {p.name: p.fingerprint for p in load_profiles_dir(CASE_PROFILES)}
        second = {p.name: p.fingerprint for p in load_profiles_dir(CASE_PROFILES)}
        assert first == second
        assert len(set(first.values())) == 7

    async def test_articulo_faq_sample_matches_the_reference_preview(self) -> None:
        """Reference: validation_report.md (10 blocks, keywords 2, related_aids 7).

        The sample is gitignored raw data: the test only runs where it was downloaded.
        """
        sample = CASE_PROFILES.parent / "samples" / "articulos_html" / "cms-aid-3590.html"
        if not sample.exists():
            pytest.skip("case sample not downloaded")
        profile = load_profile(CASE_PROFILES / "articulo_faq.yaml")
        parsed = await SimpleParser().parse(
            SourceFile(sample.name, "text/html", sample.read_bytes())
        )
        for spec in profile.cleaners:
            parsed = await build_cleaners()[spec.key].clean(parsed, spec.params)
        title = "¿Cuánto me deben descontar de mi sueldo para el seguro de cesantía?"
        assert parsed.title == title
        assert [s.headings[1:] for s in parsed.sections] == [
            (),
            ("Personas con contrato indefinido",),
            ("Personas con contrato a plazo, obra, faena o servicio determinado",),
        ]
        assert all(s.headings[0] == title for s in parsed.sections)
        text_blocks = sum(len(s.text.split("\n\n")) for s in parsed.sections)
        assert text_blocks + 1 + 2 == 10  # + the title and the two sub-headings
        assert dict(parsed.collected) == {
            "keywords": ("seguro de cesantía", "remuneraciones"),
            "related_aids": (3714, 3715, 3600, 3605, 3566, 2686, 7856),
        }
        flat = " ".join(s.text for s in parsed.sections)
        assert "- Un 0,6% de su remuneración imponible" in flat
        for gone in ("Aumentar", "Glosario", "Otras preguntas"):
            assert gone not in flat

    async def test_articulo_faq_end_to_end_ingest(self) -> None:
        profile = load_profile(CASE_PROFILES / "articulo_faq.yaml")
        chunks = InMemoryChunkRepository()
        ingest = IngestDocument(
            InMemoryDocumentRepository(),
            chunks,
            {"simple": SimpleParser()},
            build_chunkers(WordCounter()),
            DeterministicFakeEmbedding(size=8),
            FixedClock(),
            NullIngestLock(),
            cleaners=build_cleaners(),
            metadata=MetadataResolver(build_extractors(), build_transforms(), FixedClock()),
        )
        html = (
            "<html><head><title>Site</title></head><body><nav>Menu</nav>"
            '<div id="article_i__SP_ar_articulo_titulo_1"><h1>  Como   cotizo?  </h1></div>'
            '<div id="article_i__SP_ar_articulo_resumen_y_cuerpo_1">'
            "<p>Debe   informar al empleador.</p>"
            '<div style="display:none">cotizar,  afp</div></div>'
            "<footer>Pie</footer></body></html>"
        )
        result = await ingest.execute(
            TenantId("tenant"),
            profile,
            SourceFile(
                "https://www.spensiones.cl/portal/faq/w3-article-3590.html",
                "text/html",
                html.encode(),
                document_metadata={
                    "id": "cms-aid-3590",
                    "aid": "3590",
                    "area": "cesantia+pensiones",
                    "section": ("Atención de usuarios > Empleadores", "Educación > Seguro"),
                    "title": "Como cotizo?",
                },
            ),
        )
        assert result.chunk_count == 1
        ((stored, _),) = chunks.items
        assert stored.text == "Como cotizo?\n\nDebe informar al empleador."
        assert dict(stored.metadata) == {
            "heading": "Como cotizo?",
            "source_id": "cms-aid-3590",
            "doc_family": "articulo_faq",
            "aid": 3590,
            "area": "cesantia+pensiones",
            "audiences": ("Empleadores", "Seguro"),
            "is_question": "yes",
            "keywords": ("cotizar", "afp"),
        }
