"""CI must run the same DB image definition and TEI image as docker-compose.yml."""

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
TEI_IMAGE = re.compile(r"ghcr\.io/huggingface/text-embeddings-inference:\S+")
DOCKER_BUILD = re.compile(r"^\s*docker build .*?(\S+)\s*$", re.MULTILINE)
FROM_IMAGE = re.compile(r"^FROM\s+(\S+)", re.MULTILINE)


def _load(path: str) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    return data


def _integration_steps() -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = _load(".github/workflows/ci.yml")["jobs"]["integration"]["steps"]
    return steps


def test_ci_builds_the_compose_db_image() -> None:
    build = _load("docker-compose.yml")["services"]["db"]["build"]
    context = Path(build if isinstance(build, str) else build["context"])
    found = [Path(m) for s in _integration_steps() for m in DOCKER_BUILD.findall(s.get("run", ""))]
    assert found == [context], f"ci.yml builds {found!r}, docker-compose.yml db builds {context!r}"


def test_db_image_builds_and_runs_on_the_same_base() -> None:
    """The extension is compiled against the build stage's server headers."""
    dockerfile = (ROOT / "docker/postgres/Dockerfile").read_text(encoding="utf-8")
    bases = FROM_IMAGE.findall(dockerfile)
    assert len(bases) == 2, bases
    assert len(set(bases)) == 1, f"build and runtime stages differ: {bases!r}"


def test_ci_tei_image_matches_compose() -> None:
    compose = _load("docker-compose.yml")["services"]["embeddings"]["image"]
    found = [m.group(0) for s in _integration_steps() if (m := TEI_IMAGE.search(s.get("run", "")))]
    assert found, "no TEI image found in any ci.yml integration step"
    assert found == [compose], f"ci.yml TEI image {found!r} != docker-compose.yml {compose!r}"


def test_the_gpu_reranker_override_runs_the_compose_tei_version() -> None:
    cpu = _load("docker-compose.yml")["services"]["embeddings"]["image"]
    gpu = _load("docker-compose.gpu.yml")["services"]["reranker"]["image"]
    default_tag = re.fullmatch(r".*:\$\{RERANKER_GPU_TAG:-(\S+)\}", gpu)
    assert default_tag, f"unexpected GPU reranker image {gpu!r}"
    # cpu-1.9.4 -> 1.9.4; the GPU tags are <compute capability>-1.9.4 or plain 1.9.4.
    version = cpu.rsplit(":", 1)[1].removeprefix("cpu-")
    assert default_tag.group(1).rsplit("-", 1)[-1] == version
