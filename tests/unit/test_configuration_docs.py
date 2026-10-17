"""The docs/configuration*.md rows must match the environment variables `Settings` accepts."""

import re
from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings

from agintis.bootstrap.settings import Settings

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
DOCS = [DOCS_DIR / "configuration.md", DOCS_DIR / "configuration.es.md"]
COMPOSE_HEADING = r"## (?:Compose-only variables|Variables solo de Compose)"


def _is_group(field: FieldInfo) -> bool:
    annotation = field.annotation
    return isinstance(annotation, type) and issubclass(annotation, BaseSettings)


def _env_name(prefix: str, name: str, field: FieldInfo) -> str:
    alias = field.validation_alias
    return (alias if isinstance(alias, str) else f"{prefix}{name}").upper()


def settings_env_names() -> set[str]:
    """Flat env names: top-level fields as-is, nested groups with their own `env_prefix`."""
    names: set[str] = set()
    for name, field in Settings.model_fields.items():
        if _is_group(field):
            group = field.annotation
            assert isinstance(group, type)
            assert issubclass(group, BaseModel)
            prefix = str(group.model_config.get("env_prefix", ""))
            names.update(_env_name(prefix, n, f) for n, f in group.model_fields.items())
        else:
            names.add(_env_name("", name, field))
    return names


def documented_settings_names(doc: Path) -> set[str]:
    """Names in the table rows (`| `NAME` |`) of the sections before "Compose-only variables"."""
    text = re.split(COMPOSE_HEADING, doc.read_text(encoding="utf-8"))[0]
    return set(re.findall(r"^\| `([A-Z][A-Z0-9_]*)` \|", text, flags=re.MULTILINE))


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_documented_rows_match_settings_env_vars(doc: Path) -> None:
    names = settings_env_names()
    documented = documented_settings_names(doc)
    assert len(names) > 50, "the walk found suspiciously few variables"

    assert not names - documented, f"missing rows: {', '.join(sorted(names - documented))}"
    assert not documented - names, f"stale rows: {', '.join(sorted(documented - names))}"
