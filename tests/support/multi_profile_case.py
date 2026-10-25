"""Synthetic multi-profile case (neutral content) for tests of multi-profile features."""

from tests.support.postgres import PROJECT_ROOT

FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "multi_profile_case"
PROFILES_DIR = FIXTURE_ROOT / "profiles"
AGENTS_DIR = FIXTURE_ROOT / "agents"
# The shipped example case: its profiles must keep validating against the real registries.
EXAMPLE_PROFILES_DIR = PROJECT_ROOT / "case" / "fastapi_es" / "profiles"
