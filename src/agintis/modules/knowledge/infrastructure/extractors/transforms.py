"""Value transforms for metadata fields (registered by key, configured by params)."""

import re
import unicodedata
from collections.abc import Callable
from datetime import date
from typing import Literal

from pydantic import Field

from agintis.modules.knowledge.application import RawValue
from agintis.modules.knowledge.infrastructure.extractors.base import NoParams, ParamsTransform
from agintis.modules.knowledge.profiles import ParamsModel

type _Scalar = str | int | float | bool | None


def _each(value: RawValue, transform: Callable[[_Scalar], _Scalar]) -> RawValue:
    """Apply `transform` to a scalar, or to every item of a tuple (None results are dropped)."""
    if isinstance(value, tuple):
        mapped = tuple(item for item in map(transform, value) if item is not None)
        return mapped or None
    return transform(value)


class ValueMapParams(ParamsModel):
    map: dict[str, str] = Field(min_length=1)


class ValueMap(ParamsTransform[ValueMapParams]):
    """Replace known strings (`area: billing -> BL`); unknown values pass through."""

    key = "value_map"
    params_model = ValueMapParams

    def _apply(self, value: RawValue, params: ValueMapParams, today: date) -> RawValue:
        return _each(value, lambda v: params.map.get(v, v) if isinstance(v, str) else v)


_YEAR = re.compile(r"\d{4}")
_YEAR_MONTH = re.compile(r"\d{4}-\d{2}")
_FULL_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class DateFloor(ParamsTransform[NoParams]):
    """Complete a partial ISO date to its first day ("2021" -> "2021-01-01").

    An ISO-shaped date that is not a real calendar date ("2021-13", "2021-02-30") is dropped
    instead of being passed on to fail the whole ingest later.
    """

    key = "date_floor"
    version = 2  # impossible dates are dropped
    params_model = NoParams

    def _apply(self, value: RawValue, params: NoParams, today: date) -> RawValue:
        def floor(item: _Scalar) -> _Scalar:
            if not isinstance(item, str):
                return item
            if _YEAR.fullmatch(item):
                item = f"{item}-01-01"
            elif _YEAR_MONTH.fullmatch(item):
                item = f"{item}-01"
            elif not _FULL_DATE.fullmatch(item):
                return item
            try:
                date.fromisoformat(item)
            except ValueError:
                return None
            return item

        return _each(value, floor)


class RejectAfterParams(ParamsModel):
    limit: Literal["ingest_date"] = "ingest_date"


class RejectAfter(ParamsTransform[RejectAfterParams]):
    """Drop an ISO date later than the day of the ingest (a typo or a placeholder date)."""

    key = "reject_after"
    params_model = RejectAfterParams

    def _apply(self, value: RawValue, params: RejectAfterParams, today: date) -> RawValue:
        limit = today.isoformat()
        return _each(value, lambda v: None if isinstance(v, str) and v > limit else v)


class PathSegmentParams(ParamsModel):
    separator: str = Field(default=">", min_length=1)
    take: Literal["first", "last"] = "last"


class PathSegment(ParamsTransform[PathSegmentParams]):
    """Keep one segment of "A > B > C" paths; a list becomes its sorted distinct segments."""

    key = "path_segment"
    version = 2  # mixed str/int lists no longer fail
    params_model = PathSegmentParams

    def _apply(self, value: RawValue, params: PathSegmentParams, today: date) -> RawValue:
        def segment(item: _Scalar) -> _Scalar:
            if not isinstance(item, str):
                return item
            parts = item.split(params.separator)
            return (parts[-1] if params.take == "last" else parts[0]).strip()

        mapped = _each(value, segment)
        if not isinstance(mapped, tuple):
            return mapped
        # Sorted by (type, text): a list mixing strings and numbers has no natural order.
        return tuple(sorted(set(mapped), key=lambda item: (type(item).__name__, str(item))))


def _ascii_lower(text: str) -> str:
    return unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()


class PrefixMapParams(ParamsModel):
    map: dict[str, str] = Field(min_length=1)


class PrefixMap(ParamsTransform[PrefixMapParams]):
    """Map a word to a label by its stem: accents and case are ignored, the LONGEST matching
    prefix wins and a word with no known prefix has no value ("Reemplácese" -> "replace")."""

    key = "prefix_map"
    params_model = PrefixMapParams

    def _apply(self, value: RawValue, params: PrefixMapParams, today: date) -> RawValue:
        stems = sorted(
            ((_ascii_lower(k), v) for k, v in params.map.items()), key=lambda i: -len(i[0])
        )

        def lemma(item: _Scalar) -> _Scalar:
            if not isinstance(item, str):
                return None
            word = _ascii_lower(item)
            return next((label for stem, label in stems if word.startswith(stem)), None)

        return _each(value, lemma)
