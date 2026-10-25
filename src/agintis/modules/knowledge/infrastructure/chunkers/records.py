"""Record chunker (key "records"): index-like documents made of one record per line.

A jurisprudence index, a rates table, a glossary: lines such as
`7.12.4. Widget gasket   Ref. 17141 of 02-08-18` grouped under headings. Lines are
classified one at a time:

* `record_pattern` match -> a record line of the current group;
* a `group_levels` match -> context (e.g. subject > branch > topic). The level named by
  `group_by` also STARTS a new group (its own line is the group's first line); shallower
  levels only update the context. A level resets every level declared after it;
* anything else -> appended to the current group (a wrapped continuation).

One chunk per group, split into several when over budget, each starting with `header` = the
context values named by `header_groups` joined by " > " (when `repeat_group_header`). The header
is capped at `budget // MAX_HEADING_BUDGET_SHARE` tokens (the `heading` metadata keeps it whole),
so the lines always have room. A single line over budget is cut with `BudgetSplitter`. `groups`
carries the context named groups (`group:<name>` metadata); `page`/`page_end` come from the line
offsets.

Lines that precede the first group (an introduction, notes) are the PREAMBLE: with `preamble:
keep` (default) they become chunks headed by `preamble_heading` (none when absent); `drop`
discards them. Record lines before the first group belong to the preamble too.
"""

import asyncio
import re
from dataclasses import dataclass
from typing import Final, Literal, Self

from pydantic import Field, field_validator, model_validator

from agintis.modules.knowledge.application import ChunkDraft, Chunker, JoinedText, ParsedDocument
from agintis.modules.knowledge.domain import MetadataValue, RagProfile
from agintis.modules.knowledge.infrastructure.chunkers.splitting import (
    MAX_HEADING_BUDGET_SHARE,
    MIN_CHUNK_MAX_TOKENS,
    BudgetSplitter,
    chunk_budget,
)
from agintis.modules.knowledge.infrastructure.chunkers.tokens import TokenCounter
from agintis.modules.knowledge.profiles import (
    Flag,
    ParamsModel,
    Pattern,
    ProfileRegex,
    compile_profile_regex,
    validate_params,
)

RECORDS_CHUNKER_KEY: Final = "records"
HEADER_SEPARATOR: Final = " > "
_NAME: Final = re.compile(r"[a-z][a-z0-9_]*")


class GroupLevelParams(ParamsModel):
    name: str
    pattern: Pattern

    @field_validator("name")
    @classmethod
    def _identifier(cls, value: str) -> str:
        if not _NAME.fullmatch(value):
            raise ValueError("group level names must match [a-z][a-z0-9_]*")
        return value


class RecordsParams(ParamsModel):
    group_levels: list[GroupLevelParams] = Field(min_length=1)
    record_pattern: Pattern
    group_by: str
    header_groups: list[str] = Field(default_factory=list)
    repeat_group_header: Flag = True
    preamble: Literal["keep", "drop"] = "keep"
    preamble_heading: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        names = [level.name for level in self.group_levels]
        if len(set(names)) != len(names):
            raise ValueError("group level names must be unique")
        if self.group_by not in names:
            raise ValueError(f"group_by {self.group_by!r} is not a declared group level")
        unknown = [g for g in self.header_groups if g not in self.group_names()]
        if unknown:
            raise ValueError(f"header_groups names unknown regex groups {unknown}")
        return self

    def group_names(self) -> frozenset[str]:
        patterns = [level.pattern for level in self.group_levels] + [self.record_pattern]
        return frozenset(name for pattern in patterns for name in re.compile(pattern).groupindex)


@dataclass
class _Line:
    start: int
    end: int
    text: str


@dataclass
class _Group:
    context: dict[str, str]
    lines: list[_Line]
    preamble: bool = False


class RecordsChunker(Chunker):
    # 2: preamble lines are kept (or dropped on purpose), header capped to the budget share.
    version = 2
    min_chunk_max_tokens = MIN_CHUNK_MAX_TOKENS
    params_model = RecordsParams

    def __init__(self, counter: TokenCounter) -> None:
        self._counter = counter

    async def chunk(self, parsed: ParsedDocument, profile: RagProfile) -> list[ChunkDraft]:
        budget = chunk_budget(profile.chunk_max_tokens)
        params = validate_params(
            RecordsParams, profile.chunker_params, where="records chunker params"
        )
        return await asyncio.to_thread(self._chunk_all, parsed.joined(), params, budget)

    def _chunk_all(
        self, joined: JoinedText, params: RecordsParams, budget: int
    ) -> list[ChunkDraft]:
        splitter = BudgetSplitter(self._counter)
        drafts: list[ChunkDraft] = []
        for group in self._groups(joined.text, params):
            if group.preamble:
                if params.preamble == "drop":
                    continue
                header = params.preamble_heading or ""
            else:
                header = HEADER_SEPARATOR.join(
                    group.context[name] for name in params.header_groups if name in group.context
                )
            drafts.extend(self._pack(group, header, params, budget, joined, splitter))
        return drafts

    @staticmethod
    def _groups(text: str, params: RecordsParams) -> list[_Group]:
        levels: list[tuple[str, ProfileRegex]] = [
            (level.name, compile_profile_regex(level.pattern)) for level in params.group_levels
        ]
        record = compile_profile_regex(params.record_pattern)
        context: dict[str, dict[str, str]] = {}
        groups: list[_Group] = []
        position = 0
        for raw in text.split("\n"):
            start, position = position, position + len(raw) + 1
            line = raw.strip()
            if not line:
                continue
            entry = _Line(start, start + len(raw), line)
            if record.match(line):
                if not groups:
                    groups.append(_Group({}, [], preamble=True))
                groups[-1].lines.append(entry)
                continue
            for index, (name, regex) in enumerate(levels):
                match = regex.match(line)
                if match is None:
                    continue
                context[name] = {k: v for k, v in match.groupdict().items() if v}
                for deeper, _ in levels[index + 1 :]:
                    context.pop(deeper, None)
                if name == params.group_by:
                    merged = {
                        k: v
                        for level_name, _ in levels
                        for k, v in context.get(level_name, {}).items()
                    }
                    groups.append(_Group(merged, [entry]))
                break
            else:
                if not groups:
                    groups.append(_Group({}, [], preamble=True))
                groups[-1].lines.append(entry)
        return groups

    def _pack(
        self,
        group: _Group,
        header: str,
        params: RecordsParams,
        budget: int,
        joined: JoinedText,
        splitter: BudgetSplitter,
    ) -> list[ChunkDraft]:
        with_header = (params.repeat_group_header or group.preamble) and bool(header)
        shown = self._capped(header, budget, splitter) if with_header else ""
        prefix = [shown] if with_header else []
        drafts: list[ChunkDraft] = []
        current: list[_Line] = []

        def flush() -> None:
            if current:
                drafts.append(self._draft(header, shown, group, current, joined))
                current.clear()

        for line in group.lines:
            candidate = "\n".join([*prefix, *(entry.text for entry in current), line.text])
            if splitter.count(candidate) <= budget:
                current.append(line)
                continue
            flush()
            alone = "\n".join([*prefix, line.text])
            if splitter.count(alone) <= budget:
                current.append(line)
                continue
            room = max(budget - (splitter.count(shown) + 1 if with_header else 0), 1)
            for piece in splitter.split(line.text, room):
                drafts.append(
                    self._draft(header, shown, group, [_Line(line.start, line.end, piece)], joined)
                )
        flush()
        return drafts

    def _capped(self, header: str, budget: int, splitter: BudgetSplitter) -> str:
        """The header as printed in the chunk text: at most its share of the budget."""
        cap = max(budget // MAX_HEADING_BUDGET_SHARE, 1)
        if splitter.count(header) <= cap:
            return header
        return self._counter.split(header, cap)[0]

    @staticmethod
    def _draft(
        header: str, shown: str, group: _Group, lines: list[_Line], joined: JoinedText
    ) -> ChunkDraft:
        body = "\n".join(entry.text for entry in lines)
        metadata: dict[str, MetadataValue] = {}
        if pages := joined.pages_between(lines[0].start, lines[-1].end):
            metadata["page"], metadata["page_end"] = pages
        if header:
            metadata["heading"] = header
        return ChunkDraft(
            text=f"{shown}\n{body}" if shown else body,
            metadata=metadata,
            groups={key: (value,) for key, value in group.context.items()},
            body=body,
        )
