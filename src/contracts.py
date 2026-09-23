"""
Structured hand-offs between the pipeline's three phases.

Before this, Researcher -> Writer -> Editor passed plain Markdown strings
plus phase-specific dataclasses (ResearchResult, WriterResult) that only the
producing module understood. These wrap those existing dataclasses rather
than replacing them, so `record["research"]`/`record["writer"]` (built via
`.as_record()`) are unaffected — this module adds structure on top of the
pipeline, it doesn't change what gets logged.
"""
from dataclasses import dataclass, field


@dataclass
class ResearchPackage:
    """What the writer is allowed to know: the brief and nothing else."""
    topic: str
    brief_markdown: str
    retrieved_urls: list
    allowed_citation_urls: set
    evidence_gaps: list
    grounding_level: str
    research_result: object  # ResearchResult — kept for .as_record()


@dataclass
class DraftPackage:
    research_package: ResearchPackage
    draft_markdown: str
    checks_run: list = field(default_factory=list)
    unresolved_issues: list = field(default_factory=list)
    revision_history: list = field(default_factory=list)
    writer_result: object = None  # WriterResult — kept for .as_record()


@dataclass
class FinalPackage:
    draft_package: DraftPackage
    final_markdown: str
    edit_summary: list = field(default_factory=list)
    safety_checks: dict = field(default_factory=dict)
    publish_ready: bool = True
