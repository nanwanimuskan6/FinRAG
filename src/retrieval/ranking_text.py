"""Prepare scoring inputs without changing cited source text."""

import re
from collections.abc import Sequence


_REPORT_FOOTER = re.compile(
    r"(?im)^\s*\d+[^\n]*\bAnnual Report\b[^\n]*\d+\s*$"
)
_CORPORATE_NAME = re.compile(
    r"\b[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*\s+"
    r"(?:Limited|Ltd\.?|Inc\.?|Corporation|PLC)(?:['\u2019]s)?"
)


def scoring_text(text: str) -> str:
    """Exclude page-numbered annual-report footers from relevance scoring."""
    return _REPORT_FOOTER.sub("", text).strip()


def scoring_query(query: str, candidates: Sequence[dict]) -> str:
    """Omit a redundant issuer name only within a single matching report.

    A company name repeated in footers can dominate the requested metric.
    Keep names for multi-source searches and for entities other than the
    report issuer; keep all metric, year, and consolidation qualifiers.
    """
    if any("source_filename" not in c for c in candidates):
        return query
    if len({c["source_filename"] for c in candidates}) != 1:
        return query
    footers = " ".join(
        match.group()
        for candidate in candidates
        for match in _REPORT_FOOTER.finditer(candidate["text"])
    )

    def remove_issuer(match: re.Match) -> str:
        name = re.sub(r"['\u2019]s$", "", match.group())
        return "" if name in footers else match.group()

    return " ".join(_CORPORATE_NAME.sub(remove_issuer, query).split())


_NON_LOOKUP_QUERY = re.compile(
    r"\b(growth|increase|decrease|change|difference|why|explain|describe|"
    r"define|definition|meaning|policy|policies)\b|\bhow (does|do|did|is|are)\b",
    re.IGNORECASE,
)


def metric_row_patterns(query: str) -> list[tuple[int, re.Pattern]]:
    """Find query phrases that could label numeric statement rows.

    Preserve internal words (e.g. 'income from services'), reject question
    filler and years at phrase boundaries, and leave comparative/conceptual
    questions to semantic ranking. No report-specific metric list is needed.
    """
    from src.retrieval.bm25_retriever import tokenize, _QUERY_FILLER_TOKENS

    if _NON_LOOKUP_QUERY.search(query):
        return []
    tokens = tokenize(query)
    ignored = _QUERY_FILLER_TOKENS | {"fy", "s", "much", "on", "were"}
    patterns = []
    for length in range(1, 5):
        for start in range(len(tokens) - length + 1):
            span = tokens[start:start + length]
            if (span[0] in ignored or span[-1] in ignored
                    or any(token.isdigit() for token in span)):
                continue
            pattern = (
                r"\b" + r"\s+".join(map(re.escape, span))
                + r"\b[ \t:*#^()%\-]*[\u20b9`$\u00a3\u20ac]?\s*\(?\d"
            )
            patterns.append((length, re.compile(pattern, re.IGNORECASE)))
    return patterns


def metric_row_strength(text: str, patterns: list[tuple[int, re.Pattern]]) -> int:
    """Return the longest query phrase immediately followed by a numeric value."""
    return max((length for length, pattern in patterns if pattern.search(text)), default=0)
