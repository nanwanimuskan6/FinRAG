from src.retrieval.ranking_text import scoring_query, scoring_text


def candidate(source="report.pdf"):
    return {"source_filename": source,
            "text": "Total income 123 crore.\n112 Example Industries Limited Integrated Annual Report 2024-25 113"}


def test_footer_removed_only_from_scoring_text():
    original = candidate()["text"]
    assert scoring_text(original) == "Total income 123 crore."
    assert "Annual Report" in original
    assert scoring_text("The Annual Report describes revenue.") == "The Annual Report describes revenue."


def test_single_report_query_keeps_metric_scope_and_year():
    query = "What was Example Industries Limited's standalone total income in FY 2024-25?"
    assert scoring_query(query, [candidate()]) == "What was standalone total income in FY 2024-25?"


def test_other_entities_and_multiple_reports_keep_names():
    query = "What was Another Company Limited's revenue?"
    assert scoring_query(query, [candidate()]) == query
    issuer_query = "What was Example Industries Limited's revenue?"
    assert scoring_query(issuer_query, [candidate(), candidate("other.pdf")]) == issuer_query
    assert scoring_query(issuer_query, []) == issuer_query


def test_company_name_without_matching_footer_is_preserved():
    query = "What was Example Industries Limited's revenue?"
    assert scoring_query(query, [{"source_filename": "report.pdf", "text": "Revenue 123"}]) == query


def test_numeric_rows_require_metric_and_adjacent_number():
    from src.retrieval.ranking_text import metric_row_patterns, metric_row_strength
    patterns = metric_row_patterns("What was total income in FY 2025-26?")
    assert metric_row_strength("Total income: 123 crore", patterns) == 2
    assert metric_row_strength("Total comprehensive income: 123 crore", patterns) < 2
    assert metric_row_strength("Total income is discussed elsewhere. Page 123", patterns) == 0
    assert metric_row_strength("Total income tax policy", patterns) == 0


def test_comparative_and_conceptual_questions_do_not_promote_value_rows():
    from src.retrieval.ranking_text import metric_row_patterns
    for query in ("What was revenue growth?", "How did revenue change?", "Explain total assets", "What is the revenue policy?"):
        assert metric_row_patterns(query) == []


def test_internal_stopwords_are_preserved_in_metric_rows():
    from src.retrieval.ranking_text import metric_row_patterns, metric_row_strength
    patterns = metric_row_patterns("What was income from services?")
    assert metric_row_strength("Income from services 123", patterns) == 3


def test_currency_symbols_and_curly_possessives():
    from src.retrieval.ranking_text import metric_row_patterns, metric_row_strength
    patterns = metric_row_patterns("total income")
    for symbol in ("\u20b9", "\u00a3", "\u20ac", "$", "`"):
        assert metric_row_strength(f"Total income: {symbol}123", patterns) == 2
    query = "Example Industries Limited\u2019s total income"
    assert scoring_query(query, [candidate()]) == "total income"
