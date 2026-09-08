import pytest

from src.evaluation.evaluate_retrieval import result_is_relevant, metrics, evidence_hit


@pytest.mark.parametrize("text,answer,expected", [
    ("Revenue: 10,71,174 crore", "\u20b910,71,174 crore", True),
    ("Growth 18.6%", "8.6%", False),
    ("Growth 86%", "8.6%", False),
    ("Capital 16,766", "6,766", False),
    ("Ratio 8.65", "8.6", False),
    ("Growth 8.6%", "8.6%", True),
    ("Anything", "", False),
])
def test_answer_matching_preserves_numeric_boundaries(text, answer, expected):
    assert result_is_relevant({"text": text}, answer) is expected


def test_page_labels_are_measured_separately_from_answer_occurrence():
    result = metrics([{"text": "Revenue 100", "page_number": 2}],
                     {"answer": "100", "pages": [3]})
    assert result["answer_hit@1"] == 1
    assert result["page_hit@1"] == 0


def test_calculated_answers_require_both_operands():
    item = {"answer": "71,052", "evidence_answers": ["10,71,174", "10,00,122"]}
    first = {"text": "Current revenue 10,71,174"}
    second = {"text": "Previous revenue 10,00,122"}
    assert evidence_hit([first], item) == 0
    assert evidence_hit([first, second], item) == 1
