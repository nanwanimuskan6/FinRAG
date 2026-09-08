from app.rag_app import validate_llm_answer, format_answer


def test_one_supported_number_does_not_validate_other_invented_numbers():
    evidence = [{"text": "Revenue 1,000 crore; growth 8.6%"}]
    assert not validate_llm_answer("Revenue 1,000 crore; growth 9.2%", evidence)
    assert validate_llm_answer("Revenue 1000 crore; growth 8.6%", evidence)


def test_abstention_does_not_get_an_arbitrary_citation():
    answer = "I could not find the answer in the provided document."
    assert format_answer(answer, [{"text": "Unrelated", "page_number": 1}]) == answer
