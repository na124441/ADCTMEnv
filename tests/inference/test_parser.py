import pytest

from inference.parser import parse_llm_response


def test_parse_llm_response_parses_embedded_json():
    action = parse_llm_response('prefix {"cooling":[0.1, 0.2, 1.7]} suffix', 3)
    assert action.cooling == [0.1, 0.2, 1.0]


def test_parse_llm_response_rejects_missing_json():
    with pytest.raises(ValueError, match="No JSON found"):
        parse_llm_response("hello", 3)


def test_parse_llm_response_rejects_wrong_length():
    with pytest.raises(ValueError, match="Expected 'cooling'"):
        parse_llm_response('{"cooling":[0.1]}', 3)
