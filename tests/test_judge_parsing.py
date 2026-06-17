"""
Regression tests for judge JSON extraction robustness.

The judge stage frequently produced "Degraded analysis due to parse failure"
because (a) JSON mode was only enabled for model names containing "openai"/"ollama"
(so DeepSeek free-formed its output) and (b) a small judge max_tokens truncated the
JSON mid-output. These tests cover the extractor's tolerance: fences, prose, reasoning
preambles, trailing commas, and truncation recovery.
"""
import pytest

from omnifusion.fusion.judge import extract_json_from_text


def test_plain_json():
    assert extract_json_from_text('{"consensus": "a"}') == {"consensus": "a"}


def test_markdown_fenced_json():
    text = '```json\n{"consensus": "a", "x": 1}\n```'
    assert extract_json_from_text(text) == {"consensus": "a", "x": 1}


def test_bare_fenced_json():
    text = '```\n{"consensus": "a"}\n```'
    assert extract_json_from_text(text) == {"consensus": "a"}


def test_prose_wrapped_json():
    text = 'Here is the analysis:\n{"consensus": "a"}\nHope that helps!'
    assert extract_json_from_text(text) == {"consensus": "a"}


def test_reasoning_think_preamble():
    text = '<think>The models agree.</think>\n{"consensus": "ok"}'
    assert extract_json_from_text(text) == {"consensus": "ok"}


def test_trailing_commas():
    text = '{"consensus": "a", "list": [1, 2, 3,],}'
    assert extract_json_from_text(text) == {"consensus": "a", "list": [1, 2, 3]}


def test_nested_object_in_prose():
    text = 'noise {"a": {"b": 1}} more noise'
    assert extract_json_from_text(text) == {"a": {"b": 1}}


def test_truncated_mid_string_value_is_repaired():
    """Judge hit max_tokens mid-string: close the string + braces, salvage prior fields."""
    truncated = (
        '{"consensus": "Both agree on Paris.", '
        '"disagreements": "None.", '
        '"recommended_final_answer_plan": "Synthesize a single clear sentence about Pari'
    )
    out = extract_json_from_text(truncated)
    assert out["consensus"] == "Both agree on Paris."
    assert out["disagreements"] == "None."


def test_truncated_mid_key_is_repaired_to_last_complete_field():
    """Truncation right at the start of a new key trims back to the last complete pair."""
    truncated = (
        '{"consensus": "Agreed.", '
        '"disagreements": "Minor wording.", '
        '"likely_err'
    )
    out = extract_json_from_text(truncated)
    assert out["consensus"] == "Agreed."
    assert out["disagreements"] == "Minor wording."


def test_unrecoverable_raises():
    with pytest.raises((ValueError, Exception)):
        extract_json_from_text("this is not json at all, no braces")
