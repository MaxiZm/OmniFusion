from omnifusion.fusion.prompts import render_judge_prompt, render_final_prompt
from omnifusion.fusion.types import JudgeAnalysis


def test_render_judge_prompt():
    user_prompt = "Compare A and B"
    panel_answers = {"MODEL_A": "Answer A", "MODEL_B": "Answer B"}
    prompt = render_judge_prompt(user_prompt, panel_answers, nonce="test-nonce")

    assert "--- START OF USER_PROMPT (ID: test-nonce) ---" in prompt
    assert "Compare A and B" in prompt
    assert "--- END OF USER_PROMPT (ID: test-nonce) ---" in prompt

    assert "--- START OF PANEL_ANSWERS (ID: test-nonce) ---" in prompt
    assert "[MODEL_A]" in prompt
    assert "Answer A" in prompt
    assert "[END OF MODEL_A (ID: test-nonce)]" in prompt

    assert "[MODEL_B]" in prompt
    assert "Answer B" in prompt


def test_render_final_prompt():
    panel_answers = {"MODEL_A": "Answer A", "MODEL_B": "Answer B"}
    judge_analysis = JudgeAnalysis(
        consensus="Agree on X",
        disagreements="Disagree on Y",
        likely_errors="None",
        missing_information="Z",
        recommended_final_answer_plan="Follow A",
    )
    prompt = render_final_prompt(panel_answers, judge_analysis, nonce="test-nonce")

    assert "--- START OF PANEL_ANSWERS (ID: test-nonce) ---" in prompt
    assert "[MODEL_A]" in prompt
    assert "Answer A" in prompt

    assert "--- START OF JUDGE_ANALYSIS (ID: test-nonce) ---" in prompt
    assert "Consensus: Agree on X" in prompt
    assert "Recommended Plan: Follow A" in prompt
    assert "--- END OF JUDGE_ANALYSIS (ID: test-nonce) ---" in prompt
