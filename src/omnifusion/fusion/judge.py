import json
import re
from typing import List, Optional
from .types import Preset, PanelResult, JudgeAnalysis
from .runtime.executor import BudgetedExecutor
from .prompts import render_judge_prompt


def _strip_trailing_commas(s: str) -> str:
    """Remove trailing commas before } or ] — a common cause of strict-JSON failures."""
    return re.sub(r",(\s*[}\]])", r"\1", s)


def _extract_balanced_json(text: str) -> Optional[dict]:
    """Scan for the first balanced {...} object, respecting strings/escapes.

    More robust than first-brace/last-brace slicing when the model emits prose or
    multiple objects around the JSON (common with reasoning models).
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    for attempt in (candidate, _strip_trailing_commas(candidate)):
                        try:
                            return json.loads(attempt)
                        except json.JSONDecodeError:
                            continue
                    return None
    return None


def _repair_truncated_json(text: str) -> Optional[dict]:
    """Best-effort recovery of JSON that was cut off mid-output (e.g. the judge hit
    its max_tokens cap). Closes any open string and brackets; if that doesn't parse,
    trims back to each earlier comma and retries — salvaging the fields that did
    complete instead of discarding the whole analysis.
    """
    start = text.find("{")
    if start == -1:
        return None
    s = text[start:]

    candidates = [s]
    idx = len(s)
    while True:
        comma = s.rfind(",", 0, idx)
        if comma == -1:
            break
        candidates.append(s[:comma])
        idx = comma

    for chunk in candidates:
        stack = []
        in_str = False
        esc = False
        for c in chunk:
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    stack.append("}")
                elif c == "[":
                    stack.append("]")
                elif c in "}]":
                    if stack:
                        stack.pop()
        repaired = chunk
        if in_str:
            repaired += '"'
        repaired = repaired.rstrip().rstrip(",").rstrip()
        for closer in reversed(stack):
            repaired += closer
        try:
            return json.loads(_strip_trailing_commas(repaired))
        except json.JSONDecodeError:
            continue
    return None


def extract_json_from_text(text: str) -> dict:
    """Best-effort extraction of a JSON object from a model response.

    Handles markdown fences, reasoning-model <think> preambles, prose around the
    JSON, trailing commas, and truncated output. Raises ValueError if nothing
    parseable can be recovered.
    """
    if not text or not text.strip():
        raise ValueError("empty response")
    text = text.strip()

    # Reasoning models (e.g. R1) may prepend a <think>...</think> block.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```).
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # 1. Direct parse.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Direct parse with trailing-comma cleanup.
    try:
        return json.loads(_strip_trailing_commas(text))
    except json.JSONDecodeError:
        pass

    # 3. Balanced-brace scan (handles surrounding prose / multiple blocks).
    obj = _extract_balanced_json(text)
    if obj is not None:
        return obj

    # 4. Widest { ... } slice with trailing-comma cleanup.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(_strip_trailing_commas(text[start : end + 1]))
        except json.JSONDecodeError:
            pass

    # 5. Repair truncated JSON (model cut off mid-output, e.g. hit max_tokens).
    repaired = _repair_truncated_json(text)
    if repaired is not None:
        return repaired

    raise ValueError("no JSON object found in response")


async def run_judge(
    run_id: str, preset: Preset, messages: list, panel_results: List[PanelResult]
) -> JudgeAnalysis:
    def _mfield(m, name):
        # Safe across dicts and pydantic ChatMessage objects (whose attrs may be
        # None — e.g. an assistant tool-call message with content=None — which would
        # otherwise fall through to .get() and crash on the object).
        return m.get(name) if isinstance(m, dict) else getattr(m, name, None)

    conversation_lines = []
    for m in messages:
        role = _mfield(m, "role")
        content = _mfield(m, "content")
        if role and content:
            conversation_lines.append(f"[{role.upper()}]: {content}")
    user_prompt = "\n\n".join(conversation_lines)

    panel_answers = {}
    for idx, res in enumerate([r for r in panel_results if r.status == "ok"]):
        label = f"MODEL_{chr(65 + idx)}"  # MODEL_A, MODEL_B, etc.
        panel_answers[label] = res.content

    prompt = render_judge_prompt(
        user_prompt,
        panel_answers,
        run_id,
        prompt_config=getattr(preset, "prompts", None),
    )

    judge_messages = [{"role": "user", "content": prompt}]

    analysis = None
    actual_cost_usd = 0.0
    executor = BudgetedExecutor(run_id)
    try:
        # Request OpenAI-style JSON mode by default — it is the single biggest
        # reduction in judge parse failures and is broadly supported (DeepSeek
        # V4, OpenAI, Groq, OpenRouter, Ollama, LM Studio). llm/client.filter_params()
        # drops response_format for providers that don't support it (anthropic,
        # gemini), and the explicit retry below covers any model that rejects it.
        kwargs = {
            "timeout": preset.judge.timeout,
            "response_format": {"type": "json_object"},
        }

        try:
            response = await executor.call(
                "judge",
                provider_id="default",
                model=preset.judge_model,
                messages=judge_messages,
                max_tokens=preset.judge.max_tokens,
                **kwargs,
            )
        except Exception:
            kwargs.pop("response_format", None)
            response = await executor.call(
                "judge",
                provider_id="default",
                model=preset.judge_model,
                messages=judge_messages,
                max_tokens=preset.judge.max_tokens,
                **kwargs,
            )

        content = response.choices[0].message.content
        actual_cost_usd = getattr(response, "_omnifusion_cost_usd", 0.0)

        # Capture judge token usage so it can be aggregated into the response.
        judge_usage = getattr(response, "usage", None)
        judge_prompt_tokens = int(getattr(judge_usage, "prompt_tokens", 0) or 0)
        judge_completion_tokens = int(getattr(judge_usage, "completion_tokens", 0) or 0)

        try:
            data = extract_json_from_text(content)
            analysis = JudgeAnalysis(
                consensus=data.get("consensus", ""),
                disagreements=data.get("disagreements", ""),
                strongest_points_by_model=data.get("strongest_points_by_model", {}),
                missing_information=data.get("missing_information", ""),
                likely_errors=data.get("likely_errors", ""),
                recommended_final_answer_plan=data.get(
                    "recommended_final_answer_plan", ""
                ),
                cost_usd=actual_cost_usd,
                prompt_tokens=judge_prompt_tokens,
                completion_tokens=judge_completion_tokens,
            )
        except Exception:
            analysis = JudgeAnalysis(
                consensus="Degraded analysis due to parse failure.",
                recommended_final_answer_plan="Synthesize the best available information.",
                cost_usd=actual_cost_usd,
                prompt_tokens=judge_prompt_tokens,
                completion_tokens=judge_completion_tokens,
            )
        return analysis

    except Exception:
        return JudgeAnalysis(
            consensus="Judge failed to execute.",
            recommended_final_answer_plan="Synthesize the best available information.",
            cost_usd=0.0,
        )
