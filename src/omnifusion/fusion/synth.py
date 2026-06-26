from typing import List, AsyncGenerator, Union, Any
from .types import Preset, PanelResult, JudgeAnalysis
from .runtime.executor import BudgetedExecutor
from ..api.schemas import ChatCompletionRequest


async def run_synthesis(
    run_id: str,
    preset: Preset,
    request: ChatCompletionRequest,
    panel_results: List[PanelResult],
    judge_analysis: JudgeAnalysis,
    context: dict,
) -> Union[AsyncGenerator[Any, None], Any]:

    # 1. Assemble panel answers dictionary
    panel_answers = {}
    for idx, res in enumerate([r for r in panel_results if r.status == "ok"]):
        label = f"MODEL_{chr(65 + idx)}"
        panel_answers[label] = res.content

    # 2. Render system content using prompts template
    from .prompts import render_final_prompt

    system_content = render_final_prompt(panel_answers, judge_analysis, run_id)

    # 3. Assemble messages, preserving original history.
    # If the user sent a system message, merge it into the synthesis system prompt
    # rather than appending it as a duplicate system turn.
    user_system_parts = []
    non_system_messages = []
    for m in request.messages:
        m_dict = m.model_dump(exclude_none=True)
        if m_dict.get("role") == "system":
            user_system_parts.append(m_dict.get("content", ""))
        else:
            non_system_messages.append(m_dict)

    if user_system_parts:
        merged_user_system = "\n".join(user_system_parts)
        system_content = f"{merged_user_system}\n\n{system_content}"

    final_messages = [{"role": "system", "content": system_content}]
    final_messages.extend(non_system_messages)

    # 4. Clamp user request max_tokens to preset's final stage cap (Invariant #9)
    max_tokens = preset.final.max_tokens
    if request.max_tokens is not None:
        max_tokens = min(request.max_tokens, preset.final.max_tokens)

    kwargs = {
        "timeout": preset.final.timeout,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "stop": request.stop,
    }
    # Filter out None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    executor = BudgetedExecutor(run_id)
    if request.stream:
        return await executor.stream(
            "final",
            provider_id="default",
            model=preset.final_model,
            messages=final_messages,
            max_tokens=max_tokens,
            **kwargs,
        )

    return await executor.call(
        "final",
        provider_id="default",
        model=preset.final_model,
        messages=final_messages,
        max_tokens=max_tokens,
        **kwargs,
    )
