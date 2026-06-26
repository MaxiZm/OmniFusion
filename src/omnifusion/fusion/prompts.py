import os
import uuid
from jinja2 import Environment, FileSystemLoader

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
jinja_env = Environment(loader=FileSystemLoader(PROMPTS_DIR))


def new_prompt_nonce() -> str:
    return uuid.uuid4().hex


def _prefix_role_prompt(rendered: str, prompt_config, role: str) -> str:
    if not prompt_config:
        return rendered
    parts = [
        getattr(prompt_config, "global_prompt", "").strip(),
        getattr(prompt_config, "role_prompts", {}).get(role, "").strip(),
    ]
    prefix = "\n\n".join(part for part in parts if part)
    if not prefix:
        return rendered
    return f"{prefix}\n\n{rendered}"


def render_judge_prompt(
    user_prompt: str,
    panel_answers: dict,
    nonce: str = "default-nonce",
    prompt_config=None,
) -> str:
    template = jinja_env.get_template("judge_b.j2")
    rendered = template.render(
        user_prompt=user_prompt, panel_answers=panel_answers, nonce=nonce
    )
    return _prefix_role_prompt(rendered, prompt_config, "judge")


def render_final_prompt(
    panel_answers: dict,
    judge_analysis,
    nonce: str = "default-nonce",
    prompt_config=None,
) -> str:
    template = jinja_env.get_template("final_b.j2")
    rendered = template.render(
        panel_answers=panel_answers, judge_analysis=judge_analysis, nonce=nonce
    )
    return _prefix_role_prompt(rendered, prompt_config, "final")
