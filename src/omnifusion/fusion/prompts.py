import os
from jinja2 import Environment, FileSystemLoader

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
jinja_env = Environment(loader=FileSystemLoader(PROMPTS_DIR))


def render_judge_prompt(user_prompt: str, panel_answers: dict, nonce: str = "default-nonce") -> str:
    template = jinja_env.get_template("judge_b.j2")
    return template.render(
        user_prompt=user_prompt, panel_answers=panel_answers, nonce=nonce
    )


def render_final_prompt(panel_answers: dict, judge_analysis, nonce: str = "default-nonce") -> str:
    template = jinja_env.get_template("final_b.j2")
    return template.render(
        panel_answers=panel_answers, judge_analysis=judge_analysis, nonce=nonce
    )
