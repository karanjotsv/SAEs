from openai import OpenAI
from pydantic import BaseModel
from typing import Literal


_openai_client = None

JUDGE_PROMPT = """You are tasked with evaluating a model response to see if it meets a specific criteria.

The criteria will always be YES/NO evaluation.

The model response is as follows:

<MODEL_RESPONSE>

{}

</MODEL_RESPONSE>

The criteria that the model response must meet is as follows. Be VERY STRICT!:

<CRITERIA>

{}

</CRITERIA>

Print your reasoning followed by your verdict, either "YES" or "NO"."""


class Judge(BaseModel):
    """
    Structured response schema for the LLM judge.

    Attributes:
        reasoning (str): The explanation provided by the model for its decision.
        verdict (Literal["YES", "NO"]): The final binary judgment.
    """
    reasoning: str
    verdict: Literal["YES", "NO"]


def get_openai_client():
    """
    Lazily initialize and return a singleton OpenAI client.

    Returns:
        OpenAI: An initialized OpenAI client instance.
    """
    global _openai_client

    if _openai_client is None:
        _openai_client = OpenAI()

    return _openai_client


def llm_judge(pred, cri, ref, judge_model='gpt-4o'):
    """
    Evaluate a model prediction against a binary criterion using an LLM judge.

    Args:
        pred (str): The model-generated response to evaluate.
        cri (str): The evaluation criteria (must be YES/NO based).
        ref (str): The expected (ground truth) verdict ("YES" or "NO").
        judge_model (str, optional): The model used for judging.
            Defaults to 'gpt-4o'.

    Returns:
        tuple:
            - float: 1.0 if the predicted verdict matches the reference, else 0.0.
            - str: The reasoning provided by the LLM judge.
            - str: The normalized verdict ("YES" or "NO").

    Notes:
        - The LLM is run with temperature=0 for deterministic output.
        - Verdict comparison is case-insensitive and whitespace-trimmed.
        - Uses structured parsing via Pydantic (`Judge` model).
    """
    client = get_openai_client()

    prompt = JUDGE_PROMPT.format(pred, cri)

    completion = client.beta.chat.completions.parse(
        model=judge_model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        response_format=Judge,
    )

    parsed = completion.choices[0].message.parsed
    verdict = parsed.verdict.strip().upper()
    
    gold = ref.strip().upper()

    return float(verdict == gold), parsed.reasoning, verdict
