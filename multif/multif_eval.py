import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from multif.if_eval import *

def _parse_json_if_needed(x: Any) -> Any:
    if isinstance(x, str):
        return json.loads(x)
    return x


def _parse_kwargs_list(kwargs_list: Any) -> List[Dict[str, Any]]:
    kwargs_list = _parse_json_if_needed(kwargs_list)
    if kwargs_list is None:
        return []

    out: List[Dict[str, Any]] = []
    for kw in kwargs_list:
        kw = _parse_json_if_needed(kw)
        if kw is None:
            out.append({})
        elif isinstance(kw, dict):
            out.append(kw)
        else:
            out.append(dict(kw))
    return out


def _normalize_reference(ref: Dict[str, Any], language: str = "English") -> Dict[str, Any]:
    """Normalize one per-generation Multi-IF reference block.

    Expected shape:
      {
        "instruction_id_list": [...],
        "kwargs": [...]
      }
    """
    if ref is None:
        raise ValueError("reference is None")

    if language != "English":
        return {
            "instruction_id_list": [],
            "kwargs": [],
        }

    inst_ids = _parse_json_if_needed(ref.get("instruction_id_list", []))
    kwargs = _parse_kwargs_list(ref.get("kwargs", []))

    if inst_ids is None:
        inst_ids = []

    return {
        "instruction_id_list": list(inst_ids),
        "kwargs": kwargs,
    }


def gen_acc_strict(x: Dict[str, Any]) -> Dict[str, Any]:
    """Strict instruction checks for one response."""
    resp = str(x["response"])
    inst_ids = x["instruction_id_list"]
    kwargs = x["kwargs"]
    follow: List[int] = []

    for j, inst_id in enumerate(inst_ids):
        inst_cls = INSTRUCTION_DICT[inst_id]
        inst = inst_cls(inst_id)
        inst.build_description(**kwargs[j])
        follow.append(int(bool(resp) and inst.check_following(resp)))

    return {
        "instruction_id_list": inst_ids,
        "follow_instruction_list": follow,
    }


def eval_generation_strict(
    response: str,
    ref: Dict[str, Any],
    language: str = "English",
) -> Dict[str, Any]:
    """Evaluate one generation.

    Returns exactly the per-turn fields needed inside run(...):
      1. conversation_level_strict
      2. instruction_level_strict
      3. instruction_scores
    """
    x = _normalize_reference(ref, language=language)
    inst_ids = x["instruction_id_list"]

    if not inst_ids:
        return {
            "conversation_level_strict": 0.0,
            "instruction_level_strict": 0.0,
            "instruction_scores": {},
            "instruction_id_list": [],
            "follow_instruction_list": [],
        }

    res = gen_acc_strict({
        "response": response,
        "instruction_id_list": inst_ids,
        "kwargs": x["kwargs"],
    })

    follow = [int(v) for v in res["follow_instruction_list"]]
    inst_ids = res["instruction_id_list"]

    conv_strict = float(all(follow))
    inst_strict = sum(follow) / len(inst_ids) if inst_ids else 0.0
    inst_scores = {inst_id: score for inst_id, score in zip(inst_ids, follow)}

    return {
        "conversation_level_strict": conv_strict,
        "instruction_level_strict": inst_strict,
        "instruction_scores": inst_scores,
        "instruction_id_list": inst_ids,
        "follow_instruction_list": follow,
    }


def eval_metric(metric: str, pred: str, ref: Dict[str, Any], language: str = "English") -> float:
    """Drop-in wrapper for run(...).

    For Multi-IF, returns conversation-level strict so it behaves like a turn pass/fail.
    """
    if metric != "rule_based":
        raise ValueError(f"Unsupported metric for multi_ifeval.py: {metric}")
    return eval_generation_strict(pred, ref, language=language)["conversation_level_strict"]


def parse_result(outputs: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Aggregate strict metrics over multiple examples."""
    prompt_total = 0
    prompt_correct = 0
    inst_total = 0
    inst_correct = 0

    for ex in outputs:
        follow = ex["follow_instruction_list"]
        inst_ids = ex["instruction_id_list"]

        prompt_total += 1
        if all(follow):
            prompt_correct += 1

        inst_total += len(inst_ids)
        inst_correct += sum(follow)

    if prompt_total == 0 or inst_total == 0:
        return 0.0, 0.0

    return prompt_correct / prompt_total, inst_correct / inst_total


def _extract_response(row: pd.Series) -> str:
    try:
        return json.loads(row["responses"])[0]["response"]
    except Exception:
        return str(row["responses"])


def english_strict_metrics(output_df: pd.DataFrame) -> Dict[str, Any]:
    """Repo-style English-only strict evaluation for one step CSV."""
    if output_df.empty:
        return {
            "turn_index": None,
            "num_english_examples": 0,
            "conversation_level_strict": 0.0,
            "instruction_level_strict": 0.0,
            "details_df": pd.DataFrame(),
        }

    row0 = output_df.iloc[0]
    turn_idx = int(row0["turn_index"])
    prompt_col = f"turn_{turn_idx}_prompt"
    inst_col = f"turn_{turn_idx}_instruction_id_list"
    kwargs_col = f"turn_{turn_idx}_kwargs"

    outs: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for _, row in output_df.iterrows():
        if str(row.get("language", "")) != "English":
            continue
        if str(row.get(prompt_col, "")) in {"", "None"}:
            continue

        try:
            ref = {
                "instruction_id_list": row[inst_col],
                "kwargs": row[kwargs_col],
            }
            pred = _extract_response(row)
            res = eval_generation_strict(pred, ref, language="English")
        except Exception:
            continue

        outs.append({
            "instruction_id_list": res["instruction_id_list"],
            "follow_instruction_list": res["follow_instruction_list"],
        })
        rows.append({
            "key": row.get("key", ""),
            "turn_index": turn_idx,
            "conversation_level_strict": res["conversation_level_strict"],
            "instruction_level_strict": res["instruction_level_strict"],
            "instruction_scores": json.dumps(res["instruction_scores"], ensure_ascii=False),
            "instruction_id_list": json.dumps(res["instruction_id_list"], ensure_ascii=False),
            "follow_instruction_list": json.dumps(res["follow_instruction_list"], ensure_ascii=False),
        })

    conv_strict, inst_strict = parse_result(outs)

    return {
        "turn_index": turn_idx,
        "num_english_examples": len(outs),
        "conversation_level_strict": conv_strict,
        "instruction_level_strict": inst_strict,
        "details_df": pd.DataFrame(rows),
    }
