import os
import json
import random
import argparse
import csv

# from dictionary_learning.config import random_seeds


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset", type=str, nargs="+", choices=["instruct", "multic", "multif"], required=True, help="which dataset to use"
    )
    parser.add_argument(
        "--ratio", type=float, required=True, help="test size"
    )
    # parser.add_argument(
    #     "--language", type=str, default="English", help="language filter for datasets that support it"
    # )
    # parser.add_argument(
    #     "--multi_turn", action="store_true", help="follow multi-turn fashion"
    # )

    args = parser.parse_args()
    return args


def normalize_sentence(text):
    """fix punctuation if required"""
    text = text.strip()

    if not text:
        return text

    if text[-1] not in ".?!":
        text += "."
    return text


def load_instruct(path="./instruct/dataset.json", mt=False):  # language="English"
    # read json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    instances = []
    # iterate over instances
    for t in data.keys():
        # task
        for c_id in data[t]:
            n_turn = len(data[t][c_id]) // 2

            if mt:
                for i in range(n_turn):
                    # turns
                    ques = data[t][c_id][ : i * 2 + 1]
                    ans = data[t][c_id][i * 2 + 1]

                    instances.append({
                        f'{c_id}_{i}': {
                            "messages": [
                                {
                                "role": x["role"],
                                "content": normalize_sentence(x['text'])
                                } for x in ques
                            ],
                            "response": ans["text"],
                            "reference": ans["evaluation_reference"],
                            "task": t
                        }
                    })
            else:
                i = n_turn - 1

                ques = data[t][c_id][ : i * 2 + 1]
                ans = data[t][c_id][i * 2 + 1]

                instances.append({
                    f'{c_id}_{i}': {
                        "messages": [
                                {
                                "role": x["role"],
                                "content": normalize_sentence(x['text'])
                                } for x in ques
                            ],
                        "response": ans["text"],
                        "reference": ans["evaluation_reference"],
                        "task": t,
                        "metric": ans["evaluation_metric"]
                    }
                })

    return instances


def _is_non_empty(value):
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def _parse_json_field(value, default=None):
    if not _is_non_empty(value):
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def load_multif(path="./multif/dataset.csv", mt=True, language="English"):
    instances = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for li, row in enumerate(reader):
            row_language = (row.get("language") or "").strip()
            if language and row_language != language:
                continue

            qid = row.get("key", f"item_{li}")

            prompts = [
                _parse_json_field(row.get("turn_1_prompt"), default={}),
                _parse_json_field(row.get("turn_2_prompt"), default={}),
                _parse_json_field(row.get("turn_3_prompt"), default={}),
            ]

            instruction_lists = [
                _parse_json_field(row.get("turn_1_instruction_id_list"), default=[]),
                _parse_json_field(row.get("turn_2_instruction_id_list"), default=[]),
                _parse_json_field(row.get("turn_3_instruction_id_list"), default=[]),
            ]

            kwargs_lists = [
                _parse_json_field(row.get("turn_1_kwargs"), default=[]),
                _parse_json_field(row.get("turn_2_kwargs"), default=[]),
                _parse_json_field(row.get("turn_3_kwargs"), default=[]),
            ]

            if mt:
                for i in range(3):
                    messages = []
                    for j in range(i + 1):
                        prompt = prompts[j]
                        if isinstance(prompt, dict):
                            role = prompt.get("role", "user")
                            content = prompt.get("content", "")
                        else:
                            role = "user"
                            content = str(prompt)

                        if _is_non_empty(content):
                            messages.append({
                                "role": role,
                                "content": content,
                            })

                    instance = {
                        "messages": messages,
                        "reference": {
                            "instruction_id_list": instruction_lists[i],
                            "kwargs": kwargs_lists[i],
                        },
                        "metric": "rule_based",
                        "language": row_language if row_language else language,
                    }

                    instances.append({
                        f"{qid}_{i}": instance
                    })
            else:
                i = 2
                messages = []
                for j in range(i + 1):
                    prompt = prompts[j]
                    if isinstance(prompt, dict):
                        role = prompt.get("role", "user")
                        content = prompt.get("content", "")
                    else:
                        role = "user"
                        content = str(prompt)

                    if _is_non_empty(content):
                        messages.append({
                            "role": role,
                            "content": content,
                        })

                instances.append({
                    f"{qid}_{i}": {
                        "messages": messages,
                        "reference": {
                            "instruction_id_list": instruction_lists[i],
                            "kwargs": kwargs_lists[i],
                        },
                        "metric": "rule_based",
                        "language": row_language if row_language else language,
                    }
                })

    return instances


if __name__ == '__main__':
    args = get_args()
    random.seed(0)

    func_map = {
        'instruct': load_instruct,
        'multic': load_multic,
        'multif': load_multif,
    }

    path_map = {
        'instruct': "./instruct/dataset.json",
        'multic': "./multic/dataset.jsonl",
        'multif': "./multif/multiIF_20241018.csv",
    }

    for dataset in args.dataset:
        instances = func_map[dataset](path=path_map[dataset])  # language=args.language

        if args.ratio > 0:
            random.shuffle(instances)

            i = int(args.ratio * len(instances))
            print(f"test size: {i}")

            ins = {
                'train': instances[i : ],
                'test': instances[ : i]
            }

            for d in ins.keys():
                fname = f'{dataset}_{d}.json'
                with open(fname, "w", encoding="utf-8") as f:
                    json.dump(ins[d], f, ensure_ascii=False, indent=4)
        else:
            fname = f'{dataset}.json'
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(instances, f, ensure_ascii=False, indent=4)


# python3 dataset.py --dataset alpaca
