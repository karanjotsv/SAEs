import os
import json
import random
import argparse

# from dictionary_learning.config import random_seeds


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset", type=str, nargs="+", choices=["instruct", "multic"], required=True, help="which dataset to use"
    )
    parser.add_argument(
        "--ratio", type=float, required=True, help="test size"
    )
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


def load_multic(path="./multic/dataset.jsonl", mt=False):
    instances = []

    with open(path, "r", encoding="utf-8") as f:
        for li, l in enumerate(f):
            l = l.strip()

            row = json.loads(l)

            qid = row.get("QUESTION_ID", f"item_{li}")
            axis = row.get("AXIS", "")
            conv = row.get("CONVERSATION", [])
            target_question = row.get("TARGET_QUESTION", "")
            pass_criteria = row.get("PASS_CRITERIA", "")

            messages = [
                {
                    "role": turn["role"],
                    "content": normalize_sentence(turn["content"]),
                }
                for turn in conv
            ]

            if mt:
                user_turn_indices = [
                    i for i, turn in enumerate(messages) if turn["role"] == "user"
                ]

                for j, end_idx in enumerate(user_turn_indices):
                    partial_messages = messages[: end_idx + 1]

                    instances.append({
                        f"{qid}_{j}": {
                            "messages": partial_messages,
                            "response": target_question,
                            "reference": pass_criteria,
                            "task": axis,
                            "metric": "llm_judge",
                        }
                    })
            else:
                instances.append({
                    qid: {
                        "messages": messages,
                        "response": target_question,
                        "reference": pass_criteria,
                        "task": axis,
                        "metric": "llm_judge",
                    }
                })
    return instances


def load_instruct(path="./instruct/dataset.json", mt=False):
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


if __name__ == '__main__':
    args = get_args()
    random.seed(0)

    func_map = {
        'instruct': load_instruct,
        'multic': load_multic
    }

    for dataset in args.dataset:
        instances = func_map[dataset]()

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
