import os
import json
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset", type=str, nargs="+", choices=["alpaca", "topiocqa"], required=True, help="which dataset to use"
    )
    # parser.add_argument(
    #     "--multi_turn", action="store_true", help="follow multi-turn fashion"
    # )

    args = parser.parse_args()
    return args


def load_alpaca(path="chatalpaca-20k.json", multi_turn=True):
    """
    Load a chatalpaca from a jsonl file.

    Parameters:
    path (str): Path to the jsonl file.
    multi_turn (bool): If True, the dataset will be processed as multi-turn dialogue. If False, the dataset will be processed as single-turn dialogue.

    Returns:
    list: A list of dictionaries, where each dictionary contains the id, prompt and response of a sample.
    """
    # read jsonl
    data = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # remove \n
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    # iterate over instances
    instances = []

    for i in data:
        # iterate over conversation
        if multi_turn:
            for j in range(len(i['conversations']) // 2):
                instances.append({
                    str(i['id']) + '_' + str(j): {
                        'prompt': " ".join([d['value'] for d in i['conversations'][: (j * 2) + 1]]),
                        'response': i['conversations'][(j * 2) + 1]['value']
                    }
                })
        else:
            instances.append({
                str(i['id']): {
                    'prompt': " ".join([d['value'] for d in i['conversations'][ : -1]]),
                    'response': i['conversations'][-1]['value']
                }
            })

    return instances


def load_topiocqa(path="topiocqa_train.json"):

    def normalize_sentence(text):
        """fix punctuation if required"""
        text = text.strip()
        
        if not text:
            return text

        if text[-1] not in ".?!":
            text += "."
        return text

    # read json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    instances = []
    # iterate over instances
    for i in data:
        c_id = i["conversation_no"]
        t_id = i["turn_no"]

        context = [normalize_sentence(c) for c in i.get("context", [])]
        question = normalize_sentence(i["question"])
        answer = normalize_sentence(i["answer"])
        # prompt
        prompt = " ".join(context + [question])
        # instance
        instance_id = f"{c_id}_{t_id}"

        instances.append({
            instance_id: {
                "prompt": prompt,
                "response": answer
            }
        })

    return instances


if __name__ == '__main__':
    args = get_args()

    func_map = {
        'alpaca': load_alpaca,
        'topiocqa': load_topiocqa
    }

    for dataset in args.dataset:
        instances = func_map[dataset]()

        fname = f'{dataset}.json'
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(instances, f, ensure_ascii=False, indent=4)

# python3 dataset.py --dataset alpaca 
