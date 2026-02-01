import os
import json


def load_alpaca(path="chatalpaca-20k.json", multi_turn=False):
    """
    Load a dataset from a jsonl file.

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


if __name__ == '__main__':
    multi_turn = True
    instances = load_alpaca(multi_turn=multi_turn)
    
    fname = f'data_mt.json' if multi_turn else f'data.json'
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(instances, f, ensure_ascii=False, indent=4)
