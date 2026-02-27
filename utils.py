import re
import json
from tqdm import tqdm

import torch


def get_subset(data_path, tasks):
    raw = json.load(open(data_path, "r"))

    subset = [] 

    for i, item in enumerate(raw):
        for key, ex in item.items():
            if ex.get("task") in tasks:
                subset.append((i, key, ex))
    return raw, subset

# -------- metrics --------
def extract_choices(text):
    m = re.search(r"Answer:\s*([A-Z](?:\s*,\s*[A-H])*)", text, flags=re.I)
    if m:
        return set(re.findall(r"[A-Z]", m.group(1).upper()))
    letters = re.findall(r"\b([A-Z])\b", text.upper())

    return {letters[-1]} if letters else set()


def matching_all(pred_text, refs):
    pred = extract_choices(pred_text)
    gold = {r.upper() for r in refs}

    return float(len(pred & gold) > 0)


def matching_any(pred_text, refs):
    p = pred_text.lower()

    return float(any(r.lower() in p for r in refs))


def eval_metric(metric, pred_text, refs):
    if metric == "matching_any_exact":
        return matching_all(pred_text, refs)
    elif metric == "matching_any":
        return matching_any(pred_text, refs)
    else:
        raise ValueError(f"unexpected metric for selected tasks: {metric}")
# -----------------------

def get_conversations(task, outcome, num_conversations, num_turns, ids_path="./ids.json", instruct_path="./instruct_mt.json"):
    # return empty if no conversations needed
    if num_conversations <= 0: return []
    
    # load conversation ids for given task/outcome
    id_list = json.load(open(ids_path))[task][outcome]
    
    # extract unique (conversation_id, max_turn)
    conv_meta, seen_ids = [], set()
    for i in id_list:
        conv_id, max_turn = i.rsplit("_", 1)
        if conv_id not in seen_ids:
            seen_ids.add(conv_id)
            conv_meta.append((conv_id, int(max_turn)))
    
    # build mapping: conversation_id -> {turn_index: payload}
    conv_turns = {}
    for i in json.load(open(instruct_path)):
        key, payload = next(iter(i.items()))
        conv_id, turn_id = key.rsplit("_", 1)
        conv_turns.setdefault(conv_id, {})[int(turn_id)] = payload
    
    # assemble conversations in order (only exact num_turns)
    conversations = []
    for conv_id, max_turn in conv_meta:
        # ensure conversation has exactly the requested number of turns
        if max_turn + 1 != num_turns:
            continue

        turn_map = conv_turns.get(conv_id, {})
        
        # ensure all required turns exist (0 .. num_turns-1)
        if all(i in turn_map for i in range(num_turns)):
            conversations.append([turn_map[i]["prompt"] for i in range(num_turns)])
            
            # stop
            if len(conversations) == num_conversations: break

    return conversations


@torch.no_grad()
def hidden_states_and_mask(text, tokenizer, model, hs_index, device):
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    hs = outputs.hidden_states[hs_index]
    mask = inputs["attention_mask"].bool()

    return hs, mask

@torch.no_grad()
def sae_features(sae, hs):
    bsz, seq_len, dim = hs.shape
    flat = hs.reshape(bsz * seq_len, dim)

    _, flat_feat = sae(flat, output_features=True)
    fdim = flat_feat.shape[-1]

    return flat_feat.reshape(bsz, seq_len, fdim)

@torch.no_grad()
def count_activations(fts, mask, thr: float):
    t_mask = mask.unsqueeze(-1)

    return ((fts > thr) & t_mask).sum(dim=(0, 1))

@torch.no_grad()
def last_token_activations(fts, mask, thr: float):
    last_idx = int(mask.sum(dim=1).item()) - 1
    
    return (fts[0, last_idx] > thr).to(torch.float32)

def avg_token_activation_rate(feats, mask, threshold: float):
        """
        fer-feature activation rate for a turn:
        (# valid tokens where activation > threshold) / (# valid tokens)
        """
        # handle shapes [1, T, D] or [T, D]
        if feats.dim() == 3:
            feats_ = feats[0]
        else:
            feats_ = feats

        if mask.dim() == 2:
            mask_ = mask[0]
        else:
            mask_ = mask

        valid = mask_.bool()  # [T]
        num_tokens = valid.sum().clamp_min(1)

        feats_valid = feats_[valid]  # [T_valid, D]
        fired_counts = (feats_valid > threshold).float().sum(dim=0)  # [D]

        return fired_counts / num_tokens  # [D]


def generate(model, tokenizer, prompt, max_new_tokens=4, device="cpu"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()


def run_evaluation(dataset, model, tokenizer, out_path, ids_path, tasks, device, save_results=True):
    out = []

    ids = {task: {"success": [], "fail": []} for task in tasks}
    
    for i, id, x in tqdm(dataset, total=len(dataset)):
        # metadata
        task = x["task"]
        metric = x["metric"]
        refs = x.get("reference", [])
        # run inference
        pred = re.sub(r"[^\w\s:]", "", generate(model, tokenizer, x["prompt"], device=device))
        score = eval_metric(metric, pred, refs) 
        # store output 
        x["prediction"] = pred
        x["score"] = score

        out.append({
            id: x
        })

        if score >= 1.0:
            ids[task]["success"].append(id)
        else:
            ids[task]["fail"].append(id)

    if save_results:
        with open(out_path, "w") as f: json.dump(out, f, indent=2, ensure_ascii=False)
        with open(ids_path, "w") as f: json.dump(ids, f, indent=2, ensure_ascii=False)
    # summary
    print(f"per-instance predictions to: {out_path}")
    print(f"success/fail id lists to: {ids_path}")

    for task in tasks:
        print(f"\n== {task} ==")
        print(f"success: {len(ids[task]['success'])} | fail: {len(ids[task]['fail'])}")
