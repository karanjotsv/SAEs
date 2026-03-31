import re
import os
import json
import tempfile
from tqdm import tqdm
from collections import defaultdict

import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm_judge import *
from dictionary_learning import AutoEncoder
from dictionary_learning.trainers.top_k import AutoEncoderTopK


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

def matching_none(pred_text, refs):
    p = pred_text.lower()

    return float(not any(r.lower() in p for r in refs))

def eval_metric(metric, pred_text, refs):
    if metric == "matching_any_exact":
        return matching_all(pred_text, refs)
    elif metric == "matching_any":
        return matching_any(pred_text, refs)
    elif metric == "not_matching_any":
        return matching_none(pred_text, refs)
    else:
        raise ValueError(f"unexpected metric for selected tasks: {metric}")


class generate:
    def __init__(self, model_id, sae_id, base_f, device="cpu"):
        self.base_f = base_f
        # load the tokenizer from the hub
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # ensure a padding token exists for batching
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # load model using the transformers library framework
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype='auto'
        ).to(device).eval()
        # load the autoencoder for feature extraction
        self.sae = AutoEncoderTopK.from_pretrained(
            path=sae_id, 
            load_from_sae_lens=True, 
            device=device
        )
        self.device = device
        # storage for captured activations
        self.current_activations = None
    
    def load_data(self, data_f, tasks):
        self.tasks = tasks
        # load json
        self.dataset = json.load(open(os.path.join(self.base_f, data_f), "r"))

        self.subset = [] 
        for i, item in enumerate(self.dataset):
            for key, ex in item.items():
                if ex.get("task") in tasks:
                    self.subset.append((i, key, ex))

    def hook_fn(self, module, input, output):
        # capture hidden states from the specified layer
        hidden_states = output if isinstance(output, tuple) else output
        self.current_activations = hidden_states

    def encode(self, data):
        # use the preprocessor to format and tokenize a batch
        inputs = self.tokenizer.apply_chat_template(
            data, 
            tokenize=True, 
            add_generation_prompt=True,  
            return_tensors='pt', 
            return_dict=True,
            padding=True,
            truncation=True
        ).to(self.device)
        return inputs

    def decode(self, inputs, outputs):
        input_len = inputs['input_ids'].shape[-1]
        # slice the output to get only the tokens generated after the input
        new_tokens = outputs[0, input_len:] 
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate(self, inputs, max_new_tokens=256):
        # run inference using optimized infrastructure
        with t.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return outputs
    
    def top_features_for_tokens(self, feature_activations, tokens, target_tokens, top_k=20, aggregation='mean', exact_match=True):
        target_set = set(target_tokens)

        matched_pos = []
        for i, tok in enumerate(tokens):
            if exact_match:
                if tok in target_set:
                    matched_pos.append(i)
            else:
                if any(t in tok for t in target_set):
                    matched_pos.append(i)
        if not matched_pos:
            # print("no tokens found!")
            return

        selected = feature_activations[matched_pos, :]   # [num_matches, n_features]

        if aggregation == "mean":
            scores = selected.mean(dim=0)
        elif aggregation == "max":
            scores = selected.max(dim=0).values
        else:
            raise ValueError("aggregation must be 'mean' or 'max'")

        topk = t.topk(scores, k=min(top_k, scores.shape[0]))
        return topk

    def load_results(self, data_f):
        with open(os.path.join(self.base_f, data_f), "r") as f:
            data = json.load(f)

        rt = defaultdict(lambda: {"pass": [], "fail": []})

        for i in data:
            for id, content in i.items():
                task = content.get("task", "unknown")
                score = content.get("score", 0)

                if score == 1.0:
                    rt[task]["pass"].append(id)
                else:
                    rt[task]["fail"].append(id)
        return dict(rt)

    def run(self, out_f, save_results=True):
        out = []
        self.ids = {task: {'pass': [], 'fail': []} for task in self.tasks}
    
        for i in tqdm(self.subset, desc="predictive inference"):
            # instance id
            id = i[1]
            
            input = self.encode(i[2]['messages'])
            output = self.generate(input)
            pred = self.decode(input, output)
            # messages
            i = i[-1]
            # metadata
            task = i['task']
            metric = i['metric']
            ref = i.get('reference', [])
            # evaluate model performance
            if metric == "llm_judge":
                cri = i['response']
                score, judge_reasoning, judge_verdict = llm_judge(pred, cri, ref)
                i['judge_reasoning'] = judge_reasoning
                i['judge_verdict'] = judge_verdict
            else:
                score = eval_metric(metric, pred, ref)

            i['prediction'] = pred
            i['score'] = score

            out.append({id: i})

            if score >= 1.0: self.ids[task]['pass'].append(id)
            else: self.ids[task]['fail'].append(id)
        
        if save_results:
            out_path = os.path.join(self.base_f, out_f)

            with open(out_path, 'w') as f: 
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"results saved: {out_path}")
        return out

    def run_sae(self, layer, out_f, save_results=True):
        out_path = os.path.join(self.base_f, out_f)
        # create output folder
        if not os.path.exists(out_path): os.makedirs(out_path)

        handle = self.model.model.layers[layer].register_forward_hook(self.hook_fn)
        last_saved = None

        try:
            for i in tqdm(self.subset[ : ], desc="activation inference"):
                i_id = i[1]   # format: {unique_id}_{num_turns}
                i = i[2]
                messages = i['messages']

                u_id, n_turns = i_id.rsplit("_", 1)
                n_turns = int(n_turns)

                i_results = []
                for turn_i in range(1, n_turns + 1):
                    msg_end = 2 * turn_i - 1
                    msg_prefix = messages[:msg_end]

                    # full prefix
                    self.current_activations = None
                    inputs = self.encode(msg_prefix)
                    with t.no_grad():
                        self.model(**inputs)
                    
                    if self.current_activations is None:
                        raise RuntimeError(
                            f"no activations captured for instance {i_id}, user turn {turn_i}"
                        )
                    feature_activations = self.sae.encode(self.current_activations)

                    # find token boundary for only the last user utterance
                    if len(msg_prefix) == 1:
                        prev_len = 0
                    else:
                        prev_inputs = self.encode(msg_prefix[:-1])
                        prev_len = prev_inputs['input_ids'].shape[1]

                    full_len = inputs['input_ids'].shape[1]
                    last_user_feature_activations = feature_activations[0][prev_len:full_len].detach().cpu()
                    last_user_input_ids = inputs['input_ids'][0][prev_len:full_len].detach().cpu()
                    last_user_input_tokens = self.tokenizer.convert_ids_to_tokens(
                        inputs['input_ids'][0][prev_len:full_len]
                    )
                    last_user_hidden_states = self.current_activations[0][prev_len:full_len].detach().cpu()

                    rt = {
                        'turn': turn_i,
                        'num_messages': msg_end,
                        'messages': msg_prefix,
                        # full prompt
                        'feature_activations': feature_activations[0].detach().cpu(),
                        'input_ids': inputs['input_ids'][0].detach().cpu(),
                        'input_tokens': self.tokenizer.convert_ids_to_tokens(
                            inputs['input_ids'][0]
                        ),
                        'hidden_states': self.current_activations[0].detach().cpu(),
                        # only last user utterance
                        'user_feature_activations': last_user_feature_activations,
                        'user_input_ids': last_user_input_ids,
                        'user_input_tokens': last_user_input_tokens,
                        'user_hidden_states': last_user_hidden_states,
                    }
                    i_results.append(rt)

                saved_obj = {
                    "id": i_id,
                    "results": i_results,
                }
                if save_results:
                    final_path = os.path.join(out_path, f"{i_id}.pt")
                    
                    with tempfile.NamedTemporaryFile(dir=out_path, delete=False, suffix=".pt") as tmp:
                        tmp_path = tmp.name
                    
                    t.save(saved_obj, tmp_path)
                    os.replace(tmp_path, final_path)

                last_saved = saved_obj
        finally:
            handle.remove()
        return last_saved
