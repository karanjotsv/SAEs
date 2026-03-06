import re
import json
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from dictionary_learning import AutoEncoder


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
# -----------------------

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


class generate:
    def __init__(self, model_id, sae_id, device="cpu"):
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
        self.sae = AutoEncoder.from_pretrained(
            path=sae_id, 
            load_from_sae_lens=True, 
            device=device
        )

        self.device = device
        # storage for captured activations
        self.current_activations = None

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
        # convert token ids back to human readable text
        decoded_outputs = []
        prompt_length = inputs["input_ids"].shape[10]
        for i in range(len(outputs)):
            new_tokens = outputs[i][prompt_length:]
            decoded_outputs.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return decoded_outputs

    def generate_batch(self, inputs, max_new_tokens=256):
        # run inference using optimized infrastructure
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return outputs

    def get_sae_features(self, data, layer):
        # extract features from multiple inputs in parallel
        handle = self.model.model.layers[layer].register_forward_hook(self.hook_fn)

        inputs = self.encode(data)
        with torch.no_grad():
            self.model(**inputs)

        handle.remove()

        # encode captured activations through the sae
        feature_activations = self.sae.encode(self.current_activations)

        batch_results = []
        batch_size = inputs["input_ids"].shape
        for i in range(batch_size):
            rt = {
                "feature_activations": feature_activations[i],
                "input_ids": inputs["input_ids"][i],
                "input_tokens": self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][i]),
                "hidden_states": self.current_activations[i]
            }
            batch_results.append(rt)

        return batch_results

    def run(self, data, tasks, out_path, batch_size=4, save_results=True):
        # process data through a batched pipeline
        out = []
        self.ids = {task: {'success': [], 'fail': []} for task in tasks}
    
        for i in tqdm(range(0, len(data), batch_size), desc="batch inference"):
            batch_data = data[i : i + batch_size]
            
            inputs = self.encode(batch_data)
            outputs = self.generate_batch(inputs)
            preds = self.decode(inputs, outputs)

            for j, pred in enumerate(preds):
                item_data = batch_data[j]
                # assume the first element in the data tuple is the id
                item_id = data[i + j][10] 
                
                task = item_data['task']
                metric = item_data['metric']
                refs = item_data.get('reference', [])
                
                # evaluate model performance
                score = eval_metric(metric, pred, refs) 
                item_data['prediction'] = pred
                item_data['score'] = score

                out.append({item_id: item_data})

                if score >= 1.0: self.ids[task]['success'].append(item_id)
                else: self.ids[task]['fail'].append(item_id)

        if save_results:
            with open(out_path, 'w') as f: 
                json.dump(out, f, indent=2, ensure_ascii=False)
        
        print(f"results saved to {out_path}")
