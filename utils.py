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

from multif.multif_eval import *


# ------------------------------------------------------------------
# MultiTurnInstruct evaluation metrics
# ------------------------------------------------------------------

def extract_choices(text):
    """
    Extract answer choices from text

    Supports formats like:
        "Answer: A"
        "Answer: A, C"
        trailing standalone capital letters
    """
    m = re.search(r"Answer:\s*([A-Z](?:\s*,\s*[A-H])*)", text, flags=re.I)
    if m:
        return set(re.findall(r"[A-Z]", m.group(1).upper()))

    letters = re.findall(r"\b([A-Z])\b", text.upper())
    return {letters[-1]} if letters else set()


def matching_all(pred_text, refs):
    """
    Match if any extracted prediction choice overlaps gold choices 
    """
    pred = extract_choices(pred_text)
    gold = {r.upper() for r in refs}
    return float(len(pred & gold) > 0)


def matching_any(pred_text, refs):
    """
    Return 1.0 if any reference string appears in the prediction
    """
    p = pred_text.lower()
    return float(any(r.lower() in p for r in refs))


def matching_none(pred_text, refs):
    """
    Return 1.0 if none of the reference strings appear in the prediction
    """
    p = pred_text.lower()
    return float(not any(r.lower() in p for r in refs))


def eval_metric(metric, pred_text, refs):
    """
    Dispatch simple task-specific metrics
    """
    if metric == 'matching_any_exact':
        return matching_all(pred_text, refs)
    elif metric == 'matching_any':
        return matching_any(pred_text, refs)
    elif metric == 'not_matching_any':
        return matching_none(pred_text, refs)
    else:
        raise ValueError(f"unexpected metric for selected tasks: {metric}")


class Generate:
    def __init__(self, model_id, sae_id, base_f, device='cpu', max_new_tokens=256):
        self.base_f = base_f
        self.device = device

        # tokenizer setup
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # model setup
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype='auto',
        ).to(device).eval()
        # SAE setup
        self.sae = AutoEncoderTopK.from_pretrained(
            path=sae_id,
            load_from_sae_lens=True,
            device=device,
        )
        # runtime storage for captured activations
        self.current_activations = None

        # this keeps behavior close to the model's .generate() defaults
        self.gen_cfg = self._resolve_generation_defaults()
        self.default_max_new_tokens = max_new_tokens

    # ------------------------------------------------------------------
    # generation defaults
    # ------------------------------------------------------------------

    def _resolve_generation_defaults(self):
        """
        Resolve generation defaults from model.generation_config.
        """
        cfg = getattr(self.model, "generation_config", None)

        eos_token_id = getattr(cfg, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        pad_token_id = getattr(cfg, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id

        do_sample = bool(getattr(cfg, "do_sample", False))

        temperature = getattr(cfg, "temperature", 1.0)
        if temperature is None:
            temperature = 1.0

        top_p = getattr(cfg, "top_p", 1.0)
        if top_p is None:
            top_p = 1.0

        max_length = getattr(cfg, "max_length", None)

        return {
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "do_sample": do_sample,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_length": max_length,
        }

    # ------------------------------------------------------------------
    # data loading
    # ------------------------------------------------------------------

    def load_data(self, data_f, tasks=None):
        """
        """
        self.tasks = tasks
        self.dataset = json.load(open(os.path.join(self.base_f, data_f), "r"))

        self.subset = []
        for i, item in enumerate(self.dataset):
            for key, ex in item.items():
                if tasks is None:
                    self.subset.append((i, key, ex))
                elif "task" in ex and ex.get("task") in tasks:
                    self.subset.append((i, key, ex))

    def load_multif_data(self, data_f, filter_kwargs=None):
        """
        """
        self.dataset = json.load(open(os.path.join(self.base_f, data_f), "r"))
        self.subset = []

        if filter_kwargs is None:
            for i, item in enumerate(self.dataset):
                for key, ex in item.items():
                    self.subset.append((i, key, ex))
            return

        instruction_list = filter_kwargs.get("instruction_list")
        turn = filter_kwargs.get("turn")
        match_all = filter_kwargs.get("match_all", False)

        valid_roots = {
            self._conversation_root(key)
            for item in self.dataset
            for key, ex in item.items()
            if (turn is None or self._turn_index(key) == turn) and (
                instruction_list is None
                or (
                    all(
                        instr in ex.get("reference", {}).get("instruction_id_list", [])
                        for instr in instruction_list
                    )
                    if match_all
                    else any(
                        instr in ex.get("reference", {}).get("instruction_id_list", [])
                        for instr in instruction_list
                    )
                )
            )
        }
        for i, item in enumerate(self.dataset):
            for key, ex in item.items():
                if self._conversation_root(key) in valid_roots:
                    self.subset.append((i, key, ex))

    # ------------------------------------------------------------------
    # id helpers
    # ------------------------------------------------------------------

    def _conversation_root(self, ex_id):
        """
        Extract the conversation root from ids (keys)
        """
        m = re.match(r"^(.*?)(?:_(?:turn|max_turn)_\d+|_\d+)$", ex_id)
        return m.group(1) if m else ex_id

    def _turn_index(self, ex_id):
        """
        Extract the turn index from ids (keys)
        """
        m = re.match(r"^(.*?)(?:_(?:turn|max_turn)_(\d+)|_(\d+))$", ex_id)
        if not m:
            return 0
        idx = m.group(2) or m.group(3)
        return int(idx)

    def _sorted_items(self, stitch_turns):
        """
        Sort subset by conversation and turn when rollout stitching is needed
        """
        if not stitch_turns:
            return self.subset

        return sorted(
            self.subset,
            key=lambda x: (self._conversation_root(x[1]), self._turn_index(x[1])),
        )

    # ------------------------------------------------------------------
    # hooks / encoding / decoding
    # ------------------------------------------------------------------

    def hook_fn(self, module, input, output):
        """
        Capture hidden states from the hooked layer
        """
        self.current_activations = output[0] if isinstance(output, tuple) else output

    def encode(self, data):
        """
        Tokenize chat messages
        """
        return self.tokenizer.apply_chat_template(
            data,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            truncation=True
        ).to(self.device)

    def decode(self, inputs, outputs):
        """
        Decode only the newly generated continuation, excluding the prompt
        """
        input_len = inputs["input_ids"].shape[-1]
        new_tokens = outputs[0, input_len:]

        if new_tokens.numel() == 0:
            return ""

        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # single-pass prompt forward + cached decoding
    # ------------------------------------------------------------------

    def _prefill_with_capture(self, messages, layer):
        """
        Run a prompt forward pass and capture hidden states from the selected layer
        """
        self.current_activations = None
        handle = self.model.model.layers[layer].register_forward_hook(self.hook_fn)

        try:
            inputs = self.encode(messages)
            with t.no_grad():
                self.model(
                    **inputs,
                    use_cache=False,
                    return_dict=True,
                )

            if self.current_activations is None:
                raise RuntimeError("no activations captured during forward pass")

            hidden_states = self.current_activations.detach()
            return inputs, hidden_states
        finally:
            handle.remove()

    def generate(self, inputs):
        """
        Generate with the model's standard Hugging Face generation path
        """
        gen_kwargs = {
            "max_new_tokens": self.default_max_new_tokens,
            "eos_token_id": self.gen_cfg["eos_token_id"],
            "pad_token_id": self.gen_cfg["pad_token_id"],
            "do_sample": self.gen_cfg["do_sample"],
        }

        if self.gen_cfg["do_sample"]:
            gen_kwargs["temperature"] = self.gen_cfg["temperature"]
            gen_kwargs["top_p"] = self.gen_cfg["top_p"]

        with t.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        return outputs

    def infer(self, messages, layer):
        """
        Capture hidden states on one prompt pass, then generate using 
        standard HF generate()
        """
        inputs, hidden_states = self._prefill_with_capture(messages, layer)
        generated_ids = self.generate(inputs)
        pred = self.decode(inputs, generated_ids)

        return {
            "inputs": inputs,
            "hidden_states": hidden_states,
            "prediction": pred,
            "generated_ids": generated_ids,
        }

    # ------------------------------------------------------------------
    # utility helpers
    # ------------------------------------------------------------------

    def top_features_for_tokens(
        self,
        feature_activations,
        tokens,
        target_tokens,
        top_k=20,
        aggregation="mean",
        exact_match=True,
    ):
        """
        Return top SAE features for selected token positions
        """
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
            return

        selected = feature_activations[matched_pos, :]

        if aggregation == "mean":
            scores = selected.mean(dim=0)
        elif aggregation == "max":
            scores = selected.max(dim=0).values
        else:
            raise ValueError("aggregation must be 'mean' or 'max'")

        topk = t.topk(scores, k=min(top_k, scores.shape[0]))
        return topk

    def load_results(self, data_f, mt=False):
        """
        Load saved prediction results
        """
        with open(os.path.join(self.base_f, data_f), "r") as f:
            data = json.load(f)

        records = []
        for item in data:
            for sample_id, content in item.items():
                records.append({
                    "id": sample_id,
                    "task": content.get("task", "unknown"),
                    "score": content.get("score", 0),
                    "root": self._conversation_root(sample_id),
                    "turn": self._turn_index(sample_id),
                })
        if mt:
            grouped = defaultdict(lambda: defaultdict(lambda: {"pass": [], "fail": []}))

            for rec in records:
                bucket = "pass" if rec["score"] == 1.0 else "fail"
                grouped[rec["task"]][rec["turn"]][bucket].append(rec["id"])

            return {
                task: {turn: dict(v) for turn, v in sorted(turns.items())}
                for task, turns in grouped.items()
            }
        last_by_root = {}
        for rec in records:
            root = rec["root"]
            if root not in last_by_root or rec["turn"] > last_by_root[root]["turn"]:
                last_by_root[root] = rec

        grouped = defaultdict(lambda: {"pass": [], "fail": []})
        for rec in last_by_root.values():
            bucket = "pass" if rec["score"] == 1.0 else "fail"
            grouped[rec["task"]][bucket].append(rec["id"])

        return dict(grouped)

    def load_multif_results(self, data_f, instruction_list=None, turn=0):
        """
        Load Multi-IF results and organize pass/fail by instruction and turn
        """
        with open(os.path.join(self.base_f, data_f), "r") as f:
            data = json.load(f)

        instruction_filter = set(instruction_list) if instruction_list is not None else None

        by_base = defaultdict(dict)
        all_turns = set()

        for item in data:
            for sample_id, content in item.items():
                base_id = self._conversation_root(sample_id)
                t_idx = self._turn_index(sample_id)
                by_base[base_id][t_idx] = content
                all_turns.add(t_idx)

        all_turns = sorted(all_turns)

        anchor_ids = defaultdict(list)
        for base_id, turn_map in by_base.items():
            content = turn_map.get(turn)
            if content is None:
                continue

            insts = content.get("reference", {}).get("instruction_id_list", [])
            for inst in insts:
                if instruction_filter is None or inst in instruction_filter:
                    anchor_ids[inst].append(base_id)

        results = defaultdict(lambda: defaultdict(lambda: {"pass": [], "fail": []}))

        for inst, base_ids in anchor_ids.items():
            for base_id in base_ids:
                turn_map = by_base[base_id]
                for t_idx in all_turns:
                    content = turn_map.get(t_idx)
                    sample_id = f"{base_id}_{t_idx}"

                    if content is None:
                        results[inst][t_idx]["fail"].append(sample_id)
                        continue

                    if content.get("instruction_scores", {}).get(inst, 0) == 1:
                        results[inst][t_idx]["pass"].append(sample_id)
                    else:
                        results[inst][t_idx]["fail"].append(sample_id)
        return {
            inst: {t_idx: dict(v) for t_idx, v in sorted(turns.items())}
            for inst, turns in results.items()
        }

    # ------------------------------------------------------------------
    # stitching helpers
    # ------------------------------------------------------------------

    def _build_stitched_messages(self, messages, hist):
        """
        Rebuild message list using stitched assistant history without duplicating
        assistant turns

        Behavior:
            If the dataset already contains assistant turns, replace those prior
            assistant turns with stitched history.
            If the dataset contains only user turns, insert stitched assistant turns
            after prior user turns.
        """
        has_assistant = any(m.get("role") == "assistant" for m in messages)

        # case 1: messages already contain assistant turns.
        # replace existing assistant messages instead of inserting new ones.
        if has_assistant:
            new_msgs = []
            hist_idx = 0

            for m in messages:
                if m.get("role") == "assistant":
                    if hist_idx < len(hist):
                        new_msgs.append({
                            "role": "assistant",
                            "content": hist[hist_idx],
                        })
                        hist_idx += 1
                    else:
                        new_msgs.append(m)
                else:
                    new_msgs.append(m)

            return new_msgs

        # case 2: messages are user-only.
        # insert stitched assistant replies after prior user turns.
        new_msgs = []
        for j, m in enumerate(messages):
            new_msgs.append(m)
            if j < len(hist):
                new_msgs.append({
                    "role": "assistant",
                    "content": hist[j],
                })

        return new_msgs

    def _stitch_reply(self, ex, pred, stitch_turns):
        """
        Choose the assistant reply to pass to the next turn

            Behavior:
            if stitch_turns=True:
                prefer previous prediction; fall back to gold
            if stitch_turns=False:
                prefer gold; fall back to previous prediction
        """
        gold = ex.get("response")

        if stitch_turns:
            return pred if pred is not None else gold

        return gold if gold is not None else pred

    # ------------------------------------------------------------------
    # scoring helper
    # ------------------------------------------------------------------

    def _score_prediction(self, ex, pred):
        """
        """
        score = None
        extra = {}

        metric = ex.get("metric", None)
        ref = ex.get("reference", [])

        if metric and ref:
            if metric == "llm_judge":
                cri = ex["response"]
                score, jr, jv = llm_judge(pred, cri, ref)
                extra["judge_reasoning"] = jr
                extra["judge_verdict"] = jv

            elif metric == "rule_based":
                res = eval_generation_strict(pred, ref)
                score = res["conversation_level_strict"]
                extra["conversation_level_strict"] = res["conversation_level_strict"]
                extra["instruction_level_strict"] = res["instruction_level_strict"]
                extra["instruction_scores"] = res["instruction_scores"]

            else:
                score = eval_metric(metric, pred, ref)

        return score, extra

    # ------------------------------------------------------------------
    # predictions / shared inference orchestration
    # ------------------------------------------------------------------

    def _infer_payload_if_needed(self, messages, layer, mode, stitch_turns):
        """
        Run inference once when the current configuration needs a prediction.

        This avoids recomputing the same prompt pass in ``mode="both"`` and
        still preserves the old behavior where stitching requires a generated
        reply even if predictions are not being saved.
        """
        do_predictions = mode in {"predictions", "both"}

        if do_predictions or stitch_turns:
            return self.infer(messages, layer)

        return None

    def run(
        self,
        layer,
        out_f="predictions.json",
        save_results=True,
        stitch_turns=False,
        mode="predictions",
        act_out_f=None,
    ):
        """
        Runner for predictions, activation dumps, or both.

        Parameters
        ----------
        layer : int
            Transformer layer to tap for hidden states / SAE activations.
        out_f : str
            Output JSON filename for prediction results.
        save_results : bool
            Whether to save prediction JSON when predictions are produced.
        stitch_turns : bool
            Whether to stitch previous-turn replies into the current turn context.
        mode : str
            'predictions': save prediction results only
            'activations': save SAE activation dumps only
            'both': save both predictions and activation dumps
        act_out_f : str or None
            Directory name for activation dumps when mode is 'activations' or 'both'.
        """
        if mode not in {"predictions", "activations", "both"}:
            raise ValueError("mode must be one of: 'predictions', 'activations', 'both'")

        do_predictions = mode in {"predictions", "both"}
        do_activations = mode in {"activations", "both"}

        out = []

        if getattr(self, "tasks", None):
            self.ids = {t: {"pass": [], "fail": []} for t in self.tasks}
        else:
            self.ids = {}

        self.ids["all"] = {"pass": [], "fail": []}

        items = self._sorted_items(stitch_turns)[ : 5]
        cur_k = None
        hist = []

        act_dir = None
        if do_activations:
            if act_out_f is None:
                raise ValueError("act_out_f must be provided when mode is 'activations' or 'both'")
            act_dir = os.path.join(self.base_f, act_out_f)
            if not os.path.exists(act_dir):
                os.makedirs(act_dir)

        for _, eid, ex in tqdm(items, desc=f"run [{mode}]"):
            if stitch_turns:
                k = self._conversation_root(eid)
                if k != cur_k:
                    cur_k = k
                    hist = []
                new_msgs = self._build_stitched_messages(ex["messages"], hist)
            else:
                new_msgs = ex["messages"]

            infer_payload = self._infer_payload_if_needed(
                messages=new_msgs,
                layer=layer,
                mode=mode,
                stitch_turns=stitch_turns,
            )
            pred = None if infer_payload is None else infer_payload["prediction"]

            if do_activations:
                self._save_activation_dump(
                    ex_id=eid,
                    messages=new_msgs,
                    layer=layer,
                    save_dir=act_dir,
                    cached_full_pass=infer_payload,
                )
            if do_predictions:
                score, extra = self._score_prediction(ex, pred)
                out_ex = dict(ex)
                out_ex["messages"] = new_msgs
                out_ex["prediction"] = pred
                out_ex["score"] = score
                out_ex.update(extra)

                out.append({eid: out_ex})

                if score is not None:
                    if score >= 1.0:
                        self.ids["all"]["pass"].append(eid)
                    else:
                        self.ids["all"]["fail"].append(eid)

                    task = ex.get("task")
                    if task is not None and task in self.ids:
                        if score >= 1.0:
                            self.ids[task]["pass"].append(eid)
                        else:
                            self.ids[task]["fail"].append(eid)
            if stitch_turns:
                hist.append(self._stitch_reply(ex, pred, stitch_turns))

        if do_predictions and save_results:
            out_path = os.path.join(self.base_f, out_f)
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"results saved: {out_path}")

        return out if do_predictions else None

    # ------------------------------------------------------------------
    # activations / SAE dumps only
    # ------------------------------------------------------------------

    def _get_prediction_if_needed(self, messages, layer, mode, stitch_turns):
        """
        Backward-compatible wrapper that returns only the prediction string
        """
        payload = self._infer_payload_if_needed(messages, layer, mode, stitch_turns)
        return None if payload is None else payload["prediction"]

    def _build_activation_dump(self, ex_id, messages, layer, cached_full_pass=None):
        """
        Build the SAE activation object
        """
        msg_prefix = messages

        if cached_full_pass is not None:
            inputs = cached_full_pass["inputs"]
            hidden_states = cached_full_pass["hidden_states"]
        else:
            inputs, hidden_states = self._prefill_with_capture(msg_prefix, layer)

        feature_activations = self.sae.encode(hidden_states)

        # slice out only the last user utterance from the full encoded prefix
        if len(msg_prefix) == 1:
            prev_len = 0
        else:
            prev_inputs = self.encode(msg_prefix[:-1])
            prev_len = prev_inputs["input_ids"].shape[1]

        full_len = inputs["input_ids"].shape[1]

        rt = {
            "turn": sum(1 for m in msg_prefix if m.get("role") == "user"),
            "num_messages": len(msg_prefix),
            "messages": msg_prefix,
            # full prefix
            "feature_activations": feature_activations[0].detach().cpu(),
            "input_ids": inputs["input_ids"][0].detach().cpu(),
            "input_tokens": self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
            "hidden_states": hidden_states[0].detach().cpu(),
            # only last user utterance
            "user_feature_activations": feature_activations[0][prev_len:full_len].detach().cpu(),
            "user_input_ids": inputs["input_ids"][0][prev_len:full_len].detach().cpu(),
            "user_input_tokens": self.tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0][prev_len:full_len]
            ),
            "user_hidden_states": hidden_states[0][prev_len:full_len].detach().cpu(),
        }

        return {
            "id": ex_id,
            "results": [rt],
        }

    def _save_activation_dump(self, ex_id, messages, layer, save_dir, cached_full_pass=None):
        """
        Build and save SAE activation for one example
        """
        saved_obj = self._build_activation_dump(
            ex_id=ex_id,
            messages=messages,
            layer=layer,
            cached_full_pass=cached_full_pass,
        )

        final_path = os.path.join(save_dir, f"{ex_id}.pt")
        with tempfile.NamedTemporaryFile(dir=save_dir, delete=False, suffix=".pt") as tmp:
            tmp_path = tmp.name
        t.save(saved_obj, tmp_path)
        os.replace(tmp_path, final_path)

        return saved_obj
