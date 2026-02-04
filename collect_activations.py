import os
import json
import argparse
from tqdm import trange
from dotenv import load_dotenv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from dictionary_learning import utils
from dictionary_learning import AutoEncoder


load_dotenv()
hf_token = os.getenv("HF_TOKEN") 
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset", type=str, required=True, help="which dataset to use"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, help="which language model to use"
    )
    parser.add_argument(
        "--sae_release_id", type=str, required=True, help="which SAE release to use"
    )
    parser.add_argument(
        "--sae_id", type=str, required=True, help="which SAE to use"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32
    )

    args = parser.parse_args()
    return args

def collect_activations(batch, model, tokenizer, ae):    
    # input text per instance
    inputs = [i[1] for i in batch]  # list[str]
    batch_size = len(inputs)
    # tokenize
    token_ids = tokenizer(
        inputs,
        return_tensors='pt',
        max_length=128,
        padding=True,
        truncation=True,
        add_special_tokens=True
    ).to(device)
    # tokens per instance
    tokens = [
        tokenizer.convert_ids_to_tokens(token_ids["input_ids"][i])
        for i in range(batch_size)
    ]
    # model forward
    model.eval()
    with torch.no_grad():
        outputs = model(
            **token_ids,
            output_hidden_states=True,
            return_dict=True
        )
    # last hidden state: [B, T, H]
    last_hidden = outputs.hidden_states[-1]
    # SAE forward
    outputs_hat, activations = ae.forward(
        last_hidden,
        output_features=True
    )
    # to avoid OOM
    last_hidden = last_hidden.detach().to('cpu')
    outputs_hat = outputs_hat.detach().to('cpu')
    activations = activations.detach().to('cpu')
    # outputs_hat: [B, T, H]  activations: [B, T, F]

    # pack per-instance dicts
    results = []

    for i in range(batch_size):
        results.append({
            "inputs": inputs[i],
            "tokens": tokens[i],
            "token_ids": token_ids["input_ids"][i],
            "outputs": {
                # "logits": outputs.logits[i],
                "hidden_states": [h[i] for h in outputs.hidden_states][-1],
            },
            "outputs_hat": outputs_hat[i],
            "activations": activations[i],
        })
    
    # to avoid OOM
    del token_ids, outputs, outputs_hat, activations
    if device == 'cuda':
        torch.cuda.empty_cache()

    return results

if __name__ == "__main__":
    """
    python collect_activations.py --dataset alpaca.json --model_id google/gemma-2-2b --sae_release_id gemma-scope-2b-pt-mlp-canonical --sae_id layer_12/width_16k/canonical
    """
    args = get_args()
    model_id = args.model_id 
    sae_release_id = args.sae_release_id 
    sae_id = args.sae_id 

    layer = int(sae_id.split("/")[0].split("_")[1])
    # load model
    model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token, dtype=config.LLM_CONFIG[model_id].dtype).to(device)
    print(f"loaded model {model_id} on {device}\n")
    # if layer = 12:, model = model[ : layer + 1]
    model = utils.truncate_model(model, layer)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    # if layer = 12:, submodel = model[layer]
    submodule, layer = utils.get_submodule(model, layer)
    # load SAE
    ae = AutoEncoder.from_pretrained(load_from_sae_lens=True, release=sae_release_id, sae_id=sae_id, device=device)
    print(f"\nloaded SAE {sae_release_id} hooked to layer {layer + 1} of {model_id} on {device}")
    # load dataset
    data = json.load(open(args.dataset, "r"))
    data_l = [[list(i.keys())[0], list(i.values())[0]['prompt']] for i in data]
    
    results = []
    # iterate over dataset
    for i in trange(0, len(data_l), args.batch_size, desc="collecting activations: "):
        # slice batch
        batch = data_l[i : i + args.batch_size]
        # run to collect activations for batch
        result = collect_activations(batch, model, tokenizer, ae)
        results.extend(result)

    # save results
    with open(f"activations/{args.dataset}/{args.model_id}_{args.sae_release_id}_{args.sae_id}.json", "w") as f:
        json.dump(results, f)

    print(f"saved results to activations/{args.dataset}/{args.model_id}_{args.sae_release_id}_{args.sae_id}.json")
