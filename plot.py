import os
import json
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt

from utils import *

mpl.rcParams['figure.dpi'] = 100


def collect_trainers(paths, t_class, x_k, y_k):
    d_vals = {path.split('_')[1]: {x_k: [], y_k: []} for path in paths}

    for path in paths:
        d_size = path.split('_')[1]
        # check trainers
        trainers = os.listdir(path)

        for trainer in trainers:
            # config and results
            t_cfg = json.load(open(os.path.join(path, f"{trainer}/config.json"), "r", encoding="utf-8"))["trainer"]
            t_res = json.load(open(os.path.join(path, f"{trainer}/eval_results.json"), "r", encoding="utf-8"))

            if t_cfg["trainer_class"] == t_class:
                d_vals[d_size][x_k].append(t_cfg[x_k])
                d_vals[d_size][y_k].append(round(t_res[y_k], 3))

    return d_vals


def plot_line(data, x_k, y_k, title=None, xlabel=None, ylabel=None, sort_x=True):
    plt.figure()

    for l_name, vals in data.items():
        x = vals[x_k]
        y = vals[y_k]

        if sort_x:
            xy = sorted(zip(x, y), key=lambda t: t[0])
            x, y = zip(*xy)

        plt.plot(x, y, marker='o', label=l_name)

    plt.xlabel(xlabel if xlabel else x_k)
    plt.ylabel(ylabel if ylabel else y_k)
    
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_concept_activation_count_by_turn(
    conversations,
    sae,
    model,
    tokenizer,
    layer_index: int = 6,
    feature_dim: int = 2**10,
    threshold: float = 0.5,
    mode: str = "last_token",        # "last_token" "avg_tokens"
    plot_mode: str = "per_turn",     # "all" "per_turn"
    device: str | None = None,
    figsize=(10, 4),
    a: int | None = None,
    b: int | None = None,
):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sae = sae.to(device).eval()
    model = model.to(device).eval()

    hidden_state_index = layer_index + 1
    num_conversations = len(conversations)
    if num_conversations == 0:
        return []

    num_turns = len(conversations[0])
    totals = [torch.zeros(feature_dim, device=device) for _ in range(num_turns)]

    for conv in conversations:
        if len(conv) != num_turns:
            raise ValueError(f"all conversations must have {num_turns} turns, got {len(conv)}")

        for t, text in enumerate(conv):
            hs, mask = hidden_states_and_mask(text, tokenizer, model, hidden_state_index, device)
            feats = sae_features(sae, hs)

            if mode == "last_token":
                totals[t] += last_token_activations(feats, mask, threshold)
            elif mode == "avg_tokens":
                totals[t] += avg_token_activation_rate(feats, mask, threshold)
            else:
                raise ValueError('mode must be one of: "last_token", "avg_tokens"')
    
    totals = [t / max(1, num_conversations) for t in totals]

    start = 0 if a is None else a
    end = feature_dim if b is None else b
    x = torch.arange(feature_dim)[start : end]

    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, num_turns - 1)) for i in range(num_turns)]

    if plot_mode == "all":
        plt.figure(figsize=figsize)
        for t in range(num_turns):
            plt.plot(x, totals[t][start:end].detach().cpu(), color=colors[t], alpha=0.9, label=f"Turn {t + 1}")

        plt.ylim(0, 1)
        plt.title(f"normalized concept activations by turn (mode={mode}, thr={threshold})")
        plt.xlabel("concept index")
        plt.ylabel("normalized activation (0–1)")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.show()

    elif plot_mode == "per_turn":
        for t in range(num_turns):
            plt.figure(figsize=figsize)
            plt.plot(x, totals[t][start:end].detach().cpu(), color=colors[t], alpha=0.9, label=f"Turn {t + 1}")

            plt.ylim(0, 1)
            plt.title(f"normalized concept activations (turn {t + 1}, mode={mode})")
            plt.xlabel("concept index")
            plt.ylabel("normalized activation (0–1)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

    else:
        raise ValueError('plot_mode must be "all" or "per_turn"')
