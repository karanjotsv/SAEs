import re
import torch
import torch.nn.functional as F

import pandas as pd
import altair as alt


DEFAULT_SPECIAL_TOKENS = {
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
}


def reduce_tensor(x, mode='mean'):
    x = x.float()

    if mode == 'mean':
        return x.mean(dim=0)
    if mode == 'median':
        return x.median(dim=0).values
    if mode == 'max':
        return x.max(dim=0).values

    raise ValueError("mode must be 'mean', 'median', or 'max'")


def valid_token(token, special_tokens):
    if token in special_tokens:
        return False
    return bool(re.search(r"[A-Za-z0-9]", token.replace("Ġ", "").replace("Ċ", "")))


def get_turn_representation(t_data, source, selector, pooling='mean', lexical_only=False, spl_tokens=None):
    spl_tokens = DEFAULT_SPECIAL_TOKENS if spl_tokens is None else spl_tokens

    if source == 'full':
        tokens = list(t_data['input_tokens'])
        reps = t_data['hidden_states']
    elif source == 'user':
        tokens = list(t_data['user_input_tokens'])
        reps = t_data['user_hidden_states']
    else:
        raise ValueError("source must be 'full' or 'user'")

    for i in range(len(tokens) - 1):
        if tokens[i] == "<|start_header_id|>" and tokens[i + 1] == "assistant":
            tokens = tokens[:i]
            reps = reps[:i]
            break

    while tokens:
        token = tokens[-1]
        cleaned = token.replace("Ġ", "").replace("Ċ", "")
        if token in spl_tokens or cleaned == "":
            tokens.pop()
            reps = reps[:-1]
        else:
            break

    if selector == 'all':
        positions = (
            [i for i, token in enumerate(tokens) if valid_token(token, spl_tokens)]
            if lexical_only
            else list(range(len(tokens)))
        )
    elif selector == 'last':
        positions = []
        for i in reversed(range(len(tokens))):
            token = tokens[i]
            if not valid_token(token, spl_tokens):
                if positions:
                    break
                continue
            positions.append(i)
            if i == 0 or token.startswith("Ġ"):
                break
        positions = list(reversed(positions))
    else:
        raise ValueError("selector must be 'all' or 'last'")

    if not positions:
        raise ValueError("no tokens selected")

    return reduce_tensor(reps[positions], pooling)


def summarize(vectors):
    if not vectors:
        return None

    x = torch.stack(vectors).float()
    norms = x.norm(dim=1)

    return {
        'count': x.shape[0],
        'mean': x.mean(dim=0),
        'median': x.median(dim=0).values,
        'std': x.std(dim=0, unbiased=False) if x.shape[0] > 1 else torch.zeros_like(x[0]),
        'var': x.var(dim=0, unbiased=False) if x.shape[0] > 1 else torch.zeros_like(x[0]),
        'mean_norm': norms.mean().item(),
        'std_norm': norms.std(unbiased=False).item() if x.shape[0] > 1 else 0.0,
    }


def cosine(a, b):
    return F.cosine_similarity(a.float().unsqueeze(0), b.float().unsqueeze(0)).item()


def get_stats(instances, level='turn'):
    if level not in {'turn', 'transition'}:
        raise ValueError("level must be 'turn' or 'transition'")

    grouped = {}

    if level == 'turn':
        for i in instances:
            for item in i['turns']:
                key = item['turn']
                grouped.setdefault(key, []).append(item['representation'])

        return {
            'level': level,
            'stats': {k: summarize(v) for k, v in sorted(grouped.items())},
        }

    scalar = {}
    all_deltas = []
    all_cos = []
    all_l2 = []

    for x in instances:
        vectors = [item['representation'].float() for item in x['turns']]

        for i in range(1, len(vectors)):
            prev_vec = vectors[i - 1]
            curr_vec = vectors[i]
            delta = curr_vec - prev_vec
            key = i + 1

            grouped.setdefault(key, []).append(delta)

            c = cosine(curr_vec, prev_vec)
            l2 = delta.norm().item()

            scalar.setdefault(key, {'cosine': [], 'l2': []})
            scalar[key]['cosine'].append(c)
            scalar[key]['l2'].append(l2)

            all_deltas.append(delta)
            all_cos.append(c)
            all_l2.append(l2)

    return {
        'level': level,
        'stats': {k: summarize(v) for k, v in sorted(grouped.items())},
        'scalar': {
            k: {
                'count': len(v['cosine']),
                'mean_cosine': sum(v['cosine']) / len(v['cosine']) if v['cosine'] else 0.0,
                'mean_l2': sum(v['l2']) / len(v['l2']) if v['l2'] else 0.0,
            }
            for k, v in sorted(scalar.items())
        },
        'overall': {
            'count': len(all_cos),
            'mean_cosine': sum(all_cos) / len(all_cos) if all_cos else 0.0,
            'mean_l2': sum(all_l2) / len(all_l2) if all_l2 else 0.0,
            'delta': summarize(all_deltas),
        },
    }


def standardize(vector, reference):
    return vector.float() - reference.float() + 1.0


def label_for(level, key):
    if level == 'turn':
        return f'turn {key}'
    if level == 'transition':
        return f't{key-1}-t{key}'
    return str(key)


def concept_frame(
    stats,
    groups=None,
    order_by=None,
    top_k=None,
    normalize=False,
    split='all',
    stat='mean',
):
    level = stats['level']
    stats_map = {k: v for k, v in stats['stats'].items() if v is not None}

    if groups is None:
        groups = sorted(stats_map)
    else:
        groups = [g for g in groups if g in stats_map]

    if not groups:
        return pd.DataFrame(
            columns=[
                'level', 'group', 'group_label', 'concept_id',
                'concept_rank', 'value', 'split', 'order_by',
                'normalized', 'stat'
            ]
        )

    order_by = groups[0] if order_by is None else order_by
    if order_by not in stats_map:
        raise ValueError("order_by group not found in stats")

    if stat not in stats_map[order_by]:
        raise ValueError("stat not found in stats")

    ref_vec = stats_map[order_by][stat].float().cpu()
    order_vec = standardize(ref_vec, ref_vec) if normalize else ref_vec
    order = torch.argsort(order_vec, descending=True)

    if top_k is not None:
        order = order[:top_k]

    rows = []

    for group in groups:
        if stat not in stats_map[group]:
            raise ValueError("stat not found in stats")

        vec = stats_map[group][stat].float().cpu()
        if normalize:
            vec = standardize(vec, ref_vec)
        vec = vec[order]

        for rank, concept_id in enumerate(order.tolist(), start=1):
            rows.append(
                {
                    'level': level,
                    'group': group,
                    'group_label': label_for(level, group),
                    'concept_id': concept_id,
                    'concept_rank': rank,
                    'value': vec[rank - 1].item(),
                    'split': split,
                    'order_by': order_by,
                    'normalized': normalize,
                    'stat': stat,
                }
            )

    return pd.DataFrame(rows)
