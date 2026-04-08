import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    cleaned = token.replace("Ġ", "").replace("Ċ", "")
    return bool(re.search(r"[A-Za-z0-9]", cleaned))


def _get_token_data(t_data, source):
    if source == 'full':
        return list(t_data['input_tokens']), t_data['feature_activations']
    if source == 'user':
        return list(t_data['user_input_tokens']), t_data['user_feature_activations']
    raise ValueError("source must be 'full' or 'user'")


def _get_tokens_before_assistant(tokens, reps):
    for i in range(len(tokens) - 1):
        if tokens[i] == "<|start_header_id|>" and tokens[i + 1] == "assistant":
            return tokens[:i], reps[:i]
    return tokens, reps


def _get_clean_tokens(tokens, reps, spl_tokens):
    tokens = list(tokens)

    while tokens:
        token = tokens[-1]
        cleaned = token.replace("Ġ", "").replace("Ċ", "")
        if token in spl_tokens or cleaned == "":
            tokens.pop()
            reps = reps[:-1]
        else:
            break

    return tokens, reps


def _get_token_positions(tokens, selector, spl_tokens, lexical_only=False):
    if selector == 'all':
        if lexical_only:
            return [i for i, token in enumerate(tokens) if valid_token(token, spl_tokens)]
        return list(range(len(tokens)))

    if selector == 'last':
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
        return list(reversed(positions))

    raise ValueError("selector must be 'all' or 'last'")


def get_turn_representation(t_data, source, selector, pooling='mean', lexical_only=False, spl_tokens=None):
    spl_tokens = DEFAULT_SPECIAL_TOKENS if spl_tokens is None else spl_tokens

    tokens, reps = _get_token_data(t_data, source)
    tokens, reps = _get_tokens_before_assistant(tokens, reps)
    tokens, reps = _get_clean_tokens(tokens, reps, spl_tokens)

    positions = _get_token_positions(
        tokens,
        selector,
        spl_tokens,
        lexical_only=lexical_only,
    )

    if not positions:
        raise ValueError("no tokens selected")

    return reduce_tensor(reps[positions], pooling)


def _get_id_turn(name):
    match = re.match(r"^(.*)_(\d+)$", name)
    if not match:
        return name, None
    return match.group(1), int(match.group(2))


def _get_activation_index(act_dir):
    index = defaultdict(list)

    for fname in os.listdir(act_dir):
        if not fname.endswith(".pt"):
            continue

        stem = fname[:-3]
        root, turn = _get_id_turn(stem)
        index[root].append((stem, turn, os.path.join(act_dir, fname)))

    for root in index:
        index[root].sort(key=lambda x: (x[1] is None, x[1]))

    return index


def _load_conv_data(conv_id, file_index, selected_ids=None):
    root, _ = _get_id_turn(conv_id)

    if root not in file_index:
        raise FileNotFoundError(f"No activation files found for conversation: {conv_id}")

    turns = []
    for ex_id, _, path in file_index[root]:
        if selected_ids is not None and ex_id not in selected_ids:
            continue
        obj = torch.load(path)
        turns.append(obj["results"][0])

    if not turns:
        raise FileNotFoundError(f"No activation files found for selected ids in conversation: {conv_id}")

    return root, turns


def _flatten_ids(ids_subset):
    if not isinstance(ids_subset, dict):
        raise ValueError("ids subset must be a dict")

    if all(isinstance(v, list) for v in ids_subset.values()):
        return ids_subset

    flat = defaultdict(list)
    for _, group in ids_subset.items():
        for label, values in group.items():
            flat[label].extend(values)

    return dict(flat)


def collate_ids_by_verdict_turn(ids, verdict_turn=0):
    flat = {}

    for task_name, turn_data in ids.items():
        if verdict_turn not in turn_data:
            raise ValueError(
                f"verdict_turn={verdict_turn} not found for task '{task_name}'. "
                f"Available turns: {sorted(turn_data.keys())}"
            )
        verdict_pass = set(i.rsplit('_', 1)[0] for i in turn_data[verdict_turn]['pass'])
        verdict_fail = set(i.rsplit('_', 1)[0] for i in turn_data[verdict_turn]['fail'])

        # ensure mutual exclusivity - verdict_turn's pass takes priority
        verdict_fail = verdict_fail - verdict_pass

        result = {'pass': list(verdict_pass), 'fail': list(verdict_fail)}
        flat[task_name] = result

    return flat


def get_activation_instances(
    ids,
    base_f,
    filter=None,
    act_subdir="activations_topk",
    source='user',
    selector='all',
    pooling='mean',
    lexical_only=True,
    all_turns=False,
):
    act_dir = os.path.join(base_f, act_subdir)
    file_index = _get_activation_index(act_dir)

    instances = {}

    filter_ids = ids if not filter else ids[filter]
    filter_ids = _flatten_ids(filter_ids)

    for label in filter_ids:
        processed = []
        selected_ids = None if all_turns else set(filter_ids[label])
        seen_roots = set()

        for conv_id in filter_ids[label]:
            conv_root, _ = _get_id_turn(conv_id)
            if conv_root in seen_roots:
                continue
            seen_roots.add(conv_root)

            conv_root, turn_results = _load_conv_data(
                conv_id,
                file_index,
                selected_ids=selected_ids,
            )

            turns = []
            for t_data in turn_results:
                rep = get_turn_representation(
                    t_data,
                    source=source,
                    selector=selector,
                    pooling=pooling,
                    lexical_only=lexical_only,
                )
                turns.append({
                    'turn': t_data['turn'],
                    'representation': rep,
                })

            processed.append({
                'id': conv_root,
                'turns': turns,
            })

        instances[label] = processed

    if 'pass' in instances and 'fail' in instances:
        instances['all'] = instances['pass'] + instances['fail']

    return instances


def summarize(vectors, l0_threshold=1e-6):
    if not vectors:
        return None

    x = torch.stack(vectors).float()

    mean_vec = x.mean(dim=0)
    median_vec = x.median(dim=0).values
    l0 = (x.abs() > l0_threshold).sum(dim=1).float()

    num_concepts = mean_vec.numel()
    active_mean = (mean_vec > 0).sum().item()
    active_median = (median_vec > 0).sum().item()

    return {
        "count": x.shape[0],
        "mean": mean_vec,
        "median": median_vec,
        "std": x.std(dim=0, unbiased=False) if x.shape[0] > 1 else torch.zeros_like(x[0]),
        "var": x.var(dim=0, unbiased=False) if x.shape[0] > 1 else torch.zeros_like(x[0]),
        "mean_l0": l0.mean().item(),
        "std_l0": l0.std(unbiased=False).item() if x.shape[0] > 1 else 0.0,
        "l0_distribution": l0.cpu().tolist(),
        "active_mean": active_mean,
        "active_frac_mean": active_mean / num_concepts,
        "active_median": active_median,
        "active_frac_median": active_median / num_concepts,
        "num_concepts": num_concepts,
    }


def cosine(a, b):
    return F.cosine_similarity(a.float().unsqueeze(0), b.float().unsqueeze(0)).item()


def get_stats(instances, level='turn'):
    if level not in {'turn', 'transition'}:
        raise ValueError("level must be 'turn' or 'transition'")

    grouped = {}

    if level == 'turn':
        for instance in instances:
            for item in instance['turns']:
                grouped.setdefault(item['turn'], []).append(item['representation'])

        return {
            'level': level,
            'stats': {k: summarize(v) for k, v in sorted(grouped.items())},
        }

    scalar = {}
    all_deltas = []
    all_cos = []
    all_l2 = []

    for instance in instances:
        vectors = [item['representation'].float() for item in instance['turns']]

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
    order=None,
    top_k=None,
    normalize=False,
    split='all',
    stat='mean',
    order_label=None,
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

    if order is None:
        ref_group = groups[0]
        if stat not in stats_map[ref_group]:
            raise ValueError("stat not found in stats")
        ref_vec = stats_map[ref_group][stat].float().cpu()
        order_vec = standardize(ref_vec, ref_vec) if normalize else ref_vec
        order = torch.argsort(order_vec, descending=True)
        order_label = order_label or f"{split}:{stat}:{ref_group}"

    if top_k is not None:
        order = order[:top_k]

    rows = []

    ref_vec = None
    if normalize:
        ref_group = groups[0]
        ref_vec = stats_map[ref_group][stat].float().cpu()

    for group in groups:
        if stat not in stats_map[group]:
            raise ValueError(f"stat '{stat}' not found in stats for group {group}")

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
                    'order_by': order_label,
                    'normalized': normalize,
                    'stat': stat,
                }
            )

    return pd.DataFrame(rows)


def _get_split_examples(data):
    for split in ('pass', 'fail'):
        for ex in data.get(split, []):
            yield split, ex


class FeatureImportance:
    def __init__(
        self,
        data,
        reg_values=(0.01, 0.1, 1.0, 10.0),
        n_splits=5,
        seed=44,
    ):
        self.data = data
        self.reg_values = list(reg_values)
        self.n_splits = n_splits
        self.seed = seed
        self.results = {}

    def available_turns(self):
        turns = set()

        for _, ex in _get_split_examples(self.data):
            for tdata in ex.get('turns', []):
                turns.add(tdata['turn'])

        return sorted(turns)

    def build_dataset(self, mode='turn', turn=None):
        X = []
        y = []

        for split, ex in _get_split_examples(self.data):
            label = 1 if split == 'pass' else 0
            turns = ex.get('turns', [])

            if mode == 'turn':
                if turn is None:
                    raise ValueError("turn must be provided when mode='turn'")

                rep = next((t['representation'] for t in turns if t['turn'] == turn), None)
                if rep is None:
                    continue

                X.append(rep.float().cpu().numpy())
                y.append(label)

            elif mode == 'global':
                if not turns:
                    continue

                reps = [t['representation'].float().cpu().numpy() for t in turns]
                X.append(np.mean(reps, axis=0))
                y.append(label)

            else:
                raise ValueError("mode must be 'turn' or 'global'")

        if not X:
            if mode == 'turn':
                raise ValueError(f"No data for turn {turn}")
            raise ValueError("No data found")

        return np.stack(X), np.array(y)

    def select_reg(self, X, y):
        rows = []

        for c in self.reg_values:
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(
                    C=c,
                    penalty='l1',
                    solver='liblinear',
                    max_iter=2000,
                    random_state=self.seed,
                ))
            ])

            cv = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.seed,
            )

            scores = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring={'acc': 'accuracy', 'auc': 'roc_auc'},
                return_train_score=True,
            )

            rows.append({
                'c': c,
                'acc_mean': float(np.mean(scores['test_acc'])),
                'acc_std': float(np.std(scores['test_acc'])),
                'auc_mean': float(np.mean(scores['test_auc'])),
                'auc_std': float(np.std(scores['test_auc'])),
                'train_acc_mean': float(np.mean(scores['train_acc'])),
                'train_acc_std': float(np.std(scores['train_acc'])),
                'train_auc_mean': float(np.mean(scores['train_auc'])),
                'train_auc_std': float(np.std(scores['train_auc'])),
            })

        return pd.DataFrame(rows).sort_values(
            ['auc_mean', 'acc_mean'],
            ascending=False,
        ).reset_index(drop=True)

    def compute(self, mode='turn', turn=None, c=None, n_top_features=None):
        X, y = self.build_dataset(mode=mode, turn=turn)

        if c is None:
            reg_table = self.select_reg(X, y)
            best_c = reg_table.loc[0, 'c']
        else:
            reg_table = None
            best_c = c

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=best_c,
                penalty='l1',
                solver='liblinear',
                max_iter=2000,
                random_state=self.seed,
            ))
        ])
        model.fit(X, y)

        coef = model.named_steps['clf'].coef_[0]
        imp = np.abs(coef)

        num_coef = len(coef)
        num_zero = int(np.sum(coef == 0))
        num_nonzero = int(num_coef - num_zero)

        feat_table = pd.DataFrame({
            'feature': np.arange(len(coef)),
            'coefficient': coef,
            'importance': imp,
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        top_features = feat_table['feature'].head(n_top_features).tolist()
        key = turn if mode == 'turn' else 'global'

        result = {
            'mode': mode,
            'turn': turn,
            'num_samples': len(y),
            'num_features': X.shape[1],
            'best_c': best_c,
            'regularization_table': reg_table,
            'model': model,
            'feature_table': feat_table,
            'top_features': top_features,
            'num_zero_coef': int(num_zero),
            'num_nonzero_coef': int(num_nonzero),
            'sparsity': float(num_zero / num_coef),
            'density': float(num_nonzero / num_coef),
        }

        self.results[key] = result
        return result

    def top_feature_turn_values_df(self, result_key, top_n=None):
        if result_key not in self.results:
            raise ValueError(f"result_key {result_key!r} not found in self.results")

        result = self.results[result_key]
        top_features = result['top_features']
        if top_n is not None:
            top_features = top_features[:top_n]

        rows = []

        for split in ('pass', 'fail'):
            label = 1 if split == 'pass' else 0
            ex_list = self.data.get(split, [])

            for example_idx, ex in enumerate(ex_list):
                turns = ex.get('turns', [])
                if not turns:
                    continue

                for tdata in turns:
                    rep = tdata['representation'].float().cpu().numpy()

                    row = {
                        'split': split,
                        'label': label,
                        'example_idx': example_idx,
                        'turn': tdata['turn'],
                    }
                    for feat in top_features:
                        row[f'feature_{feat}'] = float(rep[feat])

                    rows.append(row)

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        return df.sort_values(
            ['split', 'example_idx', 'turn'],
            ascending=[True, True, True],
        ).reset_index(drop=True)
