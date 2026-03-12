import os
import json
import torch

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import altair as alt
import plotly.graph_objects as go

from utils import *
from feature import *

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


def plot_concepts(
    splits,
    groups=None,
    order_split=None,
    order_group=None,
    top_k=None,
    normalize=False,
    stat='mean',
    title=None,
    width=400,
    height=250,
):
    if order_split is None:
        order_split = list(splits.keys())[0]

    if order_split not in splits:
        raise ValueError(f"order_split '{order_split}' not found in splits")

    order_stats = splits[order_split]
    order_stats_map = {k: v for k, v in order_stats['stats'].items() if v is not None}

    if groups is None:
        groups = sorted(order_stats_map)
    else:
        groups = [g for g in groups if g in order_stats_map]

    if not groups:
        raise ValueError("No valid groups found")

    if order_group is None:
        order_group = groups[0]

    if order_group not in order_stats_map:
        raise ValueError(f"order_group '{order_group}' not found in order split stats")

    if stat not in order_stats_map[order_group]:
        raise ValueError(f"stat '{stat}' not found in order split stats")

    ref_vec = order_stats_map[order_group][stat].float().cpu()
    order_vec = standardize(ref_vec, ref_vec) if normalize else ref_vec
    order = torch.argsort(order_vec, descending=True)

    if top_k is not None:
        order = order[:top_k]

    order_label = f'{order_split}:{stat}:{order_group}'

    frame = pd.concat(
        [
            concept_frame(
                stats_obj,
                groups=groups,
                order=order,
                top_k=None,
                normalize=normalize,
                split=split_name,
                stat=stat,
                order_label=order_label,
            )
            for split_name, stats_obj in splits.items()
        ],
        ignore_index=True,
    )
    y_title = (
        f'{stat} concept value standardized to {order_label}'
        if normalize and len(frame)
        else f'{stat} concept amplitude'
    )
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X(
                'concept_id:O',
                sort=alt.SortField(field='concept_rank', order='ascending'),
                title=None,
                axis=None,
            ),
            y=alt.Y('value:Q', title=y_title),
            color=alt.Color(
                'split:N',
                title='split',
                scale=alt.Scale(
                    scheme='set1'
                )
            ),
            xOffset=alt.XOffset('split:N'),
            tooltip=[
                alt.Tooltip('split:N', title='split'),
                alt.Tooltip('group_label:N', title='group'),
                alt.Tooltip('concept_id:O', title='concept id'),
                alt.Tooltip('concept_rank:Q', title='rank'),
                alt.Tooltip('value:Q', title='value'),
                alt.Tooltip('stat:N', title='stat'),
                alt.Tooltip('order_by:N', title='ordered by'),
            ],
        )
        .properties(width=width, height=height)
        .facet(column=alt.Column('group_label:N', title=None))
        .properties(title=title)
    )


def plot_transition(stats, width=400, height=250):
    if stats['level'] != 'transition':
        raise ValueError("Expected transition stats")

    frame = pd.DataFrame(
        [
            {
                'transition': f't{key-1}-t{key}',
                'step': key,
                'metric': 'mean_cosine',
                'value': values['mean_cosine'],
            }
            for key, values in stats['scalar'].items()
        ] +
        [
            {
                'transition': f't{key-1}-t{key}',
                'step': key,
                'metric': 'mean_l2',
                'value': values['mean_l2'],
            }
            for key, values in stats['scalar'].items()
        ]
    )
    return (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X('step:Q', title='transition ending at turn'),
            y=alt.Y('value:Q', title='value'),
            tooltip=['transition', 'metric', 'value'],
        )
        .properties(width=width, height=height)
        .facet(column=alt.Column('metric:N', title=None))
    )


def plot_l0(splits, groups=None, width=400, height=250):
    if isinstance(splits, dict) and 'stats' in splits:
        stats_items = [('all', splits)]
    else:
        stats_items = list(splits.items())

    rows = []

    for split_name, stats in stats_items:
        level = stats['level']
        stats_map = {k: v for k, v in stats['stats'].items() if v is not None}

        use_groups = sorted(stats_map) if groups is None else [g for g in groups if g in stats_map]

        for group in use_groups:
            s = stats_map[group]
            rows.append(
                {
                    'split': split_name,
                    'group': group,
                    'group_label': label_for(level, group),
                    'mean_l0': s['mean_l0'],
                    'std_l0': s['std_l0'],
                }
            )

    frame = pd.DataFrame(rows)

    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X('group_label:N', title=None),
            y=alt.Y('mean_l0:Q', title='mean turn-level L0'),
            color=alt.Color('split:N', title='split'),
            xOffset=alt.XOffset('split:N'),
            tooltip=[
                alt.Tooltip('split:N', title='split'),
                alt.Tooltip('group_label:N', title='group'),
                alt.Tooltip('mean_l0:Q', title='mean L0'),
                alt.Tooltip('std_l0:Q', title='std L0'),
            ],
        )
        .properties(width=width, height=height)
    )


def plot_top_feature_trajectories(
    stats,
    feature_table,
    stat='mean',
    top_n=10,
    normalize_turn=None,
    width=400,
    height=250,
):
    if not stats:
        raise ValueError("stats_by_split is empty")

    split_names = list(stats.keys())

    stats_maps = {}
    turn_lists = []

    for split_name, stats in stats.items():
        if stats['level'] != 'turn':
            raise ValueError(f"expected turn-level stats for split '{split_name}'")

        stats_map = {k: v for k, v in stats['stats'].items() if v is not None}
        if not stats_map:
            raise ValueError(f"no stats found for split '{split_name}'")

        stats_maps[split_name] = stats_map
        turn_lists.append(sorted(stats_map))

    turns = turn_lists[0]
    for split_name, stats_map in stats_maps.items():
        if sorted(stats_map) != turns:
            raise ValueError(f"all splits must have the same turns; mismatch for '{split_name}'")

    if not turns:
        raise ValueError("no turns found")

    for split_name, stats_map in stats_maps.items():
        if stat not in stats_map[turns[0]]:
            raise ValueError(f"stat '{stat}' not found for split '{split_name}'")

    if normalize_turn is not None:
        for split_name, stats_map in stats_maps.items():
            if normalize_turn not in stats_map:
                raise ValueError(f"normalize_turn {normalize_turn} not found for split '{split_name}'")

    top_feature_table = feature_table.head(top_n).copy()
    top_features = top_feature_table['feature'].tolist()
    feature_rank_map = {feat: i + 1 for i, feat in enumerate(top_features)}

    ref_vecs = {}
    if normalize_turn is not None:
        for split_name, stats_map in stats_maps.items():
            ref_vecs[split_name] = stats_map[normalize_turn][stat].float().cpu().numpy()

    rows = []

    for feat in top_features:
        feat_rank = feature_rank_map[feat]

        for split_name, stats_map in stats_maps.items():
            base_val = None
            if normalize_turn is not None:
                base_val = float(ref_vecs[split_name][feat])

            for turn in turns:
                vec = stats_map[turn][stat].float().cpu().numpy()
                value = float(vec[feat])

                if normalize_turn is not None:
                    value = np.nan if abs(base_val) < 1e-12 else value / base_val

                rows.append({
                    'turn': turn,
                    'split': split_name,
                    'feature': int(feat),
                    'feature_label': f'feature {feat}',
                    'feature_rank': feat_rank,
                    'feature_split': f'feature {feat} | {split_name}',
                    'value': value,
                })

    frame = pd.DataFrame(rows).dropna()

    y_title = f'{stat} activation'
    if normalize_turn is not None:
        y_title = f'{stat} activation relative to turn {normalize_turn}'

    return (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X('turn:O', title='turn'),
            y=alt.Y('value:Q', title=y_title),

            # important: unique line per (feature, split)
            detail='feature_split:N',
            # color by rank, not by label
            color=alt.Color(
                'feature_rank:Q',
                title='feature rank',
                scale=alt.Scale(scheme='viridis', reverse=True),
                legend=alt.Legend(values=list(range(1, top_n + 1)))
            ),
            # explicit dash patterns
            strokeDash=alt.StrokeDash(
                'split:N',
                # title='split',
                legend=None,
                scale=alt.Scale(
                    domain=split_names,
                    range=[
                        [1, 0],   # solid
                        [6, 3],   # dashed
                        [2, 2],   # dotted
                        [8, 2, 2, 2],  # dash-dot
                    ][:len(split_names)]
                )
            ),
            tooltip=[
                alt.Tooltip('feature_label:N', title='feature'),
                alt.Tooltip('feature_rank:Q', title='rank'),
                alt.Tooltip('split:N', title='split'),
                alt.Tooltip('turn:O', title='turn'),
                alt.Tooltip('value:Q', title='value'),
            ],
        )
        .properties(
            width=width,
            height=height,
            title=(
                f'top {top_n} feature trajectories across splits'
                + (
                    ''
                    if normalize_turn is None
                    else f' (normalized to turn {normalize_turn})'
                )
            ),
        )
    )
