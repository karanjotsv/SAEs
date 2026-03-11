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
    order_by=None,
    top_k=None,
    normalize=False,
    stat='mean',
    width=400,
    height=250,
):
    if isinstance(splits, dict) and 'stats' in splits:
        frame = concept_frame(
            splits,
            groups=groups,
            order_by=order_by,
            top_k=top_k,
            normalize=normalize,
            split='all',
            stat=stat,
        )
    else:
        frame = pd.concat(
            [
                concept_frame(
                    stats,
                    groups=groups,
                    order_by=order_by,
                    top_k=top_k,
                    normalize=normalize,
                    split=split_name,
                    stat=stat,
                )
                for split_name, stats in splits.items()
            ],
            ignore_index=False,
        )

    y_title = (
        f'{stat} concept value standardized to {label_for(frame["level"].iloc[0], order_by)}'
        if normalize and len(frame)
        else f'{stat} concept amplitude'
    )

    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X('concept_id:O', title=None, axis=None),
            y=alt.Y('value:Q', title=y_title),
            color=alt.Color('split:N', title='split'),
            xOffset=alt.XOffset('split:N'),
            tooltip=[
                alt.Tooltip('split:N', title='split'),
                alt.Tooltip('group_label:N', title='group'),
                alt.Tooltip('concept_id:O', title='concept id'),
                alt.Tooltip('concept_rank:Q', title='rank'),
                alt.Tooltip('value:Q', title='value'),
                alt.Tooltip('stat:N', title='stat'),
            ],
        )
        .properties(width=width, height=height)
        .facet(column=alt.Column('group_label:N', title=None))
    )


def plot_transition_scalars(stats, width=320, height=260):
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
