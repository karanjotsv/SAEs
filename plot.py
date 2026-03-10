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


def plot_per_turn(data, normalize=True, show_mean=False):
    # collect rows for dataframe
    rows = []
    # iterate through each sample
    for sample_id, feature_lists in data.items():
        # each inner list corresponds to a feature rank
        for feature_idx, values in enumerate(feature_lists):
            if len(values) == 0:
                continue
            values = list(values)
            # optionally normalize each feature trajectory by its first value
            if normalize:
                base = values[0]
                if base == 0:
                    norm_values = [np.nan if v == 0 else np.inf for v in values]
                else:
                    norm_values = [v / base for v in values]
            else:
                norm_values = values
            # add each turn as a row in the dataframe
            for turn_idx, v in enumerate(norm_values):
                rows.append({
                    "id": sample_id,
                    "feature_rank": feature_idx + 1,  # ranks start at 1
                    "turn": turn_idx + 1,
                    "activation": v
                })
    # create dataframe from collected rows
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no valid data to plot.")
    # plot individual trajectories colored by feature rank
    lines = (
        alt.Chart(df)
        .mark_line(opacity=0.4)
        .encode(
            x=alt.X("turn:Q", title="turn"),
            y=alt.Y(
                "activation:Q",
                title="relative activation" if normalize else "activation"
            ),
            color=alt.Color("feature_rank:N", title="feature rank"),
            detail=["id:N", "feature_rank:N"],
            tooltip=[
                alt.Tooltip("id:N"),
                alt.Tooltip("feature_rank:N", title="feature rank"),
                alt.Tooltip("turn:Q"),
                alt.Tooltip("activation:Q", format=".4f"),
            ]
        )
    )
    # add points to make individual observations visible
    points = (
        alt.Chart(df)
        .mark_point(opacity=0.4, size=40)
        .encode(
            x="turn:Q",
            y="activation:Q",
            color=alt.Color("feature_rank:N", title="feature rank"),
            detail=["id:N", "feature_rank:N"],
            tooltip=[
                alt.Tooltip("id:N"),
                alt.Tooltip("feature_rank:N", title="feature rank"),
                alt.Tooltip("turn:Q"),
                alt.Tooltip("activation:Q", format=".4f"),
            ]
        )
    )
    layers = [lines, points]
    # compute and plot mean trajectory per feature rank
    if show_mean:
        # group by feature rank and turn to compute mean activation
        mean_df = (
            df.groupby(["feature_rank", "turn"], as_index=False)["activation"]
            .mean()
        )
        # permanent label text: omit rank on the first turn
        mean_df["label"] = mean_df.apply(
            lambda r: (
                f"turn {int(r['turn'])}, mean {r['activation']:.4f}"
                if int(r["turn"]) == 1
                else f"rank {int(r['feature_rank'])}, turn {int(r['turn'])}, mean {r['activation']:.4f}"
            ),
            axis=1
        )
        # draw thinner black mean lines
        mean_lines = (
            alt.Chart(mean_df)
            .mark_line(color="black", strokeWidth=2, opacity=0.8)
            .encode(
                x="turn:Q",
                y="activation:Q",
                detail="feature_rank:N",
                tooltip=[
                    alt.Tooltip("feature_rank:N", title="feature rank"),
                    alt.Tooltip("turn:Q"),
                    alt.Tooltip("activation:Q", format=".4f", title="mean activation"),
                ]
            )
        )
        # add points on the mean lines
        mean_points = (
            alt.Chart(mean_df)
            .mark_point(color="black", size=80, filled=True)
            .encode(
                x="turn:Q",
                y="activation:Q",
                detail="feature_rank:N",
                tooltip=[
                    alt.Tooltip("feature_rank:N", title="feature rank"),
                    alt.Tooltip("turn:Q"),
                    alt.Tooltip("activation:Q", format=".4f", title="mean activation"),
                ]
            )
        )
        # always-visible labels for mean points
        mean_labels = (
            alt.Chart(mean_df)
            .mark_text(
                align="left",
                dx=8,
                dy=-8,
                fontSize=11,
                color="black"
            )
            .encode(
                x="turn:Q",
                y="activation:Q",
                detail="feature_rank:N",
                text="label:N"
            )
        )
        layers.extend([mean_lines, mean_points, mean_labels])
    # add horizontal reference line at 1.0 when normalized
    if normalize:
        rule = (
            alt.Chart(pd.DataFrame({"y": [1.0]}))
            .mark_rule(strokeDash=[5, 5], color="black")
            .encode(y="y:Q")
        )
        layers.append(rule)
    # combine all layers into final interactive chart
    chart = (
        alt.layer(*layers)
        .properties(
            width=950,
            height=600,
            title="feature activation across turns"
        )
        .interactive()
    )
    return chart
