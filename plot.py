import os
import json
import torch

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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


def plot_per_turn(data):
    # normalize each line by its first value
    normalized = {}
    for key, values in data.items():
        base = values[0]
        normalized[key] = [v / base for v in values]

    # compute mean line across turns (ignoring missing turns)
    max_len = max(len(v) for v in normalized.values())
    aligned = np.array([
        vals + [np.nan] * (max_len - len(vals))
        for vals in normalized.values()
    ], dtype=float)
    mean_line = np.nanmean(aligned, axis=0)

    # build interactive plot
    fig = go.Figure()

    # individual trajectories
    for key, values in normalized.items():
        x = list(range(1, len(values) + 1))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=values,
                mode="lines+markers",
                opacity=0.45,
                showlegend=False,
                hovertemplate=(
                    f"id={key}<br>"
                    "turn=%{x}<br>"
                    "relative activation=%{y:.4f}<extra></extra>"
                ),
            )
        )

    # mean trajectory
    fig.add_trace(
        go.Scatter(
            x=list(range(1, max_len + 1)),
            y=mean_line,
            mode="lines+markers",
            line=dict(width=4, color="black"),
            showlegend=False,
            hovertemplate=(
                "mean<br>"
                "turn=%{x}<br>"
                "relative activation=%{y:.4f}<extra></extra>"
            ),
        )
    )
    # reference line at 1.0
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="black",
        line_width=1
    )
    # layout with major + minor grids
    fig.update_layout(
        title="Normalized Feature Activation Across Turns",
        xaxis_title="Turn",
        yaxis_title="Relative Activation",
        template="plotly_white",
        hovermode="closest",
        width=950,
        height=600,
        showlegend=False,
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(0,0,0,0.15)",
        tickmode="linear",
        dtick=1,
        minor=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor="rgba(0,0,0,0.08)",
            dtick=0.5,
        ),
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(0,0,0,0.15)",
        minor=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor="rgba(0,0,0,0.08)",
        ),
    )

    fig.show()
