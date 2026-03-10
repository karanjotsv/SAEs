import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import altair as alt


alt.data_transformers.disable_max_rows()


def to_numpy(x):
    """convert torch tensor / numpy / list to numpy."""
    if hasattr(x, "detach"):
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def load_instance_from_pt(pt_path):
    """load one instance dict from a pt file."""
    obj = torch.load(pt_path, map_location="cpu")
    return obj


def aggregate_token_feature_activations(feature_activations, metric="amplitude"):
    """
    aggregate token-level feature activations into one vector.

    metric:
      - 'amplitude': mean activation over tokens
      - 'l0': fraction of tokens with activation > 0
    """
    a = to_numpy(feature_activations)

    if a.ndim != 2:
        raise ValueError(f"expected 2d [num_tokens, num_features], got {a.shape}")

    if metric == "amplitude":
        return a.mean(axis=0)
    elif metric == "l0":
        return (a > 0).mean(axis=0)
    else:
        raise ValueError("metric must be 'amplitude' or 'l0'")


def build_id_to_file_map(folder_path):
    """build a mapping from instance id to pt file path."""
    folder = Path(folder_path)
    pt_files = sorted(folder.glob("*.pt"))

    id_to_file = {}

    for pt_file in pt_files:
        stem = pt_file.stem
        if stem not in id_to_file:
            id_to_file[stem] = pt_file

    return id_to_file


def resolve_file_for_id(instance_id, id_to_file, folder_path):
    """
    resolve the pt file for an id.

    first tries direct filename match.
    if not found, scans the folder and checks the loaded object's 'id' field.
    """
    if instance_id in id_to_file:
        return id_to_file[instance_id]

    folder = Path(folder_path)
    for pt_file in sorted(folder.glob("*.pt")):
        try:
            obj = load_instance_from_pt(pt_file)
            if isinstance(obj, dict) and obj.get("id") == instance_id:
                id_to_file[instance_id] = pt_file
                return pt_file
        except Exception:
            continue

    raise FileNotFoundError(f"could not find pt file for id: {instance_id}")


def collect_turn_vectors_from_ids(
    folder_path,
    success_ids,
    fail_ids,
    turn,
    activation_key="user_feature_activations",
    metric="amplitude",
):
    """
    load pt files from a folder using success and fail id lists, then extract
    one aggregated feature vector per instance for the requested turn.
    """
    success_ids = list(success_ids)
    fail_ids = list(fail_ids)

    id_to_label = {instance_id: 1 for instance_id in success_ids}
    id_to_label.update({instance_id: 0 for instance_id in fail_ids})

    all_ids = success_ids + fail_ids
    id_to_file = build_id_to_file_map(folder_path)

    vectors = []
    labels = []
    kept_ids = []

    for instance_id in all_ids:
        pt_file = resolve_file_for_id(instance_id, id_to_file, folder_path)
        instance = load_instance_from_pt(pt_file)

        if not isinstance(instance, dict):
            continue

        results = instance.get("results", [])
        turn_result = None

        for r in results:
            if int(r.get("turn")) == int(turn):
                turn_result = r
                break

        if turn_result is None:
            continue

        if activation_key not in turn_result:
            continue

        vec = aggregate_token_feature_activations(
            turn_result[activation_key],
            metric=metric,
        )

        vectors.append(vec)
        labels.append(id_to_label[instance_id])
        kept_ids.append(instance_id)

    if len(vectors) == 0:
        raise ValueError(
            f"no usable instances found for turn={turn}, activation_key={activation_key}"
        )

    x_turn = np.stack(vectors, axis=0)
    y_turn = np.asarray(labels, dtype=int)

    return x_turn, y_turn, kept_ids


def prepare_common_turn_matrices(
    folder_path,
    success_ids,
    fail_ids,
    activation_key="user_feature_activations",
    metric="amplitude",
    turn1=1,
    turn2=2,
):
    """prepare aligned turn 1 and turn 2 matrices using the common ids only."""
    x1, y1, ids1 = collect_turn_vectors_from_ids(
        folder_path=folder_path,
        success_ids=success_ids,
        fail_ids=fail_ids,
        turn=turn1,
        activation_key=activation_key,
        metric=metric,
    )

    x2, y2, ids2 = collect_turn_vectors_from_ids(
        folder_path=folder_path,
        success_ids=success_ids,
        fail_ids=fail_ids,
        turn=turn2,
        activation_key=activation_key,
        metric=metric,
    )

    id_to_row1 = {instance_id: i for i, instance_id in enumerate(ids1)}
    id_to_row2 = {instance_id: i for i, instance_id in enumerate(ids2)}
    common_ids = [instance_id for instance_id in ids1 if instance_id in id_to_row2]

    if len(common_ids) == 0:
        raise ValueError("no instances contain both requested turns")

    x1 = np.stack([x1[id_to_row1[instance_id]] for instance_id in common_ids], axis=0)
    y1 = np.asarray([y1[id_to_row1[instance_id]] for instance_id in common_ids], dtype=int)

    x2 = np.stack([x2[id_to_row2[instance_id]] for instance_id in common_ids], axis=0)
    y2 = np.asarray([y2[id_to_row2[instance_id]] for instance_id in common_ids], dtype=int)

    if not np.array_equal(y1, y2):
        raise ValueError("labels for common instances differ across turns")

    return x1, x2, y1, common_ids


def compute_curves_given_order(x_turn, y_turn, order):
    """compute all/success/failure curves using a fixed feature order."""
    succ_mask = y_turn == 1
    fail_mask = y_turn == 0

    if succ_mask.sum() == 0 or fail_mask.sum() == 0:
        raise ValueError("need at least one success and one failure instance")

    all_curve = x_turn.mean(axis=0)[order]
    succ_curve = x_turn[succ_mask].mean(axis=0)[order]
    fail_curve = x_turn[fail_mask].mean(axis=0)[order]

    return all_curve, succ_curve, fail_curve


def build_barplot_dataframe(
    x1,
    x2,
    y,
    turn1=1,
    turn2=2,
    top_k=100,
):
    """
    build a long dataframe for altair bar plots.

    concepts are sorted by turn 1 overall importance.
    turn 2 uses the same ordering.
    """
    overall_turn1 = x1.mean(axis=0)
    sorted_feature_ids = np.argsort(-overall_turn1)

    if top_k is not None:
        sorted_feature_ids = sorted_feature_ids[:top_k]

    all_t1, succ_t1, fail_t1 = compute_curves_given_order(x1, y, sorted_feature_ids)
    all_t2, succ_t2, fail_t2 = compute_curves_given_order(x2, y, sorted_feature_ids)

    rows = []

    for rank_idx, feature_id in enumerate(sorted_feature_ids):
        rows.append(
            {
                "turn": f"turn {turn1}",
                "rank": rank_idx,
                "feature_id": int(feature_id),
                "split": "all",
                "value": float(all_t1[rank_idx]),
            }
        )
        rows.append(
            {
                "turn": f"turn {turn1}",
                "rank": rank_idx,
                "feature_id": int(feature_id),
                "split": "success",
                "value": float(succ_t1[rank_idx]),
            }
        )
        rows.append(
            {
                "turn": f"turn {turn1}",
                "rank": rank_idx,
                "feature_id": int(feature_id),
                "split": "failure",
                "value": float(fail_t1[rank_idx]),
            }
        )

        rows.append(
            {
                "turn": f"turn {turn2}",
                "rank": rank_idx,
                "feature_id": int(feature_id),
                "split": "all",
                "value": float(all_t2[rank_idx]),
            }
        )
        rows.append(
            {
                "turn": f"turn {turn2}",
                "rank": rank_idx,
                "feature_id": int(feature_id),
                "split": "success",
                "value": float(succ_t2[rank_idx]),
            }
        )
        rows.append(
            {
                "turn": f"turn {turn2}",
                "rank": rank_idx,
                "feature_id": int(feature_id),
                "split": "failure",
                "value": float(fail_t2[rank_idx]),
            }
        )

    df = pd.DataFrame(rows)
    return df, sorted_feature_ids


def make_altair_barplot(
    df,
    metric="amplitude",
    width=700,
    height=350,
):
    """make an altair faceted bar chart."""
    ylabel = "average code amplitude" if metric == "amplitude" else "code l0 (fraction active > 0)"

    chart = (
        alt.Chart(df)
        .mark_bar(opacity=0.8)
        .encode(
            x=alt.X(
                "rank:O",
                title="concept rank in turn 1 ordering",
                axis=alt.Axis(labels=False, ticks=False),
            ),
            y=alt.Y("value:Q", title=ylabel),
            color=alt.Color(
                "split:N",
                scale=alt.Scale(
                    domain=["all", "success", "failure"],
                    range=["#4c78a8", "#f58518", "#54a24b"],
                ),
            ),
            xOffset="split:N",
            tooltip=[
                alt.Tooltip("turn:N"),
                alt.Tooltip("split:N"),
                alt.Tooltip("rank:Q"),
                alt.Tooltip("feature_id:Q"),
                alt.Tooltip("value:Q", format=".6f"),
            ],
        )
        .properties(width=width, height=height)
        .facet(
            column=alt.Column("turn:N", title=None),
        )
        .resolve_scale(y="shared")
    )

    return chart


def plot_sorted_concepts_turn1_turn2_altair(
    folder_path,
    success_ids,
    fail_ids,
    activation_key="user_feature_activations",
    metric="amplitude",
    turn1=1,
    turn2=2,
    top_k=100,
    save_html_path=None,
):
    """
    sort concepts by turn-1 overall importance using both success and failure,
    then build altair bar plots for turn 1 and turn 2 using the same ordering.
    """
    x1, x2, y, common_ids = prepare_common_turn_matrices(
        folder_path=folder_path,
        success_ids=success_ids,
        fail_ids=fail_ids,
        activation_key=activation_key,
        metric=metric,
        turn1=turn1,
        turn2=turn2,
    )

    df, sorted_feature_ids = build_barplot_dataframe(
        x1=x1,
        x2=x2,
        y=y,
        turn1=turn1,
        turn2=turn2,
        top_k=top_k,
    )

    chart = make_altair_barplot(df, metric=metric)

    if save_html_path is not None:
        chart.save(save_html_path)

    return chart, df, sorted_feature_ids, common_ids
