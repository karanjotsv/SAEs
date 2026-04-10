"""
Plots for instruction degradation analysis.

Starting point: feature_bar_chart
  — shows mean activation of top-ranked features per turn,
    fail vs ctrl side by side, faceted by turn.

Usage:
    from multif.plot import feature_bar_chart
    feature_bar_chart(da, pattern="forgetting", vec_key="user_vecs",
                      anchor_turn=0, start=0, end=20)
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Optional

import torch
import numpy as np
import pandas as pd
import altair as alt

if TYPE_CHECKING:
    from multif.analyze import DegradationAnalysis

from multif.analyze import FORGETTING, INTERFERENCE, PASS_ALL

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

FAIL_COLOR = "#e15759"
CTRL_COLOR = "#4e79a7"

PLOTS_DIR = "multif/plots"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

# valid vec_key values
VEC_KEYS = [
    "user_mean", "user_median", "user_max",
    "full_mean", "full_median", "full_max",
]

def _collect_vecs(records: list, vec_key: str, turn: int) -> Optional[torch.Tensor]:
    """
    Stack activation vectors for all records at a given turn.

    vec_key must be one of:
      user_mean, user_median, user_max,
      full_mean, full_median, full_max
    """
    if vec_key not in VEC_KEYS:
        raise ValueError(f"vec_key must be one of {VEC_KEYS} — got {vec_key!r}")
    vecs = [
        r["vecs"][turn][vec_key]
        for r in records
        if turn in r["vecs"] and vec_key in r["vecs"][turn]
    ]
    return torch.stack(vecs) if vecs else None


def _save(chart: alt.Chart, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    chart.save(path)
    print(f"  saved -> {path}")



# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(
    da: "DegradationAnalysis",
    vec_source: str = "user",
    out_dir: str = PLOTS_DIR,
) -> dict:
    """
    Compute per-turn summary stats for all three groups:
      fail_forgetting, fail_interference, pass

    Stats are computed using all three pooling modes of vec_source
    (mean, median, max) across all examples in each group.

    Scalar summaries are saved to txt files in out_dir.
    Raw tensors (mean/median/max/std/var vectors) are NOT written
    to txt — only scalar stats are saved.

    Parameters
    ----------
    vec_source : str
        "user" or "full"
    out_dir : str
        Directory to save txt files.

    Returns
    -------
    dict with keys: fail_forgetting, fail_interference, pass
    Each value is a dict keyed by turn index with per-turn stats.
    """
    if vec_source not in ("user", "full"):
        raise ValueError(f"vec_source must be 'user' or 'full' — got {vec_source!r}")

    mean_key   = f"{vec_source}_mean"
    median_key = f"{vec_source}_median"
    max_key    = f"{vec_source}_max"

    groups = {
        "fail_forgetting":  da.get_records(pattern=FORGETTING),
        "fail_interference": da.get_records(pattern=INTERFERENCE),
        "pass":             da.get_records(pattern=PASS_ALL),
    }

    all_stats = {}
    os.makedirs(out_dir, exist_ok=True)

    for group_name, records in groups.items():
        if not records:
            print(f"  [{group_name}] no records found, skipping")
            continue

        n_turns    = max(r["n_turns"] for r in records)
        turn_stats = {}

        for t in range(n_turns):
            # collect pooled vectors across examples at turn t
            mean_vecs   = _collect_vecs(records, mean_key,   t)
            median_vecs = _collect_vecs(records, median_key, t)
            max_vecs    = _collect_vecs(records, max_key,    t)

            if mean_vecs is None:
                continue

            count = mean_vecs.shape[0]
            mv    = mean_vecs.float()    # [n, 32768]

            # aggregate vectors
            agg_mean   = mv.mean(dim=0)
            agg_median = median_vecs.float().mean(dim=0) if median_vecs is not None else None
            agg_max    = max_vecs.float().mean(dim=0)    if max_vecs    is not None else None
            agg_std    = mv.std(dim=0, unbiased=False)   if count > 1   else torch.zeros_like(agg_mean)
            agg_var    = mv.var(dim=0, unbiased=False)   if count > 1   else torch.zeros_like(agg_mean)

            # threshold to avoid floating point near-zeros being counted as active
            ACTIVE_THRESHOLD = 1e-6

            # l0 per example (number of active features above threshold)
            l0_per_example = (mv > ACTIVE_THRESHOLD).sum(dim=1).float().tolist()
            mean_l0  = float(np.mean(l0_per_example))
            std_l0   = float(np.std(l0_per_example, ddof=0)) if count > 1 else 0.0

            # active feature counts from aggregated vectors (same threshold)
            active_mean   = int((agg_mean   > ACTIVE_THRESHOLD).sum().item())
            active_median = int((agg_median > ACTIVE_THRESHOLD).sum().item()) if agg_median is not None else 0
            active_max    = int((agg_max    > ACTIVE_THRESHOLD).sum().item()) if agg_max    is not None else 0
            num_concepts  = agg_mean.numel()

            turn_stats[t] = {
                # raw tensors stored but not written to txt
                "mean":   agg_mean,
                "median": agg_median,
                "max":    agg_max,
                "std":    agg_std,
                "var":    agg_var,
                # scalars
                "count":            count,
                "mean_l0":          mean_l0,
                "std_l0":           std_l0,
                "l0_distribution":  l0_per_example,
                "active_mean":      active_mean,
                "active_frac_mean": active_mean / num_concepts,
                "active_median":    active_median,
                "active_frac_median": active_median / num_concepts if agg_median is not None else 0.0,
                "active_max":       active_max,
                "active_frac_max":  active_max / num_concepts if agg_max is not None else 0.0,
                "num_concepts":     num_concepts,
            }

        all_stats[group_name] = turn_stats

        # save scalar summary to txt
        fname = f"stats_{group_name}_{vec_source}.txt"
        fpath = os.path.join(out_dir, fname)
        _write_stats_txt(fpath, group_name, vec_source, turn_stats)
        print(f"  saved -> {fpath}")

    return all_stats


def _write_stats_txt(
    path: str,
    group_name: str,
    vec_source: str,
    turn_stats: dict,
) -> None:
    """Write scalar stats summary to a txt file."""
    lines = []
    lines.append(f"group:      {group_name}")
    lines.append(f"vec_source: {vec_source}")
    lines.append(f"turns:      {sorted(turn_stats.keys())}")
    lines.append("")

    for t in sorted(turn_stats.keys()):
        s = turn_stats[t]
        lines.append(f"{'='*40}")
        lines.append(f"turn {t}")
        lines.append(f"{'='*40}")
        lines.append(f"  count:              {s['count']}")
        lines.append(f"  num_concepts:       {s['num_concepts']}")
        lines.append("")
        lines.append(f"  mean_l0:            {s['mean_l0']:.4f}")
        lines.append(f"  std_l0:             {s['std_l0']:.4f}")
        lines.append(f"  l0_distribution:    {[round(x, 1) for x in s['l0_distribution']]}")
        lines.append("")
        lines.append(f"  active_mean:        {s['active_mean']}")
        lines.append(f"  active_frac_mean:   {s['active_frac_mean']:.6f}")
        lines.append(f"  active_median:      {s['active_median']}")
        lines.append(f"  active_frac_median: {s['active_frac_median']:.6f}")
        lines.append(f"  active_max:         {s['active_max']}")
        lines.append(f"  active_frac_max:    {s['active_frac_max']:.6f}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ---------------------------------------------------------------------------
# feature_bar_chart
# ---------------------------------------------------------------------------



def _auto_save_path(
    plot_name: str,
    pattern: Optional[str],
    vec_key: str,
    anchor_turn: int,
    start: int,
    end: int,
) -> str:
    """
    Build a descriptive filename and ensure the plots directory exists.

    Format: {plot_name}_{pattern}_{vec_key}_anchor{t}_r{start}-{end}.html
    Example: feature_bar_forgetting_user_mean_anchor0_r0-20.html
    """
    pat   = pattern if pattern else "all"
    vk    = vec_key.replace("_", "")   # usermean, fullmedian etc — compact
    fname = f"{plot_name}_{pat}_{vk}_anchor{anchor_turn}_r{start}-{end}.html"
    os.makedirs(PLOTS_DIR, exist_ok=True)
    return os.path.join(PLOTS_DIR, fname)


def feature_bar_chart(
    da: "DegradationAnalysis",
    pattern: Optional[str] = None,
    vec_key: str = "user_mean",
    anchor_turn: int = 0,
    start: int = 0,
    end: int = 20,
    width: int = 500,
    height: int = 220,
    save_path: Optional[str] = None,
) -> alt.Chart:
    """
    Grouped bar chart: mean activation of top-ranked features per turn.

    Features are ranked by raw mean activation in failing cases at
    anchor_turn, then sliced [start:end] from that ranking.

    The chart is automatically saved to multif/plots/ with a descriptive
    filename. Pass save_path to override the default path.

    Parameters
    ----------
    da : DegradationAnalysis
        Loaded analysis object.
    pattern : str or None
        "forgetting", "interference", or None (merge both).
    vec_key : str
        One of: user_mean, user_median, user_max, full_mean, full_median, full_max
    anchor_turn : int
        Turn that determines feature ranking order.
    start : int
        Rank slice start (inclusive, 0-indexed).
    end : int
        Rank slice end (exclusive).

    Returns
    -------
    alt.Chart
    """

    # --- get failing and control records ---
    if pattern is None:
        failing = (
            da.get_records(pattern=FORGETTING) +
            da.get_records(pattern=INTERFERENCE)
        )
    else:
        failing = da.get_records(pattern=pattern)

    control = da.get_records(pattern=PASS_ALL)

    if not failing:
        print(f"No failing records found for pattern={pattern!r}")
        return alt.Chart(pd.DataFrame()).mark_point()

    # --- rank features by raw fail activation at anchor_turn ---
    fail_anchor = _collect_vecs(failing, vec_key, anchor_turn)
    if fail_anchor is None:
        print(f"No activations found at anchor_turn={anchor_turn}")
        return alt.Chart(pd.DataFrame()).mark_point()

    fail_mean_anchor = fail_anchor.mean(dim=0)                     # [32768]
    ranked_idx       = fail_mean_anchor.argsort(descending=True)   # all features sorted
    selected_idx     = ranked_idx[start:end].tolist()              # slice [start:end]
    n_selected       = len(selected_idx)

    # --- collect per-turn mean activations for selected features ---
    n_turns = max(r["n_turns"] for r in failing)
    rows    = []

    for t in range(n_turns):
        fail_vecs = _collect_vecs(failing, vec_key, t)
        ctrl_vecs = _collect_vecs(control, vec_key, t)

        fail_means = (
            fail_vecs.mean(dim=0)[selected_idx].tolist()
            if fail_vecs is not None
            else [float("nan")] * n_selected
        )
        ctrl_means = (
            ctrl_vecs.mean(dim=0)[selected_idx].tolist()
            if ctrl_vecs is not None
            else [float("nan")] * n_selected
        )

        for rank_pos, (feat_id, fm, cm) in enumerate(
            zip(selected_idx, fail_means, ctrl_means)
        ):
            global_rank = start + rank_pos + 1   # 1-indexed
            for group, val in [("fail", fm), ("ctrl", cm)]:
                rows.append({
                    "turn":       f"turn {t}",
                    "turn_int":   t,
                    "feature_id": feat_id,
                    "rank":       global_rank,
                    "rank_label": f"#{global_rank}",
                    "group":      group,
                    "activation": val,
                })

    if not rows:
        return alt.Chart(pd.DataFrame()).mark_point()

    df = pd.DataFrame(rows)

    # --- build labels ---
    pattern_label = pattern if pattern else "forgetting+interference"
    title = (
        f"{pattern_label} — mean activation per feature  "
        f"(ranks {start+1}–{end}, anchor=turn {anchor_turn}, {vec_key})"
    )

    cscale = alt.Scale(
        domain=["fail", "ctrl"],
        range=[FAIL_COLOR, CTRL_COLOR],
    )

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                "rank_label:N",
                sort=alt.SortField("rank", order="ascending"),
                title="Feature rank",
                axis=alt.Axis(labelAngle=-60, labelFontSize=9),
            ),
            y=alt.Y(
                "activation:Q",
                title="Mean activation",
            ),
            color=alt.Color(
                "group:N",
                scale=cscale,
                title="Group",
            ),
            xOffset=alt.XOffset("group:N"),
            tooltip=[
                alt.Tooltip("rank:Q",       title="Rank"),
                alt.Tooltip("feature_id:Q", title="Feature ID"),
                alt.Tooltip("group:N",      title="Group"),
                alt.Tooltip("activation:Q", format=".4f", title="Mean activation"),
                alt.Tooltip("turn:N",       title="Turn"),
            ],
        )
        .properties(width=width, height=height)
        .facet(
            row=alt.Row(
                "turn:N",
                sort=alt.SortField("turn_int", order="ascending"),
                title=None,
                header=alt.Header(labelFontSize=12, labelAngle=0),
            ),
        )
        .properties(title=title)
        .resolve_scale(y="shared")
    )

    # auto-save with descriptive name; save_path overrides
    path = save_path or _auto_save_path(
        "feature_bar", pattern, vec_key, anchor_turn, start, end
    )
    _save(chart, path)

    return chart


# ---------------------------------------------------------------------------
# diff_bar_chart
# ---------------------------------------------------------------------------

def diff_bar_chart(
    da: "DegradationAnalysis",
    results: dict,
    pattern: str = "forgetting",
    source: Optional[str] = None,
    chart_type: str = "drop",
    start: int = 0,
    end: int = 20,
    width: int = 500,
    height: int = 220,
    save_path: Optional[str] = None,
):
    """
    Grouped bar chart using diff_drop or diff_gain features
    from analyze() results. Same visual style as feature_bar_chart
    but features are selected by differential signal not raw activation.

    vec_key is inferred automatically:
      forgetting  + drop -> user_mean
      interference + drop -> full_mean
      interference + gain -> user_mean

    source is auto-detected:
      "aggregate"          -> aggregate level
      name in GROUP_ORDER  -> group level
      name containing ":"  -> instruction level

    Parameters
    ----------
    results : dict
        Output of da.analyze().
    pattern : str
        "forgetting" or "interference".
    source : str or None
        None (default) runs for all available sources and returns
        a dict of {source_name: chart}.
        Otherwise: "aggregate", a group name like "combination",
        or an instruction like "startend:quotation".
    chart_type : str
        "drop"  — diff_drop features (both patterns)
        "gain"  — new_features_at_t1 (interference only)
    start : int
        Rank slice start (inclusive, 0-indexed).
    end : int
        Rank slice end (exclusive).
    """
    if chart_type not in ("drop", "gain"):
        raise ValueError(f"chart_type must be 'drop' or 'gain' — got {chart_type!r}")
    if chart_type == "gain" and pattern != "interference":
        raise ValueError("chart_type='gain' is only available for pattern='interference'")

    # --- if source=None, run for all available sources ---
    if source is None:
        from multif.analyze import GROUP_ORDER as _GO, instruction_to_group
        pat_data    = results.get(pattern, {})
        per_group   = pat_data.get("per_group", {})
        per_inst    = pat_data.get("per_instruction", {})

        # ordered: aggregate first, then groups in GROUP_ORDER,
        # then instructions sorted by their group order
        inst_sorted = sorted(
            per_inst.keys(),
            key=lambda x: (_GO.index(instruction_to_group(x))
                           if instruction_to_group(x) in _GO else len(_GO), x)
        )
        all_sources = (
            ["aggregate"]
            + [g for g in _GO if g in per_group]
            + inst_sorted
        )

        print(f"Running diff_bar_chart for {len(all_sources)} sources: {all_sources}")
        charts = {}
        for src in all_sources:
            charts[src] = diff_bar_chart(
                da, results, pattern=pattern, source=src,
                chart_type=chart_type, start=start, end=end,
                width=width, height=height,
            )
        return charts

    # --- auto vec_key ---
    if chart_type == "drop":
        vec_key = "user_mean" if pattern == "forgetting" else "full_mean"
    else:
        vec_key = "user_mean"

    # --- auto source detection ---
    from multif.analyze import GROUP_ORDER as _GROUP_ORDER
    pat_data = results.get(pattern, {})
    if source == "aggregate":
        data = pat_data.get("aggregate", {})
        source_type = "aggregate"
    elif source in _GROUP_ORDER:
        data = pat_data.get("per_group", {}).get(source, {})
        source_type = "group"
    elif ":" in source:
        data = pat_data.get("per_instruction", {}).get(source, {})
        source_type = "instruction"
    else:
        print(f"Could not identify source={source!r} — "
              f"expected 'aggregate', a group name, or an instruction (containing ':')")
        return alt.Chart(pd.DataFrame()).mark_point()

    if not data:
        print(f"No data found for pattern={pattern!r} source={source!r}")
        return alt.Chart(pd.DataFrame()).mark_point()

    # --- get feature ids ---
    if chart_type == "drop":
        feat_ids   = data.get("diff_drop_features", {}).get("feature_ids", [])
        feat_label = "diff_drop"
    else:
        feat_ids   = data.get("new_features_at_t1", {}).get("feature_ids", [])
        feat_label = "diff_gain"

    if not feat_ids:
        print(f"No features found for chart_type={chart_type!r}")
        return alt.Chart(pd.DataFrame()).mark_point()

    # apply slice
    selected_ids = feat_ids[start:end]
    n_selected   = len(selected_ids)

    # --- get records ---
    from multif.analyze import FORGETTING, INTERFERENCE, PASS_ALL
    failing = da.get_records(pattern=pattern)
    control = da.get_records(pattern=PASS_ALL)

    # --- get differential values for selected slice ---
    if chart_type == "drop":
        diff_vals  = data.get("diff_drop_features", {}).get("differential_drop", [])[start:end]
        diff_label = "diff_drop"
    else:
        diff_vals  = data.get("new_features_at_t1", {}).get("differential_gain", [])[start:end]
        diff_label = "diff_gain"

    # --- collect per-turn mean activations ---
    n_turns = max(r["n_turns"] for r in failing)
    rows    = []

    for t in range(n_turns):
        fail_vecs = _collect_vecs(failing, vec_key, t)
        ctrl_vecs = _collect_vecs(control, vec_key, t)

        fail_means = (
            fail_vecs.mean(dim=0)[selected_ids].tolist()
            if fail_vecs is not None
            else [float("nan")] * n_selected
        )
        ctrl_means = (
            ctrl_vecs.mean(dim=0)[selected_ids].tolist()
            if ctrl_vecs is not None
            else [float("nan")] * n_selected
        )

        for rank_pos, (feat_id, fm, cm) in enumerate(
            zip(selected_ids, fail_means, ctrl_means)
        ):
            global_rank = start + rank_pos + 1
            dv          = diff_vals[rank_pos] if rank_pos < len(diff_vals) else float("nan")

            for group, val in [("fail", fm), ("ctrl", cm), (diff_label, dv)]:
                rows.append({
                    "turn":       f"turn {t}",
                    "turn_int":   t,
                    "feature_id": feat_id,
                    "rank":       global_rank,
                    "rank_label": f"#{global_rank}",
                    "group":      group,
                    "activation": val,
                })

    if not rows:
        return alt.Chart(pd.DataFrame()).mark_point()

    df = pd.DataFrame(rows)

    # --- stats annotation ---
    stats_key = "stats_diff_drop" if chart_type == "drop" else "stats_gain"
    stats     = data.get(stats_key, {})
    stats_str = ""
    if stats:
        stats_str = (
            f"  |  perm_p={stats['perm_p']:.3f}  "
            f"cohen_d={stats['cohen_d']:+.2f}"
        )

    title = (
        f"{pattern} [{source_type}:{source}] — {feat_label} features "
        f"(ranks {start+1}–{end}, vec={vec_key}){stats_str}"
    )

    DIFF_COLOR = "#59a14f"
    diff_label = "diff_drop" if chart_type == "drop" else "diff_gain"
    cscale = alt.Scale(
        domain=["fail", "ctrl", diff_label],
        range=[FAIL_COLOR, CTRL_COLOR, DIFF_COLOR],
    )

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                "rank_label:N",
                sort=alt.SortField("rank", order="ascending"),
                title="Feature rank",
                axis=alt.Axis(labelAngle=-60, labelFontSize=9),
            ),
            y=alt.Y("activation:Q", title="Mean activation"),
            color=alt.Color("group:N", scale=cscale, title="Group"),
            xOffset=alt.XOffset("group:N"),
            tooltip=[
                alt.Tooltip("rank:Q",       title="Rank"),
                alt.Tooltip("feature_id:Q", title="Feature ID"),
                alt.Tooltip("group:N",      title="Group"),
                alt.Tooltip("activation:Q", format=".4f", title="Mean activation"),
                alt.Tooltip("turn:N",       title="Turn"),
            ],
        )
        .properties(width=width, height=height)
        .facet(
            row=alt.Row(
                "turn:N",
                sort=alt.SortField("turn_int", order="ascending"),
                title=None,
                header=alt.Header(labelFontSize=12, labelAngle=0),
            ),
        )
        .properties(title=title)
        .resolve_scale(y="shared")
    )

    # auto-save with descriptive name
    src_clean = source.replace(":", "-").replace(" ", "_")
    fname     = f"diff_bar_{pattern}_{source_type}_{src_clean}_{chart_type}_r{start}-{end}.html"
    path      = save_path or os.path.join(PLOTS_DIR, fname)
    _save(chart, path)

    return chart


# ---------------------------------------------------------------------------
# diff_line_grid
# ---------------------------------------------------------------------------

def diff_line_grid(
    da: "DegradationAnalysis",
    results: dict,
    pattern: str = "forgetting",
    chart_type: str = "drop",
    width: int = 400,
    height: int = 150,
    save_path: Optional[str] = None,
) -> alt.Chart:
    """
    Line chart grid showing fail vs ctrl activation trajectory
    across turns for every available source (aggregate, groups,
    instructions). All panels stacked vertically in one HTML.

    Each panel:
      x-axis : turn (0, 1, 2)
      y-axis : mean activation of that source's diff_drop/gain features
      lines  : fail (red) and ctrl (blue) with ±1 SE bands
      shared y-axis across all panels

    vec_key is inferred automatically:
      forgetting  + drop -> user_mean
      interference + drop -> full_mean
      interference + gain -> user_mean

    Parameters
    ----------
    pattern : str
        "forgetting" or "interference"
    chart_type : str
        "drop" or "gain"
    """
    if chart_type not in ("drop", "gain"):
        raise ValueError(f"chart_type must be 'drop' or 'gain' — got {chart_type!r}")
    if chart_type == "gain" and pattern != "interference":
        raise ValueError("chart_type='gain' only available for pattern='interference'")

    # auto vec_key
    if chart_type == "drop":
        vec_key = "user_mean" if pattern == "forgetting" else "full_mean"
    else:
        vec_key = "user_mean"

    # build ordered source list
    from multif.analyze import GROUP_ORDER as _GO, instruction_to_group
    pat_data  = results.get(pattern, {})
    per_group = pat_data.get("per_group", {})
    per_inst  = pat_data.get("per_instruction", {})

    inst_sorted = sorted(
        per_inst.keys(),
        key=lambda x: (
            _GO.index(instruction_to_group(x))
            if instruction_to_group(x) in _GO else len(_GO), x
        )
    )
    all_sources = (
        ["aggregate"]
        + [g for g in _GO if g in per_group]
        + inst_sorted
    )

    failing = da.get_records(pattern=pattern)
    control = da.get_records(pattern=PASS_ALL)

    # collect rows for all sources
    rows = []

    for source in all_sources:
        # get source data
        if source == "aggregate":
            data        = pat_data.get("aggregate", {})
            source_type = "aggregate"
        elif source in _GO:
            data        = per_group.get(source, {})
            source_type = "group"
        else:
            data        = per_inst.get(source, {})
            source_type = "instruction"

        if not data:
            continue

        # get feature ids for this source
        if chart_type == "drop":
            feat_ids = data.get("diff_drop_features", {}).get("feature_ids", [])
        else:
            feat_ids = data.get("new_features_at_t1", {}).get("feature_ids", [])

        if not feat_ids:
            continue

        # get failing/control records for this source
        if source == "aggregate":
            src_failing = failing
            src_control = control
        elif source_type == "group":
            src_failing = da.get_records(pattern=pattern, group=source)
            src_control = da.get_records(pattern=PASS_ALL, group=source) or control
        else:
            src_failing = da.get_records(pattern=pattern, instruction=source)
            src_control = da.get_records(pattern=PASS_ALL, instruction=source) or control

        n_turns     = max(r["n_turns"] for r in src_failing)
        source_label = source.split(":")[-1] if ":" in source else source

        # stats for annotation
        stats_key = "stats_diff_drop" if chart_type == "drop" else "stats_gain"
        stats     = data.get(stats_key, {})
        stats_str = ""
        if stats:
            stats_str = (
                f" | perm_p={stats['perm_p']:.3f} "
                f"d={stats['cohen_d']:+.2f}"
            )
        panel_label = f"[{source_type}] {source_label}{stats_str}"

        for t in range(n_turns):
            fail_vecs = _collect_vecs(src_failing, vec_key, t)
            ctrl_vecs = _collect_vecs(src_control, vec_key, t)

            for group_name, vecs in [("fail", fail_vecs), ("ctrl", ctrl_vecs)]:
                if vecs is None:
                    continue

                # per-example scores over this source's features
                scores = vecs.float().mean(dim=0)[feat_ids].mean()
                # per-example for SE
                per_ex = vecs.float()[:, feat_ids].mean(dim=1)
                n      = per_ex.shape[0]
                mean   = float(per_ex.mean())
                se     = float(per_ex.std(unbiased=True) / (n ** 0.5)) if n > 1 else 0.0

                rows.append({
                    "source":       source,
                    "source_label": panel_label,
                    "source_type":  source_type,
                    "turn":         t,
                    "group":        group_name,
                    "mean":         mean,
                    "lo":           mean - se,
                    "hi":           mean + se,
                    "n":            n,
                })

    if not rows:
        print("No data to plot")
        return alt.Chart(pd.DataFrame()).mark_point()

    df = pd.DataFrame(rows)

    # source order for facet
    source_order = [
        df[df["source"] == s]["source_label"].iloc[0]
        for s in all_sources
        if s in df["source"].values
    ]

    cscale = alt.Scale(domain=["fail", "ctrl"], range=[FAIL_COLOR, CTRL_COLOR])

    line = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("turn:O", title="Turn"),
            y=alt.Y("mean:Q", title="Mean activation"),
            color=alt.Color("group:N", scale=cscale, title="Group"),
            tooltip=[
                alt.Tooltip("source_label:N", title="Source"),
                alt.Tooltip("turn:O",         title="Turn"),
                alt.Tooltip("group:N",         title="Group"),
                alt.Tooltip("mean:Q",          format=".4f", title="Mean"),
                alt.Tooltip("n:Q",             title="n"),
            ],
        )
    )

    band = (
        alt.Chart(df)
        .mark_area(opacity=0.15)
        .encode(
            x=alt.X("turn:O"),
            y=alt.Y("lo:Q"),
            y2=alt.Y2("hi:Q"),
            color=alt.Color("group:N", scale=cscale, legend=None),
        )
    )

    chart_type_label = "diff_drop" if chart_type == "drop" else "diff_gain"
    title = f"{pattern} — {chart_type_label} feature activation trajectory ({vec_key})"

    chart = (
        (band + line)
        .properties(width=width, height=height)
        .facet(
            row=alt.Row(
                "source_label:N",
                sort=source_order,
                title=None,
                header=alt.Header(labelFontSize=11, labelAngle=0, labelAlign="left"),
            ),
        )
        .properties(title=title)
        .resolve_scale(y="shared")
    )

    # auto-save
    fname = f"diff_line_grid_{pattern}_{chart_type}.html"
    path  = save_path or os.path.join(PLOTS_DIR, fname)
    _save(chart, path)

    return chart



# ---------------------------------------------------------------------------
# Shared token extraction helper
# ---------------------------------------------------------------------------

STRIP_TOKENS = {
    "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>",
    "<|begin_of_text|>", "assistant", "user", "system",
    "ĊĊ", "Ċ",
}


def _load_turn_tokens(
    activations_dir: str,
    conv_id: str,
    last_n_tokens: int,
) -> dict:
    """
    Load and slice user activations for all turns of a conversation.
    Strips trailing special tokens then takes last_n_tokens.

    Returns dict keyed by turn_idx:
      {
        "act":    tensor [n_actual, 32768],
        "tokens": list of token strings,
        "cols":   list of negative int indices (-n_actual to -1),
      }

    Both token_activation_grid and concept_overlap_heatmap call this
    so they always show identical tokens.
    """
    import torch

    turn_files = {}
    for fname in os.listdir(activations_dir):
        if not fname.endswith(".pt"):
            continue
        stem = fname[:-3]
        m    = re.match(r"^(.*?)_(\d+)$", stem)
        if not m:
            continue
        if m.group(1) == conv_id:
            turn_files[int(m.group(2))] = os.path.join(activations_dir, fname)

    result = {}
    for turn_idx in sorted(turn_files.keys()):
        try:
            obj      = torch.load(turn_files[turn_idx], map_location="cpu", weights_only=False)
            rt       = obj["results"][0]
            act      = rt["user_feature_activations"].float()
            tok_list = list(rt["user_input_tokens"])
        except Exception as e:
            print(f"  Warning: could not load turn {turn_idx}: {e}")
            continue

        if act.dim() == 1:
            act = act.unsqueeze(0)

        # strip trailing special tokens
        while tok_list and tok_list[-1] in STRIP_TOKENS:
            tok_list.pop()
            act = act[:-1]

        if act.shape[0] == 0:
            continue

        # take last_n_tokens
        seq_len   = act.shape[0]
        start     = max(0, seq_len - last_n_tokens)
        act_slice = act[start:]
        tok_slice = tok_list[start:]
        n_actual  = act_slice.shape[0]

        # negative indices: -n_actual to -1
        cols = list(range(-n_actual, 0))

        result[turn_idx] = {
            "act":    act_slice,
            "tokens": tok_slice,
            "cols":   cols,
        }

    return result

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# token_activation_grid
# ---------------------------------------------------------------------------

def token_activation_grid(
    conv_id: str,
    activations_dir: str,
    last_n_tokens: int = 10,
    top_k_features: int = 100,
    width: int = 200,
    height: int = 150,
    save_path: Optional[str] = None,
) -> alt.Chart:
    """
    Grid of bar charts: rows=turns, columns=token positions.

    Each cell shows top_k active SAE features for that token as bars
    at their absolute feature ID position (x-axis 0-32767).

    Uses _load_turn_tokens() — same tokens as concept_overlap_heatmap.

    Parameters
    ----------
    last_n_tokens : int
        How many tokens from end of each turn's user utterance.
        Turns shorter than last_n_tokens will have empty cells on the left.
    top_k_features : int
        How many top active features to plot per token.
    """
    import torch

    turn_data = _load_turn_tokens(activations_dir, conv_id, last_n_tokens)

    if not turn_data:
        print(f"No activation files found for conv_id={conv_id!r}")
        return alt.Chart(pd.DataFrame()).mark_point()

    rows = []

    for turn_idx, td in turn_data.items():
        act_slice = td["act"]
        tok_slice = td["tokens"]
        cols      = td["cols"]   # list of negative ints e.g. [-10,-9,...,-1]

        for pos, (col_idx, tok_label) in enumerate(zip(cols, tok_slice)):
            act_vec = act_slice[pos]
            n_active = int((act_vec > 0).sum().item())
            if n_active == 0:
                continue
            topk_vals, topk_idx = torch.topk(
                act_vec,
                k=min(top_k_features, n_active),
            )
            for feat_id, val in zip(topk_idx.tolist(), topk_vals.tolist()):
                rows.append({
                    "turn":       f"turn {turn_idx}",
                    "turn_int":   turn_idx,
                    "col":        col_idx,          # e.g. -3
                    "token":      tok_label,
                    "feature_id": feat_id,
                    "activation": float(val),
                })

    if not rows:
        print("No activation data found")
        return alt.Chart(pd.DataFrame()).mark_point()

    df = pd.DataFrame(rows)

    # global y max for shared scale across all cells
    y_max = float(df["activation"].max()) * 1.05

    title = (
        f"Token activation grid — {conv_id}  "
        f"(last {last_n_tokens} tokens/turn, top {top_k_features} features/token)"
    )

    chart = (
        alt.Chart(df)
        .mark_bar(color="#4e79a7", width=2)
        .encode(
            x=alt.X(
                "feature_id:Q",
                title="Feature ID (0–32767)",
                scale=alt.Scale(domain=[0, 32767]),
                axis=alt.Axis(labelFontSize=7, tickCount=5),
            ),
            y=alt.Y(
                "activation:Q",
                title="Activation",
                scale=alt.Scale(domain=[0, y_max]),
            ),
            tooltip=[
                alt.Tooltip("token:N",      title="Token"),
                alt.Tooltip("feature_id:Q", title="Feature ID"),
                alt.Tooltip("activation:Q", format=".4f", title="Activation"),
                alt.Tooltip("col:O",        title="Position"),
                alt.Tooltip("turn:N",       title="Turn"),
            ],
        )
        .properties(width=width, height=height)
        .facet(
            row=alt.Row(
                "turn:N",
                sort=alt.SortField("turn_int", order="ascending"),
                title=None,
                header=alt.Header(labelFontSize=11, labelAngle=0),
            ),
            column=alt.Column(
                "col:O",
                sort="ascending",
                title="token position",
                header=alt.Header(labelFontSize=9, labelAngle=-45),
            ),
            data=df,
        )
        .properties(title=title)
        .resolve_scale(y="shared")
    )

    conv_clean = conv_id.replace(":", "-")
    fname      = f"token_grid_{conv_clean}_last{last_n_tokens}_top{top_k_features}.html"
    path       = save_path or os.path.join(PLOTS_DIR, fname)
    _save(chart, path)

    return chart

# ---------------------------------------------------------------------------
# concept_overlap_heatmap
# ---------------------------------------------------------------------------

def concept_overlap_heatmap(
    conv_id: str,
    activations_dir: str,
    last_n_tokens: int = 10,
    top_k_features: int = 100,
    threshold: float = 1e-6,
    width: int = 800,
    height: int = 800,
    save_path: Optional[str] = None,
) -> alt.Chart:
    """
    Heatmap showing concept overlap between every pair of
    (turn, token) positions for a single conversation.

    Each cell (i, j) = |active(i) ∩ active(j)| / |active(i)|
    i.e. what fraction of concepts active at position i
    are also active at position j.

    Positions are ordered: turn 0 tokens first, then turn 1, then turn 2.
    Within each turn, tokens go from -last_n_tokens to -1.

    Parameters
    ----------
    conv_id : str
        Conversation root ID e.g. "1000:1:en"
    activations_dir : str
        Path to directory containing .pt files
    last_n_tokens : int
        How many tokens from end of user turn per turn
    top_k_features : int
        Top-k active features to consider per position
    threshold : float
        Activation threshold for a concept to be "active"
    """
    import os
    import torch

    # use shared helper — guarantees same tokens as token_activation_grid
    import torch
    turn_data = _load_turn_tokens(activations_dir, conv_id, last_n_tokens)

    if not turn_data:
        print(f"No files found for conv_id={conv_id!r}")
        return alt.Chart(pd.DataFrame()).mark_point()

    # --- collect active feature sets per (turn, token_pos) ---
    positions = []

    for turn_idx, td in turn_data.items():
        act_slice = td["act"]
        tok_slice = td["tokens"]
        cols      = td["cols"]

        for pos, (col_idx, tok_label) in enumerate(zip(cols, tok_slice)):
            act_vec  = act_slice[pos]
            act_mask = act_vec > threshold
            n_active = int(act_mask.sum().item())

            if n_active == 0:
                active_set = set()
            else:
                k          = min(top_k_features, n_active)
                _, top_idx = torch.topk(act_vec, k=k)
                active_set = set(top_idx.tolist())

            pos_label = f"t{turn_idx} {col_idx}|{tok_label[:6]}"
            positions.append({
                "label":      pos_label,
                "turn":       turn_idx,
                "col":        col_idx,
                "token":      tok_label,
                "active_set": active_set,
                "n_active":   len(active_set),
            })


    if not positions:
        print("No positions found")
        return alt.Chart(pd.DataFrame()).mark_point()

    n_pos = len(positions)
    rows  = []

    for i, src in enumerate(positions):
        for j, tgt in enumerate(positions):
            if src["n_active"] == 0:
                overlap = 0.0
            else:
                overlap = len(src["active_set"] & tgt["active_set"]) / src["n_active"]

            rows.append({
                "src":         src["label"],
                "tgt":         tgt["label"],
                "src_idx":     i,
                "tgt_idx":     j,
                "overlap":     overlap,
                "src_turn":    src["turn"],
                "src_col":     src["col"],
                "src_token":   src["token"],
                "tgt_turn":    tgt["turn"],
                "tgt_col":     tgt["col"],
                "tgt_token":   tgt["token"],
            })

    df         = pd.DataFrame(rows)
    pos_order  = [p["label"] for p in positions]

    title = (
        f"Concept overlap — {conv_id}  "
        f"(last {last_n_tokens} tokens, top {top_k_features} features, "
        f"threshold={threshold})"
    )

    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X(
                "tgt:N",
                sort=pos_order,
                title="Target position (turn | token pos | token)",
                axis=alt.Axis(labelAngle=-60, labelFontSize=10),
            ),
            y=alt.Y(
                "src:N",
                sort=pos_order,
                title="Source position (turn | token pos | token)",
                axis=alt.Axis(labelFontSize=10),
            ),
            color=alt.Color(
                "overlap:Q",
                title="Overlap %",
                scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]),
            ),
            tooltip=[
                alt.Tooltip("src:N",      title="Source"),
                alt.Tooltip("tgt:N",      title="Target"),
                alt.Tooltip("overlap:Q",  format=".3f", title="Overlap"),
                alt.Tooltip("src_turn:Q", title="Src turn"),
                alt.Tooltip("src_col:Q",  title="Src token pos"),
                alt.Tooltip("tgt_turn:Q", title="Tgt turn"),
                alt.Tooltip("tgt_col:Q",  title="Tgt token pos"),
            ],
        )
        .properties(width=width, height=height, title=title)
    )

    # auto-save
    conv_clean = conv_id.replace(":", "-")
    fname      = f"concept_overlap_{conv_clean}_last{last_n_tokens}_top{top_k_features}.html"
    path       = save_path or os.path.join(PLOTS_DIR, fname)
    _save(chart, path)

    return chart


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# token_concept_grid
# ---------------------------------------------------------------------------

def token_concept_grid(
    conv_id: str,
    activations_dir: str,
    last_n_tokens: int = 10,
    top_k_features: int = 50,
    cell_width: int = 250,
    cell_height: int = 180,
    save_path: Optional[str] = None,
) -> alt.Chart:
    """
    Grid of independent bar charts.
    Rows = turns, Columns = last N token positions.
    Each cell = one token, showing top_k active SAE features
    as bars at their absolute feature ID (x: 0-32767).
    Y-axis shared across all cells.

    Uses _load_turn_tokens() — same tokens as concept_overlap_heatmap.
    """
    import torch

    turn_data = _load_turn_tokens(activations_dir, conv_id, last_n_tokens)
    if not turn_data:
        print(f"No files found for {conv_id!r}")
        return alt.Chart(pd.DataFrame()).mark_point()

    n_turns = len(turn_data)

    # --- first pass: compute global y_max for shared scale ---
    global_y_max = 0.0
    for td in turn_data.values():
        act_slice = td["act"]
        for pos in range(act_slice.shape[0]):
            act_vec  = act_slice[pos]
            n_active = int((act_vec > 0).sum().item())
            if n_active == 0:
                continue
            k    = min(top_k_features, n_active)
            vals = torch.topk(act_vec, k=k).values
            global_y_max = max(global_y_max, float(vals.max()))
    global_y_max *= 1.05

    # --- build grid: one chart per (turn, token position) ---
    # rows[turn_idx][col_idx] = alt.Chart or None
    turn_rows = []

    for turn_idx in sorted(turn_data.keys()):
        td        = turn_data[turn_idx]
        act_slice = td["act"]
        tok_slice = td["tokens"]
        cols      = td["cols"]   # e.g. [-10, -9, ..., -1]

        # build a lookup: col_idx -> (df for that cell)
        col_charts = {}

        for pos, (col_idx, tok_label) in enumerate(zip(cols, tok_slice)):
            act_vec  = act_slice[pos]
            n_active = int((act_vec > 0).sum().item())

            if n_active == 0:
                cell_df = pd.DataFrame(columns=["feature_id", "activation", "token"])
            else:
                k        = min(top_k_features, n_active)
                vals, idx = torch.topk(act_vec, k=k)
                cell_df  = pd.DataFrame({
                    "feature_id": idx.tolist(),
                    "activation": vals.tolist(),
                    "token":      tok_label,
                    "col":        col_idx,
                    "turn":       f"turn {turn_idx}",
                })

            col_charts[col_idx] = cell_df

        # build one chart per token position in this turn
        # including empty placeholder for missing positions
        all_cols = list(range(-last_n_tokens, 0))
        cell_list = []

        for col_idx in all_cols:
            if col_idx in col_charts and len(col_charts[col_idx]) > 0:
                df   = col_charts[col_idx]
                tok  = df["token"].iloc[0]
                cell = (
                    alt.Chart(df)
                    .mark_bar(color="#4e79a7", width=2)
                    .encode(
                        x=alt.X(
                            "feature_id:Q",
                            title="Feature ID",
                            scale=alt.Scale(domain=[0, 32767]),
                            axis=alt.Axis(
                                labelFontSize=7,
                                tickCount=4,
                                titleFontSize=8,
                            ),
                        ),
                        y=alt.Y(
                            "activation:Q",
                            title="Activation",
                            scale=alt.Scale(domain=[0, global_y_max]),
                            axis=alt.Axis(
                                labelFontSize=7,
                                tickCount=3,
                                titleFontSize=8,
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("token:N",      title="Token"),
                            alt.Tooltip("feature_id:Q", title="Feature ID"),
                            alt.Tooltip("activation:Q", format=".4f", title="Activation"),
                            alt.Tooltip("col:O",        title="Position"),
                            alt.Tooltip("turn:N",       title="Turn"),
                        ],
                    )
                    .properties(
                        width=cell_width,
                        height=cell_height,
                        title=alt.TitleParams(
                            text=f"{col_idx}|{tok[:8]}",
                            fontSize=8,
                            anchor="start",
                        ),
                    )
                )
            else:
                # empty cell — placeholder
                empty_df = pd.DataFrame({"feature_id": [0], "activation": [0.0]})
                cell = (
                    alt.Chart(empty_df)
                    .mark_bar(opacity=0)
                    .encode(
                        x=alt.X("feature_id:Q",
                                scale=alt.Scale(domain=[0, 32767]),
                                axis=None),
                        y=alt.Y("activation:Q",
                                scale=alt.Scale(domain=[0, global_y_max]),
                                axis=None),
                    )
                    .properties(
                        width=cell_width,
                        height=cell_height,
                        title=alt.TitleParams(
                            text=f"{col_idx}|—",
                            fontSize=8,
                            anchor="start",
                        ),
                    )
                )
            cell_list.append(cell)

        # concatenate all token cells horizontally for this turn
        turn_row = alt.hconcat(*cell_list).properties(
            title=alt.TitleParams(
                text=f"turn {turn_idx}",
                fontSize=11,
                anchor="start",
            )
        )
        turn_rows.append(turn_row)

    # concatenate all turn rows vertically
    chart = alt.vconcat(*turn_rows).properties(
        title=f"Token concept grid — {conv_id} "
              f"(last {last_n_tokens} tokens, top {top_k_features} features)"
    )

    conv_clean = conv_id.replace(":", "-")
    fname      = f"token_concept_grid_{conv_clean}_last{last_n_tokens}_top{top_k_features}.html"
    path       = save_path or os.path.join(PLOTS_DIR, fname)
    _save(chart, path)

    return chart


# ---------------------------------------------------------------------------
# Matplotlib overview plots
# ---------------------------------------------------------------------------

def token_concept_grid_mpl(
    conv_id: str,
    activations_dir: str,
    last_n_tokens: int = 10,
    top_k_features: int = 50,
    cell_width: float = 1.5,
    cell_height: float = 1.0,
    dpi: int = 150,
    save_path: Optional[str] = None,
) -> str:
    """
    Static PNG overview of the token concept grid.
    All cells visible in one frame using Matplotlib.

    Parameters
    ----------
    cell_width : float
        Width of each cell in inches.
    cell_height : float
        Height of each cell in inches.
    dpi : int
        Resolution of the output PNG.

    Returns
    -------
    str — path to saved PNG.
    """
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    turn_data = _load_turn_tokens(activations_dir, conv_id, last_n_tokens)
    if not turn_data:
        print(f"No files found for {conv_id!r}")
        return ""

    n_turns = len(turn_data)
    all_cols = list(range(-last_n_tokens, 0))
    n_cols   = len(all_cols)

    # --- compute global y_max ---
    global_y_max = 0.0
    # store cell data: {(turn_idx, col_idx): (feat_ids, activations, token)}
    cell_data = {}
    for turn_idx, td in turn_data.items():
        act_slice = td["act"]
        tok_slice = td["tokens"]
        cols      = td["cols"]
        for pos, (col_idx, tok_label) in enumerate(zip(cols, tok_slice)):
            act_vec  = act_slice[pos]
            n_active = int((act_vec > 0).sum().item())
            if n_active == 0:
                cell_data[(turn_idx, col_idx)] = ([], [], tok_label)
                continue
            k        = min(top_k_features, n_active)
            vals, idx = torch.topk(act_vec, k=k)
            feat_ids  = idx.tolist()
            act_vals  = vals.tolist()
            global_y_max = max(global_y_max, max(act_vals))
            cell_data[(turn_idx, col_idx)] = (feat_ids, act_vals, tok_label)

    global_y_max *= 1.05

    # --- create figure ---
    fig_w = cell_width  * n_cols
    fig_h = cell_height * n_turns
    fig, axes = plt.subplots(
        n_turns, n_cols,
        figsize=(fig_w, fig_h),
        dpi=dpi,
    )

    # ensure axes is always 2D
    if n_turns == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    turns_sorted = sorted(turn_data.keys())

    for row, turn_idx in enumerate(turns_sorted):
        for col_pos, col_idx in enumerate(all_cols):
            ax = axes[row][col_pos]

            key = (turn_idx, col_idx)
            if key in cell_data:
                feat_ids, act_vals, tok_label = cell_data[key]
            else:
                feat_ids, act_vals, tok_label = [], [], "—"

            if feat_ids:
                ax.bar(feat_ids, act_vals, width=80, color="#4e79a7")

            ax.set_xlim(0, 32767)
            ax.set_ylim(0, global_y_max)
            ax.set_title(
                f"{col_idx}|{tok_label[:8]}",
                fontsize=5, pad=2, loc="left",
            )

            # only show axis labels on edges
            if row == n_turns - 1:
                ax.set_xlabel("feat", fontsize=4)
                ax.tick_params(axis="x", labelsize=3)
            else:
                ax.set_xticks([])

            if col_pos == 0:
                ax.set_ylabel(f"t{turn_idx}", fontsize=5)
                ax.tick_params(axis="y", labelsize=3)
            else:
                ax.set_yticks([])

    fig.suptitle(
        f"Token concept grid — {conv_id} "
        f"(last {last_n_tokens} tokens, top {top_k_features} features)",
        fontsize=8, y=1.01,
    )
    plt.tight_layout()

    conv_clean = conv_id.replace(":", "-")
    fname      = f"token_concept_grid_overview_{conv_clean}_last{last_n_tokens}_top{top_k_features}.png"
    path       = save_path or os.path.join(PLOTS_DIR, fname)
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")
    return path


def concept_overlap_heatmap_mpl(
    conv_id: str,
    activations_dir: str,
    last_n_tokens: int = 10,
    top_k_features: int = 100,
    threshold: float = 1e-6,
    figsize: tuple = (14, 12),
    dpi: int = 150,
    save_path: Optional[str] = None,
) -> str:
    """
    Static PNG version of concept_overlap_heatmap.
    Same data, higher resolution, faster to render than HTML.

    Returns path to saved PNG.
    """
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    turn_data = _load_turn_tokens(activations_dir, conv_id, last_n_tokens)
    if not turn_data:
        print(f"No files found for {conv_id!r}")
        return ""

    # --- collect active feature sets per position ---
    positions = []
    for turn_idx in sorted(turn_data.keys()):
        td        = turn_data[turn_idx]
        act_slice = td["act"]
        tok_slice = td["tokens"]
        cols      = td["cols"]
        for pos, (col_idx, tok_label) in enumerate(zip(cols, tok_slice)):
            act_vec  = act_slice[pos]
            act_mask = act_vec > threshold
            n_active = int(act_mask.sum().item())
            if n_active == 0:
                active_set = set()
            else:
                k          = min(top_k_features, n_active)
                _, top_idx = torch.topk(act_vec, k=k)
                active_set = set(top_idx.tolist())
            positions.append({
                "label":      f"t{turn_idx}{col_idx}|{tok_label[:6]}",
                "active_set": active_set,
                "n_active":   len(active_set),
            })

    n = len(positions)
    if n == 0:
        print("No positions found")
        return ""

    # --- compute overlap matrix ---
    matrix = np.zeros((n, n))
    for i, src in enumerate(positions):
        for j, tgt in enumerate(positions):
            if src["n_active"] == 0:
                matrix[i, j] = 0.0
            else:
                matrix[i, j] = len(src["active_set"] & tgt["active_set"]) / src["n_active"]

    labels = [p["label"] for p in positions]

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    cmap = plt.cm.RdYlGn
    im   = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Overlap %", shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=5)
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel("Target position", fontsize=8)
    ax.set_ylabel("Source position", fontsize=8)
    ax.set_title(
        f"Concept overlap — {conv_id} "
        f"(last {last_n_tokens} tokens, top {top_k_features} features)",
        fontsize=10,
    )

    plt.tight_layout()

    conv_clean = conv_id.replace(":", "-")
    fname      = f"concept_overlap_{conv_clean}_last{last_n_tokens}_top{top_k_features}.png"
    path       = save_path or os.path.join(PLOTS_DIR, fname)
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")
    return path
