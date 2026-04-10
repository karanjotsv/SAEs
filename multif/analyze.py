"""
Instruction degradation analysis — core loader.

Loads predictions JSON and SAE activation .pt files,
builds one record per (conversation x instruction) with
trajectory classification and activation vectors.

Usage:
    from multif.analyze import DegradationAnalysis
    da = DegradationAnalysis("multif/predictions.json", "multif/activations_topk")
    da.load()
    records = da.get_records(pattern="forgetting")
"""

import os
import json
import re
from collections import defaultdict
from typing import Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Trajectory patterns
# ---------------------------------------------------------------------------

PASS_ALL     = "pass_all"
FAIL_ALL     = "fail_all"
FORGETTING   = "forgetting"
INTERFERENCE = "interference"
MIXED        = "mixed"
SINGLE_TURN  = "single_turn"

# ---------------------------------------------------------------------------
# Instruction group mapping
# ---------------------------------------------------------------------------

INSTRUCTION_GROUPS: Dict[str, str] = {
    "detectable_format":  "format",
    "length_constraints": "length",
    "keywords":           "keywords",
    "startend":           "startend",
    "combination":        "combination",
    "detectable_content": "content",
    "punctuation":        "punctuation",
}

GROUP_ORDER = [
    "combination", "format", "length",
    "keywords", "startend", "content", "punctuation", "other",
]


def instruction_to_group(instruction: str) -> str:
    prefix = instruction.split(":")[0]
    return INSTRUCTION_GROUPS.get(prefix, "other")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conv_root(ex_id: str) -> str:
    """'1000:1:en_2' -> '1000:1:en'"""
    m = re.match(r"^(.*?)_(\d+)$", ex_id)
    return m.group(1) if m else ex_id


def _turn_index(ex_id: str) -> int:
    """'1000:1:en_2' -> 2"""
    m = re.match(r"^(.*?)_(\d+)$", ex_id)
    return int(m.group(2)) if m else 0


def _classify(scores: List[int]) -> str:
    """Classify a per-turn score trajectory."""
    if len(scores) == 1:
        return SINGLE_TURN
    if all(s == 1 for s in scores):
        return PASS_ALL
    if all(s == 0 for s in scores):
        return FAIL_ALL
    if scores[0] != 1:
        return MIXED
    if scores[-1] == 0 and scores[1] == 1:
        return FORGETTING
    if scores[1] == 0:
        return INTERFERENCE
    return MIXED


def _pool(t: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Pool [seq_len, dict_size] -> [dict_size] using mean, median, or max.
    If already 1D, returns as-is.
    """
    t = t.float()
    if t.dim() == 1:
        return t
    if mode == "mean":
        return t.mean(dim=0)
    if mode == "median":
        return t.median(dim=0).values
    if mode == "max":
        return t.max(dim=0).values
    raise ValueError(f"mode must be mean, median or max — got {mode!r}")


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class DegradationAnalysis:
    """
    Loads predictions + SAE activations and builds per-instruction
    trajectory records.

    Each record contains:
        conv_root   : str
        instruction : str
        group       : str
        n_turns     : int
        scores      : List[int]   per-turn pass/fail
        pattern     : str         trajectory classification
        vecs        : Dict[int, Dict]    all pooling modes per turn
                      keys: user_mean, user_median, user_max,
                            full_mean, full_median, full_max
        language    : str
    """

    def __init__(
        self,
        predictions_path: str,
        activations_dir: str,
        activation_key: str = "feature_activations",
        user_activation_key: str = "user_feature_activations",
    ):
        self.predictions_path    = predictions_path
        self.activations_dir     = activations_dir
        self.activation_key      = activation_key
        self.user_activation_key = user_activation_key
        self.records: List[dict] = []

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self):
        """Load predictions and activations, build records."""
        with open(self.predictions_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        by_conv = self._group_by_conversation(raw)
        self.records = self._build_records(by_conv)

        from collections import Counter
        counts = Counter(r["pattern"] for r in self.records)
        print(f"Loaded {len(by_conv)} conversations, "
              f"{len(self.records)} instruction trajectories.")
        for pat, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {pat:<20} {cnt}")

    def _group_by_conversation(self, raw: list) -> dict:
        by_conv = defaultdict(lambda: {"turns": {}})
        for item in raw:
            for ex_id, ex in item.items():
                root = _conv_root(ex_id)
                turn = _turn_index(ex_id)
                by_conv[root]["turns"][turn] = {
                    "ex_id":              ex_id,
                    "instruction_scores": ex.get("instruction_scores", {}),
                    "language":           ex.get("language", "English"),
                }
        return dict(by_conv)

    def _build_records(self, by_conv: dict) -> list:
        records = []

        for conv_root, conv in by_conv.items():
            turns_sorted = sorted(conv["turns"].items())

            # collect all instructions present at turn 0
            turn0_scores = turns_sorted[0][1]["instruction_scores"]

            # load activations for all turns
            turn_acts = self._load_activations(turns_sorted)

            for instruction in turn0_scores:
                # build score trajectory
                scores = []
                for _, t_data in turns_sorted:
                    s = t_data["instruction_scores"].get(instruction)
                    if s is None:
                        break
                    scores.append(int(s))

                if len(scores) < 2:
                    continue

                pattern = _classify(scores)
                group   = instruction_to_group(instruction)

                # collect all pooling modes per turn
                vecs = {}
                for turn_idx, _ in turns_sorted[:len(scores)]:
                    if turn_idx in turn_acts:
                        vecs[turn_idx] = turn_acts[turn_idx]

                records.append({
                    "conv_root":   conv_root,
                    "instruction": instruction,
                    "group":       group,
                    "n_turns":     len(scores),
                    "scores":      scores,
                    "pattern":     pattern,
                    "vecs":        vecs,   # {turn: {user_mean, user_median, ...}}
                    "language":    turns_sorted[0][1].get("language", "English"),
                })

        return records

    def _load_activations(self, turns_sorted: list) -> dict:
        """
        Load and pool SAE activations for each turn.
        Saves mean, median and max pooling for both user and full activations
        so vec_key can be any of:
          user_mean, user_median, user_max,
          full_mean, full_median, full_max
        """
        result = {}
        for turn_idx, t_data in turns_sorted:
            path = os.path.join(self.activations_dir, f"{t_data['ex_id']}.pt")
            if not os.path.exists(path):
                continue
            try:
                obj = torch.load(path, map_location="cpu", weights_only=False)
                rt  = obj["results"][0]

                user_raw = rt[self.user_activation_key]
                full_raw = rt[self.activation_key]

                result[turn_idx] = {
                    "user_mean":   _pool(user_raw, "mean"),
                    "user_median": _pool(user_raw, "median"),
                    "user_max":    _pool(user_raw, "max"),
                    "full_mean":   _pool(full_raw, "mean"),
                    "full_median": _pool(full_raw, "median"),
                    "full_max":    _pool(full_raw, "max"),
                }
            except Exception as e:
                print(f"  Warning: could not load {path}: {e}")
        return result


    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, out_dir: str = "multif/plots") -> dict:
        """
        Print and save a full summary of prediction data to txt.

        Covers:
          - overall counts (conversations, trajectories, patterns)
          - per group breakdown by pattern
          - per instruction breakdown by pattern
          - turn-level pass rates (what fraction pass at each turn)
          - conversation-level score distribution

        Saves to: {out_dir}/summary_predictions.txt
        """
        from collections import Counter, defaultdict
        import os

        os.makedirs(out_dir, exist_ok=True)
        lines = []

        def p(s=""):
            print(s)
            lines.append(s)

        # ── overall ──────────────────────────────────────────────────
        p("=" * 60)
        p("PREDICTION SUMMARY")
        p("=" * 60)

        n_convs   = len(set(r["conv_root"] for r in self.records))
        n_records = len(self.records)
        p(f"  conversations:          {n_convs}")
        p(f"  instruction trajectories: {n_records}")
        p()

        # ── pattern counts ───────────────────────────────────────────
        pattern_counts = Counter(r["pattern"] for r in self.records)
        p("── Pattern Counts ──")
        for pat, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            pct = 100 * cnt / n_records
            p(f"  {pat:<20} {cnt:>5}  ({pct:.1f}%)")
        p()

        # ── per group by pattern ─────────────────────────────────────
        p("── Per Group by Pattern ──")
        group_pat = defaultdict(Counter)
        for r in self.records:
            group_pat[r["group"]][r["pattern"]] += 1

        patterns = [FORGETTING, INTERFERENCE, PASS_ALL, FAIL_ALL, MIXED]
        header   = f"  {'group':<15}" + "".join(f"  {p[:12]:<12}" for p in patterns)
        p(header)
        for grp in GROUP_ORDER:
            if grp not in group_pat:
                continue
            row = f"  {grp:<15}"
            for pat in patterns:
                row += f"  {group_pat[grp].get(pat, 0):<12}"
            p(row)
        p()

        # ── per instruction by pattern ───────────────────────────────
        p("── Per Instruction by Pattern ──")
        inst_pat = defaultdict(Counter)
        for r in self.records:
            inst_pat[r["instruction"]][r["pattern"]] += 1

        # sort by total degrading (forgetting + interference)
        inst_sorted = sorted(
            inst_pat.items(),
            key=lambda x: -(x[1].get(FORGETTING, 0) + x[1].get(INTERFERENCE, 0))
        )
        p(f"  {'instruction':<50} {'forget':>7} {'interf':>7} {'pass':>7} {'fail_all':>8} {'mixed':>6} {'total':>6}")
        for inst, counts in inst_sorted:
            f  = counts.get(FORGETTING,   0)
            i  = counts.get(INTERFERENCE, 0)
            pa = counts.get(PASS_ALL,     0)
            fa = counts.get(FAIL_ALL,     0)
            m  = counts.get(MIXED,        0)
            t  = sum(counts.values())
            p(f"  {inst:<50} {f:>7} {i:>7} {pa:>7} {fa:>8} {m:>6} {t:>6}")
        p()

        # ── turn-level pass rates ────────────────────────────────────
        p("── Turn-Level Pass Rates ──")
        turn_pass   = defaultdict(list)
        turn_counts = defaultdict(int)
        for r in self.records:
            for t, score in enumerate(r["scores"]):
                turn_pass[t].append(score)
                turn_counts[t] += 1

        p(f"  {'turn':<8} {'n':>6} {'pass_rate':>10} {'mean_score':>11}")
        for t in sorted(turn_pass.keys()):
            scores    = turn_pass[t]
            n         = len(scores)
            pass_rate = sum(scores) / n
            p(f"  {t:<8} {n:>6} {pass_rate:>10.4f} {pass_rate:>11.4f}")
        p()

        # ── turn-level pass rates by pattern ─────────────────────────
        p("── Turn-Level Pass Rates by Pattern ──")
        for pat in [FORGETTING, INTERFERENCE, PASS_ALL]:
            recs = [r for r in self.records if r["pattern"] == pat]
            if not recs:
                continue
            p(f"  {pat}  (n={len(recs)})")
            turn_scores = defaultdict(list)
            for r in recs:
                for t, score in enumerate(r["scores"]):
                    turn_scores[t].append(score)
            for t in sorted(turn_scores.keys()):
                sc = turn_scores[t]
                p(f"    turn {t}: pass_rate={sum(sc)/len(sc):.4f}  n={len(sc)}")
            p()

        # ── conversation-level score distribution ────────────────────
        p("── Conversation-Level Score Distribution ──")
        conv_scores = defaultdict(list)
        for r in self.records:
            for t, score in enumerate(r["scores"]):
                conv_scores[r["conv_root"]].append(score)

        all_conv_means = [sum(v)/len(v) for v in conv_scores.values()]
        p(f"  conversations:  {len(all_conv_means)}")
        p(f"  mean score:     {sum(all_conv_means)/len(all_conv_means):.4f}")
        p(f"  min score:      {min(all_conv_means):.4f}")
        p(f"  max score:      {max(all_conv_means):.4f}")
        p()

        # ── n_turns distribution ─────────────────────────────────────
        p("── n_turns Distribution ──")
        turn_dist = Counter(r["n_turns"] for r in self.records)
        for n, cnt in sorted(turn_dist.items()):
            p(f"  {n} turns: {cnt} trajectories")
        p()

        # ── save ─────────────────────────────────────────────────────
        fpath = os.path.join(out_dir, "summary_predictions.txt")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  saved -> {fpath}")

        return {
            "pattern_counts": dict(pattern_counts),
            "group_pattern":  {g: dict(v) for g, v in group_pat.items()},
            "inst_pattern":   {i: dict(v) for i, v in inst_pat.items()},
            "turn_pass_rates":{t: sum(v)/len(v) for t, v in turn_pass.items()},
        }
    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def get_records(
        self,
        pattern:     Optional[str] = None,
        instruction: Optional[str] = None,
        group:       Optional[str] = None,
        language:    Optional[str] = "English",
    ) -> List[dict]:
        """
        Filter records. All parameters are optional.
        pattern=None returns all records regardless of pattern.
        """
        out = self.records
        if language:
            out = [r for r in out if r["language"] == language]
        if pattern:
            out = [r for r in out if r["pattern"] == pattern]
        if instruction:
            out = [r for r in out if r["instruction"] == instruction]
        if group:
            out = [r for r in out if r["group"] == group]
        return out

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

import numpy as np
from scipy import stats as scipy_stats


def _collect_vecs(records: list, vec_key: str, turn: int):
    """Stack activation vectors for all records at a given turn."""
    vecs = [r["vecs"][turn][vec_key] for r in records
            if turn in r["vecs"] and vec_key in r["vecs"][turn]]
    return torch.stack(vecs).float() if vecs else None


def _run_tests(
    fail_scores: np.ndarray,
    ctrl_scores: np.ndarray,
    n_permutations: int = 1000,
) -> dict:
    """
    Run MWU (one-sided, fail < ctrl) + permutation test + Cohen's d.
    Returns dict of scalar stats.
    """
    n_fail = len(fail_scores)
    n_ctrl = len(ctrl_scores)

    fail_mean = float(np.mean(fail_scores))
    fail_std  = float(np.std(fail_scores, ddof=1)) if n_fail > 1 else 0.0
    ctrl_mean = float(np.mean(ctrl_scores))
    ctrl_std  = float(np.std(ctrl_scores, ddof=1)) if n_ctrl > 1 else 0.0

    # Mann-Whitney U — one sided: is fail < ctrl?
    try:
        _, mwu_p = scipy_stats.mannwhitneyu(
            fail_scores, ctrl_scores, alternative="less"
        )
    except Exception:
        mwu_p = float("nan")

    # permutation test — one sided
    observed_diff = fail_mean - ctrl_mean
    all_scores    = np.concatenate([fail_scores, ctrl_scores])
    rng           = np.random.default_rng(42)
    perm_diffs    = np.array([
        np.mean(p[:n_fail]) - np.mean(p[n_fail:])
        for p in (rng.permutation(all_scores) for _ in range(n_permutations))
    ])
    perm_p = float(np.mean(perm_diffs <= observed_diff))

    # Cohen's d
    pooled_std = np.sqrt(
        ((n_fail - 1) * fail_std**2 + (n_ctrl - 1) * ctrl_std**2)
        / max(n_fail + n_ctrl - 2, 1)
    )
    cohen_d = (fail_mean - ctrl_mean) / pooled_std if pooled_std > 0 else float("nan")

    return {
        "n_fail":        n_fail,
        "n_ctrl":        n_ctrl,
        "fail_mean":     fail_mean,
        "fail_std":      fail_std,
        "ctrl_mean":     ctrl_mean,
        "ctrl_std":      ctrl_std,
        "mwu_p":         float(mwu_p),
        "perm_p":        perm_p,
        "cohen_d":       float(cohen_d),
        "observed_diff": float(observed_diff),
    }


def _compute(
    failing: list,
    control: list,
    drop_vec_key: str,
    gain_vec_key: str,
    top_k: int,
    pattern: str,
    label: str = "aggregate",
    n_permutations: int = 1000,
) -> dict:
    """
    Core computation for one (failing, control) pair.

    Returns dict with:
      diff_drop_features  — top_k feature ids by differential drop
      drift               — {turn: [top_k activations]} for fail
      ctrl_drift          — same for control
      stats_diff_drop     — statistical tests on diff_drop features at last turn
      new_features_at_t1  — (interference only) top_k gain features
      stats_gain          — (interference only) statistical tests on gain
    """
    # --- turn-0 vectors ---
    fail_t0 = _collect_vecs(failing, drop_vec_key, 0)
    ctrl_t0 = _collect_vecs(control, drop_vec_key, 0)

    if fail_t0 is None:
        return {}

    fail_mean_t0 = fail_t0.mean(dim=0)                                    # [32768]
    ctrl_mean_t0 = ctrl_t0.mean(dim=0) if ctrl_t0 is not None else torch.zeros_like(fail_mean_t0)

    # --- last turn ---
    n_turns   = max(r["n_turns"] for r in failing)
    last_turn = n_turns - 1

    fail_tlast = _collect_vecs(failing, drop_vec_key, last_turn)
    ctrl_tlast = _collect_vecs(control, drop_vec_key, last_turn)

    if fail_tlast is None:
        return {}

    # --- diff_drop feature selection ---
    fail_drop        = fail_mean_t0 - fail_tlast.mean(dim=0)
    ctrl_drop        = (ctrl_mean_t0 - ctrl_tlast.mean(dim=0)
                        if ctrl_tlast is not None
                        else torch.zeros_like(fail_drop))
    differential_drop = fail_drop - ctrl_drop
    top_drop_idx      = torch.argsort(differential_drop, descending=True)[:top_k]

    # --- drift across turns ---
    drift      = {}
    ctrl_drift = {}
    for t in range(n_turns):
        fv = _collect_vecs(failing, drop_vec_key, t)
        cv = _collect_vecs(control, drop_vec_key, t)
        if fv is not None:
            drift[t]      = fv.mean(dim=0)[top_drop_idx].tolist()
        if cv is not None:
            ctrl_drift[t] = cv.mean(dim=0)[top_drop_idx].tolist()

    # --- stats on diff_drop features at last turn ---
    stats_drop = {}
    fail_scores = np.array([
        float(r["vecs"][last_turn][drop_vec_key][top_drop_idx].mean())
        for r in failing if last_turn in r["vecs"]
    ])
    ctrl_scores = np.array([
        float(r["vecs"][last_turn][drop_vec_key][top_drop_idx].mean())
        for r in control if last_turn in r["vecs"]
    ])
    if len(fail_scores) >= 3 and len(ctrl_scores) >= 3:
        stats_drop = _run_tests(fail_scores, ctrl_scores, n_permutations)

    result = {
        "label":              label,
        "n_failing":          len(failing),
        "n_control":          len(control),
        "diff_drop_features": {
            "feature_ids":       top_drop_idx.tolist(),
            "fail_drop":         fail_drop[top_drop_idx].tolist(),
            "ctrl_drop":         ctrl_drop[top_drop_idx].tolist(),
            "differential_drop": differential_drop[top_drop_idx].tolist(),
        },
        "drift":              drift,
        "ctrl_drift":         ctrl_drift,
        "stats_diff_drop":    stats_drop,
    }

    # --- interference gain (interference pattern only) ---
    if pattern == INTERFERENCE:
        gain_fail_t0 = _collect_vecs(failing, gain_vec_key, 0)
        gain_ctrl_t0 = _collect_vecs(control, gain_vec_key, 0)
        fail_t1      = _collect_vecs(failing, gain_vec_key, 1)
        ctrl_t1      = _collect_vecs(control, gain_vec_key, 1)

        if gain_fail_t0 is not None and fail_t1 is not None:
            gain_fail_mean_t0 = gain_fail_t0.mean(dim=0)
            gain_ctrl_mean_t0 = (gain_ctrl_t0.mean(dim=0)
                                 if gain_ctrl_t0 is not None
                                 else torch.zeros_like(gain_fail_mean_t0))

            fail_gain_t1      = fail_t1.mean(dim=0) - gain_fail_mean_t0
            ctrl_gain_t1      = (ctrl_t1.mean(dim=0) - gain_ctrl_mean_t0
                                 if ctrl_t1 is not None
                                 else torch.zeros_like(fail_gain_t1))
            differential_gain = fail_gain_t1 - ctrl_gain_t1
            top_gain_idx      = torch.argsort(differential_gain, descending=True)[:top_k]

            result["new_features_at_t1"] = {
                "feature_ids":       top_gain_idx.tolist(),
                "differential_gain": differential_gain[top_gain_idx].tolist(),
                "fail_gain":         fail_gain_t1[top_gain_idx].tolist(),
                "ctrl_gain":         ctrl_gain_t1[top_gain_idx].tolist(),
                "fail_t0_activation":gain_fail_mean_t0[top_gain_idx].tolist(),
                "fail_t1_activation":fail_t1.mean(dim=0)[top_gain_idx].tolist(),
            }

            # stats on gain features
            gain_fail_scores = np.array([
                float((r["vecs"][1][gain_vec_key] - r["vecs"][0][gain_vec_key])[top_gain_idx].mean())
                for r in failing
                if 0 in r["vecs"] and 1 in r["vecs"]
            ])
            gain_ctrl_scores = np.array([
                float((r["vecs"][1][gain_vec_key] - r["vecs"][0][gain_vec_key])[top_gain_idx].mean())
                for r in control
                if 0 in r["vecs"] and 1 in r["vecs"]
            ])
            if len(gain_fail_scores) >= 3 and len(gain_ctrl_scores) >= 3:
                result["stats_gain"] = _run_tests(
                    gain_fail_scores, gain_ctrl_scores, n_permutations
                )

    # print summary
    print(f"\n  [{label}]  n_fail={len(failing)}  n_ctrl={len(control)}")
    for t in sorted(drift.keys()):
        fv = float(np.mean(drift[t]))
        cv = float(np.mean(ctrl_drift[t])) if t in ctrl_drift else float("nan")
        print(f"    turn {t}: fail={fv:.4f}  ctrl={cv:.4f}  diff={fv-cv:+.4f}")
    if stats_drop:
        s = stats_drop
        print(f"  stats: mwu_p={s['mwu_p']:.4f}  perm_p={s['perm_p']:.4f}  "
              f"cohen_d={s['cohen_d']:+.3f}  "
              f"fail={s['fail_mean']:.4f}±{s['fail_std']:.4f}  "
              f"ctrl={s['ctrl_mean']:.4f}±{s['ctrl_std']:.4f}")
    if "stats_gain" in result:
        sg = result["stats_gain"]
        print(f"  gain: mwu_p={sg['mwu_p']:.4f}  perm_p={sg['perm_p']:.4f}  "
              f"cohen_d={sg['cohen_d']:+.3f}")

    return result


# ---------------------------------------------------------------------------
# Main analyze method (added to DegradationAnalysis)
# ---------------------------------------------------------------------------

def _analyze_pattern(
    da: "DegradationAnalysis",
    pattern: str,
    drop_vec_key: str,
    gain_vec_key: str,
    top_k: int,
    min_examples: int,
) -> dict:
    """Run analysis at aggregate, group, and instruction levels."""

    failing = da.get_records(pattern=pattern)
    control = da.get_records(pattern=PASS_ALL)

    print(f"\n=== {pattern.upper()} ===")
    print(f"  drop_vec={drop_vec_key}  gain_vec={gain_vec_key}")
    print(f"  failing={len(failing)}  control={len(control)}")

    if len(failing) < min_examples:
        print(f"  skipping — too few examples (< {min_examples})")
        return {}

    # aggregate
    agg = _compute(failing, control, drop_vec_key, gain_vec_key,
                   top_k, pattern, label="aggregate")

    # per group
    per_group = {}
    for grp in GROUP_ORDER:
        grp_fail = da.get_records(pattern=pattern, group=grp)
        grp_ctrl = da.get_records(pattern=PASS_ALL, group=grp) or control
        if len(grp_fail) < min_examples:
            continue
        per_group[grp] = _compute(
            grp_fail, grp_ctrl, drop_vec_key, gain_vec_key,
            top_k, pattern, label=f"group:{grp}"
        )

    # per instruction
    per_inst = {}
    inst_map = {}
    for r in failing:
        inst_map.setdefault(r["instruction"], []).append(r)

    inst_ctrl_map = {}
    for r in control:
        inst_ctrl_map.setdefault(r["instruction"], []).append(r)

    for inst, inst_fail in inst_map.items():
        inst_ctrl = inst_ctrl_map.get(inst, control)
        if len(inst_fail) < min_examples:
            continue
        per_inst[inst] = _compute(
            inst_fail, inst_ctrl, drop_vec_key, gain_vec_key,
            top_k, pattern, label=inst
        )

    return {
        "aggregate":       agg,
        "per_group":       per_group,
        "per_instruction": per_inst,
    }


# patch onto class
def _analyze(
    self,
    top_k: int = 50,
    min_examples: int = 5,
    n_permutations: int = 1000,
) -> dict:
    """
    Compute diff_drop feature analysis at three levels:
    aggregate, per_group, per_instruction.

    Activation source split:
      forgetting  drop: user_mean  — instruction encoding given growing context
      interference drop: full_mean — full context representation shift
      interference gain: user_mean — new instruction enters in user turn

    Returns dict with keys: forgetting, interference
    Each contains: aggregate, per_group, per_instruction
    """
    return {
        "forgetting": _analyze_pattern(
            self,
            pattern=FORGETTING,
            drop_vec_key="user_mean",
            gain_vec_key="user_mean",
            top_k=top_k,
            min_examples=min_examples,
        ),
        "interference": _analyze_pattern(
            self,
            pattern=INTERFERENCE,
            drop_vec_key="full_mean",
            gain_vec_key="user_mean",
            top_k=top_k,
            min_examples=min_examples,
        ),
    }


DegradationAnalysis.analyze = _analyze


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _save_summary(self, out_dir: str = "multif/plots") -> str:
    """
    Save a detailed summary of prediction data to a txt file.

    Covers:
      - overall counts
      - per pattern counts
      - per group counts by pattern
      - per instruction counts by pattern
      - turn-level pass rates (turn 0, 1, 2)
      - conversation-level score distribution
      - per instruction pass rate change across turns

    Returns path to saved file.
    """
    import os
    from collections import Counter, defaultdict

    os.makedirs(out_dir, exist_ok=True)
    path  = os.path.join(out_dir, "summary.txt")
    lines = []

    def h(title):
        lines.append("")
        lines.append("=" * 60)
        lines.append(title)
        lines.append("=" * 60)

    def sh(title):
        lines.append("")
        lines.append(f"  --- {title} ---")

    # --------------------------------------------------------
    # 1. overall counts
    # --------------------------------------------------------
    h("OVERALL")
    total = len(self.records)
    lines.append(f"  total instruction trajectories : {total}")
    lines.append(f"  total conversations            : {len(set(r['conv_root'] for r in self.records))}")

    pattern_counts = Counter(r["pattern"] for r in self.records)
    sh("Pattern counts")
    for pat, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {pat:<20} {cnt:>5}  ({100*cnt/total:.1f}%)")

    # --------------------------------------------------------
    # 2. per group by pattern
    # --------------------------------------------------------
    h("PER GROUP BY PATTERN")
    group_pattern = defaultdict(Counter)
    for r in self.records:
        group_pattern[r["group"]][r["pattern"]] += 1

    patterns_order = [FORGETTING, INTERFERENCE, PASS_ALL, FAIL_ALL, MIXED, SINGLE_TURN]
    header = f"  {'group':<15}" + "".join(f"  {p[:12]:<12}" for p in patterns_order)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for grp in GROUP_ORDER:
        if grp not in group_pattern:
            continue
        row = f"  {grp:<15}"
        for pat in patterns_order:
            cnt = group_pattern[grp].get(pat, 0)
            row += f"  {cnt:<12}"
        lines.append(row)

    # --------------------------------------------------------
    # 3. per instruction by pattern
    # --------------------------------------------------------
    h("PER INSTRUCTION BY PATTERN")
    inst_pattern = defaultdict(Counter)
    for r in self.records:
        inst_pattern[r["instruction"]][r["pattern"]] += 1

    # sort by group order then alphabetically
    insts_sorted = sorted(
        inst_pattern.keys(),
        key=lambda x: (
            GROUP_ORDER.index(instruction_to_group(x))
            if instruction_to_group(x) in GROUP_ORDER else len(GROUP_ORDER), x
        )
    )

    header2 = f"  {'instruction':<50}" + "".join(f"  {p[:8]:<8}" for p in patterns_order)
    lines.append(header2)
    lines.append("  " + "-" * (len(header2) - 2))

    cur_grp = None
    for inst in insts_sorted:
        grp = instruction_to_group(inst)
        if grp != cur_grp:
            lines.append(f"\n  [{grp}]")
            cur_grp = grp
        row = f"  {inst:<50}"
        for pat in patterns_order:
            cnt = inst_pattern[inst].get(pat, 0)
            row += f"  {cnt:<8}"
        lines.append(row)

    # --------------------------------------------------------
    # 4. turn-level pass rates
    # --------------------------------------------------------
    h("TURN-LEVEL PASS RATES")
    sh("Fraction of instructions passing at each turn")

    # group all records by turn count
    turn_pass  = defaultdict(lambda: defaultdict(list))  # turn -> instruction -> [0/1]
    for r in self.records:
        for t, score in enumerate(r["scores"]):
            turn_pass[t][r["instruction"]].append(score)

    # overall pass rate per turn
    lines.append(f"\n  {'turn':<8} {'n_obs':>6}  {'pass_rate':>10}")
    lines.append("  " + "-" * 28)
    for t in sorted(turn_pass.keys()):
        all_scores = [s for scores in turn_pass[t].values() for s in scores]
        n          = len(all_scores)
        rate       = sum(all_scores) / n if n else 0.0
        lines.append(f"  {t:<8} {n:>6}  {rate:>10.4f}")

    # --------------------------------------------------------
    # 5. per instruction pass rate across turns
    # --------------------------------------------------------
    h("PER INSTRUCTION PASS RATE ACROSS TURNS")

    all_turns = sorted(turn_pass.keys())
    header3   = f"  {'instruction':<50}" + "".join(f"  {'t'+str(t):<8}" for t in all_turns)
    lines.append(header3)
    lines.append("  " + "-" * (len(header3) - 2))

    cur_grp = None
    for inst in insts_sorted:
        grp = instruction_to_group(inst)
        if grp != cur_grp:
            lines.append(f"\n  [{grp}]")
            cur_grp = grp
        row = f"  {inst:<50}"
        for t in all_turns:
            scores = turn_pass[t].get(inst, [])
            rate   = sum(scores) / len(scores) if scores else float("nan")
            row   += f"  {rate:<8.3f}"
        lines.append(row)

    # --------------------------------------------------------
    # 6. conversation-level score distribution
    # --------------------------------------------------------
    h("CONVERSATION-LEVEL SCORE DISTRIBUTION")
    sh("Number of instructions that degrade per conversation")

    conv_degrade = Counter()
    conv_records = defaultdict(list)
    for r in self.records:
        conv_records[r["conv_root"]].append(r)

    for conv_root, recs in conv_records.items():
        n_degrade = sum(
            1 for r in recs
            if r["pattern"] in (FORGETTING, INTERFERENCE)
        )
        conv_degrade[n_degrade] += 1

    lines.append(f"\n  {'n_degrading_instructions':<30} {'n_conversations':>15}")
    lines.append("  " + "-" * 47)
    for k in sorted(conv_degrade.keys()):
        lines.append(f"  {k:<30} {conv_degrade[k]:>15}")

    # --------------------------------------------------------
    # write
    # --------------------------------------------------------
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path


DegradationAnalysis.save_summary = _save_summary
