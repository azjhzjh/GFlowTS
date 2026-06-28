# -*- coding: utf-8 -*-
"""Global feasibility-preserving point deletion repair for trajectory candidates."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .utils.query_sketch import QueryCoverageSketch, f1_lower_bound


DEFAULT_EXACT_SEGMENT_STEP_CAP = 200
DEFAULT_PROXY_SEGMENT_STEP_CAP = 64


def _bounds_from_traj(raw_trajectory: np.ndarray) -> Dict[str, float]:
    return {
        "x_min": float(raw_trajectory[:, 0].min()),
        "x_max": float(raw_trajectory[:, 0].max()),
        "y_min": float(raw_trajectory[:, 1].min()),
        "y_max": float(raw_trajectory[:, 1].max()),
        "t_min": float(raw_trajectory[:, 2].min()),
        "t_max": float(raw_trajectory[:, 2].max()),
    }


def _to_ranges(bounds: Dict[str, float]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    return (
        (bounds["x_min"], bounds["x_max"] + 1e-9),
        (bounds["y_min"], bounds["y_max"] + 1e-9),
        (bounds["t_min"], bounds["t_max"] + 1e-9),
    )


def _normalized_indices(raw_trajectory: np.ndarray, indices: Sequence[int]) -> List[int]:
    n = len(raw_trajectory)
    return sorted(set(int(i) for i in indices if 0 <= int(i) < n))


def _empty_stats() -> Dict[str, int]:
    return {
        "repair_skipped_small": 0,
        "repair_skipped_low_slack": 0,
        "repair_proxy_checked": 0,
        "repair_exact_checked": 0,
        "repair_early_stop_count": 0,
    }


def _compute_delete_budget(num_kept: int, delete_ratio: float, max_delete: Optional[int]) -> int:
    budget = int(np.floor(num_kept * float(np.clip(delete_ratio, 0.0, 1.0))))
    if max_delete is not None:
        budget = min(budget, max(0, int(max_delete)))
    return max(0, budget)


def build_qcs_for_indices(
    raw_trajectory: np.ndarray,
    indices: Sequence[int],
    bounds: Optional[Dict[str, float]] = None,
    grid_size: int = 64,
    segment_step_cap: int = DEFAULT_EXACT_SEGMENT_STEP_CAP,
) -> QueryCoverageSketch:
    if bounds is None:
        bounds = _bounds_from_traj(raw_trajectory)
    x_range, y_range, t_range = _to_ranges(bounds)
    qcs = QueryCoverageSketch(grid_size, x_range, y_range, t_range)
    idx = _normalized_indices(raw_trajectory, indices)
    if not idx:
        return qcs
    qcs.add_point(raw_trajectory[idx[0]])
    for j in range(1, len(idx)):
        p1 = raw_trajectory[idx[j - 1]]
        p2 = raw_trajectory[idx[j]]
        if int(segment_step_cap) != DEFAULT_EXACT_SEGMENT_STEP_CAP:
            qcs.add_segment_with_step_cap(p1, p2, max_steps=int(segment_step_cap))
        else:
            qcs.add_segment(p1, p2)
    return qcs


def evaluate_indices(
    raw_trajectory: np.ndarray,
    indices: Sequence[int],
    tau: float,
    bounds: Optional[Dict[str, float]] = None,
    qcs_original: Optional[QueryCoverageSketch] = None,
    grid_size: int = 64,
    segment_step_cap: int = DEFAULT_EXACT_SEGMENT_STEP_CAP,
) -> Dict[str, float]:
    idx = _normalized_indices(raw_trajectory, indices)
    if bounds is None:
        bounds = _bounds_from_traj(raw_trajectory)
    if qcs_original is None:
        qcs_original = build_qcs_for_indices(
            raw_trajectory,
            range(len(raw_trajectory)),
            bounds=bounds,
            grid_size=grid_size,
            segment_step_cap=segment_step_cap,
        )
    qcs_candidate = build_qcs_for_indices(
        raw_trajectory,
        idx,
        bounds=bounds,
        grid_size=grid_size,
        segment_step_cap=segment_step_cap,
    )
    f1_lb = float(f1_lower_bound(qcs_candidate, qcs_original))
    cr = len(idx) / max(1, len(raw_trajectory))
    feasible = f1_lb >= float(tau)
    return {
        "indices": idx,
        "f1_lb": f1_lb,
        "cr": float(cr),
        "feasible": bool(feasible),
    }


def _compute_turning_angles(raw_trajectory: np.ndarray) -> np.ndarray:
    n = len(raw_trajectory)
    angles = np.zeros(n, dtype=np.float64)
    if n < 3:
        return angles
    for i in range(1, n - 1):
        v1 = raw_trajectory[i, :2] - raw_trajectory[i - 1, :2]
        v2 = raw_trajectory[i + 1, :2] - raw_trajectory[i, :2]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            angles[i] = 0.0
            continue
        c = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angles[i] = float(np.arccos(c))
    return angles


def _segment_proxy(raw_trajectory: np.ndarray, kept: List[int], pos: int) -> float:
    idx = kept[pos]
    prev_idx = kept[pos - 1] if pos - 1 >= 0 else idx
    next_idx = kept[pos + 1] if pos + 1 < len(kept) else idx
    d1 = float(np.linalg.norm(raw_trajectory[idx, :2] - raw_trajectory[prev_idx, :2]))
    d2 = float(np.linalg.norm(raw_trajectory[next_idx, :2] - raw_trajectory[idx, :2]))
    return 0.5 * (d1 + d2)


def _normalize_map(values: Dict[int, float]) -> Dict[int, float]:
    if not values:
        return {}
    arr = np.array(list(values.values()), dtype=np.float64)
    lo = float(arr.min())
    hi = float(arr.max())
    if abs(hi - lo) < 1e-12:
        return {k: 0.0 for k in values}
    return {k: float((v - lo) / (hi - lo)) for k, v in values.items()}


def _spatial_scale(bounds: Dict[str, float]) -> float:
    dx = float(bounds["x_max"] - bounds["x_min"])
    dy = float(bounds["y_max"] - bounds["y_min"])
    return max(1e-6, float(np.hypot(dx, dy)))


@dataclass
class RepairResult:
    indices: List[int]
    f1_lb: float
    cr: float
    feasible: bool
    num_deleted: int
    attempted_deletions: int
    stats: Dict[str, int] = field(default_factory=_empty_stats)


def _build_locked_set(n: int, locked_endpoints: bool) -> Set[int]:
    locked: Set[int] = set()
    if locked_endpoints and n > 0:
        locked.add(0)
        locked.add(n - 1)
    return locked


def _final_result(
    raw_trajectory: np.ndarray,
    kept: List[int],
    tau: float,
    current_f1: float,
    initial_count: int,
    attempted: int,
    stats: Dict[str, int],
) -> RepairResult:
    return RepairResult(
        indices=list(kept),
        f1_lb=float(current_f1),
        cr=float(len(kept) / max(1, len(raw_trajectory))),
        feasible=bool(current_f1 >= float(tau)),
        num_deleted=int(max(0, initial_count - len(kept))),
        attempted_deletions=int(attempted),
        stats=dict(stats),
    )


def _repair_candidate_global_exact(
    raw_trajectory: np.ndarray,
    indices: Sequence[int],
    tau: float,
    delete_ratio: float,
    max_delete: Optional[int],
    locked_endpoints: bool,
    bounds: Optional[Dict[str, float]],
    qcs_original: Optional[QueryCoverageSketch],
    grid_size: int,
    exact_segment_step_cap: int,
) -> RepairResult:
    n = len(raw_trajectory)
    kept = _normalized_indices(raw_trajectory, indices)
    stats = _empty_stats()
    if len(kept) <= 2:
        eval0 = evaluate_indices(
            raw_trajectory,
            kept,
            tau,
            bounds=bounds,
            qcs_original=qcs_original,
            grid_size=grid_size,
            segment_step_cap=exact_segment_step_cap,
        )
        return RepairResult(
            indices=eval0["indices"],
            f1_lb=eval0["f1_lb"],
            cr=eval0["cr"],
            feasible=bool(eval0["feasible"]),
            num_deleted=0,
            attempted_deletions=0,
            stats=stats,
        )

    if bounds is None:
        bounds = _bounds_from_traj(raw_trajectory)
    if qcs_original is None:
        qcs_original = build_qcs_for_indices(
            raw_trajectory,
            range(n),
            bounds=bounds,
            grid_size=grid_size,
            segment_step_cap=exact_segment_step_cap,
        )

    base_eval = evaluate_indices(
        raw_trajectory,
        kept,
        tau=tau,
        bounds=bounds,
        qcs_original=qcs_original,
        grid_size=grid_size,
        segment_step_cap=exact_segment_step_cap,
    )
    current_f1 = float(base_eval["f1_lb"])
    budget = _compute_delete_budget(len(kept), delete_ratio, max_delete)
    if budget <= 0:
        return RepairResult(
            indices=kept,
            f1_lb=current_f1,
            cr=base_eval["cr"],
            feasible=bool(base_eval["feasible"]),
            num_deleted=0,
            attempted_deletions=0,
            stats=stats,
        )

    locked = _build_locked_set(n, locked_endpoints)
    turning = _compute_turning_angles(raw_trajectory)

    def removable_positions(local_kept: List[int]) -> List[int]:
        out = []
        for p, idx in enumerate(local_kept):
            if idx in locked:
                continue
            if p == 0 or p == len(local_kept) - 1:
                continue
            out.append(p)
        return out

    versions: Dict[int, int] = {}
    heap: List[Tuple[float, int, int]] = []
    attempted = 0
    accepted = 0

    def rebuild_scores(local_kept: List[int], candidates: List[int], f1_ref: float):
        raw_scores = {}
        qcs = {}
        curvs = {}
        segs = {}
        for p in candidates:
            idx = local_kept[p]
            tmp = local_kept[:p] + local_kept[p + 1 :]
            e = evaluate_indices(
                raw_trajectory,
                tmp,
                tau=tau,
                bounds=bounds,
                qcs_original=qcs_original,
                grid_size=grid_size,
                segment_step_cap=exact_segment_step_cap,
            )
            qcs[idx] = max(0.0, f1_ref - float(e["f1_lb"]))
            curvs[idx] = float(turning[idx])
            segs[idx] = _segment_proxy(raw_trajectory, local_kept, p)
        qcs_n = _normalize_map(qcs)
        curv_n = _normalize_map(curvs)
        seg_n = _normalize_map(segs)
        for p in candidates:
            idx = local_kept[p]
            raw_scores[p] = 0.5 * qcs_n[idx] + 0.3 * curv_n[idx] + 0.2 * seg_n[idx]
        return raw_scores

    init_pos = removable_positions(kept)
    init_scores = rebuild_scores(kept, init_pos, current_f1)
    for p in init_pos:
        versions[p] = versions.get(p, 0) + 1
        heapq.heappush(heap, (float(init_scores[p]), p, versions[p]))

    while heap and accepted < budget:
        _, p, v = heapq.heappop(heap)
        if p >= len(kept) or versions.get(p, -1) != v:
            continue
        attempted += 1
        cand_idx = kept[p]
        if cand_idx in locked:
            continue

        trial = kept[:p] + kept[p + 1 :]
        trial_eval = evaluate_indices(
            raw_trajectory,
            trial,
            tau=tau,
            bounds=bounds,
            qcs_original=qcs_original,
            grid_size=grid_size,
            segment_step_cap=exact_segment_step_cap,
        )
        if trial_eval["f1_lb"] >= tau:
            kept = trial
            current_f1 = float(trial_eval["f1_lb"])
            accepted += 1

            lo = max(1, p - 2)
            hi = min(len(kept) - 2, p + 2)
            if hi >= lo:
                local_pos = [x for x in range(lo, hi + 1) if kept[x] not in locked]
                local_scores = rebuild_scores(kept, local_pos, current_f1)
                for np_pos in local_pos:
                    versions[np_pos] = versions.get(np_pos, 0) + 1
                    heapq.heappush(heap, (float(local_scores[np_pos]), np_pos, versions[np_pos]))

    final_eval = evaluate_indices(
        raw_trajectory,
        kept,
        tau=tau,
        bounds=bounds,
        qcs_original=qcs_original,
        grid_size=grid_size,
        segment_step_cap=exact_segment_step_cap,
    )
    return RepairResult(
        indices=final_eval["indices"],
        f1_lb=float(final_eval["f1_lb"]),
        cr=float(final_eval["cr"]),
        feasible=bool(final_eval["feasible"]),
        num_deleted=int(max(0, len(_normalized_indices(raw_trajectory, indices)) - len(final_eval["indices"]))),
        attempted_deletions=int(attempted),
        stats=stats,
    )


def _repair_candidate_global_hybrid(
    raw_trajectory: np.ndarray,
    indices: Sequence[int],
    tau: float,
    delete_ratio: float,
    max_delete: Optional[int],
    locked_endpoints: bool,
    bounds: Optional[Dict[str, float]],
    qcs_original: Optional[QueryCoverageSketch],
    grid_size: int,
    skip_below_kept: int,
    min_slack: float,
    shortlist_topk: int,
    proxy_grid_size: int,
    proxy_margin: float,
    max_reject_streak: int,
    exact_segment_step_cap: int,
    proxy_segment_step_cap: int,
) -> RepairResult:
    n = len(raw_trajectory)
    kept = _normalized_indices(raw_trajectory, indices)
    initial_count = len(kept)
    stats = _empty_stats()
    if len(kept) <= 2:
        eval0 = evaluate_indices(
            raw_trajectory,
            kept,
            tau,
            bounds=bounds,
            qcs_original=qcs_original,
            grid_size=grid_size,
            segment_step_cap=exact_segment_step_cap,
        )
        return RepairResult(
            indices=eval0["indices"],
            f1_lb=eval0["f1_lb"],
            cr=eval0["cr"],
            feasible=bool(eval0["feasible"]),
            num_deleted=0,
            attempted_deletions=0,
            stats=stats,
        )

    if bounds is None:
        bounds = _bounds_from_traj(raw_trajectory)
    if qcs_original is None:
        qcs_original = build_qcs_for_indices(
            raw_trajectory,
            range(n),
            bounds=bounds,
            grid_size=grid_size,
            segment_step_cap=exact_segment_step_cap,
        )

    base_eval = evaluate_indices(
        raw_trajectory,
        kept,
        tau=tau,
        bounds=bounds,
        qcs_original=qcs_original,
        grid_size=grid_size,
        segment_step_cap=exact_segment_step_cap,
    )
    current_f1 = float(base_eval["f1_lb"])
    budget = _compute_delete_budget(len(kept), delete_ratio, max_delete)
    if budget <= 0:
        return _final_result(raw_trajectory, kept, tau, current_f1, initial_count, 0, stats)
    if len(kept) < max(0, int(skip_below_kept)):
        stats["repair_skipped_small"] += 1
        return _final_result(raw_trajectory, kept, tau, current_f1, initial_count, 0, stats)
    if current_f1 - float(tau) < float(min_slack):
        stats["repair_skipped_low_slack"] += 1
        return _final_result(raw_trajectory, kept, tau, current_f1, initial_count, 0, stats)

    locked = _build_locked_set(n, locked_endpoints)
    turning = _compute_turning_angles(raw_trajectory)
    spatial_scale = _spatial_scale(bounds)
    versions: Dict[int, int] = {}
    heap: List[Tuple[float, int, int]] = []
    accepted = 0
    attempted = 0
    reject_streak = 0
    shortlist: List[Tuple[float, int]] = []
    proxy_grid_size = max(4, int(proxy_grid_size))
    shortlist_topk = max(1, int(shortlist_topk))
    max_reject_streak = max(1, int(max_reject_streak))

    def make_pos_map(local_kept: List[int]) -> Dict[int, int]:
        return {idx: pos for pos, idx in enumerate(local_kept)}

    pos_map = make_pos_map(kept)

    def removable_indices(local_kept: List[int], local_pos_map: Dict[int, int]) -> List[int]:
        out: List[int] = []
        for idx in local_kept:
            pos = local_pos_map.get(idx, -1)
            if idx in locked or pos <= 0 or pos >= len(local_kept) - 1:
                continue
            out.append(idx)
        return out

    def geometry_score(idx: int) -> Optional[float]:
        pos = pos_map.get(idx)
        if pos is None or pos <= 0 or pos >= len(kept) - 1 or idx in locked:
            return None
        curv_score = float(turning[idx] / math.pi)
        seg_score = min(1.0, float(_segment_proxy(raw_trajectory, kept, pos) / spatial_scale))
        return 0.65 * curv_score + 0.35 * seg_score

    def push_candidate(idx: int) -> None:
        score = geometry_score(idx)
        if score is None:
            return
        versions[idx] = versions.get(idx, 0) + 1
        heapq.heappush(heap, (float(score), int(idx), versions[idx]))

    def shortlist_limit() -> int:
        removable_count = len(removable_indices(kept, pos_map))
        if removable_count <= 0:
            return 0
        dynamic_topk = max(16, int(math.ceil(0.2 * removable_count)))
        return min(shortlist_topk, dynamic_topk, removable_count)

    def harvest_shortlist() -> List[Tuple[float, int]]:
        out: List[Tuple[float, int]] = []
        limit = shortlist_limit()
        while heap and len(out) < limit:
            score, idx, version = heapq.heappop(heap)
            pos = pos_map.get(idx)
            if versions.get(idx, -1) != version:
                continue
            if pos is None or pos <= 0 or pos >= len(kept) - 1 or idx in locked:
                continue
            out.append((float(score), int(idx)))
        return out

    for idx in removable_indices(kept, pos_map):
        push_candidate(idx)

    while accepted < budget:
        if not shortlist:
            shortlist = harvest_shortlist()
            if not shortlist:
                break

        _, idx = shortlist.pop(0)
        pos = pos_map.get(idx)
        if pos is None or pos <= 0 or pos >= len(kept) - 1 or idx in locked:
            continue

        trial = kept[:pos] + kept[pos + 1 :]
        proxy_eval = evaluate_indices(
            raw_trajectory,
            trial,
            tau=tau,
            bounds=bounds,
            qcs_original=qcs_original,
            grid_size=proxy_grid_size,
            segment_step_cap=proxy_segment_step_cap,
        )
        stats["repair_proxy_checked"] += 1
        if float(proxy_eval["f1_lb"]) < float(tau) + float(proxy_margin):
            reject_streak += 1
            if reject_streak >= max_reject_streak:
                stats["repair_early_stop_count"] += 1
                break
            continue

        exact_eval = evaluate_indices(
            raw_trajectory,
            trial,
            tau=tau,
            bounds=bounds,
            qcs_original=qcs_original,
            grid_size=grid_size,
            segment_step_cap=exact_segment_step_cap,
        )
        stats["repair_exact_checked"] += 1
        attempted += 1
        if float(exact_eval["f1_lb"]) >= float(tau):
            kept = trial
            pos_map = make_pos_map(kept)
            current_f1 = float(exact_eval["f1_lb"])
            accepted += 1
            reject_streak = 0
            shortlist = []

            lo = max(1, pos - 3)
            hi = min(len(kept) - 2, pos + 3)
            for local_pos in range(lo, hi + 1):
                push_candidate(kept[local_pos])
            continue

        reject_streak += 1
        if reject_streak >= max_reject_streak:
            stats["repair_early_stop_count"] += 1
            break

    return _final_result(raw_trajectory, kept, tau, current_f1, initial_count, attempted, stats)


def repair_candidate_global(
    raw_trajectory: np.ndarray,
    indices: Sequence[int],
    tau: float,
    delete_ratio: float = 0.30,
    max_delete: Optional[int] = None,
    locked_endpoints: bool = True,
    bounds: Optional[Dict[str, float]] = None,
    qcs_original: Optional[QueryCoverageSketch] = None,
    grid_size: int = 64,
    repair_mode: str = "hybrid",
    skip_below_kept: int = 128,
    min_slack: float = 0.02,
    shortlist_topk: int = 64,
    proxy_grid_size: int = 24,
    proxy_margin: float = 0.01,
    max_reject_streak: int = 16,
    exact_segment_step_cap: int = DEFAULT_EXACT_SEGMENT_STEP_CAP,
    proxy_segment_step_cap: int = DEFAULT_PROXY_SEGMENT_STEP_CAP,
) -> RepairResult:
    """
    Deterministic global deletion repair.

    repair_mode="exact" preserves the original all-exact scoring path.
    repair_mode="hybrid" gates repair, ranks by cheap geometry, then runs
    proxy-QCS screening before exact confirmation.
    """
    mode = str(repair_mode or "hybrid").strip().lower()
    if mode == "exact":
        return _repair_candidate_global_exact(
            raw_trajectory=raw_trajectory,
            indices=indices,
            tau=tau,
            delete_ratio=delete_ratio,
            max_delete=max_delete,
            locked_endpoints=locked_endpoints,
            bounds=bounds,
            qcs_original=qcs_original,
            grid_size=grid_size,
            exact_segment_step_cap=exact_segment_step_cap,
        )
    return _repair_candidate_global_hybrid(
        raw_trajectory=raw_trajectory,
        indices=indices,
        tau=tau,
        delete_ratio=delete_ratio,
        max_delete=max_delete,
        locked_endpoints=locked_endpoints,
        bounds=bounds,
        qcs_original=qcs_original,
        grid_size=grid_size,
        skip_below_kept=skip_below_kept,
        min_slack=min_slack,
        shortlist_topk=shortlist_topk,
        proxy_grid_size=proxy_grid_size,
        proxy_margin=proxy_margin,
        max_reject_streak=max_reject_streak,
        exact_segment_step_cap=exact_segment_step_cap,
        proxy_segment_step_cap=proxy_segment_step_cap,
    )
