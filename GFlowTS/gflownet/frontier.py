# -*- coding: utf-8 -*-
"""Frontier maintenance utilities for PRS-Dual-Frontier training/inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def jaccard_similarity(indices_a: Sequence[int], indices_b: Sequence[int]) -> float:
    """Compute Jaccard similarity between two index sets."""
    set_a = set(int(x) for x in indices_a)
    set_b = set(int(x) for x in indices_b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    inter = set_a & set_b
    return len(inter) / max(1, len(union))


@dataclass
class FrontierCandidate:
    """A candidate solution stored in frontier."""

    actions: List[int]
    indices: List[int]
    f1_lb: float
    cr: float
    feasible: bool
    reward_dual: float
    novelty: float = 0.0
    metadata: Dict[str, float] = field(default_factory=dict)

    def key(self) -> Tuple[int, ...]:
        return tuple(sorted(set(int(i) for i in self.indices)))

    def margin(self, tau: float) -> float:
        return float(self.f1_lb) - float(tau)

    def is_safe_feasible(self, tau: float, safe_margin: float = 0.01) -> bool:
        return bool(self.feasible) and (self.margin(tau) >= float(safe_margin) - 1e-12)


class FeasibleFrontierBuffer:
    """
    Per-chunk frontier with feasible-first + multi-objective pruning.

    Keep rules:
    - deduplicate by high Jaccard similarity
    - feasible candidates preferred
    - score combines F1 (high), CR (low), novelty (high)
    """

    def __init__(
        self,
        capacity: int = 64,
        jaccard_dedup: float = 0.95,
        novelty_weight: float = 0.20,
        f1_weight: float = 0.45,
        cr_weight: float = 0.35,
    ):
        self.capacity = int(max(1, capacity))
        self.jaccard_dedup = float(np.clip(jaccard_dedup, 0.0, 1.0))
        self.novelty_weight = float(max(0.0, novelty_weight))
        self.f1_weight = float(max(0.0, f1_weight))
        self.cr_weight = float(max(0.0, cr_weight))
        self._data: Dict[Tuple, List[FrontierCandidate]] = {}

    def _list(self, key: Tuple) -> List[FrontierCandidate]:
        if key not in self._data:
            self._data[key] = []
        return self._data[key]

    def _compute_novelty(self, existing: List[FrontierCandidate], candidate: FrontierCandidate) -> float:
        if not existing:
            return 1.0
        max_sim = 0.0
        for item in existing:
            sim = jaccard_similarity(candidate.indices, item.indices)
            if sim > max_sim:
                max_sim = sim
        return float(np.clip(1.0 - max_sim, 0.0, 1.0))

    @staticmethod
    def _normalize(values: List[float]) -> List[float]:
        if not values:
            return []
        v_min = min(values)
        v_max = max(values)
        if abs(v_max - v_min) < 1e-12:
            return [0.5 for _ in values]
        return [(v - v_min) / (v_max - v_min) for v in values]

    def _score(self, items: List[FrontierCandidate]) -> List[float]:
        f1_norm = self._normalize([x.f1_lb for x in items])
        cr_norm = self._normalize([x.cr for x in items])
        nov_norm = self._normalize([x.novelty for x in items])
        out: List[float] = []
        for i in range(len(items)):
            score = (
                self.f1_weight * f1_norm[i]
                - self.cr_weight * cr_norm[i]
                + self.novelty_weight * nov_norm[i]
            )
            # Small tiebreak by reward.
            score += 0.05 * float(items[i].reward_dual)
            out.append(float(score))
        return out

    @staticmethod
    def _priority_sort_key(
        item: FrontierCandidate,
        tau: float,
        safe_margin: float = 0.01,
    ) -> Tuple[float, ...]:
        if item.is_safe_feasible(tau=tau, safe_margin=safe_margin):
            return (0.0, float(item.cr), -float(item.f1_lb), -float(item.reward_dual), -float(item.novelty))
        if item.feasible:
            return (1.0, float(item.cr), -float(item.f1_lb), -float(item.reward_dual), -float(item.novelty))
        return (2.0, -float(item.f1_lb), -float(item.reward_dual), float(item.cr), -float(item.novelty))

    def _rank_candidates(
        self,
        items: List[FrontierCandidate],
        tau: float,
        safe_margin: float = 0.01,
    ) -> List[FrontierCandidate]:
        return sorted(
            items,
            key=lambda item: self._priority_sort_key(item, tau=tau, safe_margin=safe_margin),
        )

    def _dedup_replace(self, frontier: List[FrontierCandidate], candidate: FrontierCandidate) -> bool:
        """Try replacing a near-duplicate candidate. Returns True if handled."""
        for i, item in enumerate(frontier):
            sim = jaccard_similarity(item.indices, candidate.indices)
            if sim > self.jaccard_dedup:
                # Keep higher reward/quality one.
                replace = False
                if candidate.feasible and not item.feasible:
                    replace = True
                elif candidate.feasible == item.feasible:
                    if candidate.reward_dual > item.reward_dual + 1e-12:
                        replace = True
                    elif abs(candidate.reward_dual - item.reward_dual) <= 1e-12:
                        if (candidate.f1_lb > item.f1_lb + 1e-12) or (
                            abs(candidate.f1_lb - item.f1_lb) <= 1e-12 and candidate.cr < item.cr - 1e-12
                        ):
                            replace = True
                if replace:
                    frontier[i] = candidate
                return True
        return False

    def update(self, chunk_key: Tuple, candidate: FrontierCandidate) -> List[FrontierCandidate]:
        frontier = self._list(chunk_key)
        candidate.novelty = self._compute_novelty(frontier, candidate)
        if self._dedup_replace(frontier, candidate):
            return frontier

        frontier.append(candidate)
        if len(frontier) <= self.capacity:
            return frontier

        # Prune to capacity with feasible-first strategy.
        feasible = [x for x in frontier if x.feasible]
        infeasible = [x for x in frontier if not x.feasible]
        feasible_scores = self._score(feasible)
        infeasible_scores = self._score(infeasible)

        feasible_sorted = [x for _, x in sorted(zip(feasible_scores, feasible), key=lambda t: t[0], reverse=True)]
        infeasible_sorted = [x for _, x in sorted(zip(infeasible_scores, infeasible), key=lambda t: t[0], reverse=True)]

        keep: List[FrontierCandidate] = []
        keep.extend(feasible_sorted[: self.capacity])
        if len(keep) < self.capacity:
            keep.extend(infeasible_sorted[: self.capacity - len(keep)])

        self._data[chunk_key] = keep
        return self._data[chunk_key]

    def get(self, chunk_key: Tuple) -> List[FrontierCandidate]:
        return list(self._data.get(chunk_key, []))

    def top_reward(self, chunk_key: Tuple, k: int = 8) -> List[FrontierCandidate]:
        items = self.get(chunk_key)
        if not items:
            return []
        scores = self._score(items)
        sorted_items = [x for _, x in sorted(zip(scores, items), key=lambda t: t[0], reverse=True)]
        return sorted_items[: max(1, int(k))]

    def top_lowcr_feasible(
        self,
        chunk_key: Tuple,
        k: int = 8,
        tau: Optional[float] = None,
        safe_margin: float = 0.01,
    ) -> List[FrontierCandidate]:
        items = self.get(chunk_key)
        if not items:
            return []
        tau_value = 0.0 if tau is None else float(tau)
        ranked = self._rank_candidates(items, tau=tau_value, safe_margin=safe_margin)
        return ranked[: max(1, int(k))]

    def sample_top_m(
        self,
        chunk_key: Tuple,
        m: int = 8,
        tau: Optional[float] = None,
        safe_margin: float = 0.01,
    ) -> List[FrontierCandidate]:
        if tau is None:
            return self.top_reward(chunk_key, k=m)
        return self.top_lowcr_feasible(chunk_key, k=m, tau=tau, safe_margin=safe_margin)

    def dynamic_cr_cap(
        self,
        chunk_key: Tuple,
        teacher_cap: float,
        bonus: float = 0.02,
        expand_ratio: float = 1.1,
    ) -> float:
        items = self.get(chunk_key)
        feasible_cr = [x.cr for x in items if x.feasible]
        if not feasible_cr:
            return float(np.clip(teacher_cap, 1e-4, 1.0))
        med = float(np.median(feasible_cr))
        cap = min(float(teacher_cap) * float(max(1.0, expand_ratio)), med + float(bonus))
        return float(np.clip(cap, 1e-4, 1.0))

    def stats(self) -> Dict[str, float]:
        if not self._data:
            return {
                "chunks": 0.0,
                "mean_size": 0.0,
                "mean_feasible": 0.0,
                "mean_novelty": 0.0,
            }
        sizes = []
        feasible_ratios = []
        novelty = []
        for items in self._data.values():
            if not items:
                continue
            sizes.append(len(items))
            feasible_ratios.append(np.mean([1.0 if x.feasible else 0.0 for x in items]))
            novelty.append(np.mean([x.novelty for x in items]))
        if not sizes:
            return {
                "chunks": float(len(self._data)),
                "mean_size": 0.0,
                "mean_feasible": 0.0,
                "mean_novelty": 0.0,
            }
        return {
            "chunks": float(len(self._data)),
            "mean_size": float(np.mean(sizes)),
            "mean_feasible": float(np.mean(feasible_ratios)),
            "mean_novelty": float(np.mean(novelty)),
        }

    def state_dict(self) -> Dict[str, object]:
        data: List[Dict[str, object]] = []
        for chunk_key, items in self._data.items():
            serialized_items: List[Dict[str, object]] = []
            for item in items:
                serialized_items.append(
                    {
                        "actions": [int(x) for x in item.actions],
                        "indices": [int(x) for x in item.indices],
                        "f1_lb": float(item.f1_lb),
                        "cr": float(item.cr),
                        "feasible": bool(item.feasible),
                        "reward_dual": float(item.reward_dual),
                        "novelty": float(item.novelty),
                        "metadata": {
                            str(k): float(v) for k, v in dict(item.metadata).items()
                        },
                    }
                )
            data.append(
                {
                    "chunk_key": tuple(chunk_key),
                    "items": serialized_items,
                }
            )
        return {
            "capacity": int(self.capacity),
            "jaccard_dedup": float(self.jaccard_dedup),
            "novelty_weight": float(self.novelty_weight),
            "f1_weight": float(self.f1_weight),
            "cr_weight": float(self.cr_weight),
            "data": data,
        }

    def load_state_dict(self, state: Optional[Dict[str, object]]) -> None:
        self._data = {}
        if not isinstance(state, dict):
            return

        self.capacity = int(max(1, state.get("capacity", self.capacity)))
        self.jaccard_dedup = float(np.clip(state.get("jaccard_dedup", self.jaccard_dedup), 0.0, 1.0))
        self.novelty_weight = float(max(0.0, state.get("novelty_weight", self.novelty_weight)))
        self.f1_weight = float(max(0.0, state.get("f1_weight", self.f1_weight)))
        self.cr_weight = float(max(0.0, state.get("cr_weight", self.cr_weight)))

        data = state.get("data", [])
        if not isinstance(data, list):
            return

        restored: Dict[Tuple, List[FrontierCandidate]] = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            chunk_key_raw = entry.get("chunk_key")
            if not isinstance(chunk_key_raw, (list, tuple)):
                continue
            items_raw = entry.get("items", [])
            if not isinstance(items_raw, list):
                continue

            chunk_key = tuple(chunk_key_raw)
            restored_items: List[FrontierCandidate] = []
            for item in items_raw:
                if not isinstance(item, dict):
                    continue
                metadata_raw = item.get("metadata", {})
                metadata = {}
                if isinstance(metadata_raw, dict):
                    metadata = {
                        str(k): float(v) for k, v in metadata_raw.items()
                    }
                restored_items.append(
                    FrontierCandidate(
                        actions=[int(x) for x in item.get("actions", [])],
                        indices=[int(x) for x in item.get("indices", [])],
                        f1_lb=float(item.get("f1_lb", 0.0)),
                        cr=float(item.get("cr", 1.0)),
                        feasible=bool(item.get("feasible", False)),
                        reward_dual=float(item.get("reward_dual", 0.0)),
                        novelty=float(item.get("novelty", 0.0)),
                        metadata=metadata,
                    )
                )
            restored[chunk_key] = restored_items[: self.capacity]
        self._data = restored
