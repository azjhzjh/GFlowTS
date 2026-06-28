# -*- coding: utf-8 -*-
"""
GFlowNet 娴佸紡璁粌鑴氭湰 (鏈€缁堢増)

浣跨敤鏂瑰紡:
    python train_streaming.py --traj_path TrajData/Geolife_out --epochs 100

璁粌娴佺▼:
    for traj in TrajectoryStream:
        蟿 = sample_compression(traj)   # softmax
        if F1_lower_bound >= 0.95:
            update_backward_policy()   # TLM
        update_forward_policy()        # online
"""



import os
# [Fix] 闃叉 Intel MKL/Fortran 鍦?Windows 涓婃崟鑾?Ctrl+C 瀵艰嚧鐩存帴 Crash
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import argparse
import os
import sys
import time
import csv
from collections import OrderedDict
from contextlib import nullcontext
from functools import lru_cache
import numpy as np
import torch
import gc
import datetime
import json
import pickle
from typing import Dict, List, Tuple, Optional


# 娣诲姞椤圭洰璺緞
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gflownet.models import HierarchicalGFlowNet
from gflownet.gfn_env import GFlowNetTrajectoryEnv
from gflownet.train_tlm import TLMTrainer
from gflownet.frontier import jaccard_similarity
from gflownet.repair import repair_candidate_global
from data_utils import to_traj, rdp_simplify, CustomBallTree
from profiling_utils import finalize_profile, get_active_collector, increment_profile_counter, init_profile_collector, profile_scope



_CHUNK_SIZE = 1000
_TRAJECTORY_CACHE_MAXSIZE = 256
_GREEDY_CAP_CACHE_MAXSIZE = 4096
_TRAJECTORY_CACHE = OrderedDict()
_GREEDY_CAP_CACHE = OrderedDict()


def _cache_get(cache: OrderedDict, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_put(cache: OrderedDict, key, value, maxsize: int):
    if maxsize <= 0:
        return value
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max(0, int(maxsize)):
        cache.popitem(last=False)
    return value


def configure_runtime_caches(
    traj_cache_size: Optional[int] = None,
    greedy_cache_size: Optional[int] = None,
):
    global _TRAJECTORY_CACHE_MAXSIZE, _GREEDY_CAP_CACHE_MAXSIZE

    if traj_cache_size is not None:
        _TRAJECTORY_CACHE_MAXSIZE = max(0, int(traj_cache_size))
        if _TRAJECTORY_CACHE_MAXSIZE == 0:
            _TRAJECTORY_CACHE.clear()
        else:
            while len(_TRAJECTORY_CACHE) > _TRAJECTORY_CACHE_MAXSIZE:
                _TRAJECTORY_CACHE.popitem(last=False)

    if greedy_cache_size is not None:
        _GREEDY_CAP_CACHE_MAXSIZE = max(0, int(greedy_cache_size))
        if _GREEDY_CAP_CACHE_MAXSIZE == 0:
            _GREEDY_CAP_CACHE.clear()
        else:
            while len(_GREEDY_CAP_CACHE) > _GREEDY_CAP_CACHE_MAXSIZE:
                _GREEDY_CAP_CACHE.popitem(last=False)


def _add_optional_profile_time(
    phase: Optional[str],
    major_module: Optional[str],
    minor_module: str,
    elapsed_s: float,
) -> None:
    if (not phase) or (not major_module):
        return
    collector = get_active_collector()
    if collector is None:
        return
    collector.add_time(
        phase=str(phase),
        major_module=str(major_module),
        minor_module=str(minor_module),
        elapsed_s=float(max(0.0, elapsed_s)),
    )


def _add_profile_counters(counter_map: Optional[Dict[str, int]]) -> None:
    if not counter_map:
        return
    for key, value in counter_map.items():
        if int(value) != 0:
            increment_profile_counter(str(key), int(value))


def _ensure_float32_array(traj: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(traj, dtype=np.float32))


@lru_cache(maxsize=8192)
def _resolve_trajectory_path(path: str, idx: int) -> str:
    filepath = os.path.join(path, str(idx))
    if os.path.exists(filepath):
        return filepath
    txt_path = filepath + '.txt'
    if os.path.exists(txt_path):
        return txt_path
    return ''


def _cache_key_for_traj(path: str, idx: int) -> Tuple[str, int]:
    return (os.path.normcase(path), int(idx))


def load_preprocessed_trajectory_record(path: str, idx: int) -> Optional[Dict[str, object]]:
    cache_key = _cache_key_for_traj(path, idx)
    cached = _cache_get(_TRAJECTORY_CACHE, cache_key)
    if cached is not None:
        return cached

    traj = load_single_trajectory(path, idx)
    if traj is None:
        return None

    traj = pre_simplify_if_needed(traj)
    if traj is None or len(traj) < 3:
        return None

    traj = _ensure_float32_array(traj)
    record = {
        'traj': traj,
        'bounds': _compute_chunk_bounds(traj),
        'chunk_ranges': tuple(_build_chunk_ranges(len(traj), chunk_size=_CHUNK_SIZE)),
        'global_qcs_original': None,
    }
    return _cache_put(_TRAJECTORY_CACHE, cache_key, record, _TRAJECTORY_CACHE_MAXSIZE)


def load_preprocessed_trajectory(path: str, idx: int) -> Optional[np.ndarray]:
    record = load_preprocessed_trajectory_record(path, idx)
    if record is None:
        return None
    return record['traj']


def _prepare_norm_stats(stats: dict) -> Tuple[np.ndarray, np.ndarray]:
    t_min = stats.get('_norm_min')
    inv_scale = stats.get('_norm_inv_scale')
    if t_min is None or inv_scale is None:
        t_min = np.array([stats['x_min'], stats['y_min'], stats['t_min']], dtype=np.float32)
        t_max = np.array([stats['x_max'], stats['y_max'], stats['t_max']], dtype=np.float32)
        inv_scale = (1.0 / np.maximum(t_max - t_min, 1e-9)).astype(np.float32, copy=False)
        stats['_norm_min'] = t_min
        stats['_norm_inv_scale'] = inv_scale
    return stats['_norm_min'], stats['_norm_inv_scale']





def load_single_trajectory(path: str, idx: int) -> np.ndarray:
    """鍔犺浇鍗曟潯杞ㄨ抗"""
    filepath = _resolve_trajectory_path(path, int(idx))
    if filepath == '':
        return None

    traj = to_traj(filepath)
    if traj is None:
        return None
    return _ensure_float32_array(traj)


def normalize_trajectory(traj: np.ndarray, stats: dict = None) -> np.ndarray:
    """褰掍竴鍖栬建杩瑰埌 [0, 1]"""
    traj = _ensure_float32_array(traj)

    if stats is None:
        t_min = traj.min(axis=0).astype(np.float32, copy=False)
        t_max = traj.max(axis=0).astype(np.float32, copy=False)
        inv_scale = (1.0 / np.maximum(t_max - t_min, 1e-9)).astype(np.float32, copy=False)
    else:
        t_min, inv_scale = _prepare_norm_stats(stats)

    return ((traj - t_min) * inv_scale).astype(np.float32, copy=False)
def pre_simplify_if_needed(traj: np.ndarray) -> np.ndarray:
    """Pre-simplify trajectory using RDP if length > 20000. Returns potentially simplified trajectory."""
    if len(traj) > 20000:
        traj = rdp_simplify(traj, epsilon=1e-5)
    return traj

def scan_valid_indices(path: str, start: int, end: int) -> list:
    """鎵弿鏈夋晥杞ㄨ抗绱㈠紩骞堕璁＄畻鑼冨洿 (鐢ㄤ簬 QCS)"""
    valid = []
    for idx in range(start, end):
        traj_record = load_preprocessed_trajectory_record(path, idx)
        if traj_record is not None:
            valid.append((idx, traj_record['bounds']))
    return valid


def compute_global_stats(path: str, valid_infos: list, sample_size: int = 100) -> dict:
    """璁＄畻鍏ㄥ眬缁熻閲?(閫氳繃閲囨牱)"""
    print(f"[Data] Computing global stats (sampling {sample_size} trajectories)...")
    indices = [info[0] for info in valid_infos]
    sample_indices = np.random.choice(indices, min(sample_size, len(indices)), replace=False)
    
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    t_min, t_max = float('inf'), float('-inf')
    
    for idx in sample_indices:
        traj = load_single_trajectory(path, idx)
        if traj is None or len(traj) == 0:
            continue
        
        x_min = min(x_min, traj[:, 0].min())
        x_max = max(x_max, traj[:, 0].max())
        y_min = min(y_min, traj[:, 1].min())
        y_max = max(y_max, traj[:, 1].max())
        t_min = min(t_min, traj[:, 2].min())
        t_max = max(t_max, traj[:, 2].max())
    
    return {
        'x_min': x_min, 'x_max': x_max,
        'y_min': y_min, 'y_max': y_max,
        't_min': t_min, 't_max': t_max
    }


def build_global_qcs_original(raw_trajectory, local_stats=None):
    """Build and cacheable original trajectory QCS for global F1 computation."""
    from gflownet.utils.query_sketch import QueryCoverageSketch

    if local_stats is not None:
        x_range = (local_stats['x_min'], local_stats['x_max'] + 1e-9)
        y_range = (local_stats['y_min'], local_stats['y_max'] + 1e-9)
        t_range = (local_stats['t_min'], local_stats['t_max'] + 1e-9)
    else:
        x_range = (raw_trajectory[:, 0].min(), raw_trajectory[:, 0].max() + 1e-9)
        y_range = (raw_trajectory[:, 1].min(), raw_trajectory[:, 1].max() + 1e-9)
        t_range = (raw_trajectory[:, 2].min(), raw_trajectory[:, 2].max() + 1e-9)

    qcs_original = QueryCoverageSketch(64, x_range, y_range, t_range)
    qcs_original.add_point(raw_trajectory[0])
    for i in range(1, len(raw_trajectory)):
        qcs_original.add_segment(raw_trajectory[i-1], raw_trajectory[i])
    return qcs_original


def calculate_global_f1(raw_trajectory, selected_indices, local_stats=None, qcs_original=None):
    """璁＄畻瀹屾暣杞ㄨ抗鐨?F1 鍒嗘暟 (Shape Aware)"""
    from gflownet.utils.query_sketch import QueryCoverageSketch, f1_lower_bound

    if qcs_original is None:
        qcs_original = build_global_qcs_original(raw_trajectory, local_stats=local_stats)

    qcs_compressed = QueryCoverageSketch(
        qcs_original.grid_size,
        qcs_original.x_range,
        qcs_original.y_range,
        qcs_original.t_range,
    )
    if len(selected_indices) > 0:
        # 纭繚绱㈠紩鏈夊簭涓斿敮涓€
        indices = sorted(list(set(selected_indices)))
        qcs_compressed.add_point(raw_trajectory[indices[0]])
        for i in range(1, len(indices)):
            qcs_compressed.add_segment(raw_trajectory[indices[i-1]], raw_trajectory[indices[i]])
            
    return f1_lower_bound(qcs_compressed, qcs_original)


def repair_combined_trajectory(
    raw_trajectory,
    selected_indices,
    tau: float,
    args,
    bounds: Optional[dict] = None,
    qcs_original=None,
):
    """Apply repair once after chunk indices are merged into a trajectory-level path."""
    unique_indices = sorted(set(int(i) for i in selected_indices if 0 <= int(i) < len(raw_trajectory)))
    if len(unique_indices) <= 2:
        return unique_indices, 0, {}
    if bool(getattr(args, 'legacy_single_path', False)) or (not bool(getattr(args, 'repair_enable', False))):
        return unique_indices, 0, {}

    rep = repair_candidate_global(
        raw_trajectory=raw_trajectory,
        indices=unique_indices,
        tau=float(tau),
        delete_ratio=float(getattr(args, 'repair_delete_ratio', 0.0)),
        max_delete=int(getattr(args, 'repair_max_delete', 0)),
        locked_endpoints=True,
        bounds=bounds,
        qcs_original=qcs_original,
        repair_mode=str(getattr(args, 'repair_mode', 'hybrid')),
        skip_below_kept=int(getattr(args, 'repair_skip_below_kept', 128)),
        min_slack=float(getattr(args, 'repair_min_slack', 0.02)),
        shortlist_topk=int(getattr(args, 'repair_shortlist_topk', 64)),
        proxy_grid_size=int(getattr(args, 'repair_proxy_grid_size', 24)),
        proxy_margin=float(getattr(args, 'repair_proxy_margin', 0.01)),
    )
    return rep.indices, int(rep.num_deleted), dict(getattr(rep, 'stats', {}) or {})


def _compute_chunk_bounds(chunk: np.ndarray) -> dict:
    return {
        'x_min': float(chunk[:, 0].min()), 'x_max': float(chunk[:, 0].max()),
        'y_min': float(chunk[:, 1].min()), 'y_max': float(chunk[:, 1].max()),
        't_min': float(chunk[:, 2].min()), 't_max': float(chunk[:, 2].max())
    }


def _build_chunk_ranges(traj_len: int, chunk_size: int = _CHUNK_SIZE) -> List[Tuple[int, int]]:
    """Build [start, end) ranges for chunked training with 1-point overlap."""
    if traj_len < 3:
        return []
    if traj_len <= chunk_size:
        return [(0, traj_len)]

    ranges: List[Tuple[int, int]] = []
    for start in range(0, traj_len - 2, chunk_size - 1):
        end = min(start + chunk_size, traj_len)
        if end - start < 3:
            break
        ranges.append((start, end))
    return ranges


def parse_length_bucket_bounds(spec: str) -> List[int]:
    """Parse bucket bounds like '256,512,1000' into sorted positive ints."""
    values: List[int] = []
    for tok in str(spec).split(','):
        tok = tok.strip()
        if tok == '':
            continue
        try:
            val = int(tok)
        except ValueError:
            continue
        if val >= 3:
            values.append(val)
    if not values:
        values = [256, 512, 1000]
    values = sorted(set(values))
    return values


def _chunk_bucket_id(n_points: int, bounds: List[int]) -> int:
    if not bounds:
        return 0
    for i, upper in enumerate(bounds):
        if n_points <= int(upper):
            return i
    return len(bounds) - 1


def _derive_greedy_cap(
    chunk,
    chunk_norm,
    keep_start,
    keep_end,
    args,
    global_stats,
    device,
    relax_factor: float = 1.0,
):
    """Derive chunk-level CR cap from greedy teacher (or target fallback)."""
    chunk_bounds = _compute_chunk_bounds(chunk)
    env = GFlowNetTrajectoryEnv(
        trajectory=chunk_norm,
        raw_trajectory=chunk,
        alpha=0.5,
        beta=0.5,
        f1_threshold=args.f1_threshold,
        target_compression=args.target_compression,
        device=device,
        global_stats=global_stats,
        local_stats=chunk_bounds,
        keep_start=keep_start,
        keep_end=keep_end
    )
    greedy_actions = env.greedy_grid_simplify()
    env.close()

    greedy_indices = sorted({int(a) for a in greedy_actions if 0 <= int(a) < len(chunk)})
    if keep_start:
        greedy_indices.append(0)
    if keep_end and len(chunk) > 1:
        greedy_indices.append(len(chunk) - 1)
    greedy_indices = sorted(set(greedy_indices))

    if args.cr_cap_source == 'greedy':
        base_max_keep = min(len(chunk), max(2, len(greedy_indices)))
    else:
        base_max_keep = min(len(chunk), max(2, int(np.ceil(len(chunk) * args.target_compression))))

    greedy_max_keep = min(len(chunk), max(2, int(np.ceil(base_max_keep * max(1e-6, relax_factor)))))

    greedy_cr_cap = greedy_max_keep / max(1, len(chunk))
    return greedy_actions, greedy_indices, greedy_max_keep, greedy_cr_cap, chunk_bounds


def _make_greedy_cache_key(
    traj_path: str,
    idx: int,
    start: int,
    end: int,
    keep_start: bool,
    keep_end: bool,
    args,
) -> Tuple[object, ...]:
    return (
        os.path.normcase(str(traj_path)),
        int(idx),
        int(start),
        int(end),
        bool(keep_start),
        bool(keep_end),
        str(args.cr_cap_source),
        float(args.target_compression),
        float(args.f1_threshold),
    )


def _derive_greedy_cap_cached(
    cache_key,
    chunk,
    chunk_norm,
    keep_start,
    keep_end,
    args,
    global_stats,
    device,
    relax_factor: float = 1.0,
):
    if cache_key is None or _GREEDY_CAP_CACHE_MAXSIZE <= 0:
        return _derive_greedy_cap(
            chunk,
            chunk_norm,
            keep_start,
            keep_end,
            args,
            global_stats,
            device,
            relax_factor=relax_factor,
        )

    cached = _cache_get(_GREEDY_CAP_CACHE, cache_key)
    if cached is None:
        greedy_actions, greedy_indices, base_max_keep, _, chunk_bounds = _derive_greedy_cap(
            chunk,
            chunk_norm,
            keep_start,
            keep_end,
            args,
            global_stats,
            device,
            relax_factor=1.0,
        )
        cached = {
            'greedy_actions': tuple(int(a) for a in greedy_actions),
            'greedy_indices': tuple(int(i) for i in greedy_indices),
            'base_max_keep': int(base_max_keep),
            'chunk_bounds': chunk_bounds,
        }
        _cache_put(_GREEDY_CAP_CACHE, cache_key, cached, _GREEDY_CAP_CACHE_MAXSIZE)

    base_max_keep = int(cached['base_max_keep'])
    greedy_max_keep = min(len(chunk), max(2, int(np.ceil(base_max_keep * max(1e-6, relax_factor)))))
    greedy_cr_cap = greedy_max_keep / max(1, len(chunk))
    return (
        list(cached['greedy_actions']),
        list(cached['greedy_indices']),
        greedy_max_keep,
        greedy_cr_cap,
        cached['chunk_bounds'],
    )


def _build_action_pool_from_env(env, valid_actions: np.ndarray, pool_size: int, explore_ratio: float = 0.2) -> np.ndarray:
    """Build candidate pool from valid point actions (exclude stop)."""
    valid_idx = np.where(valid_actions[:-1])[0]
    if len(valid_idx) <= 1:
        return valid_idx.astype(np.int64)
    k = int(max(1, pool_size))
    if len(valid_idx) <= k:
        return valid_idx.astype(np.int64)

    turn = np.abs(getattr(env, 'turning_angles', np.zeros(len(valid_idx)))[valid_idx])
    vel = np.abs(getattr(env, 'velocities', np.zeros(len(valid_idx)))[valid_idx])
    score = turn + 0.2 * vel

    explore_ratio = float(np.clip(explore_ratio, 0.0, 1.0))
    num_top = int(max(1, round(k * (1.0 - explore_ratio))))
    num_rand = int(max(0, k - num_top))
    if num_top >= len(valid_idx):
        top_idx = valid_idx
        rem_idx = np.array([], dtype=np.int64)
    else:
        top_pos = np.argpartition(-score, num_top - 1)[:num_top]
        top_idx = valid_idx[top_pos]
        rem_mask = np.ones(len(valid_idx), dtype=bool)
        rem_mask[top_pos] = False
        rem_idx = valid_idx[rem_mask]

    if num_rand > 0 and len(rem_idx) > 0:
        pick = min(num_rand, len(rem_idx))
        rand_idx = np.random.choice(rem_idx, size=pick, replace=False)
        merged = np.concatenate([top_idx.astype(np.int64), rand_idx.astype(np.int64)])
    else:
        merged = top_idx.astype(np.int64)
    if len(merged) > k:
        merged = merged[:k]
    return np.unique(merged).astype(np.int64)


def _create_chunk_eval_env(
    chunk,
    chunk_norm,
    keep_start,
    keep_end,
    tau,
    cr_cap_ratio,
    global_stats,
    device,
    use_proxy_eval: bool = False,
    proxy_grid_size: int = 24,
    proxy_stride: int = 4,
):
    return GFlowNetTrajectoryEnv(
        trajectory=chunk_norm,
        raw_trajectory=chunk,
        alpha=0.5,
        beta=0.5,
        f1_threshold=tau,
        target_compression=cr_cap_ratio,
        device=device,
        global_stats=global_stats,
        local_stats=None,
        keep_start=keep_start,
        keep_end=keep_end,
        proxy_grid_size=int(max(0, proxy_grid_size if use_proxy_eval else 0)),
        proxy_stride=int(max(1, proxy_stride)),
    )


def run_chunk_inference_const(
    model,
    chunk,
    chunk_norm,
    keep_start,
    keep_end,
    max_keep,
    min_keep,
    cr_cap_ratio,
    args,
    global_stats,
    device,
):
    """Run deterministic chunk inference with forward_policy_const and explicit valid-action masking."""
    env = GFlowNetTrajectoryEnv(
        trajectory=chunk_norm,
        raw_trajectory=chunk,
        alpha=0.5,
        beta=0.5,
        f1_threshold=args.f1_threshold,
        target_compression=cr_cap_ratio,
        device=device,
        global_stats=global_stats,
        local_stats=None,
        keep_start=keep_start,
        keep_end=keep_end
    )

    termination_reason = 'unknown'

    with torch.inference_mode():
        traj_tensor = torch.as_tensor(chunk_norm, dtype=torch.float32, device=device).unsqueeze(0)
        traj_emb = model.traj_encoder(traj_tensor)
        valid_lens = torch.tensor([len(chunk)], dtype=torch.long, device=device)

        done = False
        step = 0
        state_tensor = env.get_state_tensor()

        while not done and step < len(chunk) * 2:
            step += 1
            forced_reason = None
            current_kept = int(env.mask.sum())
            current_min_keep = 2 if min_keep is None else max(2, int(min_keep))
            if max_keep is not None and current_kept >= max_keep:
                action_idx = len(chunk)
                forced_reason = 'max_keep'
            else:
                s_t = torch.as_tensor(state_tensor, dtype=torch.float32, device=device).unsqueeze(0)
                valid_actions = env.get_valid_actions()
                allow_stop = current_kept >= current_min_keep
                valid_actions[-1] = bool(allow_stop)
                pool_size = int(getattr(args, 'action_pool_size', 0))
                use_pool = (pool_size > 0) and (int(valid_actions[:-1].sum()) > max(2, pool_size))

                if use_pool:
                    pool_idx = _build_action_pool_from_env(
                        env,
                        valid_actions,
                        pool_size=pool_size,
                        explore_ratio=float(getattr(args, 'action_pool_explore_ratio', 0.2)),
                    )
                    valid_small = np.zeros(len(chunk) + 1, dtype=bool)
                    if len(pool_idx) > 0:
                        valid_small[pool_idx] = True
                    valid_small[-1] = bool(allow_stop)
                    if int(valid_small.sum()) <= 1:
                        action_idx = len(chunk)
                        forced_reason = 'no_valid'
                    else:
                        cand_t = torch.as_tensor(pool_idx, dtype=torch.long, device=device).unsqueeze(0)
                        logits_small = model.forward_policy_const_candidates(traj_emb, s_t, cand_t, valid_lens)
                        if not allow_stop:
                            logits_small[:, -1] = -1e9
                        logits_small = torch.clamp(logits_small, min=-100.0, max=100.0)
                        local_idx = int(torch.argmax(logits_small[0]).item())
                        action_idx = len(chunk) if local_idx >= len(pool_idx) else int(pool_idx[local_idx])
                        if action_idx == len(chunk):
                            forced_reason = 'model_stop'
                else:
                    logits = model.forward_policy_const(traj_emb, s_t, valid_lens)
                    invalid_actions = torch.as_tensor(~valid_actions, dtype=torch.bool, device=device).unsqueeze(0)
                    logits = logits.masked_fill(invalid_actions, -1e9)
                    logits = torch.clamp(logits, min=-100.0, max=100.0)
                    if int(valid_actions.sum()) <= 1:
                        action_idx = len(chunk)
                        forced_reason = 'no_valid'
                    else:
                        action_idx = int(torch.argmax(logits[0]).item())
                        if action_idx == len(chunk):
                            forced_reason = 'model_stop'

                if int(valid_actions.sum()) <= 1 and forced_reason is None:
                    action_idx = len(chunk)
                    forced_reason = 'no_valid'

            env_action = action_idx if action_idx < len(chunk) else -1
            if env_action == -1:
                _, _, done, _ = env.step(-1)
            else:
                _, _, done, _ = env.lightweight_step(env_action)
            if done:
                termination_reason = forced_reason or 'model_stop'
            state_tensor = env.get_state_tensor()

        if not done:
            # Safety termination (rare): ensure terminal F1/CR are available.
            _, _, _, _ = env.step(-1)
            termination_reason = 'step_limit'

    selected_indices = np.where(env.mask)[0].tolist()
    info = {
        'num_points': len(selected_indices),
        'f1': float(getattr(env, '_last_f1', 0.0)),
        'sparsity': float(getattr(env, '_last_sparsity', 0.0)),
        'cr': len(selected_indices) / max(1, len(chunk)),
        'termination_reason': termination_reason,
    }
    env.close()
    return selected_indices, info


def _sample_chunk_candidate(
    model,
    chunk,
    chunk_norm,
    keep_start,
    keep_end,
    max_keep,
    min_keep,
    cr_cap_ratio,
    tau,
    global_stats,
    device,
    temperature: float = 0.7,
    dual_lambda: float = 1.0,
    action_pool_size: int = 0,
    action_pool_explore_ratio: float = 0.2,
    use_proxy_eval: bool = False,
    proxy_grid_size: int = 24,
    proxy_stride: int = 4,
    shared_env=None,
    traj_emb=None,
    valid_lens=None,
    profile_phase: Optional[str] = None,
    profile_major_module: Optional[str] = None,
):
    env = shared_env
    owns_env = env is None
    if env is None:
        env = _create_chunk_eval_env(
            chunk=chunk,
            chunk_norm=chunk_norm,
            keep_start=keep_start,
            keep_end=keep_end,
            tau=tau,
            cr_cap_ratio=cr_cap_ratio,
            global_stats=global_stats,
            device=device,
            use_proxy_eval=use_proxy_eval,
            proxy_grid_size=proxy_grid_size,
            proxy_stride=proxy_stride,
        )
    else:
        env.reset(keep_start=keep_start, keep_end=keep_end)

    n = len(chunk)
    sample_pool_elapsed = 0.0
    sample_rollout_elapsed = 0.0
    sample_eval_elapsed = 0.0
    try:
        with torch.inference_mode():
            if traj_emb is None or valid_lens is None:
                traj_tensor = torch.as_tensor(chunk_norm, dtype=torch.float32, device=device).unsqueeze(0)
                traj_emb = model.traj_encoder(traj_tensor)
                valid_lens = torch.tensor([n], dtype=torch.long, device=device)

            state_tensor = env.get_state_tensor()
            done = False
            step = 0
            actions = []
            while not done and step < n * 2:
                step_started = time.perf_counter()
                step_pool_elapsed = 0.0
                step += 1
                valid_actions = env.get_valid_actions().astype(bool, copy=True)
                cur_kept = int(env.mask.sum())
                cur_min_keep = 2 if min_keep is None else max(2, int(min_keep))
                if max_keep is not None and cur_kept >= int(max_keep):
                    valid_actions = np.zeros(n + 1, dtype=bool)
                    valid_actions[-1] = True
                    action_idx = n
                else:
                    allow_stop = cur_kept >= cur_min_keep
                    valid_actions[-1] = bool(allow_stop)
                    s_t = torch.as_tensor(state_tensor, dtype=torch.float32, device=device).unsqueeze(0)
                    use_pool = (int(action_pool_size) > 0) and (int(valid_actions[:-1].sum()) > max(2, int(action_pool_size)))
                    if use_pool:
                        pool_started = time.perf_counter()
                        pool_idx = _build_action_pool_from_env(
                            env,
                            valid_actions,
                            pool_size=int(action_pool_size),
                            explore_ratio=float(action_pool_explore_ratio),
                        )
                        pool_elapsed = time.perf_counter() - pool_started
                        step_pool_elapsed += pool_elapsed
                        sample_pool_elapsed += pool_elapsed
                        valid_small = np.zeros(n + 1, dtype=bool)
                        if len(pool_idx) > 0:
                            valid_small[pool_idx] = True
                        valid_small[-1] = bool(allow_stop)
                        if int(valid_small.sum()) <= 1:
                            action_idx = n
                        elif len(pool_idx) == 1 and (not allow_stop):
                            action_idx = int(pool_idx[0])
                        else:
                            cand_t = torch.as_tensor(pool_idx, dtype=torch.long, device=device).unsqueeze(0)
                            logits_small = model.forward_policy_const_candidates(traj_emb, s_t, cand_t, valid_lens)
                            if not allow_stop:
                                logits_small[:, -1] = -1e9
                            logits_small = torch.clamp(logits_small, min=-100.0, max=100.0)
                            dist = torch.distributions.Categorical(logits=logits_small / max(1e-4, float(temperature)))
                            local_idx = int(dist.sample().item())
                            action_idx = n if local_idx >= len(pool_idx) else int(pool_idx[local_idx])
                    else:
                        logits = model.forward_policy_const(traj_emb, s_t, valid_lens)
                        invalid_actions = torch.as_tensor(~valid_actions, dtype=torch.bool, device=device).unsqueeze(0)
                        logits = logits.masked_fill(invalid_actions, -1e9)
                        logits = torch.clamp(logits, min=-100.0, max=100.0)
                        if int(valid_actions.sum()) <= 1:
                            action_idx = n
                        else:
                            dist = torch.distributions.Categorical(logits=logits / max(1e-4, float(temperature)))
                            action_idx = int(dist.sample().item())

                actions.append(int(action_idx))
                env_action = action_idx if action_idx < n else -1
                if env_action == -1:
                    _, _, done, _ = env.step(-1)
                else:
                    _, _, done, _ = env.lightweight_step(env_action)
                state_tensor = env.get_state_tensor()
                sample_rollout_elapsed += max(0.0, time.perf_counter() - step_started - step_pool_elapsed)

            if not done:
                finalize_started = time.perf_counter()
                _, _, done, _ = env.step(-1)
                actions.append(n)
                sample_rollout_elapsed += time.perf_counter() - finalize_started

        indices = np.where(env.mask)[0].tolist()
        eval_started = time.perf_counter()
        if use_proxy_eval:
            eval_info = env.evaluate_indices_proxy(indices, tau=tau, stride=int(max(1, proxy_stride)))
        else:
            eval_info = env.evaluate_indices(indices, tau=tau)
        violation = max(0.0, float(tau) - float(eval_info["f1_lb"]))
        reward_dual = float(np.exp(-(float(eval_info["cr"]) + float(dual_lambda) * violation)) + 1e-9)
        sample_eval_elapsed += time.perf_counter() - eval_started
        _add_optional_profile_time(profile_phase, profile_major_module, "per_chunk_prs_sample_pool", sample_pool_elapsed)
        _add_optional_profile_time(profile_phase, profile_major_module, "per_chunk_prs_sample_rollout", sample_rollout_elapsed)
        _add_optional_profile_time(profile_phase, profile_major_module, "per_chunk_prs_sample_eval", sample_eval_elapsed)
        return {
            "indices": sorted(set(int(i) for i in indices)),
            "actions": actions,
            "f1_lb": float(eval_info["f1_lb"]),
            "cr": float(eval_info["cr"]),
            "feasible": bool(eval_info["feasible"]),
            "reward_dual": reward_dual,
            "proxy_eval": bool(use_proxy_eval),
        }
    finally:
        if owns_env:
            env.close()


def _candidate_index_key(candidate: Dict[str, object]) -> Tuple[int, ...]:
    return tuple(sorted(set(int(i) for i in candidate.get("indices", []))))


def _candidate_is_safe_feasible(
    candidate: Dict[str, object],
    tau: float,
    safe_margin: float,
) -> bool:
    f1_lb = float(candidate.get("f1_lb", 0.0))
    feasible = bool(candidate.get("feasible", f1_lb >= tau - 1e-12))
    return feasible and (f1_lb >= float(tau) + float(safe_margin) - 1e-12)


def _candidate_priority_sort_key(
    candidate: Dict[str, object],
    tau: float,
    safe_margin: float,
) -> Tuple[float, ...]:
    f1_lb = float(candidate.get("f1_lb", 0.0))
    cr = float(candidate.get("cr", 1.0))
    reward_dual = float(candidate.get("reward_dual", 0.0))
    feasible = bool(candidate.get("feasible", f1_lb >= tau - 1e-12))
    if _candidate_is_safe_feasible(candidate, tau=tau, safe_margin=safe_margin):
        return (0.0, cr, -f1_lb, -reward_dual)
    if feasible:
        return (1.0, cr, -f1_lb, -reward_dual)
    return (2.0, -f1_lb, -reward_dual, cr)


def _select_best_prs_candidate(
    candidates: List[Dict[str, object]],
    tau: float,
    safe_margin: float,
) -> Optional[Dict[str, object]]:
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda cand: _candidate_priority_sort_key(cand, tau=tau, safe_margin=safe_margin),
    )
    return ranked[0]


def _build_prs_exact_shortlist(
    candidates: List[Dict[str, object]],
    tau: float,
    safe_margin: float,
    lowcr_topk: int,
    reward_topk: int,
    f1_topk: int,
) -> List[Dict[str, object]]:
    if not candidates:
        return []

    shortlist: List[Dict[str, object]] = []
    seen = set()

    def add_ranked(items: List[Dict[str, object]], limit: int) -> None:
        for item in items[: max(0, int(limit))]:
            c_key = _candidate_index_key(item)
            if c_key in seen:
                continue
            seen.add(c_key)
            shortlist.append(item)

    lowcr_ranked = sorted(
        candidates,
        key=lambda cand: _candidate_priority_sort_key(cand, tau=tau, safe_margin=safe_margin),
    )
    reward_ranked = sorted(
        candidates,
        key=lambda cand: (
            float(cand.get("reward_dual", 0.0)),
            1.0 if _candidate_is_safe_feasible(cand, tau=tau, safe_margin=safe_margin) else 0.0,
            float(cand.get("f1_lb", 0.0)),
            -float(cand.get("cr", 1.0)),
        ),
        reverse=True,
    )
    f1_ranked = sorted(
        candidates,
        key=lambda cand: (
            float(cand.get("f1_lb", 0.0)),
            1.0 if _candidate_is_safe_feasible(cand, tau=tau, safe_margin=safe_margin) else 0.0,
            float(cand.get("reward_dual", 0.0)),
            -float(cand.get("cr", 1.0)),
        ),
        reverse=True,
    )

    add_ranked(lowcr_ranked, lowcr_topk)
    add_ranked(reward_ranked, reward_topk)
    add_ranked(f1_ranked, f1_topk)
    return shortlist if shortlist else candidates[:1]


def run_chunk_inference_prs(
    model,
    chunk,
    chunk_norm,
    keep_start,
    keep_end,
    max_keep,
    min_keep,
    cr_cap_ratio,
    args,
    global_stats,
    device,
    infer_k: Optional[int] = None,
    infer_k_max: Optional[int] = None,
    profile_phase: Optional[str] = None,
    profile_major_module: Optional[str] = None,
):
    tau = float(args.f1_threshold)
    safe_margin = float(getattr(args, 'f1_safe_margin', 0.01))
    k_target = int(max(1, args.infer_k if infer_k is None else infer_k))
    k_max = int(max(k_target, args.infer_k_max if infer_k_max is None else infer_k_max))
    dedup_thr = float(np.clip(args.repair_jaccard_dedup, 0.0, 1.0))
    dual_lambda = float(args.infer_dual_lambda)
    use_multifidelity = not bool(getattr(args, 'multifidelity_disable', False))
    mf_topk_exact = int(max(1, getattr(args, 'multifidelity_topk_exact', 2)))
    mf_proxy_grid_size = int(max(0, getattr(args, 'multifidelity_proxy_grid_size', 24)))
    mf_proxy_stride = int(max(1, getattr(args, 'multifidelity_proxy_stride', 4)))
    prs_exact_lowcr_topk = int(max(0, getattr(args, 'prs_exact_lowcr_topk', 2)))
    prs_exact_reward_topk = int(max(0, getattr(args, 'prs_exact_reward_topk', 2)))
    prs_exact_f1_topk = int(max(0, getattr(args, 'prs_exact_f1_topk', 2)))
    shared_env = _create_chunk_eval_env(
        chunk=chunk,
        chunk_norm=chunk_norm,
        keep_start=keep_start,
        keep_end=keep_end,
        tau=tau,
        cr_cap_ratio=cr_cap_ratio,
        global_stats=global_stats,
        device=device,
        use_proxy_eval=use_multifidelity,
        proxy_grid_size=mf_proxy_grid_size,
        proxy_stride=mf_proxy_stride,
    )
    try:
        with torch.inference_mode():
            traj_tensor = torch.as_tensor(chunk_norm, dtype=torch.float32, device=device).unsqueeze(0)
            traj_emb = model.traj_encoder(traj_tensor)
            valid_lens = torch.tensor([len(chunk)], dtype=torch.long, device=device)

        candidates = []
        attempts = 0
        dedup_elapsed = 0.0
        with profile_scope(
            profile_phase or "noop",
            profile_major_module or "noop",
            "per_chunk_prs_sampling",
        ) if (profile_phase and profile_major_module) else nullcontext():
            while len(candidates) < k_target and attempts < k_max:
                attempts += 1
                cand = _sample_chunk_candidate(
                    model=model,
                    chunk=chunk,
                    chunk_norm=chunk_norm,
                    keep_start=keep_start,
                    keep_end=keep_end,
                    max_keep=max_keep,
                    min_keep=min_keep,
                    cr_cap_ratio=cr_cap_ratio,
                    tau=tau,
                    global_stats=global_stats,
                    device=device,
                    temperature=args.infer_temperature,
                    dual_lambda=dual_lambda,
                    action_pool_size=int(getattr(args, 'action_pool_size', 0)),
                    action_pool_explore_ratio=float(getattr(args, 'action_pool_explore_ratio', 0.2)),
                    use_proxy_eval=use_multifidelity,
                    proxy_grid_size=mf_proxy_grid_size,
                    proxy_stride=mf_proxy_stride,
                    shared_env=shared_env,
                    traj_emb=traj_emb,
                    valid_lens=valid_lens,
                    profile_phase=profile_phase,
                    profile_major_module=profile_major_module,
                )
                dedup_started = time.perf_counter()
                replaced = False
                for i, prev in enumerate(candidates):
                    if jaccard_similarity(prev["indices"], cand["indices"]) > dedup_thr:
                        replaced = True
                        if cand["reward_dual"] > prev["reward_dual"] + 1e-12:
                            candidates[i] = cand
                        break
                if not replaced:
                    candidates.append(cand)
                dedup_elapsed += time.perf_counter() - dedup_started
        _add_optional_profile_time(profile_phase, profile_major_module, "per_chunk_prs_sample_dedup", dedup_elapsed)

        # Multi-fidelity exact reranking for top proxy candidates.
        if use_multifidelity and len(candidates) > 0:
            with profile_scope(
                profile_phase or "noop",
                profile_major_module or "noop",
                "per_chunk_multifidelity_exact",
            ) if (profile_phase and profile_major_module) else nullcontext():
                exact_subset = _build_prs_exact_shortlist(
                    candidates=candidates,
                    tau=tau,
                    safe_margin=safe_margin,
                    lowcr_topk=prs_exact_lowcr_topk,
                    reward_topk=prs_exact_reward_topk,
                    f1_topk=prs_exact_f1_topk,
                )
                if len(exact_subset) < max(1, mf_topk_exact):
                    ranked_idx = sorted(
                        range(len(candidates)),
                        key=lambda j: (
                            float(candidates[j].get("reward_dual", 0.0)),
                            float(candidates[j].get("f1_lb", 0.0)),
                            -float(candidates[j].get("cr", 1.0)),
                        ),
                        reverse=True,
                    )
                    seen_keys = {_candidate_index_key(c) for c in exact_subset}
                    for j in ranked_idx:
                        cand = candidates[j]
                        cand_key = _candidate_index_key(cand)
                        if cand_key in seen_keys:
                            continue
                        exact_subset.append(cand)
                        seen_keys.add(cand_key)
                        if len(exact_subset) >= max(1, mf_topk_exact):
                            break
                for c in exact_subset:
                    exact = shared_env.evaluate_indices(c["indices"], tau=tau)
                    c["f1_lb"] = float(exact["f1_lb"])
                    c["cr"] = float(exact["cr"])
                    c["feasible"] = bool(exact["feasible"])
                    violation = max(0.0, tau - c["f1_lb"])
                    c["reward_dual"] = float(np.exp(-(c["cr"] + dual_lambda * violation)) + 1e-9)
                    c["proxy_eval"] = False
    finally:
        shared_env.close()

    feasible = [c for c in candidates if bool(c.get("feasible", float(c.get("f1_lb", 0.0)) >= tau - 1e-12))]
    best = _select_best_prs_candidate(candidates, tau=tau, safe_margin=safe_margin)
    fallback = (best is None) or (not bool(best.get("feasible", False)))
    if best is None:
        best = {
            "indices": [0, len(chunk) - 1] if len(chunk) > 1 else [0],
            "f1_lb": 0.0,
            "cr": 1.0,
            "reward_dual": 1e-9,
            "feasible": False,
        }
        fallback = True

    best_margin = float(best.get("f1_lb", 0.0)) - tau
    best_safe_feasible = _candidate_is_safe_feasible(best, tau=tau, safe_margin=safe_margin)

    info = {
        "num_points": len(best["indices"]),
        "actions": list(best.get("actions", [])),
        "f1": float(best["f1_lb"]),
        "sparsity": float(best["cr"]),
        "cr": float(best["cr"]),
        "termination_reason": "prs_select",
        "prs_candidate_count": int(len(candidates)),
        "prs_attempts": int(attempts),
        "prs_feasible_count": int(len(feasible)),
        "prs_fallback": bool(fallback),
        "repair_avg_deleted": 0.0,
        "prs_safe_feasible_found": int(sum(1 for c in candidates if _candidate_is_safe_feasible(c, tau=tau, safe_margin=safe_margin))),
        "prs_best_safe_feasible": bool(best_safe_feasible),
        "prs_best_margin": float(best_margin),
    }
    return best["indices"], info


def validate_epoch(
    model,
    val_indices,
    args,
    global_stats,
    device='cuda',
    split_name: str = 'val',
    epoch: Optional[int] = None,
    csv_dir: Optional[str] = None,
    cap_relax_val: float = 1.0,
):
    """Validation with unified const-policy inference, per-trajectory CSV, and termination stats."""
    model.eval()

    total_f1 = 0.0
    total_pts = 0
    total_orig_pts = 0
    count = 0

    per_traj_cr = []
    per_traj_cap_cr = []
    per_traj_f1 = []
    per_traj_rows = []

    chunk_term_counts = {
        'model_stop': 0,
        'max_keep': 0,
        'no_valid': 0,
        'step_limit': 0,
        'other': 0,
    }
    total_chunks = 0
    prs_total_candidates = 0
    prs_total_feasible = 0
    prs_total_fallback = 0
    prs_repair_deleted_sum = 0.0
    prs_safe_feasible_best_count = 0
    prs_best_margin_sum = 0.0

    with torch.inference_mode():
        for i, (idx, bounds) in enumerate(val_indices):
            traj_record = load_preprocessed_trajectory_record(args.traj_path, idx)
            if traj_record is None:
                continue

            traj = traj_record['traj']
            bounds = traj_record['bounds']
            chunk_ranges = traj_record['chunk_ranges']
            qcs_original = traj_record.get('global_qcs_original')
            if qcs_original is None:
                qcs_original = build_global_qcs_original(traj, local_stats=bounds)
                traj_record['global_qcs_original'] = qcs_original

            all_selected_indices = set()
            all_cap_indices = set()
            chunk_count = 0

            for start, end in chunk_ranges:
                chunk = traj[start:end]
                chunk_norm = normalize_trajectory(chunk, stats=global_stats)
                keep_start = (start == 0)
                keep_end = (end == len(traj))
                greedy_cache_key = _make_greedy_cache_key(
                    args.traj_path,
                    idx,
                    start,
                    end,
                    keep_start,
                    keep_end,
                    args,
                )

                greedy_actions, greedy_indices, greedy_max_keep, greedy_cr_cap, _ = _derive_greedy_cap_cached(
                    greedy_cache_key,
                    chunk,
                    chunk_norm,
                    keep_start,
                    keep_end,
                    args,
                    global_stats,
                    device,
                    relax_factor=cap_relax_val,
                )
                del greedy_actions
                min_keep = max(2, int(np.ceil(greedy_max_keep * args.min_keep_ratio)))
                min_keep = min(min_keep, greedy_max_keep)

                if args.legacy_single_path:
                    chunk_indices, chunk_info = run_chunk_inference_const(
                        model=model,
                        chunk=chunk,
                        chunk_norm=chunk_norm,
                        keep_start=keep_start,
                        keep_end=keep_end,
                        max_keep=greedy_max_keep,
                        min_keep=min_keep,
                        cr_cap_ratio=greedy_cr_cap,
                        args=args,
                        global_stats=global_stats,
                        device=device,
                    )
                else:
                    chunk_indices, chunk_info = run_chunk_inference_prs(
                        model=model,
                        chunk=chunk,
                        chunk_norm=chunk_norm,
                        keep_start=keep_start,
                        keep_end=keep_end,
                        max_keep=greedy_max_keep,
                        min_keep=min_keep,
                        cr_cap_ratio=greedy_cr_cap,
                        args=args,
                        global_stats=global_stats,
                        device=device,
                        infer_k=int(getattr(args, 'val_infer_k', args.infer_k)),
                        infer_k_max=int(getattr(args, 'val_infer_k_max', args.infer_k_max)),
                    )

                reason = chunk_info.get('termination_reason', 'other')
                if reason in chunk_term_counts:
                    chunk_term_counts[reason] += 1
                else:
                    chunk_term_counts['other'] += 1
                total_chunks += 1
                chunk_count += 1
                prs_total_candidates += int(chunk_info.get('prs_candidate_count', 0))
                prs_total_feasible += int(chunk_info.get('prs_feasible_count', 0))
                prs_total_fallback += int(1 if chunk_info.get('prs_fallback', False) else 0)
                prs_safe_feasible_best_count += int(1 if chunk_info.get('prs_best_safe_feasible', False) else 0)
                prs_best_margin_sum += float(chunk_info.get('prs_best_margin', 0.0))

                all_selected_indices.update(j + start for j in chunk_indices)
                all_cap_indices.update(j + start for j in greedy_indices)

            unique_indices = sorted(idx_g for idx_g in all_selected_indices if 0 <= idx_g < len(traj))
            unique_cap_indices = sorted(idx_g for idx_g in all_cap_indices if 0 <= idx_g < len(traj))
            with profile_scope("train", "train_validation_output", "per_traj_repair"):
                unique_indices, traj_repair_deleted, repair_stats = repair_combined_trajectory(
                    raw_trajectory=traj,
                    selected_indices=unique_indices,
                    tau=float(args.f1_threshold),
                    args=args,
                    bounds=bounds,
                    qcs_original=qcs_original,
                )
            _add_profile_counters(repair_stats)
            prs_repair_deleted_sum += float(traj_repair_deleted)

            f1 = calculate_global_f1(traj, unique_indices, bounds, qcs_original=qcs_original)
            pts = len(unique_indices)
            cr = pts / max(1, len(traj))
            cap_cr = len(unique_cap_indices) / max(1, len(traj))

            total_f1 += f1
            total_pts += pts
            total_orig_pts += len(traj)
            count += 1

            per_traj_f1.append(f1)
            per_traj_cr.append(cr)
            per_traj_cap_cr.append(cap_cr)
            per_traj_rows.append({
                'idx': int(idx),
                'N': int(len(traj)),
                'F1': float(f1),
                'CR': float(cr),
                'cap_CR': float(cap_cr),
                'CR-gap': float(cr - cap_cr),
                'chunk_count': int(chunk_count),
            })

    avg_f1 = total_f1 / max(1, count)
    weighted_cr = total_pts / max(1, total_orig_pts)
    traj_mean_cr = float(np.mean(per_traj_cr)) if per_traj_cr else 0.0

    f1_hard_ok_rate = float(np.mean([f >= args.f1_threshold for f in per_traj_f1])) if per_traj_f1 else 0.0
    cr_p50 = float(np.percentile(per_traj_cr, 50)) if per_traj_cr else 0.0
    cr_p90 = float(np.percentile(per_traj_cr, 90)) if per_traj_cr else 0.0
    cr_p95 = float(np.percentile(per_traj_cr, 95)) if per_traj_cr else 0.0
    avg_cap_cr = float(np.mean(per_traj_cap_cr)) if per_traj_cap_cr else 0.0
    avg_gap = float(np.mean(np.array(per_traj_cr) - np.array(per_traj_cap_cr))) if per_traj_cr else 0.0
    model_stop_ratio = chunk_term_counts['model_stop'] / max(1, total_chunks)
    max_keep_ratio = chunk_term_counts['max_keep'] / max(1, total_chunks)
    prs_avg_candidates = prs_total_candidates / max(1, total_chunks)
    prs_avg_feasible = prs_total_feasible / max(1, total_chunks)
    prs_fallback_rate = prs_total_fallback / max(1, total_chunks)
    prs_repair_avg_deleted = prs_repair_deleted_sum / max(1, count)
    safe_feasible_rate = prs_safe_feasible_best_count / max(1, total_chunks)
    best_margin_avg = prs_best_margin_sum / max(1, total_chunks)

    csv_path = ''
    if csv_dir is not None and epoch is not None:
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f'{split_name}_epoch{epoch+1:04d}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['idx', 'N', 'F1', 'CR', 'cap_CR', 'CR-gap', 'chunk_count']
            )
            writer.writeheader()
            writer.writerows(per_traj_rows)

    return {
        'avg_f1': avg_f1,
        'avg_cr': weighted_cr,
        'weighted_cr': weighted_cr,
        'traj_mean_cr': traj_mean_cr,
        'avg_pts': (total_pts / max(1, count)),
        'f1_hard_ok_rate': f1_hard_ok_rate,
        'cr_p50': cr_p50,
        'cr_p90': cr_p90,
        'cr_p95': cr_p95,
        'avg_cap_cr': avg_cap_cr,
        'cr_vs_cap_gap': avg_gap,
        'count': count,
        'csv_path': csv_path,
        'chunk_count': total_chunks,
        'chunk_term_counts': chunk_term_counts,
        'chunk_term_model_stop_ratio': model_stop_ratio,
        'chunk_term_max_keep_ratio': max_keep_ratio,
        'prs_avg_candidates': prs_avg_candidates,
        'prs_avg_feasible': prs_avg_feasible,
        'prs_fallback_rate': prs_fallback_rate,
        'prs_repair_avg_deleted': prs_repair_avg_deleted,
        'safe_feasible_count': int(prs_safe_feasible_best_count),
        'safe_feasible_rate': safe_feasible_rate,
        'best_margin_avg': best_margin_avg,
    }


def train_one_epoch(
    epoch,
    trainer,
    indices,
    args,
    global_stats,
    device,
    cap_relax_train: float = 1.0,
    temperature_floor: float = 0.3,
    hard_chunk_cache: Optional[set] = None,
    phase2_lowcr_active: bool = False,
    phase2_cap_scale: float = 1.0,
    summary_label: Optional[str] = None,
):
    """Chunk-bucketed training loop for a single epoch."""
    trainer.epoch = epoch

    if args.strict_f1:
        current_f1_threshold = args.f1_threshold
    else:
        current_f1_threshold = min(args.f1_threshold, 0.8 + epoch * 0.01)
    label = summary_label or f"Epoch {epoch+1}"
    print(f"[{label}] Current F1 Threshold: {current_f1_threshold:.2f}")

    if hard_chunk_cache is None:
        hard_chunk_cache = set()

    bucket_bounds = list(getattr(args, 'length_bucket_bounds', [256, 512, 1000]))
    if len(bucket_bounds) == 0:
        bucket_bounds = [256, 512, 1000]
    bucket_batch_size = max(1, int(getattr(args, 'bucket_batch_size', 8)))
    base_num_samples = max(1, int(getattr(args, 'num_samples_train', 2)))
    hard_num_samples = max(base_num_samples, int(getattr(args, 'hard_num_samples_train', 4)))
    hard_f1_margin = float(getattr(args, 'hard_f1_margin', 0.01))
    base_frontier_top_m = max(1, int(getattr(args, 'frontier_top_m', 4)))

    epoch_start = time.time()
    epoch_success = 0
    epoch_chunks = 0
    epoch_traj_count = 0
    epoch_f1_sum = 0.0
    epoch_cr_sum = 0.0
    epoch_cap_cr_sum = 0.0
    epoch_loss_f_sum = 0.0
    epoch_loss_b_sum = 0.0
    epoch_loss_expert_forward_sum = 0.0
    epoch_expert_forward_total = 0
    epoch_expert_forward_skipped = 0
    epoch_dual_sum = 0.0
    epoch_dual_count = 0
    epoch_dual_values = []
    epoch_frontier_size_sum = 0.0
    epoch_hard_chunk_used = 0
    epoch_hard_promoted = 0
    epoch_hard_cleared = 0
    epoch_actual_samples_sum = 0.0
    epoch_exact_evals_sum = 0.0

    epoch_indices = list(indices)
    np.random.shuffle(epoch_indices)
    epoch_temperature = max(float(temperature_floor), args.temperature * (0.995 ** epoch) * 1.5)

    # Stage-1: build chunk tasks and cache trajectories once.
    traj_records = {}
    traj_order = []
    bucket_tasks = [[] for _ in range(len(bucket_bounds))]
    bucket_task_counts = [0 for _ in range(len(bucket_bounds))]

    with profile_scope("train", "train_loop", "epoch_build_tasks"):
        for idx, _ in epoch_indices:
            traj_record_cached = load_preprocessed_trajectory_record(args.traj_path, idx)
            if traj_record_cached is None:
                continue
            traj = traj_record_cached['traj']
            bounds = traj_record_cached['bounds']
            chunk_ranges = traj_record_cached['chunk_ranges']
            if len(chunk_ranges) == 0:
                continue

            idx_i = int(idx)
            traj_order.append(idx_i)
            traj_records[idx_i] = {
                'traj': traj,
                'bounds': bounds,
                'global_qcs_original': traj_record_cached.get('global_qcs_original'),
                'all_selected_indices': set(),
                'all_cap_indices': set(),
                'loss_f_sum': 0.0,
                'loss_b_sum': 0.0,
                'train_chunk_count': 0,
            }

            for chunk_idx, (start_in_traj, end_in_traj) in enumerate(chunk_ranges):
                n_points = int(end_in_traj - start_in_traj)
                b_id = _chunk_bucket_id(n_points, bucket_bounds)
                bucket_tasks[b_id].append({
                    'idx': idx_i,
                    'chunk_idx': int(chunk_idx),
                    'start': int(start_in_traj),
                    'end': int(end_in_traj),
                    'n_points': n_points,
                })
                bucket_task_counts[b_id] += 1

    # Stage-2: train chunk tasks by length bucket, batched by bucket only.
    for b_id, tasks in enumerate(bucket_tasks):
        if len(tasks) == 0:
            continue
        np.random.shuffle(tasks)
        lower = 3 if b_id == 0 else (bucket_bounds[b_id - 1] + 1)
        upper = bucket_bounds[b_id]
        print(
            f"  [Bucket {b_id+1}/{len(bucket_bounds)}] "
            f"len=[{lower},{upper}] chunks={len(tasks)} batch={bucket_batch_size}"
        )

        for k in range(0, len(tasks), bucket_batch_size):
            batch_tasks = tasks[k:k + bucket_batch_size]
            for task in batch_tasks:
                idx_i = int(task['idx'])
                start_in_traj = int(task['start'])
                end_in_traj = int(task['end'])
                chunk_idx = int(task['chunk_idx'])

                record = traj_records.get(idx_i)
                if record is None:
                    continue
                traj = record['traj']
                if traj is None:
                    continue
                if not (0 <= start_in_traj < end_in_traj <= len(traj)):
                    continue

                chunk = traj[start_in_traj:end_in_traj]
                if len(chunk) < 3:
                    continue
                chunk_norm = normalize_trajectory(chunk, stats=global_stats)

                keep_start = (start_in_traj == 0)
                keep_end = (end_in_traj == len(traj))
                greedy_cache_key = _make_greedy_cache_key(
                    args.traj_path,
                    idx_i,
                    start_in_traj,
                    end_in_traj,
                    keep_start,
                    keep_end,
                    args,
                )

                with profile_scope("train", "train_loop", "epoch_greedy_cap"):
                    greedy_actions, greedy_indices, greedy_max_keep, greedy_cr_cap, chunk_bounds = _derive_greedy_cap_cached(
                        greedy_cache_key,
                        chunk,
                        chunk_norm,
                        keep_start,
                        keep_end,
                        args,
                        global_stats,
                        device,
                        relax_factor=cap_relax_train,
                    )
                hard_key = (idx_i, start_in_traj, end_in_traj)
                use_hard_sampling = hard_key in hard_chunk_cache
                num_samples_train = hard_num_samples if use_hard_sampling else base_num_samples
                if use_hard_sampling:
                    epoch_hard_chunk_used += 1

                train_max_keep = int(greedy_max_keep)
                train_cr_cap = float(greedy_cr_cap)
                dynamic_cap_expand_ratio = 1.1
                if phase2_lowcr_active and (not use_hard_sampling):
                    tightened_cap = float(np.clip(train_cr_cap * phase2_cap_scale, 1e-4, train_cr_cap))
                    tightened_max_keep = int(np.ceil(len(chunk) * tightened_cap))
                    train_max_keep = max(2, min(train_max_keep, tightened_max_keep))
                    train_cr_cap = min(tightened_cap, train_max_keep / max(1, len(chunk)))
                    dynamic_cap_expand_ratio = 1.0

                min_keep = max(2, int(np.ceil(train_max_keep * args.min_keep_ratio)))
                min_keep = min(min_keep, train_max_keep)

                try:
                    trainer.f1_threshold = current_f1_threshold
                    with profile_scope("train", "train_loop", "epoch_chunk_train_total"):
                        if args.legacy_single_path:
                            metrics = trainer.train_single_trajectory(
                                trajectory=chunk_norm,
                                raw_trajectory=chunk,
                                local_stats=chunk_bounds,
                                temperature=epoch_temperature,
                                greedy_actions=greedy_actions,
                                keep_start=keep_start,
                                keep_end=keep_end,
                                max_keep=train_max_keep,
                                min_keep=min_keep,
                                cr_cap_ratio=train_cr_cap,
                                expert_forward_weight=args.expert_forward_weight,
                                expert_bc_ratio_cap=args.expert_bc_ratio_cap,
                            )
                        else:
                            metrics = trainer.train_single_trajectory_distributional(
                                trajectory=chunk_norm,
                                raw_trajectory=chunk,
                                local_stats=chunk_bounds,
                                temperature=epoch_temperature,
                                greedy_actions=greedy_actions,
                                keep_start=keep_start,
                                keep_end=keep_end,
                                max_keep=train_max_keep,
                                min_keep=min_keep,
                                cr_cap_ratio=train_cr_cap,
                                chunk_key=(idx_i, start_in_traj, end_in_traj, f"{current_f1_threshold:.2f}"),
                                num_samples_train=num_samples_train,
                                frontier_top_m=base_frontier_top_m,
                                tau=current_f1_threshold,
                                dynamic_cap_expand_ratio=dynamic_cap_expand_ratio,
                                expert_forward_weight=args.distributional_expert_forward_weight,
                            )
                except Exception as e:
                    print(f"[Warn] Failed to train trajectory {idx_i} chunk {chunk_idx}: {e}")
                    continue

                if not args.legacy_single_path:
                    cur_f1 = float(metrics.get('f1', 0.0))
                    cur_success = bool(metrics.get('success', False))
                    still_hard = (not cur_success) or (cur_f1 < (current_f1_threshold + hard_f1_margin))
                    if still_hard:
                        if hard_key not in hard_chunk_cache:
                            epoch_hard_promoted += 1
                        hard_chunk_cache.add(hard_key)
                    else:
                        if hard_key in hard_chunk_cache:
                            hard_chunk_cache.remove(hard_key)
                            epoch_hard_cleared += 1

                chunk_indices = metrics.get('indices', [])
                valid_chunk_indices = [idx_c for idx_c in chunk_indices if 0 <= idx_c < len(chunk)]
                record['all_selected_indices'].update(idx_c + start_in_traj for idx_c in valid_chunk_indices)
                record['all_cap_indices'].update(idx_c + start_in_traj for idx_c in greedy_indices)
                record['loss_f_sum'] += float(metrics.get('loss_forward', 0.0))
                record['loss_b_sum'] += float(metrics.get('loss_backward', 0.0))
                record['train_chunk_count'] += 1

                epoch_chunks += 1
                epoch_loss_f_sum += metrics.get('loss_forward', 0.0)
                epoch_loss_b_sum += metrics.get('loss_backward', 0.0)
                epoch_loss_expert_forward_sum += metrics.get('loss_expert_forward', 0.0)
                if 'expert_forward_skipped' in metrics:
                    epoch_expert_forward_total += 1
                    if bool(metrics.get('expert_forward_skipped', False)):
                        epoch_expert_forward_skipped += 1
                if 'dual_lambda' in metrics:
                    epoch_dual_sum += float(metrics.get('dual_lambda', 0.0))
                    epoch_dual_count += 1
                    epoch_dual_values.append(float(metrics.get('dual_lambda', 0.0)))
                if 'frontier_mean_size' in metrics:
                    epoch_frontier_size_sum += float(metrics.get('frontier_mean_size', 0.0))
                if 'actual_samples' in metrics:
                    epoch_actual_samples_sum += float(metrics.get('actual_samples', 0.0))
                if 'exact_evals' in metrics:
                    epoch_exact_evals_sum += float(metrics.get('exact_evals', 0.0))

    # Stage-3: aggregate trajectory-level metrics.
    with profile_scope("train", "train_loop", "epoch_aggregate_metrics"):
        for i, idx_i in enumerate(traj_order):
            record = traj_records.get(idx_i)
            if record is None:
                continue
            traj = record['traj']
            bounds = record['bounds']
            if traj is None or len(traj) < 3:
                continue
            qcs_original = record.get('global_qcs_original')
            if qcs_original is None:
                qcs_original = build_global_qcs_original(traj, local_stats=bounds)
                record['global_qcs_original'] = qcs_original

            unique_indices = sorted(idx_g for idx_g in record['all_selected_indices'] if 0 <= idx_g < len(traj))
            unique_cap_indices = sorted(idx_g for idx_g in record['all_cap_indices'] if 0 <= idx_g < len(traj))
            traj_chunk_count = max(1, int(record.get('train_chunk_count', 0)))
            traj_loss_f_avg = float(record.get('loss_f_sum', 0.0)) / traj_chunk_count
            traj_loss_b_avg = float(record.get('loss_b_sum', 0.0)) / traj_chunk_count

            global_f1 = calculate_global_f1(traj, unique_indices, bounds, qcs_original=qcs_original)
            traj_cr = len(unique_indices) / max(1, len(traj))
            cap_cr = len(unique_cap_indices) / max(1, len(traj))

            epoch_f1_sum += global_f1
            epoch_cr_sum += traj_cr
            epoch_cap_cr_sum += cap_cr
            epoch_traj_count += 1

            if args.strict_f1:
                success = (global_f1 >= args.f1_threshold) and (traj_cr <= cap_cr + 1e-9)
            else:
                success = global_f1 >= current_f1_threshold

            if success:
                epoch_success += 1

            if i % args.log_every == 0:
                print(
                    f"  [{i+1}/{len(traj_order)}] "
                    f"N={len(traj)} "
                    f"F1={global_f1:.3f} "
                    f"CR={traj_cr:.3f} (cap={cap_cr:.3f}) "
                    f"Pts={len(unique_indices)} "
                    f"Loss_F={traj_loss_f_avg:.4f} "
                    f"Loss_B={traj_loss_b_avg:.4f} "
                    f"{'OK' if success else ''}"
                )

            record['traj'] = None

    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    epoch_time = time.time() - epoch_start
    avg_f1 = epoch_f1_sum / max(1, epoch_traj_count)
    success_rate = epoch_success / max(1, epoch_traj_count)
    avg_cr = epoch_cr_sum / max(1, epoch_traj_count)
    avg_cap_cr = epoch_cap_cr_sum / max(1, epoch_traj_count)
    expert_forward_skipped_ratio = epoch_expert_forward_skipped / max(1, epoch_expert_forward_total)
    avg_dual_lambda = epoch_dual_sum / max(1, epoch_dual_count)
    dual_p50 = float(np.percentile(epoch_dual_values, 50)) if epoch_dual_values else 0.0
    dual_p90 = float(np.percentile(epoch_dual_values, 90)) if epoch_dual_values else 0.0
    dual_max = float(np.max(epoch_dual_values)) if epoch_dual_values else 0.0
    avg_frontier_size = epoch_frontier_size_sum / max(1, epoch_dual_count)
    avg_actual_samples = epoch_actual_samples_sum / max(1, epoch_chunks)
    avg_exact_evals = epoch_exact_evals_sum / max(1, epoch_chunks)

    print(
        f"\n[{label} Summary] "
        f"Time: {epoch_time:.1f}s | "
        f"Avg F1: {avg_f1:.4f} | "
        f"Avg CR: {avg_cr:.4f} (cap={avg_cap_cr:.4f}) | "
        f"ExpertSkip: {expert_forward_skipped_ratio*100:.1f}% | "
        f"Dual(avg/p50/p90/max): {avg_dual_lambda:.3f}/{dual_p50:.3f}/{dual_p90:.3f}/{dual_max:.3f} | "
        f"Frontier: {avg_frontier_size:.1f} | "
        f"AnytimeSamples: {avg_actual_samples:.2f} | "
        f"MF-Exact: {avg_exact_evals:.2f} | "
        f"Success Rate: {success_rate*100:.1f}% | "
        f"Loss F: {epoch_loss_f_sum:.2f} | "
        f"Loss B: {epoch_loss_b_sum:.2f} | "
        f"Loss ExpertF: {epoch_loss_expert_forward_sum:.2f}"
    )
    print(
        f"             Buckets: {bucket_task_counts} | "
        f"HardSamples(used/promoted/cleared/cache): "
        f"{epoch_hard_chunk_used}/{epoch_hard_promoted}/{epoch_hard_cleared}/{len(hard_chunk_cache)}"
    )
    if phase2_lowcr_active:
        print(
            f"             Phase2LowCR(active/scale): on/{phase2_cap_scale:.4f} "
            f"(non-hard chunks only)"
        )

    return {
        'success_rate': success_rate,
        'avg_f1': avg_f1,
        'avg_cr': avg_cr,
        'avg_cap_cr': avg_cap_cr,
        'epoch_trajs': epoch_traj_count,
        'epoch_chunks': epoch_chunks,
        'epoch_success': epoch_success,
        'expert_forward_skipped_ratio': expert_forward_skipped_ratio,
        'expert_forward_skipped': epoch_expert_forward_skipped,
        'expert_forward_total': epoch_expert_forward_total,
        'loss_expert_forward': epoch_loss_expert_forward_sum,
        'avg_dual_lambda': avg_dual_lambda,
        'dual_lambda_p50': dual_p50,
        'dual_lambda_p90': dual_p90,
        'dual_lambda_max': dual_max,
        'avg_frontier_size': avg_frontier_size,
        'avg_actual_samples': avg_actual_samples,
        'avg_exact_evals': avg_exact_evals,
        'hard_chunk_used': epoch_hard_chunk_used,
        'hard_chunk_promoted': epoch_hard_promoted,
        'hard_chunk_cleared': epoch_hard_cleared,
        'hard_cache_size': len(hard_chunk_cache),
        'bucket_task_counts': bucket_task_counts,
    }


def main():
    parser = argparse.ArgumentParser(description='GFlowNet streaming training')
    parser.add_argument('--traj_path', type=str, default='TrajData/Geolife_out', help='Trajectory path')
    parser.add_argument('--model_path', type=str, default='checkpoints/gflownet_streaming.pt', help='Model save path')
    parser.add_argument('--start_idx', type=int, default=0, help='Train start index')
    parser.add_argument('--end_idx', type=int, default=3000, help='Train end index')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dim')
    parser.add_argument('--lr', type=float, default=1e-3, help='Forward LR')
    parser.add_argument('--lr_backward', type=float, default=1e-4, help='Backward LR')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--val_start_idx', type=int, default=None, help='Validation start index')
    parser.add_argument('--val_end_idx', type=int, default=None, help='Validation end index')
    parser.add_argument('--val2_start_idx', type=int, default=None, help='Validation-2 start index (separate split)')
    parser.add_argument('--val2_end_idx', type=int, default=None, help='Validation-2 end index (separate split)')
    parser.add_argument('--f1_threshold', type=float, default=0.9, help='Hard F1 threshold')
    parser.add_argument('--target_compression', type=float, default=0.06, help='Fallback target compression ratio')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--log_every', type=int, default=10, help='Log interval')
    parser.add_argument('--save_every', type=int, default=10, help='Checkpoint interval (epoch)')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_path', type=str, default='TrajData/best_model.pt', help='Legacy best save path')

    parser.add_argument('--groups', type=int, default=10, help='Enable full-coverage grouped training when >0')
    parser.add_argument('--group_size', type=int, default=300, help='Trajectory count per group inside each full epoch')
    parser.add_argument('--group_epochs', type=int, default=1, help='Legacy option kept for compatibility; ignored in full-coverage grouped mode')

    parser.add_argument('--cr_cap_source', type=str, default='greedy', choices=['greedy', 'target'], help='CR cap source')
    parser.add_argument('--cap_relax_train', type=float, default=1.15, help='Multiplier for train-time max_keep before F1 threshold is met')
    parser.add_argument('--cap_relax_val', type=float, default=1.05, help='Multiplier for val-time max_keep before F1 threshold is met')
    parser.add_argument('--expert_forward_weight', type=float, default=0.9, help='Expert forward BC loss weight')
    parser.add_argument('--expert_bc_ratio_cap', type=float, default=0.9, help='Skip expert forward BC if greedy CR exceeds this cap')
    parser.add_argument('--distributional_expert_forward_weight', type=float, default=0.5, help='Expert forward BC weight for distributional PRS/frontier training')
    parser.add_argument('--temperature_min', type=float, default=0.3, help='Lower bound for sampling temperature')
    parser.add_argument('--min_keep_ratio', type=float, default=0.2, help='Disallow terminate before this fraction of max_keep is selected')
    parser.add_argument('--strict_f1', dest='strict_f1', action='store_true', help='Use strict hard-F1 mode')
    parser.add_argument('--no_strict_f1', dest='strict_f1', action='store_false', help='Enable mild F1 warmup')
    parser.add_argument('--num_samples_train', type=int, default=6, help='Base candidate rollouts per chunk during training')
    parser.add_argument('--hard_num_samples_train', type=int, default=12, help='Candidate rollouts for hard chunks')
    parser.add_argument('--hard_f1_margin', type=float, default=0.01, help='Hard-chunk margin: mark hard if chunk F1 < tau + margin')
    parser.add_argument('--anytime_disable', action='store_true', help='Disable anytime adaptive sampling budget')
    parser.add_argument('--anytime_min_samples_train', type=int, default=4, help='Minimum rollouts before anytime early-stop')
    parser.add_argument('--anytime_patience', type=int, default=2, help='Anytime patience on reward gain stagnation')
    parser.add_argument('--anytime_gain_epsilon', type=float, default=0.01, help='Anytime minimum gain threshold')
    parser.add_argument('--anytime_uncertain_margin', type=float, default=0.01, help='Only uncertain chunks keep extra rollout budget')
    parser.add_argument('--multifidelity_disable', action='store_true', help='Disable multi-fidelity reward evaluation')
    parser.add_argument('--multifidelity_topk_exact', type=int, default=2, help='Top-k proxy candidates to re-evaluate exactly')
    parser.add_argument('--multifidelity_proxy_grid_size', type=int, default=24, help='Proxy sketch grid size (0 disables proxy)')
    parser.add_argument('--multifidelity_proxy_stride', type=int, default=4, help='Subsampling stride for proxy sketch evaluation')
    parser.add_argument('--action_pool_size', type=int, default=64, help='Candidate-pool size for action re-parameterization (0 disables)')
    parser.add_argument('--action_pool_explore_ratio', type=float, default=0.2, help='Random exploration ratio inside action pool')
    parser.add_argument('--frontier_size', type=int, default=64, help='Per-chunk frontier capacity')
    parser.add_argument('--frontier_top_m', type=int, default=4, help='Top-M frontier paths used for set-level updates')
    parser.add_argument('--length_bucket_bounds', type=str, default='256,512,1000', help='Chunk length bucket upper bounds, e.g. 256,512,1000')
    parser.add_argument('--bucket_batch_size', type=int, default=8, help='Number of chunks processed per same-length bucket batch')
    parser.add_argument('--traj_cache_size', type=int, default=256, help='Preprocessed trajectory cache size (0 disables)')
    parser.add_argument('--greedy_cache_size', type=int, default=4096, help='Greedy-cap chunk cache size (0 disables)')
    parser.add_argument('--dual_eta', type=float, default=0.01, help='Dual lambda step size')
    parser.add_argument('--dual_xi', type=float, default=0.02, help='Constraint tolerance for mean violation')
    parser.add_argument('--dual_lambda_max', type=float, default=16.0, help='Maximum dual lambda')
    parser.add_argument('--phase2_dual_target', type=float, default=8.0, help='Phase-2 dual lambda target')
    parser.add_argument('--phase2_dual_decay', type=float, default=0.95, help='Phase-2 dual lambda decay factor')
    parser.add_argument('--stop_aux_weight', type=float, default=0.2, help='Auxiliary stop loss weight')
    parser.add_argument('--forward_only_ablation', action='store_true', help='Ablation: keep only forward objectives and disable backward-policy training')
    parser.add_argument('--set_loss_weight_mode', type=str, default='reward_novelty', help='Set-level weighting mode')
    parser.add_argument('--f1_safe_margin', type=float, default=0.01, help='Safe-feasible margin above tau')
    parser.add_argument('--best_f1ok_min', type=float, default=0.80, help='Minimum validation F1OK rate before best-model tie-break can consider CR')
    parser.add_argument('--infer_k', type=int, default=32, help='Top-k candidate count for PRS inference')
    parser.add_argument('--infer_k_max', type=int, default=64, help='Maximum sampling attempts for PRS inference')
    parser.add_argument('--val_infer_k', type=int, default=8, help='Top-k candidate count for PRS validation inference')
    parser.add_argument('--val_infer_k_max', type=int, default=16, help='Maximum sampling attempts for PRS validation inference')
    parser.add_argument('--infer_temperature', type=float, default=0.7, help='Sampling temperature for PRS inference')
    parser.add_argument('--prs_exact_lowcr_topk', type=int, default=2, help='Exact-rerank shortlist size for low-CR PRS candidates')
    parser.add_argument('--prs_exact_reward_topk', type=int, default=2, help='Exact-rerank shortlist size for high-reward PRS candidates')
    parser.add_argument('--prs_exact_f1_topk', type=int, default=2, help='Exact-rerank shortlist size for high-F1 PRS candidates')
    parser.add_argument('--infer_dual_lambda', type=float, default=1.0, help='Dual lambda used in inference-time candidate scoring')
    parser.add_argument('--val2_every', type=int, default=5, help='Run validation-2 every N epochs (<=0 to disable)')
    parser.add_argument('--phase2_trigger_f1ok', type=float, default=0.88, help='Validation F1OK threshold required before phase-2 low-CR mode can activate')
    parser.add_argument('--phase2_trigger_patience', type=int, default=2, help='Stable validation epochs required before phase-2 activates')
    parser.add_argument('--phase2_cap_gamma', type=float, default=0.995, help='Per-epoch cap tightening factor during phase-2')
    parser.add_argument('--phase2_cap_floor_ratio', type=float, default=0.95, help='Minimum cap scale ratio during phase-2')
    parser.add_argument('--phase2_min_best_margin', type=float, default=0.04, help='Minimum validation best-margin average required for phase-2 gating')
    parser.add_argument('--phase2_max_fallback_rate', type=float, default=0.20, help='Maximum validation PRS fallback rate allowed before phase-2 gating')
    parser.add_argument('--repair_enable', action='store_true', help='Enable global feasibility-preserving deletion repair')
    parser.add_argument('--repair_delete_ratio', type=float, default=0.3, help='Maximum deletion ratio during repair')
    parser.add_argument('--repair_max_delete', type=int, default=1000000, help='Absolute deletion cap during repair')
    parser.add_argument('--repair_jaccard_dedup', type=float, default=0.95, help='Dedup threshold for candidate Jaccard similarity')
    parser.add_argument('--repair_mode', type=str, default='hybrid', choices=['exact', 'hybrid'], help='Repair strategy for trajectory-level deletion repair')
    parser.add_argument('--repair_skip_below_kept', type=int, default=128, help='Skip repair when kept-point count is below this threshold')
    parser.add_argument('--repair_min_slack', type=float, default=0.02, help='Skip repair when base F1 slack is below this threshold')
    parser.add_argument('--repair_shortlist_topk', type=int, default=64, help='Maximum shortlist size for hybrid repair screening')
    parser.add_argument('--repair_proxy_grid_size', type=int, default=24, help='Proxy QCS grid size for hybrid repair')
    parser.add_argument('--repair_proxy_margin', type=float, default=0.01, help='Proxy F1 safety margin before exact confirmation')
    parser.add_argument('--legacy_single_path', action='store_true', help='Use legacy single-path training/inference')
    parser.add_argument('--profile_json', type=str, default='', help='Optional profiling output JSON path')
    parser.add_argument('--profile_label', type=str, default='', help='Optional profiling label')
    parser.set_defaults(strict_f1=True)

    args = parser.parse_args()

    if args.val_start_idx is None:
        args.val_start_idx = 9000
    if args.val_end_idx is None:
        args.val_end_idx = 9030
    if args.val2_start_idx is None:
        args.val2_start_idx = args.val_end_idx
    if args.val2_end_idx is None:
        args.val2_end_idx = args.val2_start_idx + max(1, (args.val_end_idx - args.val_start_idx))

    args.length_bucket_bounds = parse_length_bucket_bounds(args.length_bucket_bounds)
    args.bucket_batch_size = max(1, int(args.bucket_batch_size))
    args.distributional_expert_forward_weight = max(0.0, float(args.distributional_expert_forward_weight))
    args.num_samples_train = max(1, int(args.num_samples_train))
    args.hard_num_samples_train = max(args.num_samples_train, int(args.hard_num_samples_train))
    args.anytime_min_samples_train = max(1, int(args.anytime_min_samples_train))
    args.anytime_patience = max(1, int(args.anytime_patience))
    args.anytime_gain_epsilon = max(0.0, float(args.anytime_gain_epsilon))
    args.anytime_uncertain_margin = max(0.0, float(args.anytime_uncertain_margin))
    args.multifidelity_topk_exact = max(1, int(args.multifidelity_topk_exact))
    args.multifidelity_proxy_grid_size = max(0, int(args.multifidelity_proxy_grid_size))
    args.multifidelity_proxy_stride = max(1, int(args.multifidelity_proxy_stride))
    args.action_pool_size = max(0, int(args.action_pool_size))
    args.action_pool_explore_ratio = float(np.clip(args.action_pool_explore_ratio, 0.0, 1.0))
    args.frontier_top_m = max(1, int(args.frontier_top_m))
    args.traj_cache_size = max(0, int(args.traj_cache_size))
    args.greedy_cache_size = max(0, int(args.greedy_cache_size))
    args.phase2_dual_target = max(0.0, float(args.phase2_dual_target))
    args.phase2_dual_decay = float(np.clip(args.phase2_dual_decay, 0.0, 1.0))
    args.f1_safe_margin = max(0.0, float(args.f1_safe_margin))
    args.best_f1ok_min = float(np.clip(args.best_f1ok_min, 0.0, 1.0))
    args.val_infer_k = max(1, int(args.val_infer_k))
    args.val_infer_k_max = max(args.val_infer_k, int(args.val_infer_k_max))
    args.prs_exact_lowcr_topk = max(0, int(args.prs_exact_lowcr_topk))
    args.prs_exact_reward_topk = max(0, int(args.prs_exact_reward_topk))
    args.prs_exact_f1_topk = max(0, int(args.prs_exact_f1_topk))
    args.repair_mode = str(getattr(args, 'repair_mode', 'hybrid')).strip().lower()
    if args.repair_mode not in ('exact', 'hybrid'):
        args.repair_mode = 'hybrid'
    args.repair_skip_below_kept = max(0, int(getattr(args, 'repair_skip_below_kept', 128)))
    args.repair_min_slack = max(0.0, float(getattr(args, 'repair_min_slack', 0.02)))
    args.repair_shortlist_topk = max(1, int(getattr(args, 'repair_shortlist_topk', 64)))
    args.repair_proxy_grid_size = max(4, int(getattr(args, 'repair_proxy_grid_size', 24)))
    args.repair_proxy_margin = max(0.0, float(getattr(args, 'repair_proxy_margin', 0.01)))
    args.val2_every = int(args.val2_every)
    args.phase2_trigger_f1ok = float(np.clip(args.phase2_trigger_f1ok, 0.0, 1.0))
    args.phase2_trigger_patience = max(1, int(args.phase2_trigger_patience))
    args.phase2_cap_gamma = float(np.clip(args.phase2_cap_gamma, 0.0, 1.0))
    args.phase2_cap_floor_ratio = float(np.clip(args.phase2_cap_floor_ratio, 0.0, 1.0))
    args.phase2_min_best_margin = float(getattr(args, 'phase2_min_best_margin', 0.0))
    args.phase2_max_fallback_rate = float(np.clip(getattr(args, 'phase2_max_fallback_rate', 0.35), 0.0, 1.0))
    configure_runtime_caches(
        traj_cache_size=args.traj_cache_size,
        greedy_cache_size=args.greedy_cache_size,
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    collector = init_profile_collector(
        profile_json=args.profile_json,
        script_name="train_streaming.py",
        profile_label=args.profile_label,
        extra_metadata={
            "traj_path": args.traj_path,
            "start_idx": int(args.start_idx),
            "end_idx": int(args.end_idx),
            "val_start_idx": int(args.val_start_idx),
            "val_end_idx": int(args.val_end_idx),
            "val2_start_idx": int(args.val2_start_idx),
            "val2_end_idx": int(args.val2_end_idx),
            "epochs": int(args.epochs),
            "seed": int(args.seed),
        },
    )

    device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    print(f"[Device] {device}")
    if collector is not None:
        collector.set_metadata(device=device, torch_cuda_available=bool(torch.cuda.is_available()))

    print(f"[Data] Scanning train trajectories {args.start_idx} ~ {args.end_idx}...")
    with profile_scope("train", "train_prepare", "scan_train_indices"):
        valid_indices = scan_valid_indices(args.traj_path, args.start_idx, args.end_idx)
    print(f"[Data] Found {len(valid_indices)} valid train trajectories")
    if len(valid_indices) == 0:
        print('[Error] No valid trajectories found.')
        finalize_profile(args.profile_json)
        return

    val_indices = []
    if args.val_start_idx is not None and args.val_end_idx is not None:
        with profile_scope("train", "train_prepare", "scan_val1_indices"):
            val_indices = scan_valid_indices(args.traj_path, args.val_start_idx, args.val_end_idx)
        print(f"[Data] Validation trajectories: {len(val_indices)} ({args.val_start_idx}-{args.val_end_idx})")

    val2_indices = []
    if args.val2_start_idx is not None and args.val2_end_idx is not None:
        val2_indices = scan_valid_indices(args.traj_path, args.val2_start_idx, args.val2_end_idx)
        print(f"[Data] Validation-2 trajectories: {len(val2_indices)} ({args.val2_start_idx}-{args.val2_end_idx})")

    if (args.val_start_idx < args.val2_end_idx) and (args.val2_start_idx < args.val_end_idx):
        print(
            f"[Warn] Validation split overlap detected: "
            f"val1[{args.val_start_idx},{args.val_end_idx}) vs "
            f"val2[{args.val2_start_idx},{args.val2_end_idx})"
        )

    with profile_scope("train", "train_prepare", "compute_global_stats"):
        global_stats = compute_global_stats(args.traj_path, valid_indices, sample_size=200)
    print(
        f"  [Stats] X:[{global_stats['x_min']:.2f}, {global_stats['x_max']:.2f}] "
        f"Y:[{global_stats['y_min']:.2f}, {global_stats['y_max']:.2f}] "
        f"T:[{global_stats['t_min']:.0f}, {global_stats['t_max']:.0f}]"
    )

    with profile_scope("train", "train_prepare", "model_init_resume"):
        print('[Model] Initializing HierarchicalGFlowNet...')
        model = HierarchicalGFlowNet(
            input_dim=3,
            hidden_dim=args.hidden_dim,
            num_layers=2,
            dropout=0.1
        ).to(device)

        trainer = TLMTrainer(
            model=model,
            device=device,
            lr_forward=args.lr,
            lr_backward=args.lr_backward,
            f1_threshold=args.f1_threshold,
            target_compression=args.target_compression,
            global_stats=global_stats,
            frontier_size=args.frontier_size,
            frontier_top_m=args.frontier_top_m,
            dual_eta=args.dual_eta,
            dual_xi=args.dual_xi,
            dual_lambda_max=args.dual_lambda_max,
            phase2_dual_target=args.phase2_dual_target,
            phase2_dual_decay=args.phase2_dual_decay,
            stop_aux_weight=args.stop_aux_weight,
            forward_only=args.forward_only_ablation,
            set_loss_weight_mode=args.set_loss_weight_mode,
            anytime_enable=(not args.anytime_disable),
            anytime_min_samples_train=args.anytime_min_samples_train,
            anytime_patience=args.anytime_patience,
            anytime_gain_epsilon=args.anytime_gain_epsilon,
            anytime_uncertain_margin=args.anytime_uncertain_margin,
            multifidelity_enable=(not args.multifidelity_disable),
            multifidelity_topk_exact=args.multifidelity_topk_exact,
            multifidelity_proxy_grid_size=args.multifidelity_proxy_grid_size,
            multifidelity_proxy_stride=args.multifidelity_proxy_stride,
            action_pool_size=args.action_pool_size,
            action_pool_explore_ratio=args.action_pool_explore_ratio,
            f1_safe_margin=args.f1_safe_margin,
            prs_exact_lowcr_topk=args.prs_exact_lowcr_topk,
            prs_exact_reward_topk=args.prs_exact_reward_topk,
            prs_exact_f1_topk=args.prs_exact_f1_topk,
        )

        start_epoch = 0
        resume_checkpoint = None
        if args.resume and os.path.exists(args.resume):
            print(f"[Model] Resuming from {args.resume} via Trainer...")
            try:
                resume_checkpoint = trainer.load(args.resume)
                start_epoch = trainer.epoch + 1
                print(f"[Model] Resuming from Epoch {start_epoch}")
            except Exception as e:
                print(f"[Warn] Resume failed: {e}")

    checkpoint_dir = os.path.dirname(args.model_path)
    if checkpoint_dir == '':
        checkpoint_dir = '.'
    os.makedirs(checkpoint_dir, exist_ok=True)
    validation_csv_dir = os.path.join(checkpoint_dir, 'validation_csv')
    os.makedirs(validation_csv_dir, exist_ok=True)

    config_snapshot = {
        'cr_cap_source': args.cr_cap_source,
        'cap_relax_train': float(args.cap_relax_train),
        'cap_relax_val': float(args.cap_relax_val),
        'strict_f1': bool(args.strict_f1),
        'target_compression': float(args.target_compression),
        'expert_forward_weight': float(args.expert_forward_weight),
        'expert_bc_ratio_cap': float(args.expert_bc_ratio_cap),
        'distributional_expert_forward_weight': float(args.distributional_expert_forward_weight),
        'temperature': float(args.temperature),
        'temperature_min': float(args.temperature_min),
        'min_keep_ratio': float(args.min_keep_ratio),
        'num_samples_train': int(args.num_samples_train),
        'hard_num_samples_train': int(args.hard_num_samples_train),
        'hard_f1_margin': float(args.hard_f1_margin),
        'anytime_disable': bool(args.anytime_disable),
        'anytime_min_samples_train': int(args.anytime_min_samples_train),
        'anytime_patience': int(args.anytime_patience),
        'anytime_gain_epsilon': float(args.anytime_gain_epsilon),
        'anytime_uncertain_margin': float(args.anytime_uncertain_margin),
        'multifidelity_disable': bool(args.multifidelity_disable),
        'multifidelity_topk_exact': int(args.multifidelity_topk_exact),
        'multifidelity_proxy_grid_size': int(args.multifidelity_proxy_grid_size),
        'multifidelity_proxy_stride': int(args.multifidelity_proxy_stride),
        'action_pool_size': int(args.action_pool_size),
        'action_pool_explore_ratio': float(args.action_pool_explore_ratio),
        'frontier_size': int(args.frontier_size),
        'frontier_top_m': int(args.frontier_top_m),
        'length_bucket_bounds': [int(v) for v in args.length_bucket_bounds],
        'bucket_batch_size': int(args.bucket_batch_size),
        'traj_cache_size': int(args.traj_cache_size),
        'greedy_cache_size': int(args.greedy_cache_size),
        'dual_eta': float(args.dual_eta),
        'dual_xi': float(args.dual_xi),
        'dual_lambda_max': float(args.dual_lambda_max),
        'phase2_dual_target': float(args.phase2_dual_target),
        'phase2_dual_decay': float(args.phase2_dual_decay),
        'stop_aux_weight': float(args.stop_aux_weight),
        'forward_only_ablation': bool(args.forward_only_ablation),
        'f1_safe_margin': float(args.f1_safe_margin),
        'best_f1ok_min': float(args.best_f1ok_min),
        'infer_k': int(args.infer_k),
        'infer_k_max': int(args.infer_k_max),
        'val_infer_k': int(args.val_infer_k),
        'val_infer_k_max': int(args.val_infer_k_max),
        'prs_exact_lowcr_topk': int(args.prs_exact_lowcr_topk),
        'prs_exact_reward_topk': int(args.prs_exact_reward_topk),
        'prs_exact_f1_topk': int(args.prs_exact_f1_topk),
        'infer_temperature': float(args.infer_temperature),
        'val2_every': int(args.val2_every),
        'phase2_trigger_f1ok': float(args.phase2_trigger_f1ok),
        'phase2_trigger_patience': int(args.phase2_trigger_patience),
        'phase2_cap_gamma': float(args.phase2_cap_gamma),
        'phase2_cap_floor_ratio': float(args.phase2_cap_floor_ratio),
        'phase2_min_best_margin': float(args.phase2_min_best_margin),
        'phase2_max_fallback_rate': float(args.phase2_max_fallback_rate),
        'repair_enable': bool(args.repair_enable),
        'repair_delete_ratio': float(args.repair_delete_ratio),
        'repair_max_delete': int(args.repair_max_delete),
        'repair_jaccard_dedup': float(args.repair_jaccard_dedup),
        'repair_mode': str(args.repair_mode),
        'repair_skip_below_kept': int(args.repair_skip_below_kept),
        'repair_min_slack': float(args.repair_min_slack),
        'repair_shortlist_topk': int(args.repair_shortlist_topk),
        'repair_proxy_grid_size': int(args.repair_proxy_grid_size),
        'repair_proxy_margin': float(args.repair_proxy_margin),
        'legacy_single_path': bool(args.legacy_single_path),
        'f1_threshold': float(args.f1_threshold),
        'val_start_idx': int(args.val_start_idx),
        'val_end_idx': int(args.val_end_idx),
        'val2_start_idx': int(args.val2_start_idx),
        'val2_end_idx': int(args.val2_end_idx),
        'seed': int(args.seed),
    }
    snapshot_path = os.path.join(
        checkpoint_dir,
        f"train_config_snapshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    )
    with profile_scope("train", "train_validation_output", "checkpoint_save"):
        with open(snapshot_path, 'w', encoding='utf-8') as f:
            json.dump(config_snapshot, f, ensure_ascii=True, indent=2)
    print(f"[Config] Snapshot saved: {snapshot_path}")

    total_trajs = 0
    total_success = 0

    best_success_rate = 0.0
    best_has_target_f1ok = False
    best_val_f1ok = -1.0
    best_val_f1 = -1.0
    best_val_cr = float('inf')
    recent_val_summaries = []
    cap_relax_active = (args.cap_relax_train > 1.0) or (args.cap_relax_val > 1.0)
    hard_chunk_cache = set()
    phase2_lowcr_active = False
    phase2_trigger_streak = 0
    phase2_epochs_run = 0
    if isinstance(resume_checkpoint, dict):
        phase2_state = resume_checkpoint.get('phase2_state', {})
        if isinstance(phase2_state, dict):
            phase2_lowcr_active = bool(phase2_state.get('active', False))
            phase2_trigger_streak = int(max(0, phase2_state.get('trigger_streak', 0)))
            phase2_epochs_run = int(max(0, phase2_state.get('epochs_run', 0)))
            if phase2_lowcr_active:
                print(
                    f"[Resume] Phase2LowCR restored: active={phase2_lowcr_active} "
                    f"streak={phase2_trigger_streak} epochs_run={phase2_epochs_run}"
                )
        hard_chunk_cache_state = resume_checkpoint.get('hard_chunk_cache')
        if isinstance(hard_chunk_cache_state, (list, tuple)):
            restored_hard_chunk_cache = set()
            for item in hard_chunk_cache_state:
                if not isinstance(item, (list, tuple)) or len(item) != 3:
                    continue
                restored_hard_chunk_cache.add(tuple(int(v) for v in item))
            hard_chunk_cache = restored_hard_chunk_cache
            if hard_chunk_cache:
                print(f"[Resume] Hard chunk cache restored: {len(hard_chunk_cache)} entries")
    trainer.set_phase2_lowcr(
        phase2_lowcr_active,
        dual_target=args.phase2_dual_target,
        dual_decay=args.phase2_dual_decay,
    )

    def is_better_candidate(summary):
        nonlocal best_has_target_f1ok, best_val_f1ok, best_val_f1, best_val_cr
        cur_f1 = float(summary.get('avg_f1', 0.0))
        cur_cr = float(summary.get('avg_cr', 1.0))
        cur_f1ok = float(summary.get('f1_hard_ok_rate', 0.0))
        cur_meets_target = cur_f1ok >= args.best_f1ok_min

        if cur_meets_target and not best_has_target_f1ok:
            return True
        if cur_meets_target and best_has_target_f1ok:
            if cur_f1ok > best_val_f1ok + 1e-12:
                return True
            if abs(cur_f1ok - best_val_f1ok) <= 1e-12 and cur_f1 > best_val_f1 + 1e-12:
                return True
            if (
                abs(cur_f1ok - best_val_f1ok) <= 1e-12
                and abs(cur_f1 - best_val_f1) <= 1e-12
                and cur_cr < best_val_cr - 1e-12
            ):
                return True
            return False
        if (not cur_meets_target) and best_has_target_f1ok:
            return False
        if (not cur_meets_target) and (not best_has_target_f1ok):
            if cur_f1ok > best_val_f1ok + 1e-12:
                return True
            if abs(cur_f1ok - best_val_f1ok) <= 1e-12 and cur_f1 > best_val_f1 + 1e-12:
                return True
            if (
                abs(cur_f1ok - best_val_f1ok) <= 1e-12
                and abs(cur_f1 - best_val_f1) <= 1e-12
                and cur_cr < best_val_cr - 1e-12
            ):
                return True
        return False

    def update_best_state(summary):
        nonlocal best_has_target_f1ok, best_val_f1ok, best_val_f1, best_val_cr
        best_has_target_f1ok = float(summary.get('f1_hard_ok_rate', 0.0)) >= args.best_f1ok_min
        best_val_f1ok = float(summary.get('f1_hard_ok_rate', 0.0))
        best_val_f1 = float(summary.get('avg_f1', 0.0))
        best_val_cr = float(summary.get('avg_cr', 1.0))

    def build_phase2_state():
        return {
            'active': bool(phase2_lowcr_active),
            'trigger_streak': int(phase2_trigger_streak),
            'epochs_run': int(phase2_epochs_run),
        }

    def build_hard_chunk_cache_state():
        return sorted(tuple(int(v) for v in item) for item in hard_chunk_cache)

    def aggregate_train_metrics(metric_list):
        if len(metric_list) == 1:
            return metric_list[0]

        total_trajs_local = sum(int(m.get('epoch_trajs', 0)) for m in metric_list)
        total_chunks_local = sum(int(m.get('epoch_chunks', 0)) for m in metric_list)
        total_success_local = sum(int(m.get('epoch_success', 0)) for m in metric_list)
        total_expert_forward = sum(int(m.get('expert_forward_total', 0)) for m in metric_list)
        total_expert_skipped = sum(int(m.get('expert_forward_skipped', 0)) for m in metric_list)
        max_bucket_count = max(len(m.get('bucket_task_counts', [])) for m in metric_list)
        bucket_task_counts = [0 for _ in range(max_bucket_count)]
        for m in metric_list:
            for i, count in enumerate(m.get('bucket_task_counts', [])):
                bucket_task_counts[i] += int(count)

        weighted_by_traj = lambda key: sum(
            float(m.get(key, 0.0)) * max(1, int(m.get('epoch_trajs', 0)))
            for m in metric_list
        ) / max(1, total_trajs_local)
        weighted_by_chunk = lambda key: sum(
            float(m.get(key, 0.0)) * max(1, int(m.get('epoch_chunks', 0)))
            for m in metric_list
        ) / max(1, total_chunks_local)

        return {
            'success_rate': total_success_local / max(1, total_trajs_local),
            'avg_f1': weighted_by_traj('avg_f1'),
            'avg_cr': weighted_by_traj('avg_cr'),
            'avg_cap_cr': weighted_by_traj('avg_cap_cr'),
            'epoch_trajs': total_trajs_local,
            'epoch_chunks': total_chunks_local,
            'epoch_success': total_success_local,
            'expert_forward_skipped_ratio': total_expert_skipped / max(1, total_expert_forward),
            'expert_forward_skipped': total_expert_skipped,
            'expert_forward_total': total_expert_forward,
            'loss_expert_forward': sum(float(m.get('loss_expert_forward', 0.0)) for m in metric_list),
            'avg_dual_lambda': weighted_by_chunk('avg_dual_lambda'),
            'dual_lambda_p50': weighted_by_chunk('dual_lambda_p50'),
            'dual_lambda_p90': weighted_by_chunk('dual_lambda_p90'),
            'dual_lambda_max': max(float(m.get('dual_lambda_max', 0.0)) for m in metric_list),
            'avg_frontier_size': weighted_by_chunk('avg_frontier_size'),
            'avg_actual_samples': weighted_by_chunk('avg_actual_samples'),
            'avg_exact_evals': weighted_by_chunk('avg_exact_evals'),
            'hard_chunk_used': sum(int(m.get('hard_chunk_used', 0)) for m in metric_list),
            'hard_chunk_promoted': sum(int(m.get('hard_chunk_promoted', 0)) for m in metric_list),
            'hard_chunk_cleared': sum(int(m.get('hard_chunk_cleared', 0)) for m in metric_list),
            'hard_cache_size': int(metric_list[-1].get('hard_cache_size', 0)),
            'bucket_task_counts': bucket_task_counts,
        }

    def phase2_gate_is_stable(summary):
        window = max(1, int(args.phase2_trigger_patience))
        phase2_history = (recent_val_summaries + [summary])[-window:]
        if len(phase2_history) < window:
            return False

        for item in phase2_history:
            if float(item.get('avg_f1', 0.0)) < float(args.f1_threshold):
                return False
            if float(item.get('f1_hard_ok_rate', 0.0)) < float(args.phase2_trigger_f1ok):
                return False
            if float(item.get('best_margin_avg', 0.0)) < float(args.phase2_min_best_margin):
                return False
            if float(item.get('prs_fallback_rate', 1.0)) > float(args.phase2_max_fallback_rate):
                return False
        return True

    print(f"\n{'='*60}")
    print('Start training')
    print(
        f"Length buckets={args.length_bucket_bounds}, "
        f"bucket_batch_size={args.bucket_batch_size}, "
        f"num_samples(base/hard)={args.num_samples_train}/{args.hard_num_samples_train}, "
        f"anytime={'off' if args.anytime_disable else 'on'}(min/pat/eps/margin="
        f"{args.anytime_min_samples_train}/{args.anytime_patience}/{args.anytime_gain_epsilon:.4f}/{args.anytime_uncertain_margin:.4f}), "
        f"multifidelity={'off' if args.multifidelity_disable else 'on'}(topk/grid/stride="
        f"{args.multifidelity_topk_exact}/{args.multifidelity_proxy_grid_size}/{args.multifidelity_proxy_stride}), "
        f"pool(size/explore)={args.action_pool_size}/{args.action_pool_explore_ratio:.2f}, "
        f"cache(traj/chunk)={args.traj_cache_size}/{args.greedy_cache_size}, "
        f"frontier_top_m={args.frontier_top_m}, "
        f"dist_expert_w={args.distributional_expert_forward_weight:.2f}, "
        f"safe_margin={args.f1_safe_margin:.3f}, "
        f"prs_exact(lowcr/reward/f1)={args.prs_exact_lowcr_topk}/{args.prs_exact_reward_topk}/{args.prs_exact_f1_topk}, "
        f"repair(mode/skip/slack/topk/proxy/margin)={args.repair_mode}/{args.repair_skip_below_kept}/"
        f"{args.repair_min_slack:.3f}/{args.repair_shortlist_topk}/{args.repair_proxy_grid_size}/{args.repair_proxy_margin:.3f}, "
        f"phase2(trigger_f1ok/pat/gamma/floor/dual/min_margin/max_fallback)="
        f"{args.phase2_trigger_f1ok:.2f}/{args.phase2_trigger_patience}/"
        f"{args.phase2_cap_gamma:.3f}/{args.phase2_cap_floor_ratio:.3f}/{args.phase2_dual_target:.2f}/"
        f"{args.phase2_min_best_margin:.3f}/{args.phase2_max_fallback_rate:.2f}, "
        f"val_prs(k/kmax)={args.val_infer_k}/{args.val_infer_k_max}, "
        f"val2_every={args.val2_every}"
    )
    print(f"{'='*60}\n")

    def run_epoch_once(epoch, train_indices, train_groups=None):
        nonlocal total_trajs, total_success, best_success_rate
        nonlocal cap_relax_active, hard_chunk_cache
        nonlocal phase2_lowcr_active, phase2_trigger_streak, phase2_epochs_run
        nonlocal recent_val_summaries

        current_cap_relax_train = args.cap_relax_train if cap_relax_active else 1.0
        current_cap_relax_val = args.cap_relax_val if cap_relax_active else 1.0
        phase2_active_for_epoch = bool(phase2_lowcr_active)
        current_phase2_cap_scale = 1.0
        trainer.set_phase2_lowcr(
            phase2_active_for_epoch,
            dual_target=args.phase2_dual_target,
            dual_decay=args.phase2_dual_decay,
        )
        if phase2_active_for_epoch:
            trainer.apply_phase2_dual_decay()
            current_phase2_cap_scale = max(
                args.phase2_cap_floor_ratio,
                args.phase2_cap_gamma ** max(1, phase2_epochs_run + 1),
            )
        print(
            f"             CapRelax(train/val): "
            f"{current_cap_relax_train:.3f}/{current_cap_relax_val:.3f}"
        )
        print(
            f"             Phase2LowCR(active/streak/scale): "
            f"{'on' if phase2_active_for_epoch else 'off'}/"
            f"{phase2_trigger_streak}/"
            f"{current_phase2_cap_scale:.4f}"
        )

        with profile_scope("train", "train_loop", "epoch_total"):
            if train_groups:
                per_group_metrics = []
                print(f"             Full-coverage grouped passes: {len(train_groups)}")
                for group_idx, group_indices in enumerate(train_groups):
                    print(
                        f"             [Epoch {epoch+1} Group {group_idx+1}/{len(train_groups)}] "
                        f"Trajectories: {len(group_indices)}"
                    )
                    group_metrics = train_one_epoch(
                        epoch,
                        trainer,
                        group_indices,
                        args,
                        global_stats,
                        device,
                        cap_relax_train=current_cap_relax_train,
                        temperature_floor=args.temperature_min,
                        hard_chunk_cache=hard_chunk_cache,
                        phase2_lowcr_active=phase2_active_for_epoch,
                        phase2_cap_scale=current_phase2_cap_scale,
                        summary_label=f"Epoch {epoch+1} Group {group_idx+1}",
                    )
                    per_group_metrics.append(group_metrics)
                metrics = aggregate_train_metrics(per_group_metrics)
            else:
                metrics = train_one_epoch(
                    epoch,
                    trainer,
                    train_indices,
                    args,
                    global_stats,
                    device,
                    cap_relax_train=current_cap_relax_train,
                    temperature_floor=args.temperature_min,
                    hard_chunk_cache=hard_chunk_cache,
                    phase2_lowcr_active=phase2_active_for_epoch,
                    phase2_cap_scale=current_phase2_cap_scale,
                )
        success_rate = metrics['success_rate']
        avg_f1 = metrics['avg_f1']
        total_trajs += metrics['epoch_trajs']
        total_success += metrics['epoch_success']
        best_success_rate = max(best_success_rate, success_rate)
        print(
            f"             Train expert_forward_skipped: "
            f"{metrics['expert_forward_skipped_ratio']*100:.1f}% "
            f"({metrics['expert_forward_skipped']}/{metrics['expert_forward_total']})"
        )
        print(
            f"             HardSamples(used/promoted/cleared/cache): "
            f"{metrics.get('hard_chunk_used', 0)}/"
            f"{metrics.get('hard_chunk_promoted', 0)}/"
            f"{metrics.get('hard_chunk_cleared', 0)}/"
            f"{metrics.get('hard_cache_size', 0)}"
        )
        print(
            f"             Anytime(avg_samples/chunk)={metrics.get('avg_actual_samples', 0.0):.2f} | "
            f"MF(exact_evals/chunk)={metrics.get('avg_exact_evals', 0.0):.2f}"
        )

        if len(val_indices) > 0:
            print(f"[Validation] Evaluating on {len(val_indices)} trajectories...")
            with profile_scope("train", "train_validation_output", "validation_val1_total"):
                val_summary = validate_epoch(
                    model,
                    val_indices,
                    args,
                    global_stats,
                    device,
                    split_name='val1',
                    epoch=epoch,
                    csv_dir=validation_csv_dir,
                    cap_relax_val=current_cap_relax_val,
                )
        else:
            val_summary = {
                'avg_f1': avg_f1,
                'avg_cr': metrics.get('avg_cr', 0.0),
                'weighted_cr': metrics.get('avg_cr', 0.0),
                'traj_mean_cr': metrics.get('avg_cr', 0.0),
                'avg_pts': 0.0,
                'f1_hard_ok_rate': 0.0,
                'cr_p50': metrics.get('avg_cr', 0.0),
                'cr_p90': metrics.get('avg_cr', 0.0),
                'cr_p95': metrics.get('avg_cr', 0.0),
                'avg_cap_cr': metrics.get('avg_cap_cr', 0.0),
                'cr_vs_cap_gap': 0.0,
                'count': 0,
                'csv_path': '',
                'chunk_count': 0,
                'chunk_term_counts': {'model_stop': 0, 'max_keep': 0, 'no_valid': 0, 'step_limit': 0, 'other': 0},
                'chunk_term_model_stop_ratio': 0.0,
                'chunk_term_max_keep_ratio': 0.0,
                'prs_avg_candidates': 0.0,
                'prs_avg_feasible': 0.0,
                'prs_fallback_rate': 0.0,
                'prs_repair_avg_deleted': 0.0,
                'safe_feasible_count': 0,
                'safe_feasible_rate': 0.0,
                'best_margin_avg': 0.0,
            }

        print(
            f"             Val F1: {val_summary['avg_f1']:.4f} | "
            f"CR weighted/traj_mean: {val_summary['weighted_cr']*100:.2f}%/{val_summary['traj_mean_cr']*100:.2f}% | "
            f"CR p50/p90/p95: {val_summary['cr_p50']*100:.2f}%/{val_summary['cr_p90']*100:.2f}%/{val_summary['cr_p95']*100:.2f}% | "
            f"F1OK: {val_summary['f1_hard_ok_rate']*100:.1f}% | "
            f"CR-gap: {val_summary['cr_vs_cap_gap']*100:.2f}%"
        )
        print(
            f"             Terminate(model_stop/max_keep): "
            f"{val_summary['chunk_term_model_stop_ratio']*100:.1f}%/"
            f"{val_summary['chunk_term_max_keep_ratio']*100:.1f}% | "
            f"PRS(avg_k/feasible/fallback/repair_del_traj): "
            f"{val_summary.get('prs_avg_candidates', 0.0):.1f}/"
            f"{val_summary.get('prs_avg_feasible', 0.0):.1f}/"
            f"{val_summary.get('prs_fallback_rate', 0.0)*100:.1f}%/"
            f"{val_summary.get('prs_repair_avg_deleted', 0.0):.2f} | "
            f"SafeFeasible(count/rate): "
            f"{val_summary.get('safe_feasible_count', 0)}/"
            f"{val_summary.get('safe_feasible_rate', 0.0)*100:.1f}% | "
            f"BestMarginAvg: {val_summary.get('best_margin_avg', 0.0):.4f} | "
            f"DetailCSV: {val_summary['csv_path']}"
        )

        trainer.decay_backward_lr()

        val_f1 = val_summary['avg_f1']
        val_cr = val_summary['avg_cr']
        val_f1_ok_rate = val_summary['f1_hard_ok_rate']
        phase2_gate_ok = phase2_gate_is_stable(val_summary)
        phase2_trigger_streak = (phase2_trigger_streak + 1) if phase2_gate_ok else 0
        recent_val_summaries.append({
            'avg_f1': float(val_f1),
            'avg_cr': float(val_cr),
            'f1_hard_ok_rate': float(val_f1_ok_rate),
            'best_margin_avg': float(val_summary.get('best_margin_avg', 0.0)),
            'prs_fallback_rate': float(val_summary.get('prs_fallback_rate', 1.0)),
        })
        recent_val_summaries = recent_val_summaries[-max(1, int(args.phase2_trigger_patience)):]

        if cap_relax_active and (val_f1 >= args.f1_threshold):
            cap_relax_active = False
            print("             [CapRelax] F1 threshold reached, fallback to 1.000/1.000 from next epoch")

        if (not phase2_lowcr_active) and (phase2_trigger_streak >= args.phase2_trigger_patience):
            phase2_lowcr_active = True
            phase2_epochs_run = 0
            trainer.set_phase2_lowcr(
                True,
                dual_target=args.phase2_dual_target,
                dual_decay=args.phase2_dual_decay,
            )
            print(
                f"             [Phase2LowCR] Activated from next epoch "
                f"(streak={phase2_trigger_streak}, trigger_f1ok={args.phase2_trigger_f1ok:.2f}, "
                f"max_fallback={args.phase2_max_fallback_rate:.2f}, "
                f"min_margin={args.phase2_min_best_margin:.3f})"
            )

        if is_better_candidate(val_summary):
            update_best_state(val_summary)
            best_path = args.model_path.replace(
                '.pt',
                f"_best_f1{val_f1:.4f}_f1ok{val_f1_ok_rate:.3f}_cr{val_cr:.4f}.pt"
            )
            with profile_scope("train", "train_validation_output", "checkpoint_save"):
                trainer.save(
                    best_path,
                    success_rate=success_rate,
                    avg_f1=avg_f1,
                    val_summary=val_summary,
                    phase2_state=build_phase2_state(),
                    hard_chunk_cache=build_hard_chunk_cache_state(),
                )
            print(f"  [Save] Saved best model by stability criterion (F1OK -> F1 -> CR): {best_path}")

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = args.model_path.replace(
                '.pt',
                f"_epoch{epoch+1}_f1{val_f1:.4f}_f1ok{val_f1_ok_rate:.3f}_cr{val_cr:.4f}.pt"
            )
            with profile_scope("train", "train_validation_output", "checkpoint_save"):
                trainer.save(
                    ckpt_path,
                    success_rate=success_rate,
                    val_summary=val_summary,
                    phase2_state=build_phase2_state(),
                    hard_chunk_cache=build_hard_chunk_cache_state(),
                )
            print(f"  [Save] Saved checkpoint to {ckpt_path}")

        run_val2 = (args.val2_every > 0) and ((epoch + 1) % args.val2_every == 0)
        if run_val2:
            print(
                f"\n[Validation-2] Evaluating trajectories "
                f"{args.val2_start_idx}-{args.val2_end_idx}..."
            )

        if run_val2 and len(val2_indices) > 0:
            custom_summary = validate_epoch(
                model,
                val2_indices,
                args,
                global_stats,
                device,
                split_name='val2',
                epoch=epoch,
                csv_dir=validation_csv_dir,
                cap_relax_val=current_cap_relax_val,
            )
            print(
                f"  [Val2] F1: {custom_summary['avg_f1']:.4f} | "
                f"CR weighted/traj_mean: {custom_summary['weighted_cr']*100:.2f}%/{custom_summary['traj_mean_cr']*100:.2f}% | "
                f"F1OK: {custom_summary['f1_hard_ok_rate']*100:.1f}% | "
                f"Terminate(model_stop/max_keep): "
                f"{custom_summary['chunk_term_model_stop_ratio']*100:.1f}%/"
                f"{custom_summary['chunk_term_max_keep_ratio']*100:.1f}% | "
                f"PRS(fallback/repair_del_traj): "
                f"{custom_summary.get('prs_fallback_rate', 0.0)*100:.1f}%/"
                f"{custom_summary.get('prs_repair_avg_deleted', 0.0):.2f} | "
                f"DetailCSV: {custom_summary['csv_path']}"
            )

            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            model_name = (
                f"{timestamp}_ep{epoch+1:04d}_valf1{val_f1:.4f}_"
                f"f1ok{val_f1_ok_rate:.3f}_valcr{val_cr:.4f}.pt"
            )
            save_path = os.path.join(checkpoint_dir, model_name)
            with profile_scope("train", "train_validation_output", "checkpoint_save"):
                trainer.save(
                    save_path,
                    avg_f1=custom_summary['avg_f1'],
                    avg_pts=custom_summary['avg_pts'],
                    custom_val=custom_summary,
                    phase2_state=build_phase2_state(),
                    hard_chunk_cache=build_hard_chunk_cache_state(),
                )
            print(f"  [Save] Model saved to {save_path}")
        elif (not run_val2):
            if args.val2_every <= 0:
                print("\n[Validation-2] Skipped (disabled by --val2_every <= 0)")
            else:
                print(
                    f"\n[Validation-2] Skipped this epoch "
                    f"(runs every {args.val2_every} epochs)"
                )
        else:
            print("  [Val2] Skipped (no valid trajectories in split)")

        if phase2_active_for_epoch:
            phase2_epochs_run += 1

    if args.groups > 0:
        print(f"\n[Mode] Full-coverage grouped training: epochs={args.epochs}, requested_groups={args.groups}, group_size={args.group_size}")
        if args.group_epochs != 1:
            print(
                f"[Info] group_epochs={args.group_epochs} is ignored in full-coverage mode. "
                f"Each global epoch now covers the full training set once."
            )
        actual_group_size = max(1, int(args.group_size))
        for epoch in range(start_epoch, args.epochs):
            epoch_indices = list(valid_indices)
            np.random.shuffle(epoch_indices)
            epoch_groups = [
                epoch_indices[i:i + actual_group_size]
                for i in range(0, len(epoch_indices), actual_group_size)
            ]
            print(
                f"\n[Epoch {epoch+1}] Full-coverage grouped pass: "
                f"{len(epoch_groups)} groups, total trajectories={len(epoch_indices)}"
            )
            run_epoch_once(epoch, epoch_indices, train_groups=epoch_groups)
    else:
        for epoch in range(start_epoch, args.epochs):
            run_epoch_once(epoch, valid_indices)

    print(f"\n{'='*60}")
    print('Training finished!')
    print(f"Total traj: {total_trajs}, success: {total_success} ({total_success/max(1,total_trajs)*100:.1f}%)")
    print(f"Best success rate (legacy metric): {best_success_rate*100:.1f}%")
    print(f"Best val status: f1_ok={best_has_target_f1ok}, val_f1={best_val_f1:.4f}, val_cr={best_val_cr:.4f}")
    print(f"{'='*60}")
    profile_path = finalize_profile(args.profile_json)
    if profile_path:
        print(f"[Profile] Saved profiling JSON: {profile_path}")


if __name__ == '__main__':
    main()
