# -*- coding: utf-8 -*-
"""
GFlowNet TLM (Trajectory Likelihood Maximization) 璁粌鍣?

瀹炵幇 ICLR 2025 璁烘枃涓殑 Algorithm 1 璁粌绛栫暐锛?
- Backward Policy 浣跨敤 TLM Loss + 鏇村皬瀛︿範鐜?+ EMA target network
- Forward Policy 浣跨敤 TB Loss + Replay Buffer
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from collections import deque
import copy
import gc

from .gfn_env import GFlowNetTrajectoryEnv, BatchGFlowNetEnv
from .models import HierarchicalGFlowNet, TrajectoryEncoder
from .frontier import FeasibleFrontierBuffer, FrontierCandidate
from profiling_utils import profile_scope


class SuccessfulTrajectoryCache:
    """
    鎴愬姛杞ㄨ抗缂撳瓨 (鑺傜渷 10-100脳 鍐呭瓨)
    
    鍙瓨鍌ㄦ垚鍔熻建杩圭殑鍘嬬缉鎽樿锛屼笉瀛樺偍瀹屾暣鏁版嵁鎴?log_pf/log_pb/rewards
    
    瀛樺偍鍐呭 (鏋佸皬):
    - trajectory_length: 鍘熷杞ㄨ抗闀垮害
    - sketch: 鏈€缁?QCS sketch (鍑犲崄 bytes)
    - actions: 鍔ㄤ綔搴忓垪
    
    閲囨牱绛栫暐: FIFO (deque) 鎴?Reservoir Sampling
    """
    
    def __init__(self, capacity: int = 10000, use_reservoir: bool = False):
        self.capacity = capacity
        self.use_reservoir = use_reservoir
        self.buffer = deque(maxlen=capacity) if not use_reservoir else []
        self.total_seen = 0  # 鐢ㄤ簬 reservoir sampling
    
    def push(self, trajectory_data: Dict):
        """娣诲姞鎴愬姛杞ㄨ抗鎽樿"""
        minimal_data = {
            'trajectory_length': trajectory_data.get('trajectory_length', 0),
            'actions': trajectory_data.get('actions', []),
            'num_points': len(trajectory_data.get('actions', []))
        }
        
        if 'sketch' in trajectory_data:
            minimal_data['sketch'] = trajectory_data['sketch']
        
        if self.use_reservoir:
            self.total_seen += 1
            if len(self.buffer) < self.capacity:
                self.buffer.append(minimal_data)
            else:
                idx = np.random.randint(0, self.total_seen)
                if idx < self.capacity:
                    self.buffer[idx] = minimal_data
        else:
            self.buffer.append(minimal_data)
    
    def sample(self, batch_size: int) -> List[Dict]:
        """闅忔満閲囨牱"""
        buffer_list = list(self.buffer) if isinstance(self.buffer, deque) else self.buffer
        indices = np.random.choice(len(buffer_list), min(batch_size, len(buffer_list)), replace=False)
        return [buffer_list[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        if self.use_reservoir:
            self.buffer = []
            self.total_seen = 0
        else:
            self.buffer.clear()


# 鍒悕锛屼繚鎸佸悜鍚庡吋瀹?
ReplayBuffer = SuccessfulTrajectoryCache


class TLMTrainer:
    """
    TLM 璁粌鍣?- Algorithm 1 瀹炵幇
    """
    
    def __init__(
        self,
        model: HierarchicalGFlowNet,
        device: str = 'cpu',
        lr_forward: float = 1e-3,
        lr_backward: float = 1e-4,  # 鏇村皬瀛︿範鐜?
        alpha: float = 0.5,
        beta: float = 0.1,
        f1_threshold: float = 0.95,
        target_compression: float = 0.03,
        ema_decay: float = 0.995,  # EMA 琛板噺鐜?
        replay_buffer_size: int = 10000,
        lr_decay_rate: float = 0.99,  # 瀛︿範鐜囪“鍑忕巼
        global_stats: Optional[dict] = None,
        frontier_size: int = 64,
        frontier_top_m: int = 8,
        dual_eta: float = 0.05,
        dual_xi: float = 0.01,
        dual_lambda_max: float = 50.0,
        stop_aux_weight: float = 0.2,
        set_loss_weight_mode: str = "reward_novelty",
        anytime_enable: bool = True,
        anytime_min_samples_train: int = 1,
        anytime_patience: int = 2,
        anytime_gain_epsilon: float = 0.01,
        anytime_uncertain_margin: float = 0.01,
        multifidelity_enable: bool = True,
        multifidelity_topk_exact: int = 2,
        multifidelity_proxy_grid_size: int = 24,
        multifidelity_proxy_stride: int = 4,
        action_pool_size: int = 64,
        action_pool_explore_ratio: float = 0.2,
        f1_safe_margin: float = 0.01,
        prs_exact_lowcr_topk: int = 2,
        prs_exact_reward_topk: int = 2,
        prs_exact_f1_topk: int = 2,
        phase2_dual_target: float = 8.0,
        phase2_dual_decay: float = 0.95,
        forward_only: bool = False,
    ):
        self.device = device
        self.model = model.to(device)
        self.alpha = alpha
        self.beta = beta
        self.f1_threshold = f1_threshold
        self.target_compression = target_compression
        self.ema_decay = ema_decay
        self.lr_decay_rate = lr_decay_rate
        self.forward_only = bool(forward_only)
        self.global_stats = global_stats
        self.frontier_top_m = int(max(1, frontier_top_m))
        self.dual_eta = float(max(0.0, dual_eta))
        self.dual_xi = float(max(0.0, dual_xi))
        self.dual_lambda_max = float(max(0.0, dual_lambda_max))
        self.stop_aux_weight = float(max(0.0, stop_aux_weight))
        self.set_loss_weight_mode = str(set_loss_weight_mode)
        self.anytime_enable = bool(anytime_enable)
        self.anytime_min_samples_train = int(max(1, anytime_min_samples_train))
        self.anytime_patience = int(max(1, anytime_patience))
        self.anytime_gain_epsilon = float(max(0.0, anytime_gain_epsilon))
        self.anytime_uncertain_margin = float(max(0.0, anytime_uncertain_margin))
        self.multifidelity_enable = bool(multifidelity_enable)
        self.multifidelity_topk_exact = int(max(1, multifidelity_topk_exact))
        self.multifidelity_proxy_grid_size = int(max(0, multifidelity_proxy_grid_size))
        self.multifidelity_proxy_stride = int(max(1, multifidelity_proxy_stride))
        self.action_pool_size = int(max(0, action_pool_size))
        self.action_pool_explore_ratio = float(np.clip(action_pool_explore_ratio, 0.0, 1.0))
        self.f1_safe_margin = float(max(0.0, f1_safe_margin))
        self.prs_exact_lowcr_topk = int(max(1, prs_exact_lowcr_topk))
        self.prs_exact_reward_topk = int(max(1, prs_exact_reward_topk))
        self.prs_exact_f1_topk = int(max(1, prs_exact_f1_topk))
        self.phase2_lowcr_active = False
        self.phase2_dual_target = float(max(0.0, phase2_dual_target))
        self.phase2_dual_decay = float(np.clip(phase2_dual_decay, 0.0, 1.0))
        
        self.env: Optional[BatchGFlowNetEnv] = None
        self.traj_emb: Optional[torch.Tensor] = None
        self.valid_lens: Optional[torch.Tensor] = None
        self.max_len: int = 0
        
        # 鍙涔犵殑 log Z (鍒嗗尯鍑芥暟)
        self.log_Z = nn.Parameter(torch.tensor(0.0, device=device))
        
        # ====== EMA Target Network for Backward Policy ======
        self.target_model = copy.deepcopy(model).to(device)
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad = False
        
        # ====== Replay Buffer for Forward Policy ======
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
        self.frontier_buffer = FeasibleFrontierBuffer(capacity=int(max(1, frontier_size)))
        self.dual_lambda_map: Dict[str, float] = {}
        
        # ====== 浼樺寲鍣?======
        self.opt_forward = optim.Adam(
            list(model.parameters()) + [self.log_Z],
            lr=lr_forward
        )
        
        backward_lr_effective = 0.0 if self.forward_only else float(lr_backward)
        self.opt_backward = optim.Adam(
            model.parameters(),
            lr=backward_lr_effective
        )
        
        self.scheduler_backward = optim.lr_scheduler.ExponentialLR(
            self.opt_backward, 
            gamma=lr_decay_rate
        )
        
        self.stats = {
            'loss_forward': [],
            'loss_backward': [],
            'mean_reward': [],
            'mean_f1': [],
            'mean_sparsity': [],
            'success_rate': [],
            'replay_buffer_size': [],
            'dual_lambda': [],
            'frontier_mean_size': [],
            'actual_samples': [],
        }
        
        self.epoch = 0
    
    def update_env(
        self,
        trajectories: List[np.ndarray],
        raw_trajectories: Optional[List[np.ndarray]] = None,
        queries: Optional[List] = None,
        batch_gt_hits: Optional[List[set]] = None
    ):
        """鏇存柊鐜鍜岄璁＄畻宓屽叆"""
        B = len(trajectories)
        
        self.env = BatchGFlowNetEnv(
            trajectories=trajectories,
            raw_trajectories=None,
            device=self.device,
            alpha=self.alpha,
            beta=self.beta,
            f1_threshold=self.f1_threshold,
            target_compression=self.target_compression,
            queries=queries,
            batch_gt_hits=batch_gt_hits,
            global_stats=self.global_stats
        )
        
        self.max_len = self.env.max_len
        feature_dim = trajectories[0].shape[1] if len(trajectories) > 0 else 3
        
        traj_tensor = torch.zeros((B, self.max_len, feature_dim), dtype=torch.float32, device=self.device)
        self.valid_lens = torch.tensor([len(t) for t in trajectories], device=self.device)
        
        for i, t in enumerate(trajectories):
            traj_tensor[i, :len(t)] = torch.tensor(t, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            self.traj_emb = self.model.traj_encoder(traj_tensor)

    
    def update_target_network(self):
        if self.forward_only:
            return
        with torch.no_grad():
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def decay_backward_lr(self):
        if self.forward_only:
            return
        self.scheduler_backward.step()

    def get_backward_lr(self) -> float:
        if self.forward_only:
            return 0.0
        return float(self.opt_backward.param_groups[0]['lr'])

    def _tau_bucket(self, tau: float) -> str:
        return f"{float(tau):.2f}"

    def _get_dual_lambda(self, tau: float) -> float:
        key = self._tau_bucket(tau)
        if key not in self.dual_lambda_map:
            self.dual_lambda_map[key] = 1.0
        return float(self.dual_lambda_map[key])

    def set_phase2_lowcr(
        self,
        active: bool,
        dual_target: Optional[float] = None,
        dual_decay: Optional[float] = None,
    ) -> None:
        self.phase2_lowcr_active = bool(active)
        if dual_target is not None:
            self.phase2_dual_target = float(max(0.0, dual_target))
        if dual_decay is not None:
            self.phase2_dual_decay = float(np.clip(dual_decay, 0.0, 1.0))

    def apply_phase2_dual_decay(self) -> Dict[str, float]:
        if (not self.phase2_lowcr_active) or (not self.dual_lambda_map):
            return {}
        target = float(max(0.0, self.phase2_dual_target))
        decay = float(np.clip(self.phase2_dual_decay, 0.0, 1.0))
        updated: Dict[str, float] = {}
        for key, value in list(self.dual_lambda_map.items()):
            current = float(value)
            if current <= target + 1e-12:
                next_value = current
            else:
                next_value = max(target, current * decay)
            next_value = float(np.clip(next_value, 0.0, self.dual_lambda_max))
            self.dual_lambda_map[key] = next_value
            updated[key] = next_value
        return updated

    def _update_dual_lambda(self, tau: float, mean_violation: float) -> float:
        key = self._tau_bucket(tau)
        cur = self._get_dual_lambda(tau)
        if self.phase2_lowcr_active:
            return cur
        nxt = cur + self.dual_eta * (float(mean_violation) - self.dual_xi)
        nxt = float(np.clip(nxt, 0.0, self.dual_lambda_max))
        self.dual_lambda_map[key] = nxt
        return nxt

    @staticmethod
    def _dual_reward(cr: float, f1_lb: float, tau: float, lam: float) -> float:
        violation = max(0.0, float(tau) - float(f1_lb))
        return float(np.exp(-(float(cr) + float(lam) * violation)) + 1e-9)

    @staticmethod
    def _default_chunk_key(
        n_points: int,
        tau: float,
        local_stats: Optional[dict] = None,
    ) -> Tuple:
        if local_stats is None:
            return ("chunk", int(n_points), f"{float(tau):.2f}")
        return (
            "chunk",
            int(n_points),
            f"{float(tau):.2f}",
            round(float(local_stats.get("x_min", 0.0)), 5),
            round(float(local_stats.get("x_max", 0.0)), 5),
            round(float(local_stats.get("y_min", 0.0)), 5),
            round(float(local_stats.get("y_max", 0.0)), 5),
            round(float(local_stats.get("t_min", 0.0)), 3),
            round(float(local_stats.get("t_max", 0.0)), 3),
        )

    def _compute_set_weights(self, candidates: List[FrontierCandidate], temperature: float = 0.2) -> torch.Tensor:
        if len(candidates) == 0:
            return torch.zeros(0, dtype=torch.float32, device=self.device)
        rewards = np.array([float(c.reward_dual) for c in candidates], dtype=np.float64)
        novelty = np.array([float(c.novelty) for c in candidates], dtype=np.float64)
        if rewards.max() - rewards.min() < 1e-12:
            rewards_norm = np.full_like(rewards, 0.5)
        else:
            rewards_norm = (rewards - rewards.min()) / (rewards.max() - rewards.min())
        if novelty.max() - novelty.min() < 1e-12:
            novelty_norm = np.full_like(novelty, 0.5)
        else:
            novelty_norm = (novelty - novelty.min()) / (novelty.max() - novelty.min())
        rank = 0.7 * rewards_norm + 0.3 * novelty_norm
        rank_t = torch.tensor(rank, dtype=torch.float32, device=self.device)
        return torch.softmax(rank_t / max(1e-6, float(temperature)), dim=0)

    def _candidate_priority_key(self, candidate: FrontierCandidate, tau: float) -> Tuple[float, ...]:
        if candidate.is_safe_feasible(tau=tau, safe_margin=self.f1_safe_margin):
            return (0.0, float(candidate.cr), -float(candidate.f1_lb), -float(candidate.reward_dual), -float(candidate.novelty))
        if candidate.feasible:
            return (1.0, float(candidate.cr), -float(candidate.f1_lb), -float(candidate.reward_dual), -float(candidate.novelty))
        return (2.0, -float(candidate.f1_lb), -float(candidate.reward_dual), float(candidate.cr), -float(candidate.novelty))

    def _sort_candidates_by_priority(
        self,
        candidates: List[FrontierCandidate],
        tau: float,
    ) -> List[FrontierCandidate]:
        return sorted(candidates, key=lambda cand: self._candidate_priority_key(cand, tau=tau))

    def _select_best_candidate(
        self,
        candidates: List[FrontierCandidate],
        tau: float,
    ) -> Optional[FrontierCandidate]:
        if not candidates:
            return None
        ranked = self._sort_candidates_by_priority(candidates, tau=tau)
        return ranked[0]

    def _build_exact_shortlist(
        self,
        bank_items: List[Dict[str, Any]],
        tau: float,
    ) -> List[Dict[str, Any]]:
        if not bank_items:
            return []

        shortlist: List[Dict[str, Any]] = []
        seen = set()

        def add_ranked(items: List[Dict[str, Any]], limit: int) -> None:
            for item in items[: max(0, int(limit))]:
                key = item["candidate"].key()
                if key in seen:
                    continue
                seen.add(key)
                shortlist.append(item)

        lowcr_ranked = sorted(
            bank_items,
            key=lambda item: self._candidate_priority_key(item["candidate"], tau=tau),
        )
        reward_ranked = sorted(
            bank_items,
            key=lambda item: (
                float(item["candidate"].reward_dual),
                1.0 if item["candidate"].is_safe_feasible(tau=tau, safe_margin=self.f1_safe_margin) else 0.0,
                float(item["candidate"].f1_lb),
                -float(item["candidate"].cr),
            ),
            reverse=True,
        )
        f1_ranked = sorted(
            bank_items,
            key=lambda item: (
                float(item["candidate"].f1_lb),
                1.0 if item["candidate"].is_safe_feasible(tau=tau, safe_margin=self.f1_safe_margin) else 0.0,
                float(item["candidate"].reward_dual),
                -float(item["candidate"].cr),
            ),
            reverse=True,
        )

        add_ranked(lowcr_ranked, self.prs_exact_lowcr_topk)
        add_ranked(reward_ranked, self.prs_exact_reward_topk)
        add_ranked(f1_ranked, self.prs_exact_f1_topk)
        return shortlist if shortlist else bank_items[:1]

    def _build_action_pool(self, env: GFlowNetTrajectoryEnv, valid: np.ndarray, pool_size: int) -> np.ndarray:
        """Build a small candidate action pool from valid point actions."""
        valid_idx = np.where(valid[:-1])[0]
        if len(valid_idx) <= 1:
            return valid_idx.astype(np.int64)
        k = int(max(1, pool_size))
        if len(valid_idx) <= k:
            return valid_idx.astype(np.int64)

        # Priority = curvature + speed, with optional random exploration.
        turn = np.abs(env.turning_angles[valid_idx]) if hasattr(env, "turning_angles") else np.zeros(len(valid_idx))
        vel = np.abs(env.velocities[valid_idx]) if hasattr(env, "velocities") else np.zeros(len(valid_idx))
        score = turn + 0.2 * vel

        num_top = int(max(1, round(k * (1.0 - self.action_pool_explore_ratio))))
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

    def _rollout_trace_with_actions(
        self,
        trajectory: np.ndarray,
        raw_trajectory: np.ndarray,
        local_stats: dict,
        actions_list: List[int],
        keep_start: bool,
        keep_end: bool,
        effective_cr_cap: float,
        tau: Optional[float] = None,
    ) -> Dict[str, Any]:
        env = GFlowNetTrajectoryEnv(
            trajectory=trajectory,
            raw_trajectory=raw_trajectory,
            alpha=self.alpha,
            beta=self.beta,
            f1_threshold=self.f1_threshold,
            target_compression=effective_cr_cap,
            device=self.device,
            global_stats=self.global_stats,
            local_stats=local_stats,
            keep_start=keep_start,
            keep_end=keep_end,
            proxy_grid_size=self.multifidelity_proxy_grid_size if self.multifidelity_enable else 0,
            proxy_stride=self.multifidelity_proxy_stride,
        )
        n = len(trajectory)
        states = [env.get_state_tensor()]
        valid_actions = []
        executed_actions = []
        done = False
        for a in actions_list:
            if done:
                break
            valid = env.get_valid_actions().astype(bool, copy=True)
            valid_actions.append(valid)
            a_int = int(a)
            if a_int < 0:
                a_int = n
            if a_int > n:
                a_int = n
            if a_int < n and not bool(valid[a_int]):
                continue
            if a_int == n and not bool(valid[n]):
                continue
            env_action = a_int if a_int < n else -1
            _, _, done, _ = env.step(env_action)
            executed_actions.append(a_int)
            states.append(env.get_state_tensor())
        if not done:
            valid = env.get_valid_actions().astype(bool, copy=True)
            valid_actions.append(valid)
            _, _, done, _ = env.step(-1)
            executed_actions.append(n)
            states.append(env.get_state_tensor())

        indices = np.where(env.mask)[0].tolist()
        eval_tau = self.f1_threshold if tau is None else float(tau)
        eval_info = env.evaluate_indices(indices, tau=eval_tau)
        env.close()
        return {
            "states": states,
            "valid_actions": valid_actions,
            "actions": executed_actions,
            "indices": sorted(set(int(i) for i in indices)),
            "f1_lb": float(eval_info["f1_lb"]),
            "cr": float(eval_info["cr"]),
            "feasible": bool(eval_info["feasible"]),
            "reward_env": float(eval_info["reward"]),
        }

    def _rollout_single_candidate(
        self,
        trajectory: np.ndarray,
        raw_trajectory: np.ndarray,
        local_stats: dict,
        temperature: float,
        keep_start: bool,
        keep_end: bool,
        max_keep: Optional[int],
        min_keep: Optional[int],
        effective_cr_cap: float,
        tau: float,
        traj_emb_sample: torch.Tensor,
        valid_lens: torch.Tensor,
        eval_mode: str = "exact",
        action_pool_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        env = GFlowNetTrajectoryEnv(
            trajectory=trajectory,
            raw_trajectory=raw_trajectory,
            alpha=self.alpha,
            beta=self.beta,
            f1_threshold=tau,
            target_compression=effective_cr_cap,
            device=self.device,
            global_stats=self.global_stats,
            local_stats=local_stats,
            keep_start=keep_start,
            keep_end=keep_end,
            proxy_grid_size=self.multifidelity_proxy_grid_size if self.multifidelity_enable else 0,
            proxy_stride=self.multifidelity_proxy_stride,
        )
        n = len(trajectory)
        states = [env.get_state_tensor()]
        valid_actions = []
        actions = []
        sampled_from_pool = []
        done = False
        step = 0
        pool_k = self.action_pool_size if action_pool_size is None else int(max(0, action_pool_size))
        use_pool = pool_k > 0
        while not done and step < n * 2:
            step += 1
            valid = env.get_valid_actions().astype(bool, copy=True)
            decision_valid = valid.copy()
            cur_kept = int(env.mask.sum())
            cur_min_keep = 2 if min_keep is None else max(2, int(min_keep))
            if max_keep is not None and cur_kept >= int(max_keep):
                decision_valid = np.zeros(n + 1, dtype=bool)
                decision_valid[-1] = True
                action_idx = n
                sampled_from_pool.append(False)
            else:
                allow_stop = cur_kept >= cur_min_keep
                decision_valid[-1] = bool(allow_stop)
                s_t = torch.tensor(states[-1], dtype=torch.float32, device=self.device).unsqueeze(0)
                # Candidate-pool re-parameterization (O(K) action head).
                should_use_pool = use_pool and int(valid[:-1].sum()) > max(2, pool_k)
                if should_use_pool:
                    pool_idx = self._build_action_pool(env, valid, pool_k)
                    decision_valid = np.zeros(n + 1, dtype=bool)
                    if len(pool_idx) > 0:
                        decision_valid[pool_idx] = True
                    decision_valid[-1] = bool(allow_stop)

                    if len(pool_idx) == 0:
                        action_idx = n
                    elif len(pool_idx) == 1 and (not allow_stop):
                        action_idx = int(pool_idx[0])
                    else:
                        cand_tensor = torch.tensor(pool_idx, dtype=torch.long, device=self.device).unsqueeze(0)
                        logits_pool = self.model.forward_policy_const_candidates(
                            traj_emb_sample,
                            s_t,
                            cand_tensor,
                            valid_lens,
                        )
                        if not allow_stop:
                            logits_pool[:, -1] = -1e9
                        logits_pool = torch.clamp(logits_pool, min=-100.0, max=100.0)
                        if int(decision_valid.sum()) <= 1:
                            action_idx = n
                        else:
                            dist = Categorical(logits=logits_pool / max(1e-4, float(temperature)))
                            local_idx = int(dist.sample().item())
                            if local_idx >= len(pool_idx):
                                action_idx = n
                            else:
                                action_idx = int(pool_idx[local_idx])
                    sampled_from_pool.append(True)
                else:
                    if not allow_stop:
                        decision_valid[-1] = False
                    logits = self.model.forward_policy_const(traj_emb_sample, s_t, valid_lens)
                    invalid = torch.tensor(~decision_valid, dtype=torch.bool, device=self.device).unsqueeze(0)
                    logits = logits.masked_fill(invalid, -1e9)
                    logits = torch.clamp(logits, min=-100.0, max=100.0)
                    if int(decision_valid.sum()) <= 1:
                        action_idx = n
                    else:
                        dist = Categorical(logits=logits / max(1e-4, float(temperature)))
                        action_idx = int(dist.sample().item())
                    sampled_from_pool.append(False)
            valid_actions.append(decision_valid)
            actions.append(action_idx)
            env_action = action_idx if action_idx < n else -1
            _, _, done, _ = env.step(env_action)
            states.append(env.get_state_tensor())
        if not done:
            valid = env.get_valid_actions().astype(bool, copy=True)
            valid_actions.append(valid)
            _, _, done, _ = env.step(-1)
            actions.append(n)
            sampled_from_pool.append(False)
            states.append(env.get_state_tensor())

        indices = np.where(env.mask)[0].tolist()
        mode = str(eval_mode).lower()
        if mode == "proxy":
            eval_info = env.evaluate_indices_proxy(indices, tau=tau, stride=self.multifidelity_proxy_stride)
        else:
            eval_info = env.evaluate_indices(indices, tau=tau)
        env.close()
        return {
            "states": states,
            "valid_actions": valid_actions,
            "actions": actions,
            "indices": sorted(set(int(i) for i in indices)),
            "f1_lb": float(eval_info["f1_lb"]),
            "cr": float(eval_info["cr"]),
            "feasible": bool(eval_info["feasible"]),
            "reward_env": float(eval_info["reward"]),
            "eval_mode": mode,
            "used_pool_steps": int(sum(1 for x in sampled_from_pool if x)),
        }

    def _compute_forward_subtb_loss(
        self,
        rollout: Dict[str, Any],
        traj_emb: torch.Tensor,
        valid_lens: torch.Tensor,
        reward_dual: float,
        temperature: float,
    ) -> torch.Tensor:
        actions = rollout.get("actions", [])
        states = rollout.get("states", [])
        valid_actions = rollout.get("valid_actions", [])
        t_len = len(actions)
        if t_len <= 0 or len(states) <= 1 or len(valid_actions) <= 0:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        state_tensor_stack = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
        valid_actions_stack = torch.tensor(np.stack(valid_actions[:t_len]), dtype=torch.bool, device=self.device)
        actions_tensor = torch.tensor(actions[:t_len], dtype=torch.long, device=self.device)

        log_flows = self.model.forward_flow(state_tensor_stack).squeeze(-1)
        log_r = torch.log(torch.tensor(max(float(reward_dual), 1e-9), dtype=torch.float32, device=self.device))
        log_r = torch.clamp(log_r, min=-100.0, max=100.0)

        log_f_t = log_flows[:t_len]
        log_f_next = log_flows[1 : t_len + 1]
        target = torch.empty_like(log_f_t)
        if t_len > 1:
            target[:-1] = log_f_next[:-1]
        target[-1] = log_r

        traj_emb_batch = traj_emb.repeat(t_len, 1, 1)
        valid_l_batch = valid_lens.repeat(t_len)
        logits = self.model.forward_policy_const(traj_emb_batch, state_tensor_stack[:t_len], valid_l_batch)
        logits = logits.masked_fill(~valid_actions_stack, -1e9)
        logits = torch.clamp(logits, min=-100.0, max=100.0)
        dist = Categorical(logits=logits / max(1e-4, float(temperature)))
        log_pf = dist.log_prob(actions_tensor)
        diff = torch.clamp(log_f_t + log_pf - target, min=-100.0, max=100.0)
        return (diff ** 2).mean()

    def _compute_forward_bc_loss(
        self,
        rollout: Dict[str, Any],
        traj_emb: torch.Tensor,
        valid_lens: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        actions = rollout.get("actions", [])
        states = rollout.get("states", [])
        valid_actions = rollout.get("valid_actions", [])
        n = int(traj_emb.shape[1])
        selected_steps = []
        selected_actions = []
        t_len = min(len(actions), len(states) - 1, len(valid_actions))
        for i in range(t_len):
            action = int(actions[i])
            if action < 0 or action >= n:
                continue
            valid = valid_actions[i]
            if action >= len(valid) or not bool(valid[action]):
                continue
            selected_steps.append(i)
            selected_actions.append(action)

        if not selected_steps:
            return torch.zeros((), dtype=torch.float32, device=self.device)

        state_tensor_stack = torch.tensor(
            np.stack([states[i] for i in selected_steps]),
            dtype=torch.float32,
            device=self.device,
        )
        valid_actions_stack = torch.tensor(
            np.stack([valid_actions[i] for i in selected_steps]),
            dtype=torch.bool,
            device=self.device,
        )
        actions_tensor = torch.tensor(selected_actions, dtype=torch.long, device=self.device)
        batch_size = len(selected_steps)
        traj_emb_batch = traj_emb.repeat(batch_size, 1, 1)
        valid_l_batch = valid_lens.repeat(batch_size)

        logits = self.model.forward_policy_const(traj_emb_batch, state_tensor_stack, valid_l_batch)
        logits = logits.masked_fill(~valid_actions_stack, -1e9)
        logits = torch.clamp(logits, min=-100.0, max=100.0)
        dist = Categorical(logits=logits / max(1e-4, float(temperature)))
        log_pf = dist.log_prob(actions_tensor)
        if not torch.isfinite(log_pf).all():
            return torch.zeros((), dtype=torch.float32, device=self.device)
        return -log_pf.mean()

    def _compute_backward_path_loss(
        self,
        trajectory: np.ndarray,
        raw_trajectory: np.ndarray,
        local_stats: dict,
        actions_list: List[int],
        traj_emb: torch.Tensor,
        valid_lens: torch.Tensor,
        keep_start: bool,
        keep_end: bool,
        effective_cr_cap: float,
    ) -> torch.Tensor:
        if self.forward_only:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        if len(actions_list) == 0:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        env = GFlowNetTrajectoryEnv(
            trajectory=trajectory,
            raw_trajectory=raw_trajectory,
            f1_threshold=self.f1_threshold,
            target_compression=effective_cr_cap,
            local_stats=local_stats,
            global_stats=self.global_stats,
            device=self.device,
            keep_start=keep_start,
            keep_end=keep_end,
        )
        n = len(trajectory)
        masks = []
        cleaned_actions = []
        for a in actions_list:
            a_int = int(a)
            if a_int < 0:
                a_int = n
            valid = env.get_valid_actions()
            if a_int >= n:
                if valid[n]:
                    env.lightweight_step(-1)
                    masks.append(torch.tensor(env.mask, dtype=torch.float32, device=self.device).unsqueeze(0))
                    cleaned_actions.append(n)
                break
            if 0 <= a_int < n and bool(valid[a_int]):
                env.lightweight_step(a_int)
                masks.append(torch.tensor(env.mask, dtype=torch.float32, device=self.device).unsqueeze(0))
                cleaned_actions.append(a_int)

        if len(cleaned_actions) == 0:
            env.close()
            return torch.zeros((), dtype=torch.float32, device=self.device)

        total = torch.zeros((), dtype=torch.float32, device=self.device)
        for t, a in enumerate(cleaned_actions):
            mask_t = masks[t]
            logits_b = self.model.forward_backward(traj_emb, mask_t, valid_lens)
            dist_b = Categorical(logits=logits_b)
            target_action = torch.tensor([int(a)], dtype=torch.long, device=self.device)
            total = total - dist_b.log_prob(target_action).mean()
        env.close()
        return total / max(1, len(cleaned_actions))
    
    # ==========================================================================
    # 鏂扮殑娴佸紡璁粌鎺ュ彛 (鐢ㄦ埛鎸囧畾鐨勬渶缁堣缁冩祦绋?
    # ==========================================================================

    def train_single_trajectory_distributional(
        self,
        trajectory: np.ndarray,
        raw_trajectory: np.ndarray,
        local_stats: dict,
        temperature: float = 1.0,
        greedy_actions: Optional[List[int]] = None,
        keep_start: bool = True,
        keep_end: bool = True,
        max_keep: Optional[int] = None,
        min_keep: Optional[int] = None,
        cr_cap_ratio: Optional[float] = None,
        chunk_key: Optional[Tuple] = None,
        num_samples_train: int = 8,
        frontier_top_m: Optional[int] = None,
        tau: Optional[float] = None,
        dynamic_cap_expand_ratio: float = 1.1,
        expert_forward_weight: float = 0.0,
    ) -> dict:
        """
        PRS-Dual-Frontier set-level training:
        - rollout multiple candidates
        - update per-chunk frontier
        - update dual variable by feasibility violation
        - optimize forward/backward policy on a set of frontier paths
        """
        self.model.train()
        n = len(trajectory)
        if n <= 2:
            return {
                "f1": 0.0,
                "reward": 1e-9,
                "num_points": max(0, n),
                "cr": 1.0,
                "cr_cap": 1.0,
                "loss_forward": 0.0,
                "loss_backward": 0.0,
                "expert_forward_skipped": True,
                "success": False,
                "indices": list(range(n)),
                "dual_lambda": self._get_dual_lambda(self.f1_threshold if tau is None else tau),
                "frontier_mean_size": self.frontier_buffer.stats().get("mean_size", 0.0),
                "actual_samples": 0,
                "exact_evals": 0,
            }

        tau = float(self.f1_threshold if tau is None else tau)
        effective_cr_cap = float(np.clip(self.target_compression if cr_cap_ratio is None else cr_cap_ratio, 1e-4, 1.0))
        if chunk_key is None:
            chunk_key = self._default_chunk_key(n, tau, local_stats=local_stats)

        max_samples = max(1, int(num_samples_train))
        top_m = int(self.frontier_top_m if frontier_top_m is None else max(1, frontier_top_m))
        min_samples = min(
            max_samples,
            self.anytime_min_samples_train if self.anytime_enable else max_samples
        )

        traj_tensor = torch.tensor(trajectory, dtype=torch.float32, device=self.device).unsqueeze(0)
        valid_lens = torch.tensor([n], dtype=torch.long, device=self.device)
        with torch.no_grad():
            traj_emb_sample = self.model.traj_encoder(traj_tensor)

        cr_cap_dyn = self.frontier_buffer.dynamic_cr_cap(
            chunk_key,
            teacher_cap=effective_cr_cap,
            expand_ratio=float(max(1.0, dynamic_cap_expand_ratio)),
        )
        max_keep_dyn = int(np.ceil(n * cr_cap_dyn))
        max_keep_dyn = max(2, min(n, max_keep_dyn))
        if max_keep is not None:
            max_keep_dyn = min(max_keep_dyn, int(max_keep))
            max_keep_dyn = max(2, max_keep_dyn)
        min_keep_dyn = 2 if min_keep is None else max(2, int(min_keep))
        min_keep_dyn = min(min_keep_dyn, max_keep_dyn)

        rollouts_by_key: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        candidate_bank: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        violations: List[float] = []
        lam = self._get_dual_lambda(tau)
        sampling_eval_mode = "proxy" if self.multifidelity_enable else "exact"
        best_reward = -1e12
        best_abs_margin = float("inf")
        no_improve = 0
        actual_samples = 0
        teacher_trace = None

        with profile_scope("train", "train_chunk_internal", "prs_rollout_sampling"):
            while actual_samples < max_samples:
                rollout = self._rollout_single_candidate(
                    trajectory=trajectory,
                    raw_trajectory=raw_trajectory,
                    local_stats=local_stats,
                    temperature=temperature,
                    keep_start=keep_start,
                    keep_end=keep_end,
                    max_keep=max_keep_dyn,
                    min_keep=min_keep_dyn,
                    effective_cr_cap=cr_cap_dyn,
                    tau=tau,
                    traj_emb_sample=traj_emb_sample,
                    valid_lens=valid_lens,
                    eval_mode=sampling_eval_mode,
                    action_pool_size=self.action_pool_size,
                )
                actual_samples += 1
                violation = max(0.0, tau - rollout["f1_lb"])
                reward_dual = self._dual_reward(rollout["cr"], rollout["f1_lb"], tau=tau, lam=lam)
                candidate = FrontierCandidate(
                    actions=[int(a) for a in rollout["actions"]],
                    indices=[int(i) for i in rollout["indices"]],
                    f1_lb=float(rollout["f1_lb"]),
                    cr=float(rollout["cr"]),
                    feasible=bool(rollout["feasible"]),
                    reward_dual=float(reward_dual),
                    novelty=0.0,
                )
                c_key = candidate.key()
                prev_item = candidate_bank.get(c_key)
                if (prev_item is None) or (candidate.reward_dual > float(prev_item["candidate"].reward_dual) + 1e-12):
                    rollout_local = dict(rollout)
                    rollout_local["reward_dual"] = float(reward_dual)
                    rollout_local["violation"] = float(violation)
                    candidate_bank[c_key] = {
                        "candidate": candidate,
                        "rollout": rollout_local,
                    }

                if reward_dual > best_reward + self.anytime_gain_epsilon:
                    best_reward = float(reward_dual)
                    no_improve = 0
                else:
                    no_improve += 1
                best_abs_margin = min(best_abs_margin, abs(float(rollout["f1_lb"]) - tau))

                if self.anytime_enable and actual_samples >= min_samples:
                    uncertain_chunk = best_abs_margin <= self.anytime_uncertain_margin
                    if not uncertain_chunk:
                        break
                    if no_improve >= self.anytime_patience:
                        break

        exact_evals = 0
        if self.multifidelity_enable:
            bank_items = list(candidate_bank.values())
            if bank_items:
                with profile_scope("train", "train_chunk_internal", "prs_multifidelity_exact"):
                    exact_subset = self._build_exact_shortlist(bank_items, tau=tau)

                    eval_env = GFlowNetTrajectoryEnv(
                        trajectory=trajectory,
                        raw_trajectory=raw_trajectory,
                        alpha=self.alpha,
                        beta=self.beta,
                        f1_threshold=tau,
                        target_compression=cr_cap_dyn,
                        device=self.device,
                        global_stats=self.global_stats,
                        local_stats=local_stats,
                        keep_start=keep_start,
                        keep_end=keep_end,
                        proxy_grid_size=self.multifidelity_proxy_grid_size if self.multifidelity_enable else 0,
                        proxy_stride=self.multifidelity_proxy_stride,
                    )
                    try:
                        for item in exact_subset:
                            cand_proxy = item["candidate"]
                            rollout_proxy = item["rollout"]
                            exact = eval_env.evaluate_indices(cand_proxy.indices, tau=tau)
                            reward_exact = self._dual_reward(exact["cr"], exact["f1_lb"], tau=tau, lam=lam)
                            cand_exact = FrontierCandidate(
                                actions=[int(a) for a in cand_proxy.actions],
                                indices=[int(i) for i in cand_proxy.indices],
                                f1_lb=float(exact["f1_lb"]),
                                cr=float(exact["cr"]),
                                feasible=bool(exact["feasible"]),
                                reward_dual=float(reward_exact),
                                novelty=float(cand_proxy.novelty),
                            )
                            self.frontier_buffer.update(chunk_key, cand_exact)
                            rollout_exact = dict(rollout_proxy)
                            rollout_exact["f1_lb"] = float(exact["f1_lb"])
                            rollout_exact["cr"] = float(exact["cr"])
                            rollout_exact["feasible"] = bool(exact["feasible"])
                            rollout_exact["reward_env"] = float(exact["reward"])
                            rollout_exact["reward_dual"] = float(reward_exact)
                            rollouts_by_key[cand_exact.key()] = rollout_exact
                            violations.append(max(0.0, tau - float(exact["f1_lb"])))
                            exact_evals += 1
                    finally:
                        eval_env.close()
        else:
            for item in candidate_bank.values():
                cand = item["candidate"]
                rollout = item["rollout"]
                self.frontier_buffer.update(chunk_key, cand)
                rollouts_by_key[cand.key()] = rollout
                violations.append(float(rollout.get("violation", max(0.0, tau - float(cand.f1_lb)))))

        # Optional teacher action path as additional frontier item.
        if greedy_actions:
            with profile_scope("train", "train_chunk_internal", "prs_teacher_trace"):
                try:
                    teacher_trace = self._rollout_trace_with_actions(
                        trajectory=trajectory,
                        raw_trajectory=raw_trajectory,
                        local_stats=local_stats,
                        actions_list=list(greedy_actions) + [n],
                        keep_start=keep_start,
                        keep_end=keep_end,
                        effective_cr_cap=cr_cap_dyn,
                        tau=tau,
                    )
                    violation = max(0.0, tau - teacher_trace["f1_lb"])
                    reward_dual = self._dual_reward(teacher_trace["cr"], teacher_trace["f1_lb"], tau=tau, lam=lam)
                    teacher_candidate = FrontierCandidate(
                        actions=[int(a) for a in teacher_trace["actions"]],
                        indices=[int(i) for i in teacher_trace["indices"]],
                        f1_lb=float(teacher_trace["f1_lb"]),
                        cr=float(teacher_trace["cr"]),
                        feasible=bool(teacher_trace["feasible"]),
                        reward_dual=float(reward_dual),
                    )
                    self.frontier_buffer.update(chunk_key, teacher_candidate)
                    rollouts_by_key[teacher_candidate.key()] = teacher_trace
                    violations.append(float(violation))
                except Exception:
                    teacher_trace = None

        mean_violation = float(np.mean(violations)) if violations else 0.0
        lam = self._update_dual_lambda(tau, mean_violation)

        selected = self.frontier_buffer.sample_top_m(
            chunk_key,
            m=top_m,
            tau=tau,
            safe_margin=self.f1_safe_margin,
        )
        if not selected:
            return {
                "f1": 0.0,
                "reward": 1e-9,
                "num_points": 0,
                "cr": 1.0,
                "cr_cap": cr_cap_dyn,
                "loss_forward": 0.0,
                "loss_backward": 0.0,
                "expert_forward_skipped": True,
                "success": False,
                "indices": [],
                "dual_lambda": lam,
                "frontier_mean_size": self.frontier_buffer.stats().get("mean_size", 0.0),
                "actual_samples": int(actual_samples),
                "exact_evals": int(exact_evals),
            }

        weights = self._compute_set_weights(selected, temperature=0.2)
        traj_emb_fwd = self.model.traj_encoder(traj_tensor)

        # Forward set-level loss (SubTB-like).
        forward_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        stop_aux_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        expert_forward_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        expert_forward_weight = float(max(0.0, expert_forward_weight))
        for idx, cand in enumerate(selected):
            w = weights[idx]
            key = cand.key()
            rollout = rollouts_by_key.get(key)
            if rollout is None:
                rollout = self._rollout_trace_with_actions(
                    trajectory=trajectory,
                    raw_trajectory=raw_trajectory,
                    local_stats=local_stats,
                    actions_list=list(cand.actions),
                    keep_start=keep_start,
                    keep_end=keep_end,
                    effective_cr_cap=cr_cap_dyn,
                    tau=tau,
                )
            lf = self._compute_forward_subtb_loss(
                rollout=rollout,
                traj_emb=traj_emb_fwd,
                valid_lens=valid_lens,
                reward_dual=float(cand.reward_dual),
                temperature=temperature,
            )
            forward_loss = forward_loss + w * lf

            # Stop auxiliary target from f1_margin + cr_usage.
            final_state = rollout["states"][-1] if rollout.get("states") else None
            if final_state is not None:
                s_t = torch.tensor(final_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                stop_logit = self.model.forward_policy_const(traj_emb_fwd, s_t, valid_lens)[0, -1]
                f1_margin = float(cand.f1_lb - tau)
                cr_usage = float(cand.cr / max(1e-6, cr_cap_dyn))
                target = torch.sigmoid(torch.tensor(4.0 * f1_margin + 2.0 * (cr_usage - 0.5), device=self.device))
                stop_aux_loss = stop_aux_loss + w * F.binary_cross_entropy_with_logits(
                    stop_logit.unsqueeze(0),
                    target.unsqueeze(0),
                )

        expert_forward_skipped = True
        if expert_forward_weight > 0.0 and teacher_trace is not None:
            expert_forward_loss = self._compute_forward_bc_loss(
                rollout=teacher_trace,
                traj_emb=traj_emb_fwd,
                valid_lens=valid_lens,
                temperature=temperature,
            )
            expert_forward_skipped = bool(float(expert_forward_loss.detach().item()) <= 0.0)

        with profile_scope("train", "train_chunk_internal", "prs_forward_update"):
            total_forward = (
                forward_loss
                + self.stop_aux_weight * stop_aux_loss
                + expert_forward_weight * expert_forward_loss
            )
            self.opt_forward.zero_grad()
            total_forward.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt_forward.step()

        # Backward set-level TLM loss.
        backward_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if not self.forward_only:
            with profile_scope("train", "train_chunk_internal", "prs_backward_update"):
                traj_emb_back = self.model.traj_encoder(traj_tensor)
                for idx, cand in enumerate(selected):
                    w = weights[idx]
                    lb = self._compute_backward_path_loss(
                        trajectory=trajectory,
                        raw_trajectory=raw_trajectory,
                        local_stats=local_stats,
                        actions_list=list(cand.actions),
                        traj_emb=traj_emb_back,
                        valid_lens=valid_lens,
                        keep_start=keep_start,
                        keep_end=keep_end,
                        effective_cr_cap=cr_cap_dyn,
                    )
                    backward_loss = backward_loss + w * lb

                self.opt_backward.zero_grad()
                backward_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt_backward.step()

        # Select representative candidate for logging.
        best = self._select_best_candidate(selected, tau=tau)
        if best is None:
            best = selected[0]

        stats = self.frontier_buffer.stats()
        return {
            "f1": float(best.f1_lb),
            "reward": float(best.reward_dual),
            "num_points": int(len(best.indices)),
            "cr": float(best.cr),
            "cr_cap": float(cr_cap_dyn),
            "loss_forward": float(total_forward.detach().item()),
            "loss_backward": float(backward_loss.detach().item()),
            "loss_stop_aux": float(stop_aux_loss.detach().item()),
            "loss_expert_forward": float(expert_forward_loss.detach().item()),
            "expert_forward_skipped": bool(expert_forward_skipped),
            "success": bool(best.f1_lb >= tau),
            "indices": [int(i) for i in best.indices],
            "dual_lambda": float(lam),
            "frontier_mean_size": float(stats.get("mean_size", 0.0)),
            "frontier_mean_feasible": float(stats.get("mean_feasible", 0.0)),
            "frontier_mean_novelty": float(stats.get("mean_novelty", 0.0)),
            "max_keep_dyn": int(max_keep_dyn),
            "cr_cap_dyn": float(cr_cap_dyn),
            "actual_samples": int(actual_samples),
            "exact_evals": int(exact_evals),
        }

    def train_single_trajectory(
        self,
        trajectory: np.ndarray,
        raw_trajectory: np.ndarray,
        local_stats: dict,
        temperature: float = 1.0,
        greedy_actions: Optional[List[int]] = None,
        keep_start: bool = True,
        keep_end: bool = True,
        max_keep: Optional[int] = None,
        min_keep: Optional[int] = None,
        cr_cap_ratio: Optional[float] = None,
        expert_forward_weight: float = 0.2,
        expert_bc_ratio_cap: float = 0.35,
        subtb_lambda: float = 0.9,
        subtb_k: int = 16
    ) -> dict:
        """璁粌鍗曟潯杞ㄨ抗 (Sub-Trajectory Balance w/ Loss Accumulation)"""
        self.model.train()
        
        N = len(trajectory)
        effective_cr_cap = float(
            np.clip(
                self.target_compression if cr_cap_ratio is None else cr_cap_ratio,
                1e-4,
                1.0
            )
        )
        
        # 1. 鐜鍒濆鍖?
        from .gfn_env import GFlowNetTrajectoryEnv
        env = GFlowNetTrajectoryEnv(
            trajectory=trajectory,
            raw_trajectory=raw_trajectory,
            alpha=self.alpha,
            beta=self.beta,
            f1_threshold=self.f1_threshold,
            target_compression=effective_cr_cap,
            device=self.device,
            global_stats=self.global_stats,
            local_stats=local_stats,
            keep_start=keep_start,
            keep_end=keep_end
        )
        
        valid_lens = torch.tensor([N], device=self.device)
        traj_tensor = torch.tensor(trajectory, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # 2. 閲囨牱闃舵 (No Grad, 瀹屽叏 Detach)
        state_tensor_traj = [] # 瀛樺偍 ConstantState.to_tensor()
        action_traj = []
        valid_actions_traj = []  # valid action mask per decision state [T, N+1]
        
        state_tensor_traj.append(env.get_state_tensor())
        
        with torch.no_grad():
            traj_emb_sample = self.model.traj_encoder(traj_tensor)
        
        # mask = env.mask.copy() # (unused in const policy loop)
        done = False
        step = 0
        reward = 0.0
        with torch.no_grad():
            while not done and step < N * 2:
                step += 1

                current_kept = int(env.mask.sum())
                current_min_keep = 2 if min_keep is None else max(2, int(min_keep))
                valid_actions = env.get_valid_actions().astype(bool, copy=True)
                if max_keep is not None and current_kept >= max_keep:
                    # Hard-cap regime: once budget is used up, only terminate is admissible.
                    valid_actions = np.zeros(N + 1, dtype=bool)
                    valid_actions[-1] = True
                    action_idx = int(N)
                else:
                    if current_kept < current_min_keep:
                        valid_actions[-1] = False
                    # s_t -> s_t_tensor
                    s_t_tensor = torch.tensor(state_tensor_traj[-1], dtype=torch.float32, device=self.device).unsqueeze(0)

                    logits_f = self.model.forward_policy_const(traj_emb_sample, s_t_tensor, valid_lens)
                    invalid_actions = torch.tensor(~valid_actions, dtype=torch.bool, device=self.device).unsqueeze(0)
                    logits_f = logits_f.masked_fill(invalid_actions, -1e9)
                    logits_f = torch.clamp(logits_f, min=-100.0, max=100.0)
                    # Fallback to terminate if nothing usable remains.
                    if int(valid_actions.sum()) <= 1:
                        action_idx = int(N)
                    else:
                        dist_f = Categorical(logits=logits_f / temperature)
                        action_idx = int(dist_f.sample().item())

                action_traj.append(action_idx)
                valid_actions_traj.append(valid_actions)

                # Step
                env_action = action_idx if action_idx < N else -1
                next_mask, reward, done, _ = env.step(env_action)
                
                state_tensor_traj.append(env.get_state_tensor())
        
        final_reward = reward
        traj_len = len(action_traj)
        
        current_f1 = env._last_f1 if hasattr(env, '_last_f1') else 0.0
        sampled_indices = np.where(env.mask)[0].tolist()
        sampled_cr = len(sampled_indices) / max(1, N)
        success_bool = (current_f1 >= self.f1_threshold) and (sampled_cr <= effective_cr_cap + 1e-9) and (final_reward > 0)

        metrics = {
            'f1': current_f1,
            'reward': final_reward,
            'num_points': env.const_state.num_points,
            'cr': sampled_cr,
            'cr_cap': effective_cr_cap,
            'loss_forward': 0.0,
            'loss_backward': 0.0,
            'expert_forward_skipped': False,
            'success': success_bool,
            'indices': sampled_indices  # return kept point indices, not raw action sequence
        }

        def sanitize_action_sequence(actions_list: List[int]) -> List[int]:
            """
            Ensure expert actions are valid under environment constraints step-by-step.
            This is required because keep_start/keep_end pre-select endpoints, while
            greedy actions can still include them.
            """
            if not actions_list:
                return []

            s_env = GFlowNetTrajectoryEnv(
                trajectory=trajectory,
                raw_trajectory=raw_trajectory,
                f1_threshold=self.f1_threshold,
                target_compression=effective_cr_cap,
                local_stats=local_stats,
                device=self.device,
                keep_start=keep_start,
                keep_end=keep_end
            )
            cleaned: List[int] = []
            terminated = False
            try:
                for a in actions_list:
                    a_int = int(a)
                    if a_int < 0:
                        a_int = int(N)
                    valid = s_env.get_valid_actions()
                    if a_int >= N:
                        if valid[N]:
                            cleaned.append(int(N))
                        terminated = True
                        break
                    if 0 <= a_int < N and bool(valid[a_int]):
                        cleaned.append(a_int)
                        s_env.lightweight_step(a_int)
                    # Invalid expert action is skipped.

                if (not terminated) and bool(s_env.get_valid_actions()[N]):
                    cleaned.append(int(N))
            finally:
                if hasattr(s_env, 'close'):
                    s_env.close()
            return cleaned
        
        # [Fallback to Loss Accumulation with Flow]
        # 杩欐槸 SubTB(1) 鐨勭壒渚?
        
        loss_forward_total = 0.0
        
        if traj_len > 0:
            # 1. 棰勮绠楁墍鏈夌姸鎬?Flow
            state_tensor_stack = torch.tensor(np.stack(state_tensor_traj), dtype=torch.float32, device=self.device)
            valid_actions_stack = torch.tensor(np.stack(valid_actions_traj), dtype=torch.bool, device=self.device)
            log_flows = self.model.forward_flow(state_tensor_stack).squeeze(-1) # [T+1]
            
            # 2. Gradient Accumulation / Batch Loss
            log_R = torch.tensor(max(final_reward, 1e-9), device=self.device).log()
            log_R = torch.clamp(log_R, min=-100.0, max=100.0)
            
            # [Fix] 璁＄畻 traj_emb (甯︽搴? 鐢ㄤ簬璁粌
            traj_emb = self.model.traj_encoder(traj_tensor)
            
            self.opt_forward.zero_grad()
            
            # [Optimization] Vectorized SubTB Loop
            # 鍑嗗鏁版嵁
            states_batch = state_tensor_stack[:traj_len] # [T, 18]
            actions_batch = torch.tensor(action_traj, device=self.device) # [T]
            
            log_F_t = log_flows[:traj_len]
            log_F_next = log_flows[1:traj_len+1]
            
            # 鏋勯€?target_val_batch
            # 鍓?T-1 涓槸 log_F(s_{t+1}), 鏈€鍚庝竴涓槸 log_R
            if traj_len > 0:
                 target_val_batch = torch.empty_like(log_F_t)
                 if traj_len > 1:
                     target_val_batch[:-1] = log_F_next[:-1]
                 target_val_batch[-1] = log_R
                 
                 # 鎵归噺璁＄畻 log_pf
                 traj_emb_batch = traj_emb.repeat(traj_len, 1, 1)
                 valid_l_batch = valid_lens.repeat(traj_len)
                 
                 logits_batch = self.model.forward_policy_const(traj_emb_batch, states_batch, valid_l_batch)
                 logits_batch = logits_batch.masked_fill(~valid_actions_stack, -1e9)
                 logits_batch = torch.clamp(logits_batch, min=-100.0, max=100.0)
                 dist_batch = Categorical(logits=logits_batch / temperature)
                 log_pf_batch = dist_batch.log_prob(actions_batch)
                 
                 # Balance
                 diff = log_F_t + log_pf_batch - target_val_batch
                 diff = torch.clamp(diff, min=-100.0, max=100.0)
                 loss_steps = (diff ** 2) / max(1, traj_len)
                 
                 loss_accum = loss_steps.sum()
                 loss_forward_total = loss_accum.item()
                 
                 loss_accum.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt_forward.step()
            
            metrics['loss_forward'] = loss_forward_total

        # ====== Step 3: Backward Policy (TLM) Training ======
        
        loss_backward_val = 0.0
        
        def train_pb(actions_list):
            if not actions_list: return 0.0
            
            p_env = GFlowNetTrajectoryEnv(
                trajectory=trajectory,
                raw_trajectory=raw_trajectory,
                f1_threshold=self.f1_threshold,
                target_compression=effective_cr_cap,
                local_stats=local_stats,
                device=self.device,
                keep_start=keep_start,
                keep_end=keep_end
            )
            
            states = [p_env.get_state_tensor()]
            masks = [torch.tensor(p_env.mask, dtype=torch.float32, device=self.device).unsqueeze(0)]
            
            for action in actions_list:
                 env_act = action if action < N else -1
                 p_env.lightweight_step(env_act)
                 states.append(p_env.get_state_tensor())
                 masks.append(torch.tensor(p_env.mask, dtype=torch.float32, device=self.device).unsqueeze(0))
            
            total_log_pb = 0.0
            steps = len(actions_list)
            
            for t in range(steps):
                s_next_mask = masks[t+1]
                target_action = torch.tensor([actions_list[t]], device=self.device)
                
                # 浣跨敤 traj_emb (Gradient Accumulation 涔嬪墠姹傜殑, 闇€瑕侀噸鏂版眰鍚? 
                # 涓嶏紝Backward Policy 鐙珛锛屽彲浠ヤ娇鐢ㄦ柊鐨?traj_emb 鎴栬€?detach 鐨?
                # 涓轰簡绠€鍗曞拰鐪佹樉瀛橈紝鎴戜滑鍋囪 traj_emb_sample (detached) 瓒冲锛?
                # 涓嶏紝闇€瑕佹搴︺€?
                # 閲嶆柊璁＄畻 traj_emb?
                
                # [Optimization] 濡傛灉鏄惧瓨绱у紶锛屽彲浠ュ彧璁＄畻 backward pass 鐨?traj_emb
                # 浣嗘澶勬垜浠噸鐢?loss forward 闃舵鐨?traj_emb? 
                # 涓嶈锛宱pt_forward.step() 宸茬粡鏀瑰彉浜嗗弬鏁帮紝涔嬪墠鐨?traj_emb graph 澶辨晥銆?
                # 蹇呴』閲嶆柊璁＄畻銆?
                
                pass
            
            # Recompute traj_emb for backward policy
            # (Small cost compared to full rollout)
            with torch.no_grad():
                # Backward policy uses target network style? Or main model?
                # Main model.
                pass
            
            traj_emb_back = self.model.traj_encoder(traj_tensor)
            
            for t in range(steps):
                s_next_mask = masks[t+1]
                target_action = torch.tensor([actions_list[t]], device=self.device)
                
                logits_b = self.model.forward_backward(traj_emb_back, s_next_mask, valid_lens)
                dist_b = Categorical(logits=logits_b)
                total_log_pb += dist_b.log_prob(target_action)
            
            loss = -total_log_pb / max(1, steps)
            
            self.opt_backward.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt_backward.step()
            if hasattr(p_env, 'close'):
                p_env.close()
            
            return loss.item()

        # 1. Train on Greedy Actions (Expert)
        greedy_actions_full = None
        greedy_indices = []
        greedy_ratio = 0.0
        if greedy_actions:
             greedy_indices = sorted({int(a) for a in greedy_actions if 0 <= int(a) < N})
             if keep_start:
                 greedy_indices.append(0)
             if keep_end and N > 1:
                 greedy_indices.append(N - 1)
             greedy_indices = sorted(set(greedy_indices))
             greedy_ratio = len(greedy_indices) / max(1, N)
             if greedy_actions[-1] != N and greedy_actions[-1] != -1:
                 greedy_actions_full = greedy_actions + [int(N)]
             else:
                 greedy_actions_full = greedy_actions
             greedy_actions_full = sanitize_action_sequence(greedy_actions_full)
             if not self.forward_only:
                 loss_backward_val += train_pb(greedy_actions_full)
        
        # 2. Train on Sampled Actions (if successful)
        if (not self.forward_only) and metrics['success'] and len(action_traj) > 0:
             loss_backward_val += train_pb(action_traj)
             loss_backward_val /= 2.0
             
        metrics['loss_backward'] = loss_backward_val
        metrics['greedy_cr'] = greedy_ratio

        # [Fix for Pts=0] Train Forward Policy on Expert Trajectory (Teacher Forcing / Behavior Cloning)
        if greedy_actions and greedy_actions_full is not None:
             should_train_expert_forward = (expert_forward_weight > 0.0) and (greedy_ratio <= expert_bc_ratio_cap)
             metrics['expert_forward_skipped'] = not should_train_expert_forward

             def train_pf_expert(actions_list):
                 if not actions_list:
                     return 0.0
                 # Avoid over-biasing terminate logits in BC; sampled SubTB already learns stopping.
                 actions_list = [int(a) for a in actions_list if int(a) < N]
                 if len(actions_list) == 0:
                     return 0.0
                 p_env = GFlowNetTrajectoryEnv(
                    trajectory=trajectory,
                    raw_trajectory=raw_trajectory,
                    f1_threshold=self.f1_threshold,
                    target_compression=effective_cr_cap,
                    local_stats=local_stats,
                    device=self.device,
                    keep_start=keep_start,
                    keep_end=keep_end
                 )
                 
                 # 鏀堕泦 expert trajectory states
                 states = [p_env.get_state_tensor()]
                 valid_actions = []
                 for action in actions_list:
                     valid_actions.append(p_env.get_valid_actions().astype(bool, copy=True))
                     env_act = action if action < N else -1
                     p_env.lightweight_step(env_act)
                     states.append(p_env.get_state_tensor())
                 if hasattr(p_env, 'close'):
                     p_env.close()
                 
                 traj_emb_exp = self.model.traj_encoder(traj_tensor)
                 steps = len(actions_list)
                 
                 # Stack states
                 state_stack_full = torch.tensor(np.stack(states[:-1]), dtype=torch.float32, device=self.device)
                 valid_actions_full = torch.tensor(np.stack(valid_actions), dtype=torch.bool, device=self.device)
                 targets_full = torch.tensor(actions_list, device=self.device)
                 valid_l = torch.tensor([N], device=self.device)
                 
                 # Chunking to avoid OOM
                 batch_size = 64
                 total_loss = 0.0
                 self.opt_forward.zero_grad()
                 
                 for i in range(0, steps, batch_size):
                     chunk_end = min(i + batch_size, steps)
                     current_batch_size = chunk_end - i
                     
                     state_stack = state_stack_full[i:chunk_end]
                     valid_actions_batch = valid_actions_full[i:chunk_end]
                     targets = targets_full[i:chunk_end]
                     
                     # Expand embeddings for this chunk only
                     traj_emb_batch = traj_emb_exp.repeat(current_batch_size, 1, 1)
                     valid_l_batch = valid_l.repeat(current_batch_size)
                     
                     logits_batch = self.model.forward_policy_const(traj_emb_batch, state_stack, valid_l_batch)
                     logits_batch = logits_batch.masked_fill(~valid_actions_batch, -1e9)
                     
                     # Check for NaNs (Safety)
                     if torch.isnan(logits_batch).any():
                          print(f"[Values] Logits contain NaNs in expert chunk {i}!")
                          break
                          
                     dist = Categorical(logits=logits_batch)
                     log_probs = dist.log_prob(targets)
                     if not torch.isfinite(log_probs).all():
                          # Skip unstable chunk instead of poisoning gradients.
                          continue
                     
                     unweighted_loss = -log_probs.mean()
                     total_loss += unweighted_loss.item() * current_batch_size
                     
                     # Accumulate weighted gradients
                     retain = (i + batch_size < steps)
                     (unweighted_loss * expert_forward_weight).backward(retain_graph=retain)
                 
                 avg_loss = total_loss / max(1, steps)
                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                 self.opt_forward.step()
                 
                 return avg_loss

             if should_train_expert_forward:
                 exp_loss = train_pf_expert(greedy_actions_full)
                 metrics['loss_expert'] = exp_loss

             # [Log Update] If sampled trajectory was empty (Pts=0), report expert stats
             if metrics['num_points'] == 0 or not metrics['success']:
                 metrics['indices'] = greedy_indices
                 metrics['num_points'] = len(greedy_indices)

        # ====== Step 6: 鍐呭瓨娣卞害娓呯悊 ======
        import gc
        if hasattr(env, 'close'):
            env.close()
        del traj_tensor
        if 'state_tensor_stack' in locals():
            del state_tensor_stack
        if 'log_flows' in locals():
            del log_flows
        if 'traj_emb' in locals():
            del traj_emb
        
        if N > 5000:
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
        return metrics


    
    def train_step(
        self,
        n_backward_updates: int = 1,
        n_forward_updates: int = 1,
        temperature: float = 1.0,
        use_replay: bool = True,
        replay_batch_size: int = 32
    ) -> Dict[str, float]:
        """鎵ц涓€姝?Algorithm 1 璁粌 (Original API)"""
        self.model.train()
        
        # ====== Step 1: 閲囨牱鍘嬬缉杞ㄨ抗 蟿 ======
        traj_data, pad_mask, batch_f1, batch_success = self.sample_trajectories_with_f1(
            temperature=temperature
        )
        
        success_rate = batch_success.float().mean().item()
        
        # ====== Step 2: Replay Buffer ======
        if batch_success.any():
            self._add_to_replay_buffer(traj_data, batch_success)
        
        # ====== Step 3: 鏇存柊 P_B (TLM) ======
        loss_B = 0.0
        if (not self.forward_only) and batch_success.any():
            for _ in range(n_backward_updates):
                loss_backward = self.compute_tlm_loss(traj_data, pad_mask, batch_success)
                
                if torch.isnan(loss_backward) or loss_backward.item() == 0:
                    continue
                
                self.opt_backward.zero_grad()
                loss_backward.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt_backward.step()
                loss_B = loss_backward.item()
        
        # ====== Step 4: 鏇存柊 P_F (GFlowNet online) ======
        loss_TB = 0.0
        
        for _ in range(n_forward_updates):
            traj_data_new, _, _, _ = self.sample_trajectories_with_f1(
                temperature=temperature
            )
            
            loss_forward = self.compute_tb_loss(traj_data_new)
            
            if torch.isnan(loss_forward):
                continue
            
            self.opt_forward.zero_grad()
            loss_forward.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt_forward.step()
            loss_TB = max(loss_TB, loss_forward.item())
        
        # ====== Step 6: 鏇存柊 P_B target network (EMA) ======
        self.update_target_network()
        
        # 鏀堕泦鎸囨爣
        metrics = {
            'loss_backward': loss_B,
            'loss_forward': loss_TB,
            'mean_reward': traj_data['rewards'].mean().item(),
            'mean_f1': batch_f1.mean().item(),
            'success_rate': success_rate,
            'log_Z': self.log_Z.item(),
            'backward_lr': self.get_backward_lr(),
            'replay_buffer_size': len(self.replay_buffer)
        }
        
        self.stats['loss_backward'].append(loss_B)
        self.stats['loss_forward'].append(loss_TB)
        self.stats['mean_reward'].append(metrics['mean_reward'])
        self.stats['success_rate'].append(success_rate)
        self.stats['replay_buffer_size'].append(len(self.replay_buffer))
        
        return metrics
    
    def sample_trajectories_with_f1(
        self,
        temperature: float = 1.0
    ) -> Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """閲囨牱杞ㄨ抗骞惰繑鍥?F1 鍒嗘暟"""
        B = self.env.batch_size
        MaxL = self.max_len
        
        masks = self.env.reset()
        mask_t = torch.tensor(masks, dtype=torch.float32, device=self.device)
        
        done = np.zeros(B, dtype=bool)
        
        trajectory_states = [mask_t.clone()]
        trajectory_actions = []
        
        batch_log_pf = []
        batch_log_pb = []
        batch_rewards = torch.zeros(B, device=self.device)
        
        step_count = 0
        max_steps = MaxL * 2
        
        MAX_STATES_KEPT = 20
        
        while not done.all() and step_count < max_steps:
            step_count += 1
            
            # Forward Pass
            logits_f = self.model.forward_policy(self.traj_emb, mask_t, self.valid_lens)
            
            range_tensor = torch.arange(MaxL, device=self.device).unsqueeze(0).expand(B, -1)
            pad_mask = range_tensor >= self.valid_lens.unsqueeze(1)
            logits_f[:, :-1] = logits_f[:, :-1].masked_fill(pad_mask, -1e9)
            
            logits_f = torch.where(torch.isfinite(logits_f), logits_f, torch.tensor(-100.0, device=self.device))
            logits_f = torch.clamp(logits_f, min=-100.0, max=100.0)
            
            dist_f = Categorical(logits=logits_f / temperature)
            actions = dist_f.sample()
            
            log_pf = dist_f.log_prob(actions)
            trajectory_actions.append(actions.clone())
            
            cpu_actions = actions.cpu().numpy()
            env_actions = [a if a < MaxL else -1 for a in cpu_actions]
            
            next_masks, rewards, new_dones, _ = self.env.step(env_actions)
            next_mask_t = torch.tensor(next_masks, dtype=torch.float32, device=self.device)
            
            trajectory_states.append(next_mask_t.clone())
            
            if self.forward_only:
                log_pb = torch.zeros_like(log_pf)
            else:
                with torch.no_grad():
                    logits_b = self.target_model.forward_backward(self.traj_emb, next_mask_t, self.valid_lens)
                logits_b[:, :-1] = logits_b[:, :-1].masked_fill(pad_mask, -1e9)
                
                dist_b = Categorical(logits=logits_b)
                log_pb = dist_b.log_prob(actions)
            
            batch_log_pf.append(log_pf)
            batch_log_pb.append(log_pb)
            
            mask_t = next_mask_t
            done = new_dones
            
            current_rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            batch_rewards = torch.max(batch_rewards, current_rewards)
            
            if len(trajectory_states) > MAX_STATES_KEPT:
                trajectory_states = [trajectory_states[0]] + trajectory_states[-(MAX_STATES_KEPT-1):]
                trajectory_actions = trajectory_actions[-(MAX_STATES_KEPT-1):]
            
            if step_count % 50 == 0:
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
        
        batch_f1 = torch.zeros(B, device=self.device)
        for i, env in enumerate(self.env.envs):
            if hasattr(env, '_last_f1'):
                batch_f1[i] = env._last_f1
        
        batch_success = batch_f1 >= self.f1_threshold
        
        if len(batch_log_pf) > 0:
            log_pf_stack = torch.stack(batch_log_pf)
            log_pb_stack = torch.stack(batch_log_pb)
        else:
            log_pf_stack = torch.zeros(0, B, device=self.device)
            log_pb_stack = torch.zeros(0, B, device=self.device)
        
        traj_data = {
            'log_pf': log_pf_stack,
            'log_pb': log_pb_stack,
            'rewards': batch_rewards,
            'steps': step_count,
            'states': trajectory_states,
            'actions': trajectory_actions,
            'f1': batch_f1
        }
        
        return traj_data, pad_mask, batch_f1, batch_success
    
    def _add_to_replay_buffer(self, traj_data: Dict, success_mask: torch.Tensor):
        B = success_mask.size(0)
        
        for i in range(B):
            if success_mask[i]:
                actions_list = [a[i].item() for a in traj_data.get('actions', [])]
                
                single_traj = {
                    'trajectory_length': self.env.envs[i].N if self.env else 0,
                    'actions': actions_list,
                }
                
                if self.env and hasattr(self.env.envs[i], 'qcs_compressed'):
                    single_traj['sketch'] = self.env.envs[i].qcs_compressed
                
                self.replay_buffer.push(single_traj)
    
    def _sample_from_replay(self, batch_size: int) -> Optional[List[Dict]]:
        if len(self.replay_buffer) < batch_size or batch_size <= 0:
            return None
        
        samples = self.replay_buffer.sample(batch_size)
        
        if not samples:
            return None
        
        return samples
    
    def compute_tlm_loss(
        self,
        traj_data: Dict[str, torch.Tensor],
        pad_mask: torch.Tensor,
        success_mask: torch.Tensor
    ) -> torch.Tensor:
        """璁＄畻 TLM 鎹熷け (浠呯敤浜?Backward Policy)"""
        if self.forward_only:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        if success_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        states = traj_data.get('states', [])
        actions = traj_data.get('actions', [])
        
        if len(states) < 2 or len(actions) < 1:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        B = self.traj_emb.size(0)
        MaxL = self.max_len
        
        total_log_pb = torch.zeros(B, device=self.device)
        
        for t in range(1, min(len(states), len(actions) + 1)):
            current_mask = states[t]
            action = actions[t - 1]
            
            logits_b = self.model.forward_backward(self.traj_emb, current_mask, self.valid_lens)
            logits_b = torch.where(torch.isfinite(logits_b), logits_b, torch.tensor(-100.0, device=self.device))
            logits_b = torch.clamp(logits_b, min=-100.0, max=100.0)
            
            range_tensor = torch.arange(MaxL, device=self.device).unsqueeze(0).expand(B, -1)
            pad_mask_t = range_tensor >= self.valid_lens.unsqueeze(1)
            logits_b[:, :-1] = logits_b[:, :-1].masked_fill(pad_mask_t, -100.0)
            
            dist_b = torch.distributions.Categorical(logits=logits_b)
            log_pb = dist_b.log_prob(action)
            log_pb = torch.where(torch.isfinite(log_pb), log_pb, torch.zeros_like(log_pb))
            
            total_log_pb = total_log_pb + log_pb
        
        masked_log_pb = total_log_pb * success_mask.float()
        loss = -masked_log_pb.sum() / success_mask.sum().clamp(min=1)
        
        return loss

    def compute_forward_only_tb_loss(self, traj_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward-only terminal objective for the ablation setting.

        This intentionally removes the log P_B term from the online objective
        instead of merely zeroing the backward loss branch.
        """
        log_pf_sum = traj_data['log_pf'].sum(dim=0)
        rewards = traj_data['rewards']

        valid_mask = rewards > 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        log_pf_sum = torch.clamp(log_pf_sum, min=-100.0, max=100.0)
        log_R = torch.log(rewards + 1e-9)
        log_R = torch.clamp(log_R, min=-100.0, max=100.0)

        diff = self.log_Z + log_pf_sum - log_R
        diff = torch.clamp(diff, min=-50.0, max=50.0)

        masked_diff = diff * valid_mask.float()
        loss = (masked_diff ** 2).sum() / valid_mask.sum().clamp(min=1)
        loss = torch.clamp(loss, max=1000.0)
        return loss
    
    def compute_tb_loss(self, traj_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """璁＄畻 TB 鎹熷け"""
        if self.forward_only:
            return self.compute_forward_only_tb_loss(traj_data)
        log_pf_sum = traj_data['log_pf'].sum(dim=0)
        log_pb_sum = traj_data['log_pb'].sum(dim=0)
        rewards = traj_data['rewards']
        
        valid_mask = rewards > 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        log_pf_sum = torch.clamp(log_pf_sum, min=-100.0, max=100.0)
        log_pb_sum = torch.clamp(log_pb_sum, min=-100.0, max=100.0)
        
        log_R = torch.log(rewards + 1e-9)
        log_R = torch.clamp(log_R, min=-100.0, max=100.0)
        
        diff = self.log_Z + log_pf_sum - log_R - log_pb_sum
        diff = torch.clamp(diff, min=-50.0, max=50.0)
        
        masked_diff = diff * valid_mask.float()
        loss = (masked_diff ** 2).sum() / valid_mask.sum().clamp(min=1)
        loss = torch.clamp(loss, max=1000.0)
        
        return loss
    
    def end_epoch(self):
        self.decay_backward_lr()
        self.epoch += 1
    
    def clear_memory(self):
        if self.env is not None:
            self.env.clear()
            self.env = None
        
        self.traj_emb = None
        self.valid_lens = None
        self.max_len = 0
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    @torch.no_grad()
    def evaluate(
        self,
        trajectories: List[np.ndarray],
        raw_trajectories: Optional[List[np.ndarray]] = None,
        queries: Optional[List] = None,
        batch_gt_hits: Optional[List[set]] = None,
        n_samples: int = 5
    ) -> Dict[str, float]:
        self.model.eval()
        self.update_env(trajectories, raw_trajectories, queries, batch_gt_hits)
        
        all_f1 = []
        all_sparsity = []
        all_rewards = []
        
        for _ in range(n_samples):
            traj_data, _, batch_f1, _ = self.sample_trajectories_with_f1(epsilon=0.0, temperature=0.5)
            all_rewards.extend(traj_data['rewards'].cpu().numpy().tolist())
            all_f1.extend(batch_f1.cpu().numpy().tolist())
            
            for env in self.env.envs:
                if hasattr(env, '_last_sparsity'):
                    all_sparsity.append(env._last_sparsity)
        
        self.model.train()
        
        return {
            'mean_f1': np.mean(all_f1) if all_f1 else 0.0,
            'mean_sparsity': np.mean(all_sparsity) if all_sparsity else 0.0,
            'mean_reward': np.mean(all_rewards),
            'max_reward': np.max(all_rewards) if all_rewards else 0.0,
            'success_rate': np.mean([f >= self.f1_threshold for f in all_f1]) if all_f1 else 0.0
        }
    
    def save(self, path: str, **kwargs):
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': None if self.forward_only else self.target_model.state_dict(),
            'log_Z': self.log_Z.data,
            'opt_forward_state': self.opt_forward.state_dict(),
            'opt_backward_state': None if self.forward_only else self.opt_backward.state_dict(),
            'scheduler_backward_state': None if self.forward_only else self.scheduler_backward.state_dict(),
            'stats': self.stats,
            'epoch': self.epoch,
            'dual_lambda_map': self.dual_lambda_map,
            'frontier_state': self.frontier_buffer.state_dict(),
            'frontier_stats': self.frontier_buffer.stats(),
            'phase2_lowcr_active': self.phase2_lowcr_active,
            'phase2_dual_target': self.phase2_dual_target,
            'phase2_dual_decay': self.phase2_dual_decay,
            'forward_only': self.forward_only,
        }
        save_dict.update(kwargs)
        torch.save(save_dict, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        checkpoint_forward_only = checkpoint.get('forward_only')
        if checkpoint_forward_only is not None and bool(checkpoint_forward_only) != self.forward_only:
            print(
                f"[Warn] Checkpoint forward_only={bool(checkpoint_forward_only)} "
                f"but trainer forward_only={self.forward_only}. Using trainer setting."
            )
        
        # Mandatory: Model State
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("[Warn] Checkpoint missing 'model_state_dict'!")

        # Optional/Legacy: Target Model
        if (not self.forward_only) and checkpoint.get('target_model_state_dict') is not None:
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        
        # Optional/Legacy: Log Z
        if 'log_Z' in checkpoint:
            self.log_Z.data = checkpoint['log_Z']
        else:
            print("[Warn] Checkpoint missing 'log_Z'. Using current value.")

        # Optional/Legacy: Optimizers & Schedulers
        if 'opt_forward_state' in checkpoint:
            self.opt_forward.load_state_dict(checkpoint['opt_forward_state'])
        else:
            print("[Warn] Checkpoint missing 'opt_forward_state'. Optimizer reset.")

        if (not self.forward_only) and checkpoint.get('opt_backward_state') is not None:
            self.opt_backward.load_state_dict(checkpoint['opt_backward_state'])
        
        if (not self.forward_only) and checkpoint.get('scheduler_backward_state') is not None:
            self.scheduler_backward.load_state_dict(checkpoint['scheduler_backward_state'])
            
        if 'stats' in checkpoint:
            self.stats = checkpoint['stats']
            
        if 'epoch' in checkpoint:
            self.epoch = checkpoint['epoch']
        if 'dual_lambda_map' in checkpoint and isinstance(checkpoint['dual_lambda_map'], dict):
            self.dual_lambda_map = {str(k): float(v) for k, v in checkpoint['dual_lambda_map'].items()}
        if 'frontier_state' in checkpoint:
            self.frontier_buffer.load_state_dict(checkpoint['frontier_state'])
        if 'phase2_lowcr_active' in checkpoint:
            self.phase2_lowcr_active = bool(checkpoint['phase2_lowcr_active'])
        if 'phase2_dual_target' in checkpoint:
            self.phase2_dual_target = float(max(0.0, checkpoint['phase2_dual_target']))
        if 'phase2_dual_decay' in checkpoint:
            self.phase2_dual_decay = float(np.clip(checkpoint['phase2_dual_decay'], 0.0, 1.0))
        return checkpoint

