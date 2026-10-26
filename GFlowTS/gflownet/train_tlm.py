import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import copy

from .gfn_env import GFlowNetTrajectoryEnv
from .models import HierarchicalGFlowNet
from .frontier import FeasibleFrontierBuffer, FrontierCandidate
from profiling_utils import profile_scope

class TLMTrainer:

    def __init__(
        self,
        model: HierarchicalGFlowNet,
        device: str = 'cpu',
        lr_forward: float = 1e-3,
        lr_backward: float = 1e-4,
        alpha: float = 0.5,
        beta: float = 0.1,
        f1_threshold: float = 0.95,
        target_compression: float = 0.03,
        ema_decay: float = 0.995,

        lr_decay_rate: float = 0.99,
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
    ):
        self.device = device
        self.model = model.to(device)
        self.alpha = alpha
        self.beta = beta
        self.f1_threshold = f1_threshold
        self.target_compression = target_compression
        self.ema_decay = ema_decay
        self.lr_decay_rate = lr_decay_rate
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

        self.env = None
        self.traj_emb: Optional[torch.Tensor] = None
        self.valid_lens: Optional[torch.Tensor] = None
        self.max_len: int = 0

        self.log_Z = nn.Parameter(torch.tensor(0.0, device=device))

        self.target_model = copy.deepcopy(model).to(device)
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.frontier_buffer = FeasibleFrontierBuffer(capacity=int(max(1, frontier_size)))
        self.dual_lambda_map: Dict[str, float] = {}

        self.opt_forward = optim.Adam(
            list(model.parameters()) + [self.log_Z],
            lr=lr_forward
        )

        backward_lr_effective = float(lr_backward)
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
            'dual_lambda': [],
            'frontier_mean_size': [],
            'actual_samples': [],
        }

        self.epoch = 0

    def decay_backward_lr(self):
        self.scheduler_backward.step()

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

        valid_idx = np.where(valid[:-1])[0]
        if len(valid_idx) <= 1:
            return valid_idx.astype(np.int64)
        k = int(max(1, pool_size))
        if len(valid_idx) <= k:
            return valid_idx.astype(np.int64)

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
        masks = [env.mask.copy()]
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
            masks.append(env.mask.copy())
        if not done:
            valid = env.get_valid_actions().astype(bool, copy=True)
            valid_actions.append(valid)
            _, _, done, _ = env.step(-1)
            executed_actions.append(n)
            states.append(env.get_state_tensor())
            masks.append(env.mask.copy())

        indices = np.where(env.mask)[0].tolist()
        eval_tau = self.f1_threshold if tau is None else float(tau)
        eval_info = env.evaluate_indices(indices, tau=eval_tau)
        env.close()
        return {
            "states": states,
            "masks": masks,
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
        masks = [env.mask.copy()]
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
            masks.append(env.mask.copy())
        if not done:
            valid = env.get_valid_actions().astype(bool, copy=True)
            valid_actions.append(valid)
            _, _, done, _ = env.step(-1)
            actions.append(n)
            sampled_from_pool.append(False)
            states.append(env.get_state_tensor())
            masks.append(env.mask.copy())

        indices = np.where(env.mask)[0].tolist()
        mode = str(eval_mode).lower()
        if mode == "proxy":
            eval_info = env.evaluate_indices_proxy(indices, tau=tau, stride=self.multifidelity_proxy_stride)
        else:
            eval_info = env.evaluate_indices(indices, tau=tau)
        env.close()
        return {
            "states": states,
            "masks": masks,
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
        keep_start: bool,
        keep_end: bool,
    ) -> torch.Tensor:
        actions = rollout.get("actions", [])
        states = rollout.get("states", [])
        masks = rollout.get("masks", [])
        valid_actions = rollout.get("valid_actions", [])
        t_len = len(actions)
        if t_len <= 0 or len(states) <= 1 or len(masks) < t_len + 1 or len(valid_actions) <= 0:
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

        child_mask_stack = torch.tensor(
            np.stack(masks[1 : t_len + 1]), dtype=torch.float32, device=self.device
        )
        backward_logits = self.model.forward_backward(
            traj_emb_batch,
            child_mask_stack,
            valid_l_batch,
            keep_start=keep_start,
            keep_end=keep_end,
        )
        backward_logits = torch.clamp(backward_logits, min=-100.0, max=100.0)
        backward_dist = Categorical(logits=backward_logits)
        log_pb = backward_dist.log_prob(actions_tensor)

        diff = torch.clamp(
            log_f_t + log_pf - target - log_pb,
            min=-100.0,
            max=100.0,
        )
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
            logits_b = self.model.forward_backward(
                traj_emb, mask_t, valid_lens,
                keep_start=keep_start, keep_end=keep_end,
            )
            dist_b = Categorical(logits=logits_b)
            target_action = torch.tensor([int(a)], dtype=torch.long, device=self.device)
            total = total - dist_b.log_prob(target_action).mean()
        env.close()
        return total / max(1, len(cleaned_actions))

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
                keep_start=keep_start,
                keep_end=keep_end,
            )
            forward_loss = forward_loss + w * lf

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

        backward_loss = torch.zeros((), dtype=torch.float32, device=self.device)
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

    def save(self, path: str, **kwargs):
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'log_Z': self.log_Z.data,
            'opt_forward_state': self.opt_forward.state_dict(),
            'opt_backward_state': self.opt_backward.state_dict(),
            'scheduler_backward_state': self.scheduler_backward.state_dict(),
            'stats': self.stats,
            'epoch': self.epoch,
            'dual_lambda_map': self.dual_lambda_map,
            'frontier_state': self.frontier_buffer.state_dict(),
            'frontier_stats': self.frontier_buffer.stats(),
            'phase2_lowcr_active': self.phase2_lowcr_active,
            'phase2_dual_target': self.phase2_dual_target,
            'phase2_dual_decay': self.phase2_dual_decay,
        }
        save_dict.update(kwargs)
        torch.save(save_dict, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("[Warn] Checkpoint missing 'model_state_dict'!")

        if checkpoint.get('target_model_state_dict') is not None:
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])

        if 'log_Z' in checkpoint:
            self.log_Z.data = checkpoint['log_Z']
        else:
            print("[Warn] Checkpoint missing 'log_Z'. Using current value.")

        if 'opt_forward_state' in checkpoint:
            self.opt_forward.load_state_dict(checkpoint['opt_forward_state'])
        else:
            print("[Warn] Checkpoint missing 'opt_forward_state'. Optimizer reset.")

        if checkpoint.get('opt_backward_state') is not None:
            self.opt_backward.load_state_dict(checkpoint['opt_backward_state'])

        if checkpoint.get('scheduler_backward_state') is not None:
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
