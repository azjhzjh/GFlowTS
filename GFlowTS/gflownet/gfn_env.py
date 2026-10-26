import numpy as np
import math
from typing import Dict, Tuple, List, Optional

from .utils.query_sketch import ConstantSizeState, QueryCoverageSketch, f1_lower_bound

class GFlowNetTrajectoryEnv:

    def __init__(
        self,
        trajectory: np.ndarray,
        raw_trajectory: Optional[np.ndarray] = None,
        device: str = 'cpu',
        alpha: float = 1.0,
        beta: float = 1.0,
        f1_threshold: float = 0.95,
        target_compression: float = 0.03,
        global_stats: dict = None,
        local_stats: dict = None,
        prev_anchor: np.ndarray = None,
        next_anchor: np.ndarray = None,
        keep_start: bool = True,
        keep_end: bool = True,
        balltree = None,
        cover_radius: float = 0.02,
        gamma_curve: float = 0.5,
        proxy_grid_size: int = 0,
        proxy_stride: int = 4,
    ):

        self.trajectory = trajectory
        self.raw_trajectory = raw_trajectory if raw_trajectory is not None else trajectory
        self.N = len(trajectory)
        self.alpha = alpha
        self.beta = beta
        self.f1_threshold = f1_threshold
        self.target_compression = target_compression
        self.device = device
        self.prev_anchor = prev_anchor
        self.next_anchor = next_anchor
        self.keep_start = keep_start
        self.keep_end = keep_end

        self.balltree = balltree
        self.cover_radius = cover_radius
        self.gamma_curve = gamma_curve
        self.proxy_grid_size = int(max(0, proxy_grid_size))
        self.proxy_stride = int(max(1, proxy_stride))

        self._precompute_features()

        self.const_state = ConstantSizeState(
            total_points=self.N,
            grid_size=64,
            target_compression=target_compression
        )

        if local_stats is not None:
            self.raw_x_range = (local_stats['x_min'], local_stats['x_max'] + 1e-9)
            self.raw_y_range = (local_stats['y_min'], local_stats['y_max'] + 1e-9)
            self.raw_t_range = (local_stats['t_min'], local_stats['t_max'] + 1e-9)
        elif self.raw_trajectory is not None and len(self.raw_trajectory) > 0:

            x_min, x_max = self.raw_trajectory[:, 0].min(), self.raw_trajectory[:, 0].max()
            y_min, y_max = self.raw_trajectory[:, 1].min(), self.raw_trajectory[:, 1].max()
            t_min, t_max = self.raw_trajectory[:, 2].min(), self.raw_trajectory[:, 2].max()

            self.raw_x_range = (x_min, x_max + 1e-9)
            self.raw_y_range = (y_min, y_max + 1e-9)
            self.raw_t_range = (t_min, t_max + 1e-9)
        elif global_stats is not None:
            self.raw_x_range = (global_stats['x_min'], global_stats['x_max'] + 1e-9)
            self.raw_y_range = (global_stats['y_min'], global_stats['y_max'] + 1e-9)
            self.raw_t_range = (global_stats['t_min'], global_stats['t_max'] + 1e-9)
        else:
            self.raw_x_range = (0.0, 1.0)
            self.raw_y_range = (0.0, 1.0)
            self.raw_t_range = (0.0, 1.0)

        self.qcs_original = QueryCoverageSketch(
            grid_size=64,
            x_range=self.raw_x_range,
            y_range=self.raw_y_range,
            t_range=self.raw_t_range
        )
        if self.raw_trajectory is not None and len(self.raw_trajectory) > 0:
            self.qcs_original.add_point(self.raw_trajectory[0])
            for i in range(1, len(self.raw_trajectory)):
                self.qcs_original.add_segment(self.raw_trajectory[i-1], self.raw_trajectory[i])

        self.qcs_original_proxy = None
        if self.proxy_grid_size > 0 and self.raw_trajectory is not None and len(self.raw_trajectory) > 0:
            self.qcs_original_proxy = QueryCoverageSketch(
                grid_size=self.proxy_grid_size,
                x_range=self.raw_x_range,
                y_range=self.raw_y_range,
                t_range=self.raw_t_range
            )
            proxy_idx = np.arange(0, len(self.raw_trajectory), self.proxy_stride, dtype=np.int64).tolist()
            if len(proxy_idx) == 0 or proxy_idx[-1] != (len(self.raw_trajectory) - 1):
                proxy_idx.append(len(self.raw_trajectory) - 1)
            self.qcs_original_proxy.add_point(self.raw_trajectory[proxy_idx[0]])
            for i in range(1, len(proxy_idx)):
                self.qcs_original_proxy.add_segment(
                    self.raw_trajectory[proxy_idx[i - 1]],
                    self.raw_trajectory[proxy_idx[i]]
                )

        self.last_raw_point = None

        self.qcs_compressed = QueryCoverageSketch(
            grid_size=64,
            x_range=self.raw_x_range,
            y_range=self.raw_y_range,
            t_range=self.raw_t_range
        )

        self.mask = np.zeros(self.N, dtype=bool)

        self.reset(keep_start=self.keep_start, keep_end=self.keep_end)

    def _precompute_features(self):

        self.velocities = np.zeros(self.N)
        self.turning_angles = np.zeros(self.N)

        for i in range(1, self.N):

            dx = self.trajectory[i, 0] - self.trajectory[i-1, 0]
            dy = self.trajectory[i, 1] - self.trajectory[i-1, 1]
            dt = abs(self.trajectory[i, 2] - self.trajectory[i-1, 2]) + 1e-6
            self.velocities[i] = np.sqrt(dx**2 + dy**2) / dt

        for i in range(1, self.N - 1):

            v1 = self.trajectory[i] - self.trajectory[i-1]
            v2 = self.trajectory[i+1] - self.trajectory[i]
            v1_2d = v1[:2]
            v2_2d = v2[:2]

            norm1 = np.linalg.norm(v1_2d)
            norm2 = np.linalg.norm(v2_2d)

            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_angle = np.clip(np.dot(v1_2d, v2_2d) / (norm1 * norm2), -1, 1)
                self.turning_angles[i] = np.arccos(cos_angle)

    def reset(self, keep_start: Optional[bool] = None, keep_end: Optional[bool] = None) -> np.ndarray:

        k_start = keep_start if keep_start is not None else self.keep_start
        k_end = keep_end if keep_end is not None else self.keep_end

        self.mask = np.zeros(self.N, dtype=bool)

        if k_start:
            self.mask[0] = True
        if k_end and self.N > 1:
            self.mask[-1] = True

        self.done = False
        self.current_step = 0
        self.trajectory_log = []

        first_point = self.trajectory[0] if k_start else None
        self.const_state.reset(keep_first=k_start, first_point=first_point)

        if k_end and self.N > 1:
            self.const_state.add_point(self.trajectory[-1])

        self.qcs_compressed.reset()
        self.last_raw_point = None

        if k_start:
            self.qcs_compressed.add_point(self.raw_trajectory[0])
            self.last_raw_point = self.raw_trajectory[0]

        if k_end and self.N > 1:
            self.qcs_compressed.add_point(self.raw_trajectory[-1])

            pass

        self._last_kept_idx = 0 if keep_start else -1
        self._last_f1 = 0.0
        self._last_sparsity = 0.0
        self._last_num_points = 0
        self._last_cr_cap = float(np.clip(self.target_compression, 1e-4, 1.0))
        self._last_cr_excess = 0.0

        return self.mask.copy()

    def get_state_tensor(self) -> np.ndarray:

        return self.const_state.to_tensor()

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:

        if self.done:
            return self.mask.copy(), 0.0, True, {"f1": self._last_f1}

        info = {}
        if action_idx == -1 or action_idx == self.N:
            self.done = True
            reward = self.get_reward()
            info = {
                'f1': self._last_f1,
                'sparsity': self._last_sparsity,
                'cr_cap': self._last_cr_cap,
                'cr_excess': self._last_cr_excess,
                'const_state': self.const_state,
                'num_points': self.const_state.num_points
            }
            return self.mask.copy(), reward, True, info

        if action_idx < 0 or action_idx >= self.N:

            self.done = True
            return self.mask.copy(), 1e-9, True, {}

        if not self.mask[action_idx]:
            self.mask[action_idx] = True
            self._last_kept_idx = action_idx
            self.trajectory_log.append(action_idx)

            current_raw_point = self.raw_trajectory[action_idx]
            self.const_state.add_point(current_raw_point)

            if self.last_raw_point is not None:
                self.qcs_compressed.add_segment(self.last_raw_point, current_raw_point)
            else:
                self.qcs_compressed.add_point(current_raw_point)

            self.last_raw_point = current_raw_point

        self.current_step += 1
        info['const_state'] = self.const_state
        return self.mask.copy(), 0.0, False, info

    def lightweight_step(self, action_idx: int):

        if self.done:
            return self.mask.copy(), 0.0, True, {}

        if action_idx == -1 or action_idx == self.N:
            return self.mask.copy(), 0.0, True, {}

        if 0 <= action_idx < self.N:
            if not self.mask[action_idx]:
                self.mask[action_idx] = True
                self.const_state.add_point(self.raw_trajectory[action_idx])

        return self.mask.copy(), 0.0, False, {}

    def close(self):

        if hasattr(self, 'qcs_original'):
            del self.qcs_original
            self.qcs_original = None
        if hasattr(self, 'qcs_original_proxy'):
            del self.qcs_original_proxy
            self.qcs_original_proxy = None
        if hasattr(self, 'qcs_compressed'):
            del self.qcs_compressed
            self.qcs_compressed = None
        if hasattr(self, 'const_state'):
            del self.const_state
            self.const_state = None

    def get_reward(self) -> float:

        num_points = self.const_state.num_points
        sparsity = num_points / max(1, self.N)
        self._last_sparsity = sparsity
        self._last_num_points = num_points
        cr_cap = float(np.clip(self.target_compression, 1e-4, 1.0))
        self._last_cr_cap = cr_cap
        excess = max(0.0, sparsity - cr_cap)
        self._last_cr_excess = excess

        selected_indices = np.where(self.mask)[0]
        selected_indices = np.sort(selected_indices)

        temp_qcs = QueryCoverageSketch(
            grid_size=self.qcs_compressed.grid_size,
            x_range=self.qcs_compressed.x_range,
            y_range=self.qcs_compressed.y_range,
            t_range=self.qcs_compressed.t_range
        )

        if len(selected_indices) > 0:
            temp_qcs.add_point(self.raw_trajectory[selected_indices[0]])
            for i in range(1, len(selected_indices)):
                p1 = self.raw_trajectory[selected_indices[i-1]]
                p2 = self.raw_trajectory[selected_indices[i]]
                temp_qcs.add_segment(p1, p2)

        f1_lb = f1_lower_bound(temp_qcs, self.qcs_original)
        self._last_f1 = f1_lb

        if num_points <= 2:
            return 1e-9

        curve_reward = self._curvature_reward(selected_indices)

        excess_penalty = 15.0 * excess + 60.0 * (excess ** 2)
        sparsity_score = 5.0 * math.exp(-self.beta * 5.0 * sparsity)

        if f1_lb >= self.f1_threshold:
            reward = 20.0 + sparsity_score + curve_reward - excess_penalty
        else:
            grid_coverage_ratio = temp_qcs.size() / max(1, self.qcs_original.size())
            reward = 5.0 * grid_coverage_ratio + 2.0 * f1_lb + curve_reward * 0.5 - 0.5 * excess_penalty

        return max(reward, 1e-9)

    def evaluate_indices(self, indices: List[int], tau: Optional[float] = None) -> Dict[str, float]:

        if tau is None:
            tau = self.f1_threshold
        tau = float(tau)

        selected_indices = sorted(set(int(i) for i in indices if 0 <= int(i) < self.N))
        if self.keep_start and self.N > 0:
            selected_indices = sorted(set(selected_indices + [0]))
        if self.keep_end and self.N > 1:
            selected_indices = sorted(set(selected_indices + [self.N - 1]))

        temp_qcs = QueryCoverageSketch(
            grid_size=self.qcs_original.grid_size,
            x_range=self.qcs_original.x_range,
            y_range=self.qcs_original.y_range,
            t_range=self.qcs_original.t_range
        )
        if selected_indices:
            temp_qcs.add_point(self.raw_trajectory[selected_indices[0]])
            for i in range(1, len(selected_indices)):
                p1 = self.raw_trajectory[selected_indices[i - 1]]
                p2 = self.raw_trajectory[selected_indices[i]]
                temp_qcs.add_segment(p1, p2)

        f1_lb = float(f1_lower_bound(temp_qcs, self.qcs_original))
        cr = len(selected_indices) / max(1, self.N)

        curve_reward = self._curvature_reward(np.array(selected_indices, dtype=np.int64))
        excess = max(0.0, cr - float(np.clip(self.target_compression, 1e-4, 1.0)))
        excess_penalty = 15.0 * excess + 60.0 * (excess ** 2)
        sparsity_score = 5.0 * math.exp(-self.beta * 5.0 * cr)

        if f1_lb >= tau:
            reward = 20.0 + sparsity_score + curve_reward - excess_penalty
        else:
            grid_coverage_ratio = temp_qcs.size() / max(1, self.qcs_original.size())
            reward = 5.0 * grid_coverage_ratio + 2.0 * f1_lb + curve_reward * 0.5 - 0.5 * excess_penalty

        return {
            "f1_lb": f1_lb,
            "cr": float(cr),
            "feasible": bool(f1_lb >= tau),
            "reward": float(max(reward, 1e-9)),
        }

    @staticmethod
    def _subsample_sorted_indices(sorted_indices: List[int], stride: int) -> List[int]:
        if not sorted_indices:
            return []
        stride = int(max(1, stride))
        if stride <= 1 or len(sorted_indices) <= 2:
            return list(sorted_indices)
        out = [sorted_indices[0]]
        out.extend(sorted_indices[1:-1:stride])
        if out[-1] != sorted_indices[-1]:
            out.append(sorted_indices[-1])
        return out

    def evaluate_indices_proxy(
        self,
        indices: List[int],
        tau: Optional[float] = None,
        stride: Optional[int] = None
    ) -> Dict[str, float]:

        if self.qcs_original_proxy is None:
            return self.evaluate_indices(indices=indices, tau=tau)

        if tau is None:
            tau = self.f1_threshold
        tau = float(tau)

        selected_indices = sorted(set(int(i) for i in indices if 0 <= int(i) < self.N))
        if self.keep_start and self.N > 0:
            selected_indices = sorted(set(selected_indices + [0]))
        if self.keep_end and self.N > 1:
            selected_indices = sorted(set(selected_indices + [self.N - 1]))

        step = self.proxy_stride if stride is None else int(max(1, stride))
        selected_proxy = self._subsample_sorted_indices(selected_indices, step)

        temp_qcs = QueryCoverageSketch(
            grid_size=self.qcs_original_proxy.grid_size,
            x_range=self.qcs_original_proxy.x_range,
            y_range=self.qcs_original_proxy.y_range,
            t_range=self.qcs_original_proxy.t_range
        )
        if selected_proxy:
            temp_qcs.add_point(self.raw_trajectory[selected_proxy[0]])
            for i in range(1, len(selected_proxy)):
                p1 = self.raw_trajectory[selected_proxy[i - 1]]
                p2 = self.raw_trajectory[selected_proxy[i]]
                temp_qcs.add_segment(p1, p2)

        f1_lb = float(f1_lower_bound(temp_qcs, self.qcs_original_proxy))
        cr = len(selected_indices) / max(1, self.N)
        curve_reward = self._curvature_reward(np.array(selected_indices, dtype=np.int64))
        excess = max(0.0, cr - float(np.clip(self.target_compression, 1e-4, 1.0)))
        excess_penalty = 15.0 * excess + 60.0 * (excess ** 2)
        sparsity_score = 5.0 * math.exp(-self.beta * 5.0 * cr)

        if f1_lb >= tau:
            reward = 20.0 + sparsity_score + curve_reward - excess_penalty
        else:
            grid_coverage_ratio = temp_qcs.size() / max(1, self.qcs_original_proxy.size())
            reward = 5.0 * grid_coverage_ratio + 2.0 * f1_lb + curve_reward * 0.5 - 0.5 * excess_penalty

        return {
            "f1_lb": f1_lb,
            "cr": float(cr),
            "feasible": bool(f1_lb >= tau),
            "reward": float(max(reward, 1e-9)),
        }

    def _curvature_reward(self, kept_indices: np.ndarray) -> float:

        if len(kept_indices) < 2:
            return 0.0

        curv = self.turning_angles[kept_indices]
        curv_diff = np.abs(np.diff(curv))

        mean_diff = np.mean(curv_diff) if len(curv_diff) > 0 else 0.0

        return self.gamma_curve * (1.0 - mean_diff / math.pi)

    def greedy_grid_simplify(self) -> List[int]:

        covered_cells = set()
        selected_indices = []

        if self.keep_start:
            selected_indices.append(0)
            start_cell = self.qcs_compressed._query_aware_cell(self.raw_trajectory[0])
            covered_cells.add(start_cell)

        for i in range(1 if self.keep_start else 0, self.N - 1):
            cell = self.qcs_compressed._query_aware_cell(self.raw_trajectory[i])
            if cell not in covered_cells:
                selected_indices.append(i)
                covered_cells.add(cell)

        if self.keep_end and self.N > 1:
            if (self.N - 1) not in selected_indices:
                selected_indices.append(self.N - 1)

        return selected_indices

    def _line_intersects_box(self, p1, p2, x_min, x_max, y_min, y_max, t_min, t_max) -> bool:

        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        p = [-dx, dx, -dy, dy]
        q = [p1[0] - x_min, x_max - p1[0], p1[1] - y_min, y_max - p1[1]]

        u1, u2 = 0.0, 1.0

        for i in range(4):
            if p[i] == 0:
                if q[i] < 0:
                    return False
            else:
                u = q[i] / p[i]
                if p[i] < 0:
                    if u > u2:
                        return False
                    if u > u1:
                        u1 = u
                else:
                    if u < u1:
                        return False
                    if u < u2:
                        u2 = u

        return u1 <= u2

    def get_valid_actions(self) -> np.ndarray:

        valid = np.zeros(self.N + 1, dtype=bool)
        valid[:-1] = ~self.mask
        valid[-1] = True
        return valid
