# -*- coding: utf-8 -*-
"""
GFlowNet 杞ㄨ抗绠€鍖栫幆澧?- 閲嶆瀯鐗堟湰
鏀寔灞傛鍖栫姸鎬佺┖闂?(Region + Point) 鍜?GFlowNet 鍔ㄤ綔绌洪棿
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Set
from enum import IntEnum

# 瀵煎叆甯告暟澶у皬鐘舵€佸拰 QCS
from .utils.query_sketch import (
    QuerySketch, 
    ConstantSizeState,
    QueryCoverageSketch,
    f1_lower_bound
)


# ==============================================================================
# 鍔ㄤ綔绌洪棿瀹氫箟
# ==============================================================================

class Level1Action(IntEnum):
    """Level-1 (Region/BallTree 灞? 鍔ㄤ綔绌洪棿"""
    SELECT_LEFT = 0     # 閫夋嫨宸﹀瓙鐞?
    SELECT_RIGHT = 1    # 閫夋嫨鍙冲瓙鐞?
    TERMINATE = 2       # 缁堟鍖哄煙閫夋嫨


class Level2Action(IntEnum):
    """Level-2 (Point 灞? 鍔ㄤ綔绌洪棿"""
    KEEP_POINT = 0      # 淇濈暀褰撳墠鐐?
    SKIP_POINT = 1      # 璺宠繃褰撳墠鐐?
    TERMINATE = 2       # 缁堟鐐归€夋嫨


# ==============================================================================
# 鐘舵€佺┖闂村畾涔?
# ==============================================================================

@dataclass
class RegionState:
    """
    Level-1 (Region/BallTree 灞? 鐘舵€?
    鐢ㄤ簬灞傛鍖栫┖闂撮噰鏍峰喅绛?
    """
    node_id: int                    # BallTree 鑺傜偣 ID
    point_count: int                # 鑺傜偣瑕嗙洊鐐规暟
    local_complexity: float         # 灞€閮ㄨ建杩瑰鏉傚害 (鏇茬巼/鏃堕棿璺ㄥ害)
    retention_ratio: float          # 褰撳墠宸蹭繚鐣欑偣姣斾緥
    f1_estimate: float              # partial trajectory 鐨?F1 棰勬祴鍊?
    depth: int = 0                  # 鑺傜偣娣卞害
    parent_id: Optional[int] = None # 鐖惰妭鐐?ID
    
    def to_tensor(self) -> np.ndarray:
        """杞崲涓虹缁忕綉缁滆緭鍏ュ紶閲?"""
        return np.array([
            self.node_id,
            self.point_count,
            self.local_complexity,
            self.retention_ratio,
            self.f1_estimate,
            self.depth
        ], dtype=np.float32)
    
    @staticmethod
    def dim() -> int:
        return 6


@dataclass
class PointState:
    """
    Level-2 (Point 灞? 鐘舵€?
    鐢ㄤ簬鐐圭骇鍒殑淇濈暀/璺宠繃鍐崇瓥
    
    娉ㄦ剰锛氫笉鍖呭惈鍑犱綍璇樊 (SED/PED)锛屼弗鏍奸伒寰?GFlowNet 鐞嗚
    """
    point_idx: int                              # 褰撳墠鐐圭储寮?
    last_kept_pos: Tuple[float, float, float]   # 涓婁竴淇濈暀鐐逛綅缃?(x, y, t)
    local_velocity: float                       # 褰撳墠灞€閮ㄩ€熷害
    turning_angle: float                        # 杞悜瑙?
    selected_count: int                         # 宸查€夌偣鏁?
    total_count: int                            # 鎬荤偣鏁?
    budget_remaining: float                     # 鍓╀綑鍘嬬缉棰勭畻 (0~1)
    # 娉ㄦ剰锛氱姝娇鐢?sed_error 绛夊嚑浣曡宸綔涓虹姸鎬佺壒寰?
    
    def to_tensor(self) -> np.ndarray:
        """杞崲涓虹缁忕綉缁滆緭鍏ュ紶閲?"""
        return np.array([
            self.point_idx / max(self.total_count, 1),  # 褰掍竴鍖栦綅缃?
            self.last_kept_pos[0],
            self.last_kept_pos[1],
            self.last_kept_pos[2],
            self.local_velocity,
            self.turning_angle,
            self.selected_count / max(self.total_count, 1),  # 褰掍竴鍖栧凡閫夋瘮渚?
            self.budget_remaining
        ], dtype=np.float32)
    
    @staticmethod
    def dim() -> int:
        return 8



# ==============================================================================
# GFlowNet 杞ㄨ抗绠€鍖栫幆澧?
# ==============================================================================

class GFlowNetTrajectoryEnv:
    """
    GFlowNet 杞ㄨ抗绠€鍖栫幆澧?
    
    鐘舵€? 宸查€夌偣鐨勪簩杩涘埗鎺╃爜 + 灞傛鍖栫壒寰?
    鍔ㄤ綔: 娣诲姞鐐圭储寮?鎴?缁堟
    濂栧姳: F1^alpha * (1 - Sparsity)^beta (缁堟鏃惰绠?
    
    绗﹀悎 GFlowNet DAG 缁撴瀯锛氶€愭鏋勯€犲帇缂╄建杩圭偣闆?
    """
    
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
        queries: List = None,
        gt_hits: Set = None,
        keep_start: bool = True,
        keep_end: bool = True,
        balltree = None,
        cover_radius: float = 0.02,
        gamma_curve: float = 0.5,
        proxy_grid_size: int = 0,
        proxy_stride: int = 4,
    ):
        """
        鍒濆鍖栫幆澧?
        Args:
            trajectory: (N, D) numpy array (褰掍竴鍖栧潗鏍?
            raw_trajectory: (N, D) numpy array (鍘熷鍧愭爣)锛岀敤浜庢煡璇㈣瘎浼?
            alpha: F1 濂栧姳鎸囨暟
            beta: 鍘嬬缉鐜囧鍔辨寚鏁?
            f1_threshold: F1 纭害鏉熼槇鍊?(榛樿 0.95)
            target_compression: 鐩爣鍘嬬缉鐜?
            prev_anchor: 鍓嶄竴涓敋鐐?(鐢ㄤ簬璺?chunk 杈圭晫)
            next_anchor: 鍚庝竴涓敋鐐?
            queries: 鏌ヨ鍒楄〃
            gt_hits: 鍘熷杞ㄨ抗鍛戒腑鐨勬煡璇㈢储寮曢泦鍚?
            keep_start: 鏄惁寮哄埗淇濈暀璧峰鐐?
            keep_end: 鏄惁寮哄埗淇濈暀缁堟鐐?
        """
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
        self.queries = queries
        self.gt_hits = gt_hits if gt_hits is not None else set()
        self.keep_start = keep_start
        self.keep_end = keep_end
        
        # BallTree 绌洪棿绱㈠紩涓庤鐩栧弬鏁?
        self.balltree = balltree
        self.cover_radius = cover_radius
        self.gamma_curve = gamma_curve
        self.proxy_grid_size = int(max(0, proxy_grid_size))
        self.proxy_stride = int(max(1, proxy_stride))
        
        # 棰勮绠楄建杩圭壒寰?
        self._precompute_features()
        
        # ============================================
        # 甯告暟澶у皬鐘舵€?(鏇夸唬 O(n) 鐨?mask)
        # ============================================
        self.const_state = ConstantSizeState(
            total_points=self.N,
            grid_size=64,
            target_compression=target_compression
        )
        
        # ============================================
        # 鑾峰彇 QCS 鍧愭爣鑼冨洿 (浼樺厛浣跨敤鏈湴缁熻閲忎互淇濊瘉鍒嗚鲸鐜?
        # ============================================
        if local_stats is not None:
            self.raw_x_range = (local_stats['x_min'], local_stats['x_max'] + 1e-9)
            self.raw_y_range = (local_stats['y_min'], local_stats['y_max'] + 1e-9)
            self.raw_t_range = (local_stats['t_min'], local_stats['t_max'] + 1e-9)
        elif self.raw_trajectory is not None and len(self.raw_trajectory) > 0:
            # 澶囧閫夛細浠庡崟鏉¤建杩规彁鍙?(宸查€氳繃鍚戦噺鍖栦紭鍖?
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

        # ============================================
        # QCS: 绂荤嚎棰勮绠楀師濮嬭建杩圭殑 Query Coverage Sketch
        # ============================================
        # [Modify] 浣跨敤 add_segment 鏉ヨ鐩栬矾寰?(Shape-Aware)
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

        # Optional low-fidelity original sketch for cheap proxy evaluation.
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

        # 鍒濆鍖栦复鏃跺彉閲?
        self.last_raw_point = None
        
        # 鍘嬬缉杞ㄨ抗鐨?QCS (鍦ㄧ嚎鏇存柊)
        self.qcs_compressed = QueryCoverageSketch(
            grid_size=64,
            x_range=self.raw_x_range,
            y_range=self.raw_y_range,
            t_range=self.raw_t_range
        )
        
        # 淇濈暀 mask 鐢ㄤ簬鍏煎鎬?
        self.mask = np.zeros(self.N, dtype=bool)
        
        # 鐘舵€佸垵濮嬪寲
        self.reset(keep_start=self.keep_start, keep_end=self.keep_end)
    
    def _precompute_features(self):
        """棰勮绠楄建杩圭殑灞€閮ㄧ壒寰侊細閫熷害銆佽浆鍚戣銆佹洸鐜?"""
        self.velocities = np.zeros(self.N)
        self.turning_angles = np.zeros(self.N)
        
        for i in range(1, self.N):
            # 閫熷害 = 娆ф皬璺濈 / 鏃堕棿宸?
            dx = self.trajectory[i, 0] - self.trajectory[i-1, 0]
            dy = self.trajectory[i, 1] - self.trajectory[i-1, 1]
            dt = abs(self.trajectory[i, 2] - self.trajectory[i-1, 2]) + 1e-6
            self.velocities[i] = np.sqrt(dx**2 + dy**2) / dt
        
        for i in range(1, self.N - 1):
            # 杞悜瑙?
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
        """
        閲嶇疆鐜
        """
        # 浣跨敤瀹炰緥灞炴€т綔涓洪粯璁ゅ€?
        k_start = keep_start if keep_start is not None else self.keep_start
        k_end = keep_end if keep_end is not None else self.keep_end
        
        # 閲嶇疆 O(n) mask (浠呯敤浜庡吋瀹规€?
        self.mask = np.zeros(self.N, dtype=bool)
        
        if k_start:
            self.mask[0] = True
        if k_end and self.N > 1:
            self.mask[-1] = True
        
        self.done = False
        self.current_step = 0
        self.trajectory_log = []
        
        # ============================================
        # 閲嶇疆甯告暟澶у皬鐘舵€?
        # ============================================
        first_point = self.trajectory[0] if k_start else None
        self.const_state.reset(keep_first=k_start, first_point=first_point)
        
        if k_end and self.N > 1:
            self.const_state.add_point(self.trajectory[-1])
        
        # ============================================
        # 閲嶇疆鍘嬬缉 QCS
        # ============================================
        self.qcs_compressed.reset()
        self.last_raw_point = None
        
        if k_start:
            self.qcs_compressed.add_point(self.raw_trajectory[0])
            self.last_raw_point = self.raw_trajectory[0]
            
        if k_end and self.N > 1:
            self.qcs_compressed.add_point(self.raw_trajectory[-1])
            # End point handled separately? 
            # 閫氬父 End point 鏄湪缁堟鏃跺姞鍏? 
            # 浠ｇ爜鐪嬫潵鏄湪 reset 鏃跺己鍒跺姞鍏? 
            # 杩欐槸涓€涓綔鍦ㄩ棶棰橈細濡傛灉鍦ㄥ紑澶村己鍒跺姞浜?end point锛?
            # 閭ｄ箞鍦ㄦ涔嬪墠鍔犵殑鐐瑰拰 end point 涔嬮棿浼氭湁鎻掑€奸棶棰?
            # 瀹為檯涓?GFlowNet 鐨?action 鏄『搴忓姞鐐广€?
            # keep_end 鍙槸寮哄埗鏈€鍚?QCS 鍖呭惈 end point銆?
            # 鎴戜滑鍙互淇濈暀 add_point for end point锛屽洜涓哄畠涓嶄竴瀹氭槸 path 鐨勪竴閮ㄥ垎 (until we reach it)
            # 鎴栬€呮殏涓嶅姞鍏?end point锛岀瓑 process 鍒版渶鍚庯紵
            # 鐜版湁閫昏緫鏄細杩欓噷閲嶅娣诲姞 end point
            # 淇锛氶伩鍏嶉噸澶?add_point
            pass
        
        # 鍒濆鍖栬拷韪彉閲?
        self._last_kept_idx = 0 if keep_start else -1
        self._last_f1 = 0.0
        self._last_sparsity = 0.0
        self._last_num_points = 0
        self._last_cr_cap = float(np.clip(self.target_compression, 1e-4, 1.0))
        self._last_cr_excess = 0.0
        
        return self.mask.copy()
    
    def get_state(self) -> ConstantSizeState:
        """
        鑾峰彇褰撳墠甯告暟澶у皬鐘舵€?(O(1))
        
        杩斿洖 ConstantSizeState 瀵硅薄锛屽寘鍚?
        - last_kept_point: 鏈€鍚庝繚鐣欑殑鐐?
        - sketch: QuerySketch
        - num_points: 宸查€夌偣鏁?
        - budget_left: 鍓╀綑棰勭畻
        
        姝ゆ柟娉曡繑鍥炵殑鐘舵€佸ぇ灏忎笌杞ㄨ抗闀垮害鏃犲叧锛?
        """
        return self.const_state
    
    def get_state_tensor(self) -> np.ndarray:
        """
        鑾峰彇甯告暟澶у皬鐨勭姸鎬佸紶閲?(鐢ㄤ簬绁炵粡缃戠粶杈撳叆)
        
        Returns:
            state_tensor: 鍥哄畾 18 缁寸殑鐘舵€佸悜閲?
        """
        return self.const_state.to_tensor()
    
    def _compute_current_sed(self) -> float:
        """璁＄畻褰撳墠绠€鍖栬建杩圭殑 SED 璇樊"""
        selected_indices = np.where(self.mask)[0]
        if len(selected_indices) < 2:
            return 0.0
        
        total_sed = 0.0
        for i in range(len(selected_indices) - 1):
            start_idx = selected_indices[i]
            end_idx = selected_indices[i + 1]
            
            if end_idx - start_idx <= 1:
                continue
            
            # 璁＄畻涓棿鐐瑰埌绾挎鐨勮窛绂?
            p_start = self.trajectory[start_idx]
            p_end = self.trajectory[end_idx]
            
            for j in range(start_idx + 1, end_idx):
                p = self.trajectory[j]
                # 鐐瑰埌绾挎鐨勮窛绂?(绠€鍖栦负娆ф皬璺濈)
                t = (j - start_idx) / (end_idx - start_idx)
                interpolated = p_start + t * (p_end - p_start)
                sed = np.linalg.norm(p[:2] - interpolated[:2])
                total_sed += sed
        
        return total_sed / max(1, self.N)
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        鎵ц鍔ㄤ綔
        
        Args:
            action_idx: 鐐圭储寮?(0 ~ N-1) 鎴?N/-1 琛ㄧず缁堟
            
        Returns:
            next_mask: 涓嬩竴鐘舵€佹帺鐮?(鍏煎鎬?
            reward: 濂栧姳 (鍙湪缁堟鏃堕潪闆?
            done: 鏄惁缁堟
            info: 棰濆淇℃伅
        """
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
        
        # 姝ｅ父閫夌偣
        if action_idx < 0 or action_idx >= self.N:
            # Invalid action, terminate with minimal reward
            self.done = True
            return self.mask.copy(), 1e-9, True, {}
            
        if not self.mask[action_idx]:
            self.mask[action_idx] = True
            self._last_kept_idx = action_idx
            self.trajectory_log.append(action_idx)
            
            # 鏇存柊 QCS 鍜屽父鏁扮姸鎬?
            current_raw_point = self.raw_trajectory[action_idx]
            self.const_state.add_point(current_raw_point)
            
            # ============================================
            # 鏇存柊鍘嬬缉 QCS (鍦ㄧ嚎鏇存柊)
            # ============================================
            # ============================================
            # 鏇存柊鍘嬬缉 QCS (鍦ㄧ嚎鏇存柊 - Shape Aware)
            # ============================================
            
            if self.last_raw_point is not None:
                self.qcs_compressed.add_segment(self.last_raw_point, current_raw_point)
            else:
                self.qcs_compressed.add_point(current_raw_point)
            
            self.last_raw_point = current_raw_point
        
        self.current_step += 1
        info['const_state'] = self.const_state
        return self.mask.copy(), 0.0, False, info

    def lightweight_step(self, action_idx: int):
        """[Optimization] 杞婚噺绾ф墽琛岋紝浠呮洿鏂版帺鐮佸拰甯告暟鐘舵€侊紝璺宠繃 QCS 鍜屽鍔辫绠?"""
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
        """鏄惧紡閲婃斁鍐呭瓨"""
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
        """
        璁＄畻缁堟濂栧姳 (浣跨敤 QCS 涓嬬晫浼拌 + 鏇茬巼骞虫粦濂栧姳)
        
        [Fix] 褰㈢姸鎰熺煡濂栧姳锛?
        鐢变簬 GFlowNet 鏄贡搴忛€夌偣鐨勶紝鍦ㄧ嚎鏇存柊鐨?qcs_compressed 浼氳繛鍑洪敊璇殑绾挎銆?
        鍦ㄨ绠楀鍔辨椂锛屾垜浠繀椤诲厛瀵瑰凡閫夌偣杩涜鏃跺簭鎺掑簭锛岄噸鏂版瀯鍥俱€?
        """
        num_points = self.const_state.num_points
        sparsity = num_points / max(1, self.N)
        self._last_sparsity = sparsity
        self._last_num_points = num_points
        cr_cap = float(np.clip(self.target_compression, 1e-4, 1.0))
        self._last_cr_cap = cr_cap
        excess = max(0.0, sparsity - cr_cap)
        self._last_cr_excess = excess
        
        # 1. 鎻愬彇鎵€鏈夊凡閫夌偣鐨勭储寮曞苟鎺掑簭
        selected_indices = np.where(self.mask)[0]
        selected_indices = np.sort(selected_indices)
        
        # 2. 閲嶆柊鏋勫缓涓€涓鍚堟椂搴忔嫇鎵戠殑涓存椂 QCS
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
        
        # 3. 璁＄畻鍩轰簬姝ｇ‘褰㈢姸鐨?F1 涓嬬晫
        f1_lb = f1_lower_bound(temp_qcs, self.qcs_original)
        self._last_f1 = f1_lb
        
        if num_points <= 2:
            return 1e-9

        # 4. 璁＄畻鏇茬巼骞虫粦濂栧姳
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
        """
        Unified candidate evaluation for PRS-Dual-Frontier.

        Returns:
            dict with fields: f1_lb, cr, feasible, reward
        """
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

        # Reuse the reward shape from get_reward while allowing custom tau.
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
        """
        Low-fidelity candidate evaluation for multi-fidelity ranking.

        Uses a lower-resolution original sketch plus index subsampling to reduce
        evaluation cost. When proxy sketch is unavailable, falls back to exact
        evaluation.
        """
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
        """
        鏇茬巼骞虫粦濂栧姳锛氫繚鐣欑偣鐨勬洸鐜囧彉鍖栬秺灏忓鍔辫秺楂樸€?
        
        浣跨敤棰勮绠楃殑杞悜瑙?(turning_angles) 浣滀负鏇茬巼鐨勮繎浼笺€?
        """
        if len(kept_indices) < 2:
            return 0.0
        
        # 浣跨敤杞悜瑙掍綔涓烘洸鐜囪繎浼?
        curv = self.turning_angles[kept_indices]
        curv_diff = np.abs(np.diff(curv))
        
        # 鏇茬巼鍙樺寲瓒婂皬濂栧姳瓒婇珮 (璐熺殑鍧囧€煎彉鍖?-> 姝ｅ鍔?
        mean_diff = np.mean(curv_diff) if len(curv_diff) > 0 else 0.0
        
        # 褰掍竴鍖栧苟涔樹互 gamma_curve锛堟洸鐜囧彉鍖栬寖鍥寸害 0~蟺锛?
        return self.gamma_curve * (1.0 - mean_diff / math.pi)



    def greedy_grid_simplify(self) -> List[int]:
        """
        [New] 璐績缃戞牸瑕嗙洊绠楁硶 (浣滀负 Teacher Signal)
        鍘熷垯锛氬彧淇濈暀鈥滆兘鏂板缃戞牸瑕嗙洊鈥濈殑鐐?
        """
        covered_cells = set()
        selected_indices = []
        
        # 閬靛惊 keep_start 绾︽潫
        if self.keep_start:
            selected_indices.append(0)
            start_cell = self.qcs_compressed._query_aware_cell(self.raw_trajectory[0])
            covered_cells.add(start_cell)
        
        for i in range(1 if self.keep_start else 0, self.N - 1):
            cell = self.qcs_compressed._query_aware_cell(self.raw_trajectory[i])
            if cell not in covered_cells:
                selected_indices.append(i)
                covered_cells.add(cell)
        
        # 閬靛惊 keep_end 绾︽潫
        if self.keep_end and self.N > 1:
            if (self.N - 1) not in selected_indices:
                selected_indices.append(self.N - 1)
            
        return selected_indices
    
    def _check_queries(self, selected_indices: np.ndarray) -> Set[int]:
        """妫€鏌ョ畝鍖栬建杩瑰懡涓摢浜涙煡璇?"""
        hits = set()
        
        if len(selected_indices) < 1:
            return hits
        
        # 浣跨敤鍘熷鍧愭爣
        pts = self.raw_trajectory[selected_indices]
        
        # 娣诲姞閿氱偣
        check_pts = []
        if self.prev_anchor is not None:
            check_pts.append(self.prev_anchor.reshape(1, -1))
        check_pts.append(pts)
        if self.next_anchor is not None:
            check_pts.append(self.next_anchor.reshape(1, -1))
        
        full_pts = np.concatenate(check_pts, axis=0) if len(check_pts) > 1 else pts
        
        if len(full_pts) < 2:
            return hits
        
        # 杈圭晫妗嗗揩閫熷壀鏋?
        t_min_traj, t_max_traj = full_pts[:, 2].min(), full_pts[:, 2].max()
        x_min_traj, x_max_traj = full_pts[:, 0].min(), full_pts[:, 0].max()
        y_min_traj, y_max_traj = full_pts[:, 1].min(), full_pts[:, 1].max()
        
        for q_idx, query in enumerate(self.queries):
            x_min, x_max, y_min, y_max, t_min, t_max = query
            
            # 蹇€熸帓闄?
            if (t_max_traj < t_min or t_min_traj > t_max or
                x_max_traj < x_min or x_min_traj > x_max or
                y_max_traj < y_min or y_min_traj > y_max):
                continue
            
            # 鐐规鏌?
            in_x = (full_pts[:, 0] >= x_min) & (full_pts[:, 0] <= x_max)
            in_y = (full_pts[:, 1] >= y_min) & (full_pts[:, 1] <= y_max)
            in_t = (full_pts[:, 2] >= t_min) & (full_pts[:, 2] <= t_max)
            
            if np.any(in_x & in_y & in_t):
                hits.add(q_idx)
                continue
            
            # 绾挎妫€鏌?
            for i in range(len(full_pts) - 1):
                if self._line_intersects_box(
                    full_pts[i], full_pts[i+1],
                    x_min, x_max, y_min, y_max, t_min, t_max
                ):
                    hits.add(q_idx)
                    break
        
        return hits
    
    def _line_intersects_box(self, p1, p2, x_min, x_max, y_min, y_max, t_min, t_max) -> bool:
        """Liang-Barsky 绾挎瑁佸壀绠楁硶"""
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
        """杩斿洖褰撳墠鐘舵€佷笅鐨勬湁鏁堝姩浣滄帺鐮?"""
        # 鍙€? 鏈閫変腑鐨勭偣 + 缁堟鍔ㄤ綔
        valid = np.zeros(self.N + 1, dtype=bool)
        valid[:-1] = ~self.mask  # 鏈€変腑鐨勭偣鍙€?
        valid[-1] = True  # 缁堟鍔ㄤ綔濮嬬粓鍙€?
        return valid


# ==============================================================================
# 鎵归噺鐜鍖呰鍣?
# ==============================================================================

class BatchGFlowNetEnv:
    """鎵归噺 GFlowNet 鐜锛屾敮鎸佸苟琛屽鐞嗗鏉¤建杩?
    
    鍐呭瓨浼樺寲鐗堟湰锛?
    - 浣跨敤 __slots__ 鍑忓皯瀵硅薄寮€閿€
    - 娣诲姞 clear() 鏂规硶閲婃斁鍐呭瓨
    """
    
    __slots__ = ['batch_size', 'device', 'envs', 'max_len', 'done',
                 '_alpha', '_beta', '_f1_threshold', '_target_compression',
                 '_queries']
    
    def __init__(
        self,
        trajectories: List[np.ndarray],
        raw_trajectories: Optional[List[np.ndarray]] = None,
        device: str = 'cpu',
        alpha: float = 1.0,
        beta: float = 1.0,
        f1_threshold: float = 0.95,
        target_compression: float = 0.001,
        queries: Optional[List] = None,
        batch_gt_hits: Optional[List[Set[int]]] = None,
        global_stats: Optional[dict] = None
    ):
        self.batch_size = len(trajectories)
        self.device = device
        
        # 淇濆瓨閰嶇疆鍙傛暟锛堢敤浜?__slots__锛?
        self._alpha = alpha
        self._beta = beta
        self._f1_threshold = f1_threshold
        self._target_compression = target_compression
        self._queries = queries
        
        if raw_trajectories is None:
            raw_trajectories = [None] * self.batch_size
        if batch_gt_hits is None:
            batch_gt_hits = [None] * self.batch_size
        
        self.envs = [
            GFlowNetTrajectoryEnv(
                trajectory=t,
                raw_trajectory=raw_trajectories[i],
                device=device,
                alpha=alpha,
                beta=beta,
                f1_threshold=f1_threshold,
                target_compression=target_compression,
                queries=queries,
                gt_hits=batch_gt_hits[i],
                global_stats=global_stats
            )
            for i, t in enumerate(trajectories)
        ]
        
        self.max_len = max(len(t) for t in trajectories)
        self.done = np.zeros(self.batch_size, dtype=bool)
    
    def reset(self, keep_starts: Optional[List[bool]] = None, 
              keep_ends: Optional[List[bool]] = None) -> np.ndarray:
        """閲嶇疆鎵€鏈夌幆澧?"""
        self.done = np.zeros(self.batch_size, dtype=bool)
        
        if keep_starts is None:
            keep_starts = [True] * self.batch_size
        if keep_ends is None:
            keep_ends = [True] * self.batch_size
        
        masks = [env.reset(keep_starts[i], keep_ends[i]) for i, env in enumerate(self.envs)]
        
        # Padding
        padded = np.zeros((self.batch_size, self.max_len), dtype=bool)
        for i, m in enumerate(masks):
            padded[i, :len(m)] = m
        
        return padded
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """鎵ц鎵归噺鍔ㄤ綔"""
        next_masks = []
        rewards = []
        dones = []
        
        for i, env in enumerate(self.envs):
            if self.done[i]:
                next_masks.append(env.mask)
                rewards.append(0.0)
                dones.append(True)
                continue
            
            try:
                nm, r, d, _ = env.step(actions[i])
            except ValueError:
                nm, r, d = env.mask, 0.0, True
            
            next_masks.append(nm)
            rewards.append(r)
            dones.append(d)
        
        self.done = np.array(dones)
        
        # Padding
        padded = np.zeros((self.batch_size, self.max_len), dtype=bool)
        for i, m in enumerate(next_masks):
            padded[i, :len(m)] = m
        
        return padded, np.array(rewards), self.done, {}
    
    def get_valid_actions_batch(self) -> np.ndarray:
        """鑾峰彇鎵归噺鏈夋晥鍔ㄤ綔鎺╃爜"""
        # [B, MaxLen + 1]
        valid = np.zeros((self.batch_size, self.max_len + 1), dtype=bool)
        for i, env in enumerate(self.envs):
            v = env.get_valid_actions()
            valid[i, :len(v)] = v
        return valid
    
    def clear(self):
        """閲婃斁鎵€鏈夌幆澧冨唴瀛?
        
        鍦ㄥ鐞嗗畬涓€涓?batch/chunk 鍚庤皟鐢ㄦ鏂规硶閲婃斁鍐呭瓨
        """
        import gc
        
        # 娓呯悊姣忎釜鐜
        for env in self.envs:
            # 娓呯悊鐜鍐呴儴澶у璞?
            env.trajectory = None
            env.raw_trajectory = None
            env.velocities = None
            env.turning_angles = None
            env.queries = None
            env.gt_hits = None
        
        # 娓呯┖鐜鍒楄〃
        self.envs.clear()
        self.batch_size = 0
        self.max_len = 0
        
        # 寮哄埗鍨冨溇鍥炴敹
        gc.collect()


# ==============================================================================
# 澶氭櫤鑳戒綋鐜鍖呰鍣?
# ==============================================================================

class MultiAgentEnv:
    """
    澶氭櫤鑳戒綋鐜鍖呰鍣細灏嗚建杩瑰垝鍒嗕负澶氫釜瀛愭锛屾瘡涓瓙娈电敱鐙珛鏅鸿兘浣撳鐞嗐€?
    
    鐢ㄤ簬 MARL 璁粌锛屾敮鎸侊細
    - 缁熶竴鐨?reset/step/close 鎺ュ彛
    - 鍏ㄥ眬濂栧姳鑱氬悎
    - 灞€閮ㄥ鍔辫绠?
    """
    
    def __init__(
        self,
        envs: List[GFlowNetTrajectoryEnv],
        global_reward_weight: float = 0.5
    ):
        """
        鍒濆鍖栧鏅鸿兘浣撶幆澧?
        
        Parameters
        ----------
        envs : List[GFlowNetTrajectoryEnv]
            瀛愮幆澧冨垪琛紝姣忎釜瀵瑰簲涓€涓瓙娈点€?
        global_reward_weight : float
            鍏ㄥ眬濂栧姳鏉冮噸 (1 - global_reward_weight 涓哄眬閮ㄥ鍔辨潈閲?銆?
        """
        self.envs = envs
        self.num_agents = len(envs)
        self.global_reward_weight = global_reward_weight
        self.done = np.zeros(self.num_agents, dtype=bool)
    
    def reset(self) -> List[np.ndarray]:
        """閲嶇疆鎵€鏈夊瓙鐜"""
        self.done = np.zeros(self.num_agents, dtype=bool)
        return [env.reset() for env in self.envs]
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], List[dict]]:
        """
        鎵ц澶氭櫤鑳戒綋鍔ㄤ綔
        
        Parameters
        ----------
        actions : List[int]
            姣忎釜鏅鸿兘浣撶殑鍔ㄤ綔鍒楄〃銆?
        
        Returns
        -------
        masks : List[np.ndarray]
            姣忎釜瀛愮幆澧冪殑涓嬩竴鐘舵€佹帺鐮併€?
        rewards : List[float]
            姣忎釜瀛愮幆澧冪殑濂栧姳锛堟贩鍚堝眬閮ㄤ笌鍏ㄥ眬锛夈€?
        dones : List[bool]
            姣忎釜瀛愮幆澧冩槸鍚︾粓姝€?
        infos : List[dict]
            姣忎釜瀛愮幆澧冪殑棰濆淇℃伅銆?
        """
        masks = []
        local_rewards = []
        dones = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if self.done[i]:
                masks.append(env.mask.copy())
                local_rewards.append(0.0)
                dones.append(True)
                infos.append({})
                continue
            
            mask, reward, done, info = env.step(action)
            masks.append(mask)
            local_rewards.append(reward)
            dones.append(done)
            infos.append(info)
            self.done[i] = done
        
        # 璁＄畻鍏ㄥ眬濂栧姳锛堟墍鏈夊瓙鐜濂栧姳鐨勫钩鍧囷級
        if all(self.done):
            global_reward = np.mean([r for r in local_rewards if r > 0]) if any(r > 0 for r in local_rewards) else 0.0
            # 娣峰悎灞€閮ㄤ笌鍏ㄥ眬濂栧姳
            rewards = [
                self.global_reward_weight * global_reward + (1 - self.global_reward_weight) * lr
                for lr in local_rewards
            ]
        else:
            rewards = local_rewards
        
        return masks, rewards, dones, infos
    
    def close(self):
        """鍏抽棴鎵€鏈夊瓙鐜"""
        for env in self.envs:
            env.close()
    
    def get_all_selected_indices(self) -> List[int]:
        """鑾峰彇鎵€鏈夊瓙鐜鐨勫凡閫夌偣绱㈠紩锛堝叏灞€绱㈠紩锛?"""
        all_indices = []
        for env in self.envs:
            selected = np.where(env.mask)[0]
            all_indices.extend(selected.tolist())
        return sorted(set(all_indices))

