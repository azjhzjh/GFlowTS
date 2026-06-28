# -*- coding: utf-8 -*-
"""
Query Sketch - O(1) 空间复杂度的查询覆盖统计

用于替代 O(n) 的点掩码，实现常数大小的状态表示。
使用空间哈希将点映射到固定大小的网格中。
"""

import numpy as np
from typing import Tuple, Optional


class QuerySketch:
    """
    查询覆盖 Sketch - O(1) 空间复杂度
    
    使用空间哈希将轨迹点映射到固定大小的网格中，
    用于估计查询覆盖率而不需要存储所有已选点。
    
    状态大小: O(grid_size^2) = 常数
    
    用法:
        sketch = QuerySketch(grid_size=64)
        sketch.update_with_point(point)
        coverage = sketch.get_coverage_ratio()
    """
    
    __slots__ = ['cells', 'grid_size', 'x_range', 'y_range', 't_range',
                 '_num_updates', '_last_point', '_bbox']
    
    def __init__(
        self, 
        grid_size: int = 64,
        x_range: Tuple[float, float] = (0.0, 1.0),
        y_range: Tuple[float, float] = (0.0, 1.0),
        t_range: Tuple[float, float] = (0.0, 1.0)
    ):
        """
        Args:
            grid_size: 网格边长（总单元格数 = grid_size^2）
            x_range: x 坐标范围 (用于归一化哈希)
            y_range: y 坐标范围
            t_range: t 坐标范围
        """
        self.grid_size = grid_size
        self.x_range = x_range
        self.y_range = y_range
        self.t_range = t_range
        
        # 使用 set 存储已覆盖的单元格 ID
        # 可替换为 BitSet 或 Bloom Filter 以进一步优化
        self.cells = set()
        
        # 统计信息
        self._num_updates = 0
        self._last_point = None
        self._bbox = {
            'x_min': float('inf'), 'x_max': float('-inf'),
            'y_min': float('inf'), 'y_max': float('-inf'),
            't_min': float('inf'), 't_max': float('-inf')
        }
    
    def reset(self):
        """重置 Sketch 状态"""
        self.cells.clear()
        self._num_updates = 0
        self._last_point = None
        self._bbox = {
            'x_min': float('inf'), 'x_max': float('-inf'),
            'y_min': float('inf'), 'y_max': float('-inf'),
            't_min': float('inf'), 't_max': float('-inf')
        }
    
    def update_with_point(self, point: np.ndarray):
        """
        添加点到 Sketch
        
        Args:
            point: (x, y, t) 坐标数组
        """
        cell_id = self._spatial_hash(point)
        self.cells.add(cell_id)
        
        # 更新统计
        self._num_updates += 1
        self._last_point = point.copy() if isinstance(point, np.ndarray) else np.array(point)
        
        # 更新边界框
        x, y, t = float(point[0]), float(point[1]), float(point[2])
        self._bbox['x_min'] = min(self._bbox['x_min'], x)
        self._bbox['x_max'] = max(self._bbox['x_max'], x)
        self._bbox['y_min'] = min(self._bbox['y_min'], y)
        self._bbox['y_max'] = max(self._bbox['y_max'], y)
        self._bbox['t_min'] = min(self._bbox['t_min'], t)
        self._bbox['t_max'] = max(self._bbox['t_max'], t)
    
    def _spatial_hash(self, point: np.ndarray) -> int:
        """
        计算点的空间哈希值
        
        Args:
            point: (x, y, t) 坐标
        
        Returns:
            cell_id: 网格单元格 ID
        """
        x, y, t = float(point[0]), float(point[1]), float(point[2])
        
        # 归一化到 [0, grid_size) 范围
        x_norm = (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0] + 1e-9)
        y_norm = (y - self.y_range[0]) / (self.y_range[1] - self.y_range[0] + 1e-9)
        
        xi = int(np.clip(x_norm * self.grid_size, 0, self.grid_size - 1))
        yi = int(np.clip(y_norm * self.grid_size, 0, self.grid_size - 1))
        
        # 2D 网格索引
        return xi * self.grid_size + yi
    
    def get_coverage_ratio(self) -> float:
        """
        获取覆盖率估计
        
        Returns:
            覆盖的网格单元格比例 [0, 1]
        """
        total_cells = self.grid_size * self.grid_size
        return len(self.cells) / total_cells
    
    def get_num_cells(self) -> int:
        """返回已覆盖的单元格数量"""
        return len(self.cells)
    
    def get_num_points(self) -> int:
        """返回已添加的点数量"""
        return self._num_updates
    
    def get_last_point(self) -> Optional[np.ndarray]:
        """返回最后添加的点"""
        return self._last_point
    
    def get_bbox_features(self) -> np.ndarray:
        """
        获取边界框特征向量
        
        Returns:
            [x_span, y_span, t_span, x_center, y_center, t_center]
        """
        if self._num_updates == 0:
            return np.zeros(6, dtype=np.float32)
        
        x_span = self._bbox['x_max'] - self._bbox['x_min']
        y_span = self._bbox['y_max'] - self._bbox['y_min']
        t_span = self._bbox['t_max'] - self._bbox['t_min']
        
        x_center = (self._bbox['x_max'] + self._bbox['x_min']) / 2
        y_center = (self._bbox['y_max'] + self._bbox['y_min']) / 2
        t_center = (self._bbox['t_max'] + self._bbox['t_min']) / 2
        
        return np.array([x_span, y_span, t_span, x_center, y_center, t_center], 
                       dtype=np.float32)
    
    def to_tensor(self) -> np.ndarray:
        """
        转换为固定大小的特征向量（用于神经网络输入）
        
        Returns:
            [coverage_ratio, num_cells_norm, num_points_norm, 
             last_x, last_y, last_t, 
             x_span, y_span, t_span, x_center, y_center, t_center]
        """
        total_cells = self.grid_size * self.grid_size
        
        features = [
            self.get_coverage_ratio(),
            len(self.cells) / total_cells,  # 归一化单元格数
            min(1.0, self._num_updates / 100.0),  # 归一化点数（假设最多 100 点）
        ]
        
        # 最后添加的点
        if self._last_point is not None:
            features.extend([
                float(self._last_point[0]),
                float(self._last_point[1]),
                float(self._last_point[2])
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 边界框特征
        features.extend(self.get_bbox_features().tolist())
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def dim() -> int:
        """返回特征向量维度"""
        return 12  # 3 + 3 + 6
    
    def __repr__(self) -> str:
        return (f"QuerySketch(grid_size={self.grid_size}, "
                f"cells={len(self.cells)}, points={self._num_updates})")


class ConstantSizeState:
    """
    常数大小的 GFlowNet 状态
    
    替代 O(n) 的点掩码，使用统计信息和 Sketch 表示状态。
    状态大小与轨迹长度无关。
    
    状态包含:
    - last_kept_point: 最后保留的点 (x, y, t)
    - sketch: QuerySketch 对象
    - num_points: 已保留的点数
    - budget_left: 剩余压缩预算
    - current_idx: 当前处理的点索引
    - total_points: 轨迹总点数
    """
    
    __slots__ = ['last_kept_point', 'sketch', 'num_points', 'budget_left',
                 'current_idx', 'total_points']
    
    def __init__(
        self,
        total_points: int,
        grid_size: int = 64,
        target_compression: float = 0.1
    ):
        self.last_kept_point = None
        self.sketch = QuerySketch(grid_size=grid_size)
        self.num_points = 0
        self.budget_left = 1.0
        self.current_idx = 0
        self.total_points = total_points
    
    def reset(self, keep_first: bool = True, first_point: Optional[np.ndarray] = None):
        """重置状态"""
        self.sketch.reset()
        self.num_points = 0
        self.budget_left = 1.0
        self.current_idx = 0
        self.last_kept_point = None
        
        if keep_first and first_point is not None:
            self.add_point(first_point)
    
    def add_point(self, point: np.ndarray):
        """添加保留的点"""
        self.sketch.update_with_point(point)
        self.last_kept_point = point.copy() if isinstance(point, np.ndarray) else np.array(point)
        self.num_points += 1
        self.budget_left = max(0, 1.0 - self.num_points / max(1, self.total_points))
    
    def advance(self):
        """前进到下一个点"""
        self.current_idx += 1
    
    def is_done(self) -> bool:
        """检查是否处理完所有点"""
        return self.current_idx >= self.total_points
    
    def to_tensor(self) -> np.ndarray:
        """
        转换为固定大小的特征向量
        
        Returns:
            [last_x, last_y, last_t, 
             num_points_norm, budget_left, progress,
             sketch_features (12 维)]
        """
        features = []
        
        # 最后保留的点
        if self.last_kept_point is not None:
            features.extend([
                float(self.last_kept_point[0]),
                float(self.last_kept_point[1]),
                float(self.last_kept_point[2])
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 统计信息
        features.extend([
            min(1.0, self.num_points / max(1, self.total_points)),  # 归一化点数
            self.budget_left,
            self.current_idx / max(1, self.total_points)  # 进度
        ])
        
        # Sketch 特征
        features.extend(self.sketch.to_tensor().tolist())
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def dim() -> int:
        """返回特征向量维度"""
        return 3 + 3 + QuerySketch.dim()  # = 18
    
    def __repr__(self) -> str:
        return (f"ConstantSizeState(points={self.num_points}, "
                f"idx={self.current_idx}/{self.total_points}, "
                f"budget={self.budget_left:.2f})")


# ==============================================================================
# Query Coverage Sketch (QCS) - F1 下界估计
# ==============================================================================

class QueryCoverageSketch:
    """
    Query Coverage Sketch (QCS)
    
    用于计算 F1 的保守下界估计，替代精确 F1 计算。
    存储查询可能命中的空间单元格，大小固定为 O(grid_size²)。
    
    关键特性：
    - 离线计算原始轨迹的 QCS
    - 在线更新压缩轨迹的 QCS
    - 只在终止时计算 f1_lower_bound
    - 中间步骤完全不看 F1
    """
    
    __slots__ = ['cells', 'grid_size', 'x_range', 'y_range', 't_range']
    
    def __init__(
        self,
        grid_size: int = 64,
        x_range: Tuple[float, float] = (0.0, 1.0),
        y_range: Tuple[float, float] = (0.0, 1.0),
        t_range: Tuple[float, float] = (0.0, 1.0)
    ):
        """
        Args:
            grid_size: 网格边长
            x_range, y_range, t_range: 坐标范围 (用于归一化)
        """
        self.grid_size = grid_size
        self.x_range = x_range
        self.y_range = y_range
        self.t_range = t_range
        self.cells = set()
    
    def reset(self):
        """重置 QCS"""
        self.cells.clear()
    
    def add_point(self, point: np.ndarray):
        """
        添加点到 QCS
        
        Maps point to query-aware cell and adds to cell set.
        """
        cell = self._query_aware_cell(point)
        self.cells.add(cell)
    
    def _query_aware_cell(self, point: np.ndarray) -> int:
        """
        计算查询感知的空间单元格 ID
        
        使用 3D 网格 (x, y, t) 确保时空范围覆盖
        """
        x, y, t = float(point[0]), float(point[1]), float(point[2])
        
        # 归一化到 [0, grid_size) 范围
        x_norm = (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0] + 1e-9)
        y_norm = (y - self.y_range[0]) / (self.y_range[1] - self.y_range[0] + 1e-9)
        t_norm = (t - self.t_range[0]) / (self.t_range[1] - self.t_range[0] + 1e-9)
        
        xi = np.clip(x_norm * self.grid_size, 0, self.grid_size - 1).astype(np.int32)
        yi = np.clip(y_norm * self.grid_size, 0, self.grid_size - 1).astype(np.int32)
        ti = np.clip(t_norm * (self.grid_size // 4), 0, (self.grid_size // 4) - 1).astype(np.int32)
        
        return xi * self.grid_size * (self.grid_size // 4) + yi * (self.grid_size // 4) + ti

    def add_points(self, points: np.ndarray):
        """批量添加点到 QCS (向量化优化)"""
        if len(points) == 0:
            return
        
        x = points[:, 0]
        y = points[:, 1]
        t = points[:, 2]
        
        # 归一化 (向量化)
        x_norm = (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0] + 1e-9)
        y_norm = (y - self.y_range[0]) / (self.y_range[1] - self.y_range[0] + 1e-9)
        t_norm = (t - self.t_range[0]) / (self.t_range[1] - self.t_range[0] + 1e-9)
        
        xi = np.clip(x_norm * self.grid_size, 0, self.grid_size - 1).astype(np.int32)
        yi = np.clip(y_norm * self.grid_size, 0, self.grid_size - 1).astype(np.int32)
        ti = np.clip(t_norm * (self.grid_size // 4), 0, (self.grid_size // 4) - 1).astype(np.int32)
        
        cell_ids = xi * self.grid_size * (self.grid_size // 4) + yi * (self.grid_size // 4) + ti
        # cell_ids = xi * self.grid_size * (self.grid_size // 4) + yi * (self.grid_size // 4) + ti 
        # (Assuming unique grid mapping strategy, or using combined hash)
        
        self.cells.update(cell_ids.tolist())
        
    def _estimate_segment_steps(self, p1: np.ndarray, p2: np.ndarray) -> int:
        x1_norm = (p1[0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0] + 1e-9)
        y1_norm = (p1[1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0] + 1e-9)
        t1_norm = (p1[2] - self.t_range[0]) / (self.t_range[1] - self.t_range[0] + 1e-9)

        x2_norm = (p2[0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0] + 1e-9)
        y2_norm = (p2[1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0] + 1e-9)
        t2_norm = (p2[2] - self.t_range[0]) / (self.t_range[1] - self.t_range[0] + 1e-9)

        dist = max(abs(x2_norm - x1_norm), abs(y2_norm - y1_norm), abs(t2_norm - t1_norm))
        return max(2, int(dist * self.grid_size * 2))

    def add_segment_with_step_cap(self, p1: np.ndarray, p2: np.ndarray, max_steps: int = 200):
        num_steps = min(self._estimate_segment_steps(p1, p2), max(2, int(max_steps)))
        alphas = np.linspace(0, 1, num_steps).reshape(-1, 1)
        points = p1 * (1 - alphas) + p2 * alphas
        self.add_points(points)

    def add_segment(self, p1: np.ndarray, p2: np.ndarray, max_steps: int = 200):
        """
        添加线段 (通过插值)
        """
        # 简单的线性插值
        # 估算步数: 基于最大归一化距离
        
        # 归一化 p1, p2
        x1_norm = (p1[0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0] + 1e-9)
        y1_norm = (p1[1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0] + 1e-9)
        t1_norm = (p1[2] - self.t_range[0]) / (self.t_range[1] - self.t_range[0] + 1e-9)
        
        x2_norm = (p2[0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0] + 1e-9)
        y2_norm = (p2[1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0] + 1e-9)
        t2_norm = (p2[2] - self.t_range[0]) / (self.t_range[1] - self.t_range[0] + 1e-9)
        
        # 计算距离 (Chebyshev distance grid wise is enough proxy)
        dist = max(abs(x2_norm - x1_norm), abs(y2_norm - y1_norm), abs(t2_norm - t1_norm))
        
        # 步数: 保证至少每半个格子采样一次
        num_steps = max(2, int(dist * self.grid_size * 2))
        
        if num_steps > 200: # 限制最大步数防止过慢
            num_steps = 200
            
        alphas = np.linspace(0, 1, num_steps).reshape(-1, 1)
        points = p1 * (1 - alphas) + p2 * alphas
        
        self.add_points(points)
    
    def size(self) -> int:
        """返回已覆盖的单元格数量"""
        return len(self.cells)
    
    def intersection(self, other: 'QueryCoverageSketch') -> set:
        """计算两个 QCS 的交集"""
        return self.cells.intersection(other.cells)
    
    @classmethod
    def from_trajectory(
        cls,
        trajectory: np.ndarray,
        grid_size: int = 64,
        x_range: Tuple[float, float] = (0.0, 1.0),
        y_range: Tuple[float, float] = (0.0, 1.0),
        t_range: Tuple[float, float] = (0.0, 1.0)
    ) -> 'QueryCoverageSketch':
        """
        从轨迹创建 QCS（向量化优化）
        """
        qcs = cls(grid_size, x_range, y_range, t_range)
        qcs.add_points(trajectory)
        return qcs
    
    def __repr__(self) -> str:
        return f"QCS(cells={len(self.cells)}, grid={self.grid_size})"


def f1_lower_bound(qcs_compressed: QueryCoverageSketch, qcs_original: QueryCoverageSketch) -> float:
    """
    计算 F1 的保守下界
    
    F1_lb = 2 * |intersection| / (|qcs_compressed| + |qcs_original|)
    
    这是 F1 的下界估计，不是精确值。
    用于判断是否满足 F1 >= 0.95 阈值。
    
    Args:
        qcs_compressed: 压缩轨迹的 QCS
        qcs_original: 原始轨迹的 QCS
    
    Returns:
        F1 下界值 [0, 1]
    """
    if qcs_compressed.size() == 0 or qcs_original.size() == 0:
        return 0.0
    
    inter = qcs_compressed.intersection(qcs_original)
    denominator = qcs_compressed.size() + qcs_original.size()
    
    if denominator == 0:
        return 0.0
    
    return 2.0 * len(inter) / denominator
