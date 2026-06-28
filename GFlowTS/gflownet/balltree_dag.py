# -*- coding: utf-8 -*-
"""
BallTree 作为 GFlowNet DAG 结构

将 BallTree 从查询加速结构升级为 GFlowNet 的状态空间：
- BallTree 节点 = GFlowNet state
- 父子关系 = forward/backward transitions
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import IntEnum


@dataclass
class BallTreeNode:
    """
    BallTree 节点 = GFlowNet DAG 中的一个状态
    """
    node_id: int                      # 节点唯一 ID
    center: np.ndarray                # 球心坐标 (x, y, t)
    radius: float                     # 球半径
    point_indices: List[int]          # 包含的轨迹点索引
    depth: int                        # 树深度
    parent_id: Optional[int] = None   # 父节点 ID
    left_child_id: Optional[int] = None
    right_child_id: Optional[int] = None
    is_leaf: bool = False
    
    # GFlowNet 状态特征
    local_complexity: float = 0.0     # 局部轨迹复杂度
    retention_ratio: float = 0.0      # 当前保留点比例
    estimated_f1: float = 0.0         # F1 预测值
    
    def to_state_tensor(self) -> np.ndarray:
        """转换为 GFlowNet 状态张量"""
        return np.array([
            self.center[0] if len(self.center) > 0 else 0,
            self.center[1] if len(self.center) > 1 else 0,
            self.center[2] if len(self.center) > 2 else 0,
            self.radius,
            len(self.point_indices),
            self.depth,
            self.local_complexity,
            self.retention_ratio,
            self.estimated_f1
        ], dtype=np.float32)
    
    @staticmethod
    def state_dim() -> int:
        return 9


class BallTreeDAGAction(IntEnum):
    """BallTree DAG 上的动作"""
    GO_LEFT = 0       # 进入左子节点
    GO_RIGHT = 1      # 进入右子节点
    SELECT_ALL = 2    # 选择当前节点所有点
    SELECT_NONE = 3   # 跳过当前节点
    TERMINATE = 4     # 终止


class BallTreeDAG:
    """
    BallTree 作为 GFlowNet 的 DAG 结构
    
    特性：
    - 节点即状态
    - 父子关系定义 forward/backward transitions
    - 支持层次化策略
    """
    
    def __init__(
        self,
        trajectory: np.ndarray,
        max_depth: int = 10,
        min_leaf_size: int = 5
    ):
        """
        Args:
            trajectory: 轨迹点 [N, 3]
            max_depth: 最大树深度
            min_leaf_size: 叶节点最小点数
        """
        self.trajectory = trajectory
        self.N = len(trajectory)
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        
        # 节点存储
        self.nodes: Dict[int, BallTreeNode] = {}
        self.root_id: int = 0
        self.next_node_id: int = 0
        
        # 构建树
        self._build_tree()
        
        # 当前状态追踪
        self.current_node_id: int = self.root_id
        self.selected_points: Set[int] = set()
        self.visited_nodes: Set[int] = set()
    
    def _build_tree(self):
        """递归构建 BallTree"""
        all_indices = list(range(self.N))
        self.root_id = self._build_node(all_indices, depth=0, parent_id=None)
    
    def _build_node(
        self,
        indices: List[int],
        depth: int,
        parent_id: Optional[int]
    ) -> int:
        """构建单个节点"""
        node_id = self.next_node_id
        self.next_node_id += 1
        
        points = self.trajectory[indices]
        center = points.mean(axis=0)
        radius = np.max(np.linalg.norm(points - center, axis=1)) if len(points) > 0 else 0.0
        
        # 计算局部复杂度（转向角变化）
        local_complexity = self._compute_complexity(indices)
        
        node = BallTreeNode(
            node_id=node_id,
            center=center,
            radius=radius,
            point_indices=indices,
            depth=depth,
            parent_id=parent_id,
            local_complexity=local_complexity
        )
        
        # 判断是否为叶节点
        if depth >= self.max_depth or len(indices) <= self.min_leaf_size:
            node.is_leaf = True
        else:
            # 分割点
            left_indices, right_indices = self._split_points(indices, center)
            
            if len(left_indices) > 0 and len(right_indices) > 0:
                node.left_child_id = self._build_node(left_indices, depth + 1, node_id)
                node.right_child_id = self._build_node(right_indices, depth + 1, node_id)
            else:
                node.is_leaf = True
        
        self.nodes[node_id] = node
        return node_id
    
    def _split_points(
        self,
        indices: List[int],
        center: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """沿主方向分割点"""
        points = self.trajectory[indices]
        centered = points - center
        
        # PCA 找主方向
        if len(points) < 2:
            return indices, []
        
        try:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            principal = eigenvectors[:, np.argmax(eigenvalues)].real
        except:
            principal = np.array([1, 0, 0])
        
        # 投影并分割
        projections = centered @ principal
        median = np.median(projections)
        
        left = [indices[i] for i, p in enumerate(projections) if p <= median]
        right = [indices[i] for i, p in enumerate(projections) if p > median]
        
        return left, right
    
    def _compute_complexity(self, indices: List[int]) -> float:
        """计算局部轨迹复杂度"""
        if len(indices) < 3:
            return 0.0
        
        sorted_indices = sorted(indices)
        total_angle = 0.0
        
        for i in range(1, len(sorted_indices) - 1):
            if sorted_indices[i] == 0 or sorted_indices[i] >= self.N - 1:
                continue
            
            p_prev = self.trajectory[sorted_indices[i] - 1]
            p_curr = self.trajectory[sorted_indices[i]]
            p_next = self.trajectory[sorted_indices[i] + 1]
            
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            
            norm1 = np.linalg.norm(v1[:2])
            norm2 = np.linalg.norm(v2[:2])
            
            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_angle = np.clip(np.dot(v1[:2], v2[:2]) / (norm1 * norm2), -1, 1)
                total_angle += np.arccos(cos_angle)
        
        return total_angle / max(1, len(indices) - 2)
    
    def reset(self) -> BallTreeNode:
        """重置到根节点"""
        self.current_node_id = self.root_id
        self.selected_points = set()
        self.visited_nodes = set()
        self.visited_nodes.add(self.root_id)
        return self.nodes[self.root_id]
    
    def step(self, action: BallTreeDAGAction) -> Tuple[BallTreeNode, bool]:
        """
        执行动作（forward transition）
        
        Returns:
            next_node: 下一个状态节点
            done: 是否终止
        """
        current = self.nodes[self.current_node_id]
        
        if action == BallTreeDAGAction.TERMINATE:
            return current, True
        
        if action == BallTreeDAGAction.SELECT_ALL:
            self.selected_points.update(current.point_indices)
            return current, current.is_leaf
        
        if action == BallTreeDAGAction.SELECT_NONE:
            return current, current.is_leaf
        
        if action == BallTreeDAGAction.GO_LEFT and current.left_child_id is not None:
            self.current_node_id = current.left_child_id
            self.visited_nodes.add(self.current_node_id)
            return self.nodes[self.current_node_id], False
        
        if action == BallTreeDAGAction.GO_RIGHT and current.right_child_id is not None:
            self.current_node_id = current.right_child_id
            self.visited_nodes.add(self.current_node_id)
            return self.nodes[self.current_node_id], False
        
        # 无效动作，保持当前状态
        return current, False
    
    def backward_step(self, action: BallTreeDAGAction) -> Tuple[BallTreeNode, bool]:
        """
        反向转移（用于 Backward Policy）
        从当前节点回到父节点
        """
        current = self.nodes[self.current_node_id]
        
        if current.parent_id is None:
            return current, True  # 已在根节点
        
        self.current_node_id = current.parent_id
        return self.nodes[self.current_node_id], False
    
    def get_valid_actions(self) -> List[BallTreeDAGAction]:
        """获取当前节点的有效动作"""
        current = self.nodes[self.current_node_id]
        valid = [BallTreeDAGAction.TERMINATE]
        
        if current.is_leaf:
            valid.extend([BallTreeDAGAction.SELECT_ALL, BallTreeDAGAction.SELECT_NONE])
        else:
            if current.left_child_id is not None:
                valid.append(BallTreeDAGAction.GO_LEFT)
            if current.right_child_id is not None:
                valid.append(BallTreeDAGAction.GO_RIGHT)
        
        return valid
    
    def get_selected_mask(self) -> np.ndarray:
        """获取选中点的掩码"""
        mask = np.zeros(self.N, dtype=bool)
        for idx in self.selected_points:
            mask[idx] = True
        return mask
    
    def get_state_features(self) -> np.ndarray:
        """获取当前状态的特征向量"""
        current = self.nodes[self.current_node_id]
        
        # 更新保留比例
        current.retention_ratio = len(self.selected_points) / self.N
        
        return current.to_state_tensor()
    
    def get_trajectory_path(self) -> List[int]:
        """获取从根到当前节点的路径"""
        path = []
        node_id = self.current_node_id
        
        while node_id is not None:
            path.append(node_id)
            node_id = self.nodes[node_id].parent_id
        
        return list(reversed(path))


class HierarchicalBallTreePolicy(nn.Module):
    """
    层次化 BallTree 策略网络
    
    在 BallTree DAG 上进行 GFlowNet 采样
    """
    
    def __init__(
        self,
        state_dim: int = 9,
        hidden_dim: int = 64,
        num_actions: int = 5
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        # 状态编码器
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Forward Policy 头
        self.forward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        # Backward Policy 头
        self.backward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )
    
    def forward_policy(
        self,
        state: torch.Tensor,
        valid_actions_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward Policy: P_F(a | s)
        
        Args:
            state: [B, state_dim]
            valid_actions_mask: [B, num_actions], True=有效
        """
        h = self.encoder(state)
        logits = self.forward_head(h)
        
        # 屏蔽无效动作
        logits = logits.masked_fill(~valid_actions_mask, float('-inf'))
        
        return logits
    
    def backward_policy(
        self,
        state: torch.Tensor,
        valid_actions_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Backward Policy: P_B(a | s)
        """
        h = self.encoder(state)
        logits = self.backward_head(h)
        logits = logits.masked_fill(~valid_actions_mask, float('-inf'))
        return logits
