# -*- coding: utf-8 -*-
"""
Trajectory Likelihood Maximization (TLM) 损失函数

根据 ICLR 2025 TLM 论文实现：
- 仅使用成功轨迹结构
- 不依赖 reward
- 最大化后向轨迹的似然
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


def trajectory_likelihood_loss(
    trajectory_states: List[torch.Tensor],
    trajectory_actions: List[torch.Tensor],
    backward_policy: nn.Module,
    traj_embeddings: torch.Tensor,
    valid_lens: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    计算 Trajectory Likelihood Loss
    
    TLM 核心思想：最大化后向策略在成功轨迹上的似然
    
    Loss = -Σ log P_B(s_{t-1} | s_t, a_t)
    
    Args:
        trajectory_states: 状态序列 [s_0, s_1, ..., s_T]，每个是 [B, N] 的 mask
        trajectory_actions: 动作序列 [a_0, a_1, ..., a_{T-1}]，每个是 [B] 的长整型
        backward_policy: 后向策略网络
        traj_embeddings: 轨迹嵌入 [B, N, H]
        valid_lens: 有效长度 [B]
        
    Returns:
        loss: 负对数似然损失 (标量)
    """
    if len(trajectory_states) < 2:
        return torch.tensor(0.0, device=traj_embeddings.device)
    
    total_log_prob = 0.0
    count = 0
    
    # 从 t=1 开始，计算 P_B(s_{t-1} | s_t)
    for t in range(1, len(trajectory_states)):
        current_state = trajectory_states[t]      # s_t: [B, N]
        action = trajectory_actions[t - 1]        # a_{t-1}: [B]
        
        # 获取后向策略的 log 概率
        log_prob = backward_policy.log_prob(
            traj_embeddings=traj_embeddings,
            mask=current_state.float(),
            actions=action,
            valid_lens=valid_lens
        )
        
        total_log_prob = total_log_prob + log_prob
        count += 1
    
    if count == 0:
        return torch.tensor(0.0, device=traj_embeddings.device)
    
    # 负对数似然
    loss = -total_log_prob.mean()
    
    return loss


def batch_trajectory_likelihood_loss(
    batch_trajectories: List[Dict],
    backward_policy: nn.Module,
    traj_embeddings: torch.Tensor,
    valid_lens: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    批量计算 TLM 损失
    
    Args:
        batch_trajectories: 轨迹数据列表，每个包含 'states' 和 'actions'
        backward_policy: 后向策略网络
        traj_embeddings: 轨迹嵌入 [B, N, H]
        valid_lens: 有效长度 [B]
        
    Returns:
        loss: 批量平均损失
    """
    total_loss = 0.0
    valid_count = 0
    
    for traj_data in batch_trajectories:
        if 'states' not in traj_data or 'actions' not in traj_data:
            continue
        
        loss = trajectory_likelihood_loss(
            trajectory_states=traj_data['states'],
            trajectory_actions=traj_data['actions'],
            backward_policy=backward_policy,
            traj_embeddings=traj_embeddings,
            valid_lens=valid_lens
        )
        
        if not torch.isnan(loss):
            total_loss = total_loss + loss
            valid_count += 1
    
    if valid_count == 0:
        return torch.tensor(0.0, device=traj_embeddings.device)
    
    return total_loss / valid_count


class TLMLoss(nn.Module):
    """
    TLM 损失模块 (用于 nn.Module 接口)
    
    特性：
    - 仅使用成功轨迹结构
    - 不使用 reward 信号
    - 禁止 dense shaping
    """
    
    def __init__(self, backward_policy: nn.Module):
        """
        Args:
            backward_policy: 后向策略网络
        """
        super().__init__()
        self.backward_policy = backward_policy
    
    def forward(
        self,
        trajectory_states: List[torch.Tensor],
        trajectory_actions: List[torch.Tensor],
        traj_embeddings: torch.Tensor,
        valid_lens: Optional[torch.Tensor] = None,
        success_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算 TLM 损失
        
        Args:
            success_mask: 成功轨迹掩码 [B]，只有成功轨迹参与训练
            
        Returns:
            loss: TLM 损失
        """
        if len(trajectory_states) < 2:
            return torch.tensor(0.0, device=traj_embeddings.device)
        
        B = traj_embeddings.size(0)
        
        # 如果提供了成功掩码，只使用成功轨迹
        if success_mask is not None:
            if not success_mask.any():
                return torch.tensor(0.0, device=traj_embeddings.device)
        
        total_log_prob = torch.zeros(B, device=traj_embeddings.device)
        step_count = torch.zeros(B, device=traj_embeddings.device)
        
        for t in range(1, len(trajectory_states)):
            current_state = trajectory_states[t]
            action = trajectory_actions[t - 1]
            
            log_prob = self.backward_policy.log_prob(
                traj_embeddings=traj_embeddings,
                mask=current_state.float(),
                actions=action,
                valid_lens=valid_lens
            )
            
            total_log_prob = total_log_prob + log_prob
            step_count = step_count + 1
        
        # 每条轨迹的平均 log prob
        avg_log_prob = total_log_prob / step_count.clamp(min=1)
        
        # 应用成功掩码
        if success_mask is not None:
            avg_log_prob = avg_log_prob * success_mask.float()
            valid_count = success_mask.sum().clamp(min=1)
            loss = -avg_log_prob.sum() / valid_count
        else:
            loss = -avg_log_prob.mean()
        
        return loss


class SubTrajectoryTLMLoss(nn.Module):
    """
    子轨迹 TLM 损失
    
    将完整轨迹分解为子轨迹，分别计算 TLM 损失
    适用于长轨迹的高效训练
    """
    
    def __init__(
        self,
        backward_policy: nn.Module,
        sub_traj_len: int = 10
    ):
        super().__init__()
        self.backward_policy = backward_policy
        self.sub_traj_len = sub_traj_len
    
    def forward(
        self,
        trajectory_states: List[torch.Tensor],
        trajectory_actions: List[torch.Tensor],
        traj_embeddings: torch.Tensor,
        valid_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算子轨迹 TLM 损失"""
        T = len(trajectory_states)
        
        if T < 2:
            return torch.tensor(0.0, device=traj_embeddings.device)
        
        total_loss = 0.0
        n_sub = 0
        
        # 滑动窗口切分子轨迹
        for start in range(0, T - 1, self.sub_traj_len // 2):
            end = min(start + self.sub_traj_len, T)
            
            sub_states = trajectory_states[start:end]
            sub_actions = trajectory_actions[start:end-1] if end > start + 1 else []
            
            if len(sub_states) >= 2 and len(sub_actions) >= 1:
                loss = trajectory_likelihood_loss(
                    trajectory_states=sub_states,
                    trajectory_actions=sub_actions,
                    backward_policy=self.backward_policy,
                    traj_embeddings=traj_embeddings,
                    valid_lens=valid_lens
                )
                
                if not torch.isnan(loss):
                    total_loss = total_loss + loss
                    n_sub += 1
        
        if n_sub == 0:
            return torch.tensor(0.0, device=traj_embeddings.device)
        
        return total_loss / n_sub
