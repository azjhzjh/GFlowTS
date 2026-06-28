# -*- coding: utf-8 -*-
"""
GFlowNet Backward Policy 实现
根据 TLM (Trajectory Likelihood Maximization) 论文设计

Backward Policy P_B(s_prev | s_curr) 用于：
1. 从当前状态预测可能的前一步动作
2. 仅使用 TLM 损失进行训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BackwardPolicy(nn.Module):
    """
    GFlowNet Backward Policy: P(s_prev | s_curr)
    
    输入当前压缩状态，输出可能的前一步动作概率分布
    使用 softmax 输出，而非 ε-greedy
    """
    
    def __init__(
        self, 
        state_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            state_dim: 状态维度 (通常是点嵌入维度)
            hidden_dim: 隐藏层维度
            num_layers: 网络层数
            dropout: Dropout 比例
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # 状态编码器
        layers = []
        in_dim = state_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # 动作头：预测移除哪个点 (去往前一步状态)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 每点一个 logit
        )
        
        # 停止动作头 (对应 Forward 的 Stop 动作的逆)
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        traj_embeddings: torch.Tensor,
        mask: torch.Tensor,
        valid_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算 Backward Policy 的 logits
        
        Args:
            traj_embeddings: 轨迹点嵌入 [B, N, H]
            mask: 当前选择掩码 [B, N], 1=已选中, 0=未选中
            valid_lens: 每条轨迹的有效长度 [B]
            
        Returns:
            logits: 后向动作 logits [B, N+1]
                    前 N 个对应移除每个已选点
                    最后一个对应 "取消停止动作"
        """
        B, N, H = traj_embeddings.shape
        
        # 全局上下文：已选点的平均嵌入
        mask_expanded = mask.unsqueeze(-1)  # [B, N, 1]
        selected_sum = (traj_embeddings * mask_expanded).sum(dim=1)  # [B, H]
        selected_count = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
        global_context = selected_sum / selected_count  # [B, H]
        
        # 编码全局上下文
        global_encoded = self.encoder(global_context)  # [B, hidden_dim]
        
        # 拼接每点嵌入和全局上下文
        global_rep = global_encoded.unsqueeze(1).expand(-1, N, -1)  # [B, N, hidden_dim]
        
        # 对每个点计算 "移除该点" 的 logit
        combined = torch.cat([traj_embeddings, global_rep], dim=-1)  # [B, N, H + hidden_dim]
        
        # 点级别动作 logits
        point_logits = self.action_head(combined[:, :, :self.hidden_dim]).squeeze(-1)  # [B, N]
        
        # 只有已选中的点才能被移除 (mask=1 的点)
        # 未选中的点设为 -inf
        point_logits = point_logits.masked_fill(~mask.bool(), float('-inf'))
        
        # 首尾点通常不能移除
        point_logits[:, 0] = float('-inf')
        if N > 1:
            point_logits[:, -1] = float('-inf')
        
        # 停止逆动作 logit
        stop_logit = self.stop_head(global_encoded)  # [B, 1]
        
        # 拼接
        logits = torch.cat([point_logits, stop_logit], dim=1)  # [B, N+1]
        
        # Padding 掩码
        if valid_lens is not None:
            pad_mask = torch.arange(N, device=logits.device).unsqueeze(0) >= valid_lens.unsqueeze(1)
            logits[:, :-1] = logits[:, :-1].masked_fill(pad_mask, float('-inf'))
        
        return logits
    
    def sample(
        self, 
        traj_embeddings: torch.Tensor,
        mask: torch.Tensor,
        valid_lens: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从后向策略采样动作
        
        Returns:
            actions: 采样的动作 [B]
            log_probs: 对应的 log 概率 [B]
        """
        logits = self.forward(traj_embeddings, mask, valid_lens)
        
        # 温度缩放
        scaled_logits = logits / temperature
        
        # Categorical 采样
        dist = torch.distributions.Categorical(logits=scaled_logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        
        return actions, log_probs
    
    def log_prob(
        self,
        traj_embeddings: torch.Tensor,
        mask: torch.Tensor,
        actions: torch.Tensor,
        valid_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算指定动作的 log 概率
        
        Args:
            actions: 动作索引 [B]
            
        Returns:
            log_probs: log P_B(a | s) [B]
        """
        logits = self.forward(traj_embeddings, mask, valid_lens)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions)


class ContextAwareBackwardPolicy(BackwardPolicy):
    """
    上下文感知的后向策略
    额外考虑轨迹的运动特征（速度、转向角）
    注意：不使用几何误差 (SED/PED)
    """
    
    def __init__(
        self,
        state_dim: int,
        point_feature_dim: int = 8,  # PointState.dim() - 已移除 sed_error
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__(state_dim, hidden_dim, num_layers, dropout)
        
        # 点特征编码器 (速度、转向角等)
        self.point_feature_encoder = nn.Sequential(
            nn.Linear(point_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # 更新动作头以接受额外特征
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        traj_embeddings: torch.Tensor,
        mask: torch.Tensor,
        point_features: Optional[torch.Tensor] = None,
        valid_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            point_features: 每点的额外特征 [B, N, F]
        """
        B, N, H = traj_embeddings.shape
        
        # 全局上下文
        mask_expanded = mask.unsqueeze(-1)
        selected_sum = (traj_embeddings * mask_expanded).sum(dim=1)
        selected_count = mask_expanded.sum(dim=1).clamp(min=1)
        global_context = selected_sum / selected_count
        
        global_encoded = self.encoder(global_context)
        global_rep = global_encoded.unsqueeze(1).expand(-1, N, -1)
        
        # 组合嵌入
        if point_features is not None:
            point_feat_encoded = self.point_feature_encoder(point_features)
            combined = torch.cat([global_rep, point_feat_encoded], dim=-1)
        else:
            combined = global_rep
        
        # 动作 logits
        point_logits = self.action_head(combined).squeeze(-1)
        point_logits = point_logits.masked_fill(~mask.bool(), float('-inf'))
        point_logits[:, 0] = float('-inf')
        if N > 1:
            point_logits[:, -1] = float('-inf')
        
        stop_logit = self.stop_head(global_encoded)
        logits = torch.cat([point_logits, stop_logit], dim=1)
        
        if valid_lens is not None:
            pad_mask = torch.arange(N, device=logits.device).unsqueeze(0) >= valid_lens.unsqueeze(1)
            logits[:, :-1] = logits[:, :-1].masked_fill(pad_mask, float('-inf'))
        
        return logits
