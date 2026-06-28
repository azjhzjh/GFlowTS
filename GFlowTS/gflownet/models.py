# -*- coding: utf-8 -*-
"""
GFlowNet 模型定义
包含 Forward Policy、轨迹编码器和层次化策略网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

# 导入常数大小状态
try:
    from .utils.query_sketch import ConstantSizeState
except ImportError:
    ConstantSizeState = None


class ConstantStateEncoder(nn.Module):
    """
    常数大小状态编码器
    
    将 ConstantSizeState 的 18 维特征向量编码为隐藏表示。
    状态大小与轨迹长度无关，实现 O(1) 空间复杂度。
    
    输入: ConstantSizeState.to_tensor() -> [B, 18]
    输出: [B, hidden_dim]
    """
    
    def __init__(self, state_dim: int = 18, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        
        layers = []
        in_dim = state_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.hidden_dim = hidden_dim
    
    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_tensor: 常数状态向量 [B, 18] 或 [B, state_dim]
        Returns:
            hidden: 隐藏表示 [B, H]
        """
        return self.encoder(state_tensor)


class TrajectoryEncoder(nn.Module):
    """
    轨迹编码器：将原始轨迹点编码为隐藏表示
    输入: (Batch, N, D)  D=3 for (x, y, t)
    输出: (Batch, N, Hidden)
    """
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        
        # 位置编码 (可选，增强时序信息)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, hidden_dim) * 0.02)
        
        # MLP 编码器
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.hidden_dim = hidden_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 轨迹点 [B, N, D]
        Returns:
            embeddings: 点嵌入 [B, N, H]
        """
        B, N, D = x.shape
        
        # [Memory Optimization] Chunked Processing
        # 将输入展平为 [B*N, D] 并分块处理，避免一次性分配巨大的中间激活张量
        x_flat = x.view(-1, D)
        total_points = x_flat.size(0)
        chunk_size = 4096  # 每次处理 4096 个点
        
        embeddings_list = []
        for i in range(0, total_points, chunk_size):
            # 获取当前块
            chunk = x_flat[i : i + chunk_size]
            
            # 编码 (自动复用 self.encoder 的权重)
            # 使用 checkpointing 进一步节省内存 (可选)
            # if self.training:
            #     chunk_emb = torch.utils.checkpoint.checkpoint(self.encoder, chunk)
            # else:
            chunk_emb = self.encoder(chunk)
            
            embeddings_list.append(chunk_emb)
        
        # 合并所有块
        h_flat = torch.cat(embeddings_list, dim=0)
        h = h_flat.view(B, N, -1)
        
        # 添加位置编码 (仅在前 N 个位置)
        if N <= self.pos_embedding.size(1):
            h = h + self.pos_embedding[:, :N, :]
        else:
            # 如果轨迹超长，循环使用位置编码或截断
            # 这里简单地重复位置编码以覆盖由 N 定义的长度
            pos_emb = self.pos_embedding[:, : self.pos_embedding.size(1), :]
            # 这里的重复逻辑可能需要根据实际需求调整，简单起见我们只加能加的部分
            # 或者扩展 pos_embedding
            pass 

        return h



class ForwardPolicy(nn.Module):
    """
    GFlowNet Forward Policy: P_F(a | s)
    
    输出 logits 用于 softmax 采样（非 ε-greedy）
    支持多条生成路径
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 点级别策略网络
        self.policy_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 终止动作头
        self.stop_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始偏置：鼓励早期停止以实现高压缩率
        # 模型需要学习添加必要的点
        # 初始偏置：鼓励早期停止以实现高压缩率
        # 模型需要学习添加必要的点
        # [Modify] 降低偏置以鼓励探索 (原为 3.0 -> 0.0 -> -2.0)
        nn.init.constant_(self.stop_mlp[-1].bias, 0.0)
        
    def forward(
        self, 
        traj_embeddings: torch.Tensor,
        mask: torch.Tensor,
        valid_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算 Forward Policy logits
        
        Args:
            traj_embeddings: 轨迹点嵌入 [B, N, H]
            mask: 当前选择掩码 [B, N], 1=已选中, 0=未选中
            valid_lens: 有效长度 [B]
            
        Returns:
            logits: 动作 logits [B, N+1]
                    前 N 个对应选择每个点
                    最后一个对应终止
        """
        B, N, H = traj_embeddings.shape
        
        # 全局上下文：已选点的平均嵌入
        mask_expanded = mask.unsqueeze(-1)  # [B, N, 1]
        selected_sum = (traj_embeddings * mask_expanded).sum(dim=1)  # [B, H]
        selected_count = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
        global_context = selected_sum / selected_count  # [B, H]
        
        # 拼接全局上下文到每个点
        global_rep = global_context.unsqueeze(1).expand(-1, N, -1)  # [B, N, H]
        combined = torch.cat([traj_embeddings, global_rep], dim=-1)  # [B, N, 2H]
        
        # 点级别 logits
        point_logits = self.policy_mlp(combined).squeeze(-1)  # [B, N]
        
        # 已选中的点不能再选
        point_logits = point_logits.masked_fill(mask.bool(), float('-inf'))
        
        # 终止 logit
        stop_logit = self.stop_mlp(global_context)  # [B, 1]
        
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
        temperature: float = 1.0,
        epsilon: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从前向策略采样动作
        
        Args:
            temperature: 采样温度
            epsilon: 随机探索概率 (用于训练)
            
        Returns:
            actions: 采样的动作 [B]
            log_probs: log P_F(a | s) [B]
        """
        logits = self.forward(traj_embeddings, mask, valid_lens)
        B = logits.size(0)
        
        # 温度缩放
        scaled_logits = logits / temperature
        
        # 处理无效 logits
        scaled_logits = torch.where(
            torch.isfinite(scaled_logits),
            scaled_logits,
            torch.full_like(scaled_logits, -100.0)
        )
        scaled_logits = torch.clamp(scaled_logits, min=-100.0, max=100.0)
        
        # Categorical 采样
        dist = torch.distributions.Categorical(logits=scaled_logits)
        actions = dist.sample()
        
        # ε-探索 (可选)
        if epsilon > 0:
            rand_mask = torch.rand(B, device=logits.device) < epsilon
            if rand_mask.any():
                # 均匀随机选择有效动作
                uniform_logits = torch.where(
                    torch.isfinite(logits),
                    torch.zeros_like(logits),
                    torch.full_like(logits, -100.0)
                )
                rand_dist = torch.distributions.Categorical(logits=uniform_logits)
                rand_actions = rand_dist.sample()
                actions[rand_mask] = rand_actions[rand_mask]
        
        log_probs = dist.log_prob(actions)
        
        return actions, log_probs
    
    def log_prob(
        self,
        traj_embeddings: torch.Tensor,
        mask: torch.Tensor,
        actions: torch.Tensor,
        valid_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """计算指定动作的 log 概率"""
        logits = self.forward(traj_embeddings, mask, valid_lens)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions)


class HierarchicalGFlowNet(nn.Module):
    """
    层次化 GFlowNet 模型
    
    包含：
    - 轨迹编码器
    - Forward Policy (用于生成压缩轨迹)
    - Backward Policy (用于 TLM 训练)
    - 多头策略网络 (用于多智能体训练)
    - 可选的层次策略 (Region-Level)
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_agents: int = 1,
        max_len: int = 2000
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        self.max_len = max_len
        
        # 轨迹编码器
        self.traj_encoder = TrajectoryEncoder(input_dim, hidden_dim, num_layers)
        
        # ============================================
        # 常数状态编码器 (O(1) 状态 -> 隐藏表示)
        # ============================================
        self.const_state_encoder = ConstantStateEncoder(
            state_dim=ConstantSizeState.dim() if ConstantSizeState else 18,
            hidden_dim=hidden_dim,
            num_layers=2
        )
        
        # 基于常数状态的策略 (替代 mask-based 策略)
        # 输入: 点嵌入 + 常数状态嵌入
        self.const_policy_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Forward Policy (Flat) - 保留用于兼容性
        self.policy_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Stop 动作头
        self.stop_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        # [Modify] 降低偏置 (原为 1.0 -> 0.0 -> -2.0)
        nn.init.constant_(self.stop_mlp[-1].bias, 0.0)
        
        # ============================================
        # State Flow Estimator (for SubTB)
        # 输入: State Embedding (from ConstantStateEncoder)
        # 输出: Scalar Log Flow log F(s)
        # ============================================
        self.log_flow_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1)
        )

        
        # ============================================
        # 多头策略网络 (用于多智能体训练)
        # ============================================
        if num_agents > 1:
            self.multi_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1)
                )
                for _ in range(num_agents)
            ])
            self.multi_stop_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1)
                )
                for _ in range(num_agents)
            ])
            # 初始化偏置
            for head in self.multi_stop_heads:
                nn.init.constant_(head[-1].bias, -2.0)
        else:
            self.multi_heads = None
            self.multi_stop_heads = None
        
        # Backward Policy
        self.back_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # 可选：层次策略 (Region-Level)
        # 粗粒度策略：选择 BallTree 节点
        self.coarse_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 6, hidden_dim),  # +6 for RegionState
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 细粒度策略：在节点内选点
        self.fine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 9, hidden_dim),  # +9 for PointState
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward_policy_multi(
        self,
        traj_embeddings: torch.Tensor,
        mask: torch.Tensor,
        agent_id: int,
        valid_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        多头 Forward Policy: P_F(a | s, agent_id)
        
        Args:
            traj_embeddings: [B, N, H] 轨迹点嵌入
            mask: [B, N], 1=已选中
            agent_id: 智能体 ID (0 ~ num_agents-1)
            valid_lens: [B] 有效长度
            
        Returns:
            logits: [B, N+1] 动作 logits
        """
        if self.multi_heads is None or agent_id >= len(self.multi_heads):
            # 回退到单头策略
            return self.forward_policy(traj_embeddings, mask, valid_lens)
        
        B, N, H = traj_embeddings.shape
        
        # 全局上下文
        mask_expanded = mask.unsqueeze(-1)
        selected_sum = (traj_embeddings * mask_expanded).sum(dim=1)
        selected_count = mask_expanded.sum(dim=1).clamp(min=1)
        global_context = selected_sum / selected_count
        
        # 拼接
        global_rep = global_context.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([traj_embeddings, global_rep], dim=-1)
        
        # 使用对应智能体的头
        point_logits = self.multi_heads[agent_id](combined).squeeze(-1)
        point_logits = point_logits.masked_fill(mask.bool(), float('-inf'))
        
        # Stop logit
        stop_logit = self.multi_stop_heads[agent_id](global_context)
        
        logits = torch.cat([point_logits, stop_logit], dim=1)
        
        # Padding
        if valid_lens is not None:
            pad_mask = torch.arange(N, device=logits.device).unsqueeze(0) >= valid_lens.unsqueeze(1)
            point_logits_padded = logits[:, :-1].masked_fill(pad_mask, float('-inf'))
            logits = torch.cat([point_logits_padded, logits[:, -1:]], dim=1)
        
        return logits

    
    def forward_policy_const(
        self,
        traj_embeddings: torch.Tensor,
        const_state_tensor: torch.Tensor,
        valid_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        基于常数状态的 Forward Policy: P_F(a | s)
        
        使用 O(1) 的常数状态替代 O(n) 的 mask
        
        Args:
            traj_embeddings: [B, N, H] 轨迹点嵌入
            const_state_tensor: [B, 18] 常数状态向量
            valid_lens: [B] 有效长度
            
        Returns:
            logits: [B, N+1] 动作 logits
        """
        B, N, H = traj_embeddings.shape
        
        # 编码常数状态
        state_hidden = self.const_state_encoder(const_state_tensor)  # [B, H]
        
        # 拼接常数状态到每个点
        state_rep = state_hidden.unsqueeze(1).expand(-1, N, -1)  # [B, N, H]
        combined = torch.cat([traj_embeddings, state_rep], dim=-1)  # [B, N, 2H]
        
        # 点 logits
        point_logits = self.const_policy_mlp(combined).squeeze(-1)  # [B, N]
        
        # Stop logit
        stop_logit = self.stop_mlp(state_hidden)  # [B, 1]
        
        logits = torch.cat([point_logits, stop_logit], dim=1)  # [B, N+1]
        
        # Padding
        if valid_lens is not None:
            pad_mask = torch.arange(N, device=logits.device).unsqueeze(0) >= valid_lens.unsqueeze(1)
            logits[:, :-1] = logits[:, :-1].masked_fill(pad_mask, float('-inf'))
        
        return logits

    def forward_policy_const_candidates(
        self,
        traj_embeddings: torch.Tensor,
        const_state_tensor: torch.Tensor,
        candidate_indices: torch.Tensor,
        valid_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward policy on a candidate pool.

        This re-parameterizes the action space from all N points to K candidates
        (plus stop), reducing per-step policy computation to O(K).

        Args:
            traj_embeddings: [B, N, H]
            const_state_tensor: [B, 18]
            candidate_indices: [B, K] int64 global point indices, -1 means invalid slot
            valid_lens: [B]
        Returns:
            logits: [B, K+1]
        """
        B, N, H = traj_embeddings.shape
        if candidate_indices.dim() != 2:
            raise ValueError("candidate_indices must be [B, K]")
        if candidate_indices.size(0) != B:
            raise ValueError("candidate_indices batch size mismatch")

        K = candidate_indices.size(1)
        state_hidden = self.const_state_encoder(const_state_tensor)  # [B, H]

        # Gather candidate embeddings safely (clamp invalid slots to 0 first).
        safe_indices = candidate_indices.clamp(min=0, max=max(0, N - 1))
        gather_idx = safe_indices.unsqueeze(-1).expand(-1, -1, H)  # [B, K, H]
        cand_emb = torch.gather(traj_embeddings, dim=1, index=gather_idx)

        state_rep = state_hidden.unsqueeze(1).expand(-1, K, -1)
        combined = torch.cat([cand_emb, state_rep], dim=-1)  # [B, K, 2H]
        point_logits = self.const_policy_mlp(combined).squeeze(-1)  # [B, K]

        valid_mask = candidate_indices >= 0
        if valid_lens is not None:
            valid_mask = valid_mask & (candidate_indices < valid_lens.unsqueeze(1))
        point_logits = point_logits.masked_fill(~valid_mask, float("-inf"))

        stop_logit = self.stop_mlp(state_hidden)  # [B, 1]
        logits = torch.cat([point_logits, stop_logit], dim=1)  # [B, K+1]
        return logits
    
    def forward_flow(
        self,
        const_state_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate Log Flow F(s)
        
        Args:
            const_state_tensor: [B, 18]
        Returns:
            log_flow: [B, 1]
        """
        state_hidden = self.const_state_encoder(const_state_tensor)
        return self.log_flow_mlp(state_hidden)

    
    def forward_policy(
        self,
        traj_embeddings: torch.Tensor,
        mask: torch.Tensor,
        valid_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward Policy: P_F(a | s)
        
        Args:
            traj_embeddings: [B, N, H]
            mask: [B, N], 1=已选中
            valid_lens: [B]
            
        Returns:
            logits: [B, N+1]
        """
        B, N, H = traj_embeddings.shape
        
        # 全局上下文
        mask_expanded = mask.unsqueeze(-1)
        selected_sum = (traj_embeddings * mask_expanded).sum(dim=1)
        selected_count = mask_expanded.sum(dim=1).clamp(min=1)
        global_context = selected_sum / selected_count
        
        # 拼接
        global_rep = global_context.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([traj_embeddings, global_rep], dim=-1)
        
        # 点 logits
        point_logits = self.policy_mlp(combined).squeeze(-1)
        point_logits = point_logits.masked_fill(mask.bool(), float('-inf'))
        
        # Stop logit
        stop_logit = self.stop_mlp(global_context)
        
        logits = torch.cat([point_logits, stop_logit], dim=1)
        
        # Padding (non-inplace)
        if valid_lens is not None:
            pad_mask = torch.arange(N, device=logits.device).unsqueeze(0) >= valid_lens.unsqueeze(1)
            point_logits_padded = logits[:, :-1].masked_fill(pad_mask, float('-inf'))
            logits = torch.cat([point_logits_padded, logits[:, -1:]], dim=1)
        
        return logits
    
    def forward_backward(
        self,
        traj_embeddings: torch.Tensor,
        mask: torch.Tensor,
        valid_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Backward Policy: P_B(parent | child)
        选择移除哪个点以返回父状态
        
        Returns:
            logits: [B, N+1]
        """
        B, N, H = traj_embeddings.shape
        
        # 全局上下文
        mask_expanded = mask.unsqueeze(-1)
        selected_sum = (traj_embeddings * mask_expanded).sum(dim=1)
        selected_count = mask_expanded.sum(dim=1).clamp(min=1)
        global_context = selected_sum / selected_count
        
        # 拼接
        global_rep = global_context.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([traj_embeddings, global_rep], dim=-1)
        
        # 移除点 logits
        logits = self.back_mlp(combined).squeeze(-1)
        
        # 数值稳定性处理 (non-inplace)
        logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
        logits = torch.clamp(logits, min=-100.0, max=100.0)
        
        # 只有已选点可移除 (non-inplace)
        logits = logits.masked_fill(~mask.bool(), -100.0)
        
        # 首尾点不能移除 (使用 masked_fill 避免 inplace)
        first_mask = torch.zeros(B, N, device=logits.device, dtype=torch.bool)
        first_mask[:, 0] = True
        logits = logits.masked_fill(first_mask, -100.0)
        
        if N > 1:
            last_mask = torch.zeros(B, N, device=logits.device, dtype=torch.bool)
            last_mask[:, -1] = True
            logits = logits.masked_fill(last_mask, -100.0)
        
        # Stop 逆动作
        stop_back_logit = self.stop_mlp(global_context)
        stop_back_logit = torch.clamp(stop_back_logit, min=-100.0, max=100.0)
        
        logits = torch.cat([logits, stop_back_logit], dim=1)
        
        # Padding (non-inplace)
        if valid_lens is not None:
            pad_mask = torch.arange(N, device=logits.device).unsqueeze(0) >= valid_lens.unsqueeze(1)
            point_logits = logits[:, :-1].masked_fill(pad_mask, -100.0)
            logits = torch.cat([point_logits, logits[:, -1:]], dim=1)
        
        return logits
    
    def forward_coarse(
        self,
        traj_embeddings: torch.Tensor,
        region_features: torch.Tensor,
        global_context: torch.Tensor,
        valid_regions: torch.Tensor
    ) -> torch.Tensor:
        """
        粗粒度 (Region-Level) Forward Policy
        
        Args:
            region_features: [B, NumRegions, 6] (RegionState)
            valid_regions: [B, NumRegions] mask
        """
        B, R, _ = region_features.shape
        
        # 聚合轨迹嵌入到区域级别
        global_rep = global_context.unsqueeze(1).expand(-1, R, -1)
        combined = torch.cat([global_rep, region_features], dim=-1)
        
        logits = self.coarse_mlp(combined).squeeze(-1)
        logits = logits.masked_fill(~valid_regions.bool(), float('-inf'))
        
        return logits
    
    def forward_fine(
        self,
        traj_embeddings: torch.Tensor,
        point_features: torch.Tensor,
        global_context: torch.Tensor,
        region_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        细粒度 (Point-Level) Forward Policy (在选定区域内)
        
        Args:
            point_features: [B, N, 9] (PointState)
            region_mask: [B, N] 当前区域内的点
        """
        B, N, _ = point_features.shape
        
        global_rep = global_context.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([traj_embeddings, global_rep, point_features], dim=-1)
        
        logits = self.fine_mlp(combined).squeeze(-1)
        logits = logits.masked_fill(~region_mask.bool(), float('-inf'))
        
        return logits
