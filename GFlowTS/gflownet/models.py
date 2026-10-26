import torch
import torch.nn as nn
from typing import Optional

try:
    from .utils.query_sketch import ConstantSizeState
except ImportError:
    ConstantSizeState = None

class ConstantStateEncoder(nn.Module):

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

        return self.encoder(state_tensor)

class TrajectoryEncoder(nn.Module):

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, hidden_dim) * 0.02)

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

        B, N, D = x.shape

        x_flat = x.view(-1, D)
        total_points = x_flat.size(0)
        chunk_size = 4096

        embeddings_list = []
        for i in range(0, total_points, chunk_size):

            chunk = x_flat[i : i + chunk_size]

            chunk_emb = self.encoder(chunk)

            embeddings_list.append(chunk_emb)

        h_flat = torch.cat(embeddings_list, dim=0)
        h = h_flat.view(B, N, -1)

        if N <= self.pos_embedding.size(1):
            h = h + self.pos_embedding[:, :N, :]
        else:

            pos_emb = self.pos_embedding[:, : self.pos_embedding.size(1), :]

            pass

        return h

class HierarchicalGFlowNet(nn.Module):

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 2000
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.traj_encoder = TrajectoryEncoder(input_dim, hidden_dim, num_layers)

        self.const_state_encoder = ConstantStateEncoder(
            state_dim=ConstantSizeState.dim() if ConstantSizeState else 18,
            hidden_dim=hidden_dim,
            num_layers=2
        )

        self.const_policy_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.policy_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.stop_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        nn.init.constant_(self.stop_mlp[-1].bias, 0.0)

        self.log_flow_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1)
        )

        self.back_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.coarse_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.fine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 9, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward_policy_const(
        self,
        traj_embeddings: torch.Tensor,
        const_state_tensor: torch.Tensor,
        valid_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        B, N, H = traj_embeddings.shape

        state_hidden = self.const_state_encoder(const_state_tensor)

        state_rep = state_hidden.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([traj_embeddings, state_rep], dim=-1)

        point_logits = self.const_policy_mlp(combined).squeeze(-1)

        stop_logit = self.stop_mlp(state_hidden)

        logits = torch.cat([point_logits, stop_logit], dim=1)

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

        B, N, H = traj_embeddings.shape
        if candidate_indices.dim() != 2:
            raise ValueError("candidate_indices must be [B, K]")
        if candidate_indices.size(0) != B:
            raise ValueError("candidate_indices batch size mismatch")

        K = candidate_indices.size(1)
        state_hidden = self.const_state_encoder(const_state_tensor)

        safe_indices = candidate_indices.clamp(min=0, max=max(0, N - 1))
        gather_idx = safe_indices.unsqueeze(-1).expand(-1, -1, H)
        cand_emb = torch.gather(traj_embeddings, dim=1, index=gather_idx)

        state_rep = state_hidden.unsqueeze(1).expand(-1, K, -1)
        combined = torch.cat([cand_emb, state_rep], dim=-1)
        point_logits = self.const_policy_mlp(combined).squeeze(-1)

        valid_mask = candidate_indices >= 0
        if valid_lens is not None:
            valid_mask = valid_mask & (candidate_indices < valid_lens.unsqueeze(1))
        point_logits = point_logits.masked_fill(~valid_mask, float("-inf"))

        stop_logit = self.stop_mlp(state_hidden)
        logits = torch.cat([point_logits, stop_logit], dim=1)
        return logits

    def forward_flow(
        self,
        const_state_tensor: torch.Tensor
    ) -> torch.Tensor:

        state_hidden = self.const_state_encoder(const_state_tensor)
        return self.log_flow_mlp(state_hidden)

    def forward_backward(
        self,
        traj_embeddings: torch.Tensor,
        mask: torch.Tensor,
        valid_lens: Optional[torch.Tensor] = None,
        keep_start: bool = True,
        keep_end: bool = True,
    ) -> torch.Tensor:

        B, N, H = traj_embeddings.shape

        mask_expanded = mask.unsqueeze(-1)
        selected_sum = (traj_embeddings * mask_expanded).sum(dim=1)
        selected_count = mask_expanded.sum(dim=1).clamp(min=1)
        global_context = selected_sum / selected_count

        global_rep = global_context.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([traj_embeddings, global_rep], dim=-1)

        logits = self.back_mlp(combined).squeeze(-1)

        logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
        logits = torch.clamp(logits, min=-100.0, max=100.0)

        logits = logits.masked_fill(~mask.bool(), -100.0)

        if keep_start:
            first_mask = torch.zeros(B, N, device=logits.device, dtype=torch.bool)
            first_mask[:, 0] = True
            logits = logits.masked_fill(first_mask, -100.0)

        if keep_end and N > 1:
            last_mask = torch.zeros(B, N, device=logits.device, dtype=torch.bool)
            last_mask[:, -1] = True
            logits = logits.masked_fill(last_mask, -100.0)

        stop_back_logit = self.stop_mlp(global_context)
        stop_back_logit = torch.clamp(stop_back_logit, min=-100.0, max=100.0)

        logits = torch.cat([logits, stop_back_logit], dim=1)

        if valid_lens is not None:
            pad_mask = torch.arange(N, device=logits.device).unsqueeze(0) >= valid_lens.unsqueeze(1)
            point_logits = logits[:, :-1].masked_fill(pad_mask, -100.0)
            logits = torch.cat([point_logits, logits[:, -1:]], dim=1)

        return logits
