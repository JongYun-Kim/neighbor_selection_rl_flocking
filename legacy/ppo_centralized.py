# Centralized observation model for neighbor selection
# Based on legacy LazyFlockTorchModel (MJActorTest/MJCriticTest) architecture
# Uses single-pass all-to-all attention instead of per-agent local_forward

import copy
import numpy as np
from typing import Dict, List

import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType


class NeighborSelectionPPORLlibCentralized(TorchModelV2, nn.Module):
    """
    Centralized observation model for neighbor selection.
    
    Key differences from ego-centric model:
    - Input obs shape: (batch, num_agents_max, obs_dim) instead of (batch, N, N, obs_dim)
    - Single-pass forward: encodes all agents at once, then uses self-embedding for per-agent queries
    - Agent identity preserved via: query[i] = [enc[i] || global_context] @ Wq
    
    Output action format is identical to ego-centric model: (batch, N*N*2) logits
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        
        # Get model config
        if model_config is not None:
            cfg = model_config["custom_model_config"]
            self.share_layers = cfg.get("share_layers", False)
            d_subobs = cfg.get("d_subobs", 4)
            d_embed = cfg.get("d_embed_input", 128)
            n_layers_encoder = cfg.get("n_layers_encoder", 3)
            n_heads = cfg.get("num_heads", 8)
            d_ff = cfg.get("d_ff", 512)
            dr_rate = cfg.get("dr_rate", 0.0)
            norm_eps = cfg.get("norm_eps", 1e-5)
            self.scale_factor = cfg.get("scale_factor", 2e-3)  # Legacy default
        else:
            raise ValueError("model_config must be specified")
        
        # Verify action space
        action_size = action_space.shape[0]  # num_agents_max
        assert num_outputs == 2 * (action_size ** 2), \
            f"num_outputs != 2 * (action_size^2); num_output = {num_outputs}, action_size = {action_size}"
        
        self.num_agents_max = action_size
        self.d_embed = d_embed
        
        # ============================================================
        # Actor Network (following Legacy MJActorTest structure)
        # ============================================================
        
        # Embedding layer
        self.flock_embedding = nn.Linear(d_subobs, d_embed)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embed,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dr_rate,
            layer_norm_eps=norm_eps,
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers_encoder,
            enable_nested_tensor=False,
        )
        
        # Pointer attention projections
        # Query: from [enc || context] (d_embed * 2) -> d_embed
        # Key: from enc (d_embed) -> d_embed
        self.Wq = nn.Parameter(torch.randn(d_embed * 2, d_embed) * 0.02)
        self.Wk = nn.Parameter(torch.randn(d_embed, d_embed) * 0.02)
        
        # ============================================================
        # Critic Network
        # ============================================================
        if not self.share_layers:
            # Separate encoder for critic
            self.critic_flock_embedding = nn.Linear(d_subobs, d_embed)
            critic_encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_embed,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dr_rate,
                layer_norm_eps=norm_eps,
                norm_first=True,
                batch_first=True,
            )
            self.critic_encoder = nn.TransformerEncoder(
                critic_encoder_layer,
                num_layers=n_layers_encoder,
                enable_nested_tensor=False,
            )
        
        # Value head
        self.value_branch = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, 1),
        )
        
        # Cache for value function
        self._context_for_value = None
    
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        
        obs_dict = input_dict["obs"]
        
        # Get observations and masks
        # Centralized obs: (batch, num_agents_max, obs_dim)
        agents_info = obs_dict["local_agent_infos"]
        neighbor_masks = obs_dict["neighbor_masks"]  # (batch, num_agents_max, num_agents_max)
        padding_mask = obs_dict["padding_mask"]  # (batch, num_agents_max)
        
        batch_size = agents_info.shape[0]
        num_agents_max = agents_info.shape[1]
        
        # ============================================================
        # Actor Forward Pass (Legacy MJActorTest style)
        # ============================================================
        
        # 1. Embedding
        flock_embed = self.flock_embedding(agents_info)  # (batch, N, d_embed)
        
        # 2. Create padding mask for encoder (True = ignore, False = attend)
        # padding_mask: 1 = real agent, 0 = padding
        # TransformerEncoder expects: True = ignore position
        encoder_padding_mask = ~padding_mask.bool()  # (batch, N) - True where padding
        
        # 3. Encode with padding mask only (global information sharing)
        enc = self.encoder(flock_embed, src_key_padding_mask=encoder_padding_mask)  # (batch, N, d_embed)
        
        # 4. Context: masked mean of encoded embeddings
        # Exclude padding from mean computation
        mask_for_mean = padding_mask.unsqueeze(-1).float()  # (batch, N, 1)
        enc_masked = enc * mask_for_mean
        context = enc_masked.sum(dim=1) / mask_for_mean.sum(dim=1).clamp(min=1)  # (batch, d_embed)
        
        # 5. Agent context: concatenate each agent's encoding with global context
        # This is the key to preserving agent identity in centralized obs
        flock_context = context.unsqueeze(1).expand(batch_size, num_agents_max, -1)  # (batch, N, d_embed)
        agent_context = torch.cat((enc, flock_context), dim=-1)  # (batch, N, d_embed*2)
        
        # 6. Pointer attention: all-to-all
        queries = torch.matmul(agent_context, self.Wq)  # (batch, N, d_embed)
        keys = torch.matmul(enc, self.Wk)  # (batch, N, d_embed)
        D = queries.shape[-1]
        
        att_scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(D)  # (batch, N, N)
        
        # 7. Apply neighbor_masks: mask out positions where neighbor_masks == 0
        # neighbor_masks: 1 = connected, 0 = not connected
        att_scores = att_scores.masked_fill(neighbor_masks == 0, -1e9)
        
        # 8. Apply padding mask to both rows and columns
        # Rows: padded agents should output -1e9 (will be ignored)
        # Columns: can't select padded agents as neighbors
        padding_row_mask = ~padding_mask.bool()  # (batch, N) - True where padding
        padding_col_mask = padding_row_mask.unsqueeze(1)  # (batch, 1, N)
        att_scores = att_scores.masked_fill(padding_col_mask, -1e9)
        
        # 9. Convert to logits (same as ego-centric model)
        logits = self.attention_scores_to_logits(att_scores)
        
        # ============================================================
        # Critic: cache context for value function
        # ============================================================
        if self.share_layers:
            self._context_for_value = context  # (batch, d_embed)
        else:
            # Separate critic forward
            critic_embed = self.critic_flock_embedding(agents_info)
            critic_enc = self.critic_encoder(critic_embed, src_key_padding_mask=encoder_padding_mask)
            critic_enc_masked = critic_enc * mask_for_mean
            self._context_for_value = critic_enc_masked.sum(dim=1) / mask_for_mean.sum(dim=1).clamp(min=1)
        
        return logits, state
    
    def attention_scores_to_logits(self, attention_scores: TensorType) -> TensorType:
        """
        Maps attention scores to logits for binary action distribution.
        Follows the same convention as ego-centric model: [z_neg, z] ordering.
        
        :param attention_scores: (batch_size, num_agents_max, num_agents_max)
        :return: logits (batch_size, num_agents_max * num_agents_max * 2)
        """
        batch_size = attention_scores.shape[0]
        num_agents_max = attention_scores.shape[1]
        
        # Scale attention scores
        scale_factor = self.scale_factor
        attention_scores = attention_scores * scale_factor
        
        # Self-loops: force diagonal to very high positive value (always select self)
        # Using addition like ego-centric model (not subtraction like legacy)
        large_val = 1e9
        attention_scores = attention_scores + torch.diag_embed(
            attention_scores.new_full((num_agents_max,), large_val)
        )
        
        # Get negated attention scores
        negated_attention_scores = -attention_scores
        
        # Expand and concatenate: [z_neg, z] (same as ego-centric model)
        z_expanded = attention_scores.unsqueeze(-1)  # (batch, N, N, 1)
        z_neg_expanded = negated_attention_scores.unsqueeze(-1)  # (batch, N, N, 1)
        z_concatenated = torch.cat((z_neg_expanded, z_expanded), dim=-1)  # (batch, N, N, 2)
        
        # Reshape to 2D
        logits = z_concatenated.reshape(batch_size, num_agents_max * num_agents_max * 2)
        
        return logits
    
    def value_function(self) -> TensorType:
        """
        Returns the value function output.
        Uses cached context from forward pass.
        """
        assert self._context_for_value is not None, "Must call forward() before value_function()"
        value = self.value_branch(self._context_for_value).squeeze(-1)  # (batch,)
        return value
