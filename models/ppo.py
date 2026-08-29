# Everything is copy
import copy
import os
# Please let me get out of ray rllib
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
#
# From envs
from typing import List, Union, Dict
#
# Pytorch
import torch
import torch.nn as nn
#
# Custom modules
from models.modules.token_embedding import LinearEmbedding
from models.modules.multi_head_attention_layer import MultiHeadAttentionLayer
from models.modules.position_wise_feed_forward_layer import PositionWiseFeedForwardLayer
from models.modules.encoder_block import EncoderBlock
from models.modules.decoder_block import CustomDecoderBlock as DecoderBlock
from models.modules.encoder import Encoder
from models.modules.decoder import Decoder, DecoderPlaceholder
from models.modules.pointer_net import RawAttentionScoreGenerator, RawAttentionScoreGeneratorPlaceholder


class NeighborSelectionPPORLlib(TorchModelV2, nn.Module):
    # The Dynamic-k NN subclass opts into padding-aware encoder masks while
    # the legacy binary model retains its original behavior and state dict.
    respect_padding_mask = False

    @staticmethod
    def required_num_outputs(action_size: int) -> int:
        return 2 * (action_size ** 2)

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)  # Initialize nn.Module first
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # Get model config
        if model_config is not None:
            cfg = model_config["custom_model_config"]
            share_layers = cfg["share_layers"] if "share_layers" in cfg else True
            d_subobs = cfg["d_subobs"] if "d_subobs" in cfg else ValueError("d_subobs must be specified")
            d_embed_input = cfg["d_embed_input"] if "d_embed_input" in cfg else 128
            d_embed_context = cfg["d_embed_context"] if "d_embed_context" in cfg else 128
            d_model = cfg["d_model"] if "d_model" in cfg else 128
            d_model_decoder = cfg["d_model_decoder"] if "d_model_decoder" in cfg else 128
            n_layers_encoder = cfg["n_layers_encoder"] if "n_layers_encoder" in cfg else 3
            n_layers_decoder = cfg["n_layers_decoder"] if "n_layers_decoder" in cfg else 1
            h = cfg["num_heads"] if "num_heads" in cfg else 8
            d_ff = cfg["d_ff"] if "d_ff" in cfg else 512
            d_ff_decoder = cfg["d_ff_decoder"] if "d_ff_decoder" in cfg else 512
            dr_rate = cfg["dr_rate"] if "dr_rate" in cfg else 0
            norm_eps = cfg["norm_eps"] if "norm_eps" in cfg else 1e-5
            is_bias = cfg["is_bias"] if "is_bias" in cfg else True  # bias in MHA linear layers (W_q, W_k, W_v)
            use_residual_in_decoder = cfg["use_residual_in_decoder"] if "use_residual_in_decoder" in cfg else True
            use_FNN_in_decoder = cfg["use_FNN_in_decoder"] if "use_FNN_in_decoder" in cfg else True
            self.scale_factor = cfg["scale_factor"] if "scale_factor" in cfg else 1.0
            # Auxiliary task hyperparameters
            #   aux_type:
            #     "pair_embedding" (default) — for each pair (i, j), use encoder embedding of j as seen
            #                                  by i to predict j's flock-center-frame state.
            #     "context"                  — use agent i's pooled context vector h_c_N[i] to predict
            #                                  i's own flock-center-frame state.
            self.top_k = cfg["top_k"] if "top_k" in cfg else None
            self.hard_top_k = cfg.get("hard_top_k", False)
            self.distance_bias_scale = cfg["distance_bias_scale"] if "distance_bias_scale" in cfg else 0.0
            self.continuous_action = cfg.get("continuous_action", False)
            self.per_agent_credit = cfg.get("per_agent_credit", False)
            self.pa_credit_alpha = cfg.get("pa_credit_alpha", 1.0)
            self.pa_replace_ppo = cfg.get("pa_replace_ppo", False)
            self.dist_aux_coef = cfg.get("dist_aux_coef", 0.0)
            # --- Selection-head variants (study acs-c2-train) ---
            #   "legacy":    exact old behavior (top_k / hard_top_k / plain / continuous).
            #   "bernoulli": plain per-edge binary logits (same code path as legacy
            #                with top_k=None); named for config explicitness.
            #   "threshold": per-agent learned threshold tau_i over pair scores;
            #                s_ij = logit_scale*(scale_factor*att_ij - tau_i).
            self.selection_head = cfg.get("selection_head", "legacy")
            assert self.selection_head in ("legacy", "bernoulli", "threshold"), \
                f"selection_head must be legacy/bernoulli/threshold; got {self.selection_head!r}"
            self.logit_scale = cfg.get("logit_scale", 10.0)
            if self.selection_head != "legacy":
                assert not self.continuous_action and self.top_k is None and not self.hard_top_k, \
                    "bernoulli/threshold heads require continuous_action=False, top_k=None, hard_top_k=False"
            # dist_aux rework: configurable K (was hardcoded 10) + annealable
            # coefficient (mutated by callbacks on the policy's model/towers).
            self.dist_aux_k = cfg.get("dist_aux_k", 10)
            self.dist_aux_schedule = cfg.get("dist_aux_schedule", None)
            self.dist_aux_coef_current = self.dist_aux_coef
            # Global swarm-stats conditioning (obs key "global_stats", (4,))
            self.use_global_stats = cfg.get("use_global_stats", False)
            self._pretrained_weights_path = cfg.get("pretrained_weights_path", None)
            self.aux_enabled = cfg["aux_enabled"] if "aux_enabled" in cfg else False
            self.aux_loss_coef = cfg["aux_loss_coef"] if "aux_loss_coef" in cfg else 0.1
            self.aux_loss_coef_critic = cfg["aux_loss_coef_critic"] if "aux_loss_coef_critic" in cfg else 0.0
            self.aux_target_dim = cfg["aux_target_dim"] if "aux_target_dim" in cfg else 4
            self.aux_type = cfg["aux_type"] if "aux_type" in cfg else "pair_embedding"
            assert self.aux_type in ("pair_embedding", "context"), \
                f"aux_type must be 'pair_embedding' or 'context'; got {self.aux_type!r}"

            if use_residual_in_decoder != use_FNN_in_decoder:
                warning_text = "Warning: use_residual_in_decoder != use_FNN_in_decoder; may cause unexpected behavior"
                for i in range(7):
                    print(("%"*i) + warning_text + ("%"*i))
            if n_layers_decoder >= 2 and not use_residual_in_decoder:
                warning_text = "Warning: multiple decoder blocks often require residual connections"
                for i in range(7):
                    print(("%"*i) + warning_text + ("%"*i))
        else:
            raise ValueError("model_config must be specified")

        # 1. Define layers

        # 1-1. Module Level: Encoder
        # Need an embedding layer for the input; 2->128 in the case of Kool2019
        input_embed = LinearEmbedding(
            d_env=d_subobs,
            d_embed=d_embed_input,
        )
        mha_encoder = MultiHeadAttentionLayer(
            d_model=d_model,
            h=h,
            q_fc=nn.Linear(d_embed_input, d_model, is_bias),
            kv_fc=nn.Linear(d_embed_input, d_model, is_bias),
            out_fc=nn.Linear(d_model, d_embed_input, is_bias),
            dr_rate=dr_rate,
        )
        position_ff_encoder = PositionWiseFeedForwardLayer(
            fc1=nn.Linear(d_embed_input, d_ff),
            fc2=nn.Linear(d_ff, d_embed_input),
            dr_rate=dr_rate,
        )
        norm_encoder = nn.LayerNorm(d_embed_input, eps=norm_eps)
        # 1-2. Module Level: Decoder
        mha_decoder = MultiHeadAttentionLayer(
            d_model=d_model_decoder,
            h=h,
            q_fc=nn.Linear(d_embed_context, d_model_decoder, is_bias),
            kv_fc=nn.Linear(d_embed_input, d_model_decoder, is_bias),
            out_fc=nn.Linear(d_model_decoder, d_embed_context, is_bias),
            dr_rate=dr_rate,
        )
        position_ff_decoder = PositionWiseFeedForwardLayer(
            fc1=nn.Linear(d_embed_context, d_ff_decoder),
            fc2=nn.Linear(d_ff_decoder, d_embed_context),
            dr_rate=dr_rate,
        ) if use_FNN_in_decoder else None
        norm_decoder = nn.LayerNorm(d_embed_context, eps=norm_eps)

        # 1-3. Block Level
        encoder_block = EncoderBlock(
            self_attention=copy.deepcopy(mha_encoder),
            position_ff=copy.deepcopy(position_ff_encoder),
            norm=copy.deepcopy(norm_encoder),
            dr_rate=dr_rate,
        )
        decoder_block = DecoderBlock(
            self_attention=None,  # No (masked-)self-attention in the decoder_block in this case!
            cross_attention=copy.deepcopy(mha_decoder),
            position_ff=position_ff_decoder,  # No position-wise FFN in the decoder_block in most cases!
            norm=copy.deepcopy(norm_decoder),
            dr_rate=dr_rate,
            efficient=not use_residual_in_decoder,
        )

        # 1-4. Transformer Level (Encoder + Decoder + Generator)
        encoder = Encoder(
            encoder_block=encoder_block,
            n_layer=n_layers_encoder,
            norm=copy.deepcopy(norm_encoder),
        )
        decoder = Decoder(
            decoder_block=decoder_block,
            n_layer=n_layers_decoder,
            norm=copy.deepcopy(norm_decoder),
            # norm=nn.Identity(),
        )
        generator = RawAttentionScoreGenerator(
            d_model=d_model_decoder,
            q_fc=nn.Linear(d_embed_context, d_model_decoder, is_bias),
            k_fc=nn.Linear(d_embed_input, d_model_decoder, is_bias),
            dr_rate=dr_rate,
        )

        action_size = action_space.shape[0]  # num_agents_max?
        required_num_outputs = self.required_num_outputs(action_size)
        assert num_outputs == required_num_outputs, \
            f"num_outputs != {required_num_outputs}; num_output = {num_outputs}, action_size = {action_size}"

        # 2. Define policy network
        self.actor = NeighborSelectorTorch(
            src_embed=input_embed,
            encoder=encoder,
            decoder=decoder,
            generator=generator,
            d_embed_context=d_embed_context,
            global_stats_dim=4 if self.use_global_stats else 0,
            respect_padding_mask=self.respect_padding_mask,
        )

        # 2b. Per-agent threshold head (selection_head == "threshold" only).
        # Zero-init weight; bias from threshold_bias_init (default 0.0 -> tau=0,
        # p ~ 0.5 per edge at init). A NEGATIVE bias (e.g. -0.1 in scaled-att
        # units) starts the bar below the att distribution: sampling is dense
        # (p ~ 0.9) and the argmax readout starts near fully-connected — the
        # known-feasible end — so pruning is learned downward from success.
        if self.selection_head == "threshold":
            self.threshold_bias_init = cfg.get("threshold_bias_init", 0.0)
            self.threshold_head = nn.Linear(d_embed_context, 1)
            nn.init.zeros_(self.threshold_head.weight)
            nn.init.constant_(self.threshold_head.bias, self.threshold_bias_init)

        # 3. Define value network
        self.values = None
        self.share_layers = share_layers
        if not self.share_layers:
            self.critic = NeighborSelectorTorch(
                src_embed=copy.deepcopy(input_embed),
                encoder=copy.deepcopy(encoder),
                decoder=DecoderPlaceholder(),
                generator=RawAttentionScoreGeneratorPlaceholder(),
                d_embed_context=d_embed_context,
                respect_padding_mask=self.respect_padding_mask,
            )

        # 3b. Global-stats conditioning of the critic's pooled vector
        # (applies to both share_layers cases; actor-side conditioning lives
        # inside NeighborSelectorTorch via global_stats_dim).
        if self.use_global_stats:
            self.values_gs_proj = nn.Linear(d_embed_context + 4, d_embed_context)

        self.value_branch = nn.Sequential(
            nn.Linear(in_features=d_embed_context, out_features=d_embed_context),
            nn.ReLU(),
            nn.Linear(in_features=d_embed_context, out_features=1),  # state-value function
        )

        # 4. Define auxiliary branch
        #    - pair_embedding: aux_branch acts per (i, j) pair on the encoder embedding (d_embed_input).
        #                      Output dim is d_aux_target_dim; predicts j's flock-center state.
        #    - context: aux_branch acts per agent i on the pooled context vector (d_embed_context).
        #               Output dim is d_aux_target_dim; predicts i's own flock-center state.
        #    Both heads mirror the value_branch pattern (1 hidden, ReLU, input_dim == hidden_dim).
        self._num_agents_max = action_space.shape[0]
        if self.aux_enabled:
            if self.aux_type == "pair_embedding":
                aux_in = d_embed_input
            else:  # "context"
                aux_in = d_embed_context
            self.aux_branch = nn.Sequential(
                nn.Linear(in_features=aux_in, out_features=aux_in),
                nn.ReLU(),
                nn.Linear(in_features=aux_in, out_features=self.aux_target_dim),
            )
            # Critic gets its own aux_branch when share_layers=False (separate encoder).
            # When share_layers=True the shared encoder already gets aux gradient from actor side.
            if not share_layers and self.aux_loss_coef_critic > 0:
                self.aux_branch_critic = nn.Sequential(
                    nn.Linear(in_features=aux_in, out_features=aux_in),
                    nn.ReLU(),
                    nn.Linear(in_features=aux_in, out_features=self.aux_target_dim),
                )
        # Caches (set in forward(), read in custom_loss()):
        self._aux_features = None         # (B, N, d_ctx) for context  OR  (B, N, N, d_embed) for pair
        self._aux_features_critic = None  # same shape as above, from critic encoder
        self._aux_target = None           # (B, N, aux_target_dim) — same global descriptor for both modes
        self._aux_padding_mask = None     # (B, N), float
        self._aux_neighbor_masks = None   # (B, N, N), float
        self._last_aux_loss = None        # scalar for actor aux
        self._last_aux_loss_critic = None # scalar for critic aux

        if self.continuous_action:
            self.action_log_std = nn.Parameter(torch.tensor([-1.0]))

        if self._pretrained_weights_path and os.path.exists(self._pretrained_weights_path):
            state = torch.load(self._pretrained_weights_path, map_location="cpu")
            self.load_state_dict(state, strict=False)
            print(f"Loaded pretrained weights from {self._pretrained_weights_path}")
            self._pretrained_weights_path = None

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):

        obs_dict = input_dict["obs"]

        # att:                  (batch_size, num_agents_max, num_agents_max)
        # h_c_N:                (batch_size, 1, d_embed_context) - swarm-averaged context (for value branch)
        # per_agent_h_c_N:      (batch_size, num_agents_max, d_embed_context) - for context aux branch
        # per_pair_encoder_out: (batch_size, N_i, N_j, d_embed_input) - for pair_embedding aux branch
        att, h_c_N, per_agent_h_c_N, per_pair_encoder_out = self.actor(obs_dict)

        if self.distance_bias_scale > 0:
            local_obs = obs_dict["local_agent_infos"]  # (B, N, N, obs_dim)
            rel_pos = local_obs[:, :, :, :2]  # (B, N, N, 2)
            dist_sq = (rel_pos ** 2).sum(dim=-1)  # (B, N, N)
            N = dist_sq.shape[-1]
            K = min(self.top_k or 10, N - 1)
            diag_inf = torch.eye(N, device=dist_sq.device).unsqueeze(0) * 1e9
            dist_sq_nodiag = dist_sq + diag_inf
            ranks = dist_sq_nodiag.argsort(dim=-1).argsort(dim=-1).float()
            self._distance_bias = self.distance_bias_scale * (K - 0.5 - ranks)

        if self.selection_head == "threshold":
            # Per-agent selection bar from the agent's pooled context.
            self._tau = self.threshold_head(per_agent_h_c_N)  # (B, N, 1)

        x = self.attention_scores_to_logits(att)  # (batch_size, num_agents_max * num_agents_max * 2)
        self._cached_logits = x
        if self.dist_aux_coef > 0:
            self._cached_att = att  # (B, N, N) raw attention scores
            local_obs = obs_dict["local_agent_infos"]  # (B, N, N, obs_dim)
            self._cached_dist_sq = (local_obs[:, :, :, :2] ** 2).sum(dim=-1)  # (B, N, N)

        if self.share_layers:
            self.values = h_c_N.squeeze(1)                      # (batch_size, d_embed_context)
        else:
            critic_out = self.critic(obs_dict)
            self.values = critic_out[1].squeeze(1)              # (batch_size, d_embed_context)

        if self.use_global_stats:
            gs = obs_dict["global_stats"].to(self.values.dtype)  # (B, 4)
            self.values = self.values_gs_proj(torch.cat([self.values, gs], dim=-1))

        if self.aux_enabled:
            # Cache live (gradient-bearing) features for the active aux head. Same lifecycle as
            # self.values: PPO's loss() just ran forward() on the train minibatch, so these tensors
            # are the in-flight train batch with proper grad.
            if self.aux_type == "pair_embedding":
                self._aux_features = per_pair_encoder_out         # (B, N_i, N_j, d_embed_input)
            else:  # "context"
                self._aux_features = per_agent_h_c_N              # (B, N_max, d_embed_context)
            self._aux_target = obs_dict["global_agent_infos"].float()   # (B, N_max, aux_target_dim)
            self._aux_padding_mask = obs_dict["padding_mask"].float()   # (B, N_max)
            self._aux_neighbor_masks = obs_dict["neighbor_masks"].float()  # (B, N_max, N_max)
            # Critic encoder features for critic aux (only when separate encoders)
            if not self.share_layers and self.aux_loss_coef_critic > 0:
                if self.aux_type == "pair_embedding":
                    self._aux_features_critic = critic_out[3]     # (B, N_i, N_j, d_embed_input)
                else:
                    self._aux_features_critic = critic_out[2]     # (B, N_max, d_embed_context)

        return x, state

    def attention_scores_to_logits(self, attention_scores: TensorType) -> TensorType:
        """
        Maps attention scores to logits for the action distribution.
        Binary mode: (B, N*N*2) — pairs of [-score, +score] for Categorical.
        Continuous mode: (B, N*N*2) — [mean_logits, conc_logits] for Beta.
        """
        batch_size = attention_scores.shape[0]
        num_agents_max = attention_scores.shape[1]

        scale_factor = self.scale_factor
        attention_scores *= scale_factor

        if self.distance_bias_scale > 0 and hasattr(self, '_distance_bias'):
            attention_scores = attention_scores + self._distance_bias

        if self.selection_head == "threshold" and not self.continuous_action:
            # Per-agent threshold selection: an edge is selected when its
            # (scaled) attention score clears agent i's learned bar tau_i.
            # logit_scale sets decisiveness per unit margin (10 -> margin 0.1
            # gives p~0.88) WITHOUT the ±20 saturation of the old hard-top-K
            # path: gradients d(logp)/d(att) and d(logp)/d(tau) stay alive.
            s = self.logit_scale * (attention_scores - self._tau)  # (B, N, N)
            # Diagonal (self-loop) forced selected, same mechanism as the
            # plain binary path below; masked entries arrive at -1e8 already.
            s = s + torch.diag_embed(s.new_full((num_agents_max,), 1e9))
            z_concatenated = torch.cat((-s.unsqueeze(-1), s.unsqueeze(-1)), dim=-1)
            return z_concatenated.reshape(batch_size, num_agents_max * num_agents_max * 2)

        if self.hard_top_k and self.top_k is not None and not self.continuous_action:
            # HARD top-K binary selection. Each agent i selects EXACTLY top_k of its
            # highest-attention neighbors (off-diagonal) as edge=1, plus the self-loop.
            # A large constant BIG dominates the argmax so the binary decision is fixed by
            # the top-K mask, while the raw attention_scores term keeps a gradient path
            # to attention (and thus to dist_aux's ranking objective): d(+s)/d(att)=1.
            BIG = 20.0
            diag_mask = torch.eye(num_agents_max, device=attention_scores.device).bool()
            off_diag = attention_scores.masked_fill(diag_mask.unsqueeze(0), float('-inf'))
            k = min(self.top_k, num_agents_max - 1)
            _, topk_idx = off_diag.topk(k, dim=-1)
            topk_mask = torch.zeros_like(attention_scores, dtype=torch.bool)
            topk_mask.scatter_(2, topk_idx, True)
            topk_mask = topk_mask | diag_mask.unsqueeze(0)  # diagonal always selected
            # Selected: s = att + BIG (edge wins argmax); non-selected: s = att - BIG (no-edge wins)
            s = torch.where(topk_mask, attention_scores + BIG, attention_scores - BIG)
            z_concatenated = torch.cat((-s.unsqueeze(-1), s.unsqueeze(-1)), dim=-1)
            return z_concatenated.reshape(batch_size, num_agents_max * num_agents_max * 2)

        if self.top_k is not None and self.top_k < num_agents_max - 1:
            diag_mask = torch.eye(num_agents_max, device=attention_scores.device).bool()
            off_diag = attention_scores.masked_fill(diag_mask.unsqueeze(0), float('-inf'))
            _, topk_idx = off_diag.topk(self.top_k, dim=-1)
            topk_mask = torch.zeros_like(attention_scores, dtype=torch.bool)
            topk_mask.scatter_(2, topk_idx, True)
            topk_mask = topk_mask | diag_mask.unsqueeze(0)
            attention_scores = torch.where(topk_mask, attention_scores + 2.0,
                                           torch.ones_like(attention_scores) * -1e9)

        if self.continuous_action:
            attention_scores = attention_scores + torch.diag_embed(
                attention_scores.new_full((num_agents_max,), 10.0))
            mean_logits = attention_scores.reshape(batch_size, num_agents_max * num_agents_max)
            conc_logits = self.action_log_std.expand(batch_size, num_agents_max * num_agents_max)
            return torch.cat([mean_logits, conc_logits], dim=-1)
        else:
            large_val = 1e9
            attention_scores = attention_scores + torch.diag_embed(
                attention_scores.new_full((num_agents_max,), large_val))
            negated_attention_scores = -attention_scores
            z_expanded = attention_scores.unsqueeze(-1)
            z_neg_expanded = negated_attention_scores.unsqueeze(-1)
            z_concatenated = torch.cat((z_neg_expanded, z_expanded), dim=-1)
            logits = z_concatenated.reshape(batch_size, num_agents_max * num_agents_max * 2)
            return logits

    def value_function(self) -> TensorType:
        # assert self.values is not None, "self.values is None"
        # assert self.values.dim() == 2, "self.values.dim() != 2; NOT 2D"
        value = self.value_branch(self.values).squeeze(-1)  # (batch_size,)
        return value

    def _compute_aux_mse(self, branch, feats, target, pad, nm):
        """Shared MSE computation for a given aux branch and features."""
        T = self.aux_target_dim
        if self.aux_type == "pair_embedding":
            pred = branch(feats)                                      # (B, N_i, N_j, T)
            target_b = target.unsqueeze(1).expand_as(pred)            # (B, N_i, N_j, T)
            mask = (pad.unsqueeze(2) * pad.unsqueeze(1) * nm).unsqueeze(-1)
        else:  # "context"
            pred = branch(feats)                                      # (B, N, T)
            target_b = target                                         # (B, N, T)
            diag_nm = nm.diagonal(dim1=-2, dim2=-1)                   # (B, N)
            mask = (pad * diag_nm).unsqueeze(-1)                      # (B, N, 1)
        denom = (mask.sum() * T).clamp(min=1.0)
        return ((pred - target_b) ** 2 * mask).sum() / denom

    def _compute_per_agent_logp(self, logits, actions, padding_mask):
        """Compute per-agent log-probs for continuous action space.

        Returns (B, N) tensor where entry [b, i] = sum_j log p(a_{ij}).
        """
        import math
        N = self._num_agents_max
        mean_logits = logits[:, :N*N].reshape(-1, N, N)
        log_std = logits[:, N*N:].reshape(-1, N, N)
        std = torch.exp(log_std.clamp(-5.0, 2.0))
        actions_2d = actions.reshape(-1, N, N)

        x = actions_2d.clamp(1e-6, 1.0 - 1e-6)
        z = torch.log(x / (1.0 - x))
        normal_logp = (-0.5 * ((z - mean_logits) / std).pow(2)
                       - 0.5 * math.log(2.0 * math.pi)
                       - torch.log(std))
        correction = torch.log(x * (1.0 - x) + 1e-8)
        per_element_logp = normal_logp - correction  # (B, N, N)

        edge_mask = padding_mask.unsqueeze(2) * padding_mask.unsqueeze(1)
        return (per_element_logp * edge_mask).sum(-1)  # (B, N)

    def custom_loss(self, policy_loss, loss_inputs):
        """Per-agent credit assignment (additive) + auxiliary self-supervised regression.

        When per_agent_credit is enabled, adds a per-agent correction term to the
        standard PPO loss. The correction redistributes gradient among agents
        proportional to their per-agent reward deviation from mean. The standard
        PPO loss (with clipping, VF, entropy) is preserved for stability.

        RLlib v2.1.0 contract: policy_loss arrives as a list (PPO: length 1);
        return a list of the same length.
        """
        # --- Aux loss (always computed when enabled) ---
        total_aux = torch.tensor(0.0, device=policy_loss[0].device) if isinstance(policy_loss[0], torch.Tensor) else 0.0
        self._last_aux_loss = None
        self._last_aux_loss_critic = None
        self._last_pa_pg_loss = None
        self._last_dist_aux_loss = None

        # Guard on the cache CONTENT, not attribute existence: __init__ pre-sets
        # these to None, and subclasses that override forward() (e.g. the
        # dynamic-k pointer model) never populate them — aux must then no-op.
        if self.aux_enabled and getattr(self, '_aux_target', None) is not None:
            target = self._aux_target
            pad = self._aux_padding_mask
            nm = self._aux_neighbor_masks

            aux_actor = self._compute_aux_mse(self.aux_branch, self._aux_features, target, pad, nm)
            total_aux = self.aux_loss_coef * aux_actor
            self._last_aux_loss = aux_actor.detach()

            if (not self.share_layers and self.aux_loss_coef_critic > 0
                    and self._aux_features_critic is not None):
                aux_critic = self._compute_aux_mse(
                    self.aux_branch_critic, self._aux_features_critic, target, pad, nm)
                total_aux = total_aux + self.aux_loss_coef_critic * aux_critic
                self._last_aux_loss_critic = aux_critic.detach()

        # --- Distance supervision auxiliary loss ---
        if self.dist_aux_coef > 0 and hasattr(self, '_cached_att') and hasattr(self, '_aux_padding_mask'):
            att = self._cached_att  # (B, N, N)
            dist_sq = self._cached_dist_sq  # (B, N, N)
            pad = self._aux_padding_mask  # (B, N)
            edge_mask = pad.unsqueeze(2) * pad.unsqueeze(1)  # (B, N, N)
            N = att.shape[-1]
            diag = torch.eye(N, device=att.device).unsqueeze(0)
            edge_mask = edge_mask * (1.0 - diag)  # exclude diagonal

            # Rank-based target: K nearest get positive score, rest get negative
            K = self.dist_aux_k
            dist_sq_masked = dist_sq + (1.0 - edge_mask) * 1e9 + diag * 1e9
            ranks = dist_sq_masked.argsort(dim=-1).argsort(dim=-1).float()
            target_score = (K - 0.5 - ranks) / K  # +0.95 for rank 0, -0.05 for rank K, ...
            valid_count = edge_mask.sum().clamp(min=1.0)
            dist_aux = ((att - target_score) ** 2 * edge_mask).sum() / valid_count
            # Annealable coefficient (callbacks mutate dist_aux_coef_current on
            # the policy's model AND its GPU towers; falls back to the static coef).
            dist_aux_coef = getattr(self, "dist_aux_coef_current", self.dist_aux_coef)
            total_aux = total_aux + dist_aux_coef * dist_aux
            self._last_dist_aux_loss = dist_aux.detach()

        # --- Surface aux/saturation scalars through tower_stats ---
        # model.metrics() does not reach results on the 2.1.0 multi-GPU path;
        # tower_stats -> (GradLogging)stats_fn is the reliable route. Values
        # must be tensors (get_tower_stats stacks them).
        ts = getattr(self, "tower_stats", None)
        if ts is not None:
            if self._last_aux_loss is not None:
                ts["aux_mse"] = self._last_aux_loss
            if self._last_dist_aux_loss is not None:
                ts["dist_aux"] = self._last_dist_aux_loss
                ts["dist_aux_coef_current"] = torch.tensor(
                    float(getattr(self, "dist_aux_coef_current", self.dist_aux_coef)))
            if not self.continuous_action and hasattr(self, "_cached_logits") \
                    and getattr(self, "_aux_padding_mask", None) is not None:
                with torch.no_grad():
                    N = self._num_agents_max
                    lg = self._cached_logits.view(-1, N, N, 2)
                    p_sel = torch.softmax(lg, dim=-1)[..., 1]  # (B, N, N)
                    pad = self._aux_padding_mask
                    m = pad.unsqueeze(2) * pad.unsqueeze(1)
                    m = m * (1.0 - torch.eye(N, device=m.device).unsqueeze(0))
                    denom = m.sum().clamp(min=1.0)
                    # 0.0 = maximally stochastic, 0.5 = fully saturated
                    ts["sat_p_dev"] = ((p_sel - 0.5).abs() * m).sum() / denom

        # --- Per-agent credit assignment ---
        pa_loss = torch.tensor(0.0, device=policy_loss[0].device)
        if (self.per_agent_credit and self.continuous_action
                and "per_agent_rewards" in loss_inputs
                and hasattr(self, '_cached_logits')):

            per_agent_rew = loss_inputs["per_agent_rewards"]  # (B, N_max)
            if not isinstance(per_agent_rew, torch.Tensor):
                per_agent_rew = torch.as_tensor(per_agent_rew, dtype=torch.float32,
                                                device=policy_loss[0].device)

            padding_mask = self._aux_padding_mask  # (B, N_max), float
            N = self._num_agents_max
            num_real = padding_mask.sum(-1, keepdim=True).clamp(min=1.0)

            # Per-agent reward deviation: zero-mean, unit-std within each timestep
            mean_rew = (per_agent_rew * padding_mask).sum(-1, keepdim=True) / num_real
            dev = (per_agent_rew - mean_rew) * padding_mask
            dev_std = ((dev ** 2).sum(-1, keepdim=True) / num_real).sqrt().clamp(min=1e-8)
            dev_norm = dev / dev_std  # (B, N), zero-mean unit-std

            # Per-agent log-probs from current forward pass
            new_logits = self._cached_logits  # (B, 2*N*N)
            actions = loss_inputs["actions"]
            if not isinstance(actions, torch.Tensor):
                actions = torch.as_tensor(actions, dtype=torch.float32,
                                          device=policy_loss[0].device)
            per_agent_logp = self._compute_per_agent_logp(new_logits, actions, padding_mask)

            if self.pa_replace_ppo:
                # REPLACEMENT mode: per-agent REINFORCE replaces PPO surrogate
                global_adv = loss_inputs["advantages"]
                if not isinstance(global_adv, torch.Tensor):
                    global_adv = torch.as_tensor(global_adv, dtype=torch.float32,
                                                 device=policy_loss[0].device)
                per_agent_adv = global_adv.unsqueeze(-1) * (1.0 + self.pa_credit_alpha * dev_norm)
                per_agent_adv = per_agent_adv * padding_mask
                valid = padding_mask.bool()
                all_adv = per_agent_adv[valid]
                if all_adv.numel() > 1:
                    per_agent_adv = (per_agent_adv - all_adv.mean()) / all_adv.std().clamp(min=1e-8) * padding_mask

                pa_pg = -(per_agent_adv.detach() * per_agent_logp * padding_mask).sum(-1).mean()
                vf_pred = self.value_function()
                value_targets = loss_inputs["value_targets"]
                if not isinstance(value_targets, torch.Tensor):
                    value_targets = torch.as_tensor(value_targets, dtype=torch.float32,
                                                    device=vf_pred.device)
                vf_loss = 0.5 * (vf_pred - value_targets).pow(2).mean()

                self._last_pa_pg_loss = pa_pg.detach()
                return [pa_pg + vf_loss + total_aux]
            else:
                # ADDITIVE mode: per-agent correction added to standard PPO
                per_agent_logp_norm = per_agent_logp / num_real
                pa_loss = -(dev_norm.detach() * per_agent_logp_norm * padding_mask).sum(-1).mean()
                self._last_pa_pg_loss = pa_loss.detach()

        # Combine: standard PPO + aux + per-agent correction
        pa_coeff = self.pa_credit_alpha
        return [pl + total_aux + pa_coeff * pa_loss for pl in policy_loss]

    def metrics(self):
        d = {}
        if self.aux_enabled and self._last_aux_loss is not None:
            d["aux_mse"] = float(self._last_aux_loss)
            if self._last_aux_loss_critic is not None:
                d["aux_mse_critic"] = float(self._last_aux_loss_critic)
        if hasattr(self, '_last_pa_pg_loss') and self._last_pa_pg_loss is not None:
            d["pa_pg_loss"] = float(self._last_pa_pg_loss)
        if hasattr(self, '_last_dist_aux_loss') and self._last_dist_aux_loss is not None:
            d["dist_aux"] = float(self._last_dist_aux_loss)
        return d


class NeighborSelectorTorch(nn.Module):
    def __init__(self, src_embed, encoder, decoder, generator, d_embed_context,
                 global_stats_dim=0, respect_padding_mask=False):

        super().__init__()

        # Define the model components
        self.src_embed = src_embed
        self.d_v = src_embed.in_features
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.d_embed_context = d_embed_context
        self.respect_padding_mask = respect_padding_mask

        # Optional global swarm-stats conditioning of the decoder query context
        # (study acs-c2-train): h_c <- Linear([h_c || global_stats]).
        # global_stats_dim=0 (default) keeps the legacy architecture untouched.
        self.gs_proj = (nn.Linear(d_embed_context + global_stats_dim, d_embed_context)
                        if global_stats_dim > 0 else None)

    def forward(self, obs_dict: Dict[str, torch.Tensor]):
        """
        Vectorized version of the forward pass that produces
        identical output ordering to the per-agent for-loop version.
        """
        # ------------------------------------------------------------
        # 1) Unpack the inputs
        # ------------------------------------------------------------
        agent_infos = obs_dict["local_agent_infos"]  # (batch_size, num_agents_max, num_agents_max, obs_dim)
        network = obs_dict["neighbor_masks"]  # (batch_size, num_agents_max, num_agents_max)
        padding_mask = obs_dict["padding_mask"]  # (batch_size, num_agents_max)
        is_from_my_env = obs_dict["is_from_my_env"]  # (batch_size,)
        batch_size, num_agents_max, _, obs_dim = agent_infos.shape

        # Number of (non-padding) agents in each sample
        num_agents_per_sample = padding_mask.sum(dim=1).int()  # (batch_size,)

        # ------------------------------------------------------------
        # 2) Permute + reshape so that agent dimension i is “merged” with the batch
        #
        #    Original shapes (for each i in [0..num_agents_max]):
        #      local_agent_info      = (batch_size, num_agents_max, obs_dim)
        #      local_network         = (batch_size, num_agents_max)
        #      local_padding_flags   = (batch_size,)
        #
        #    We will create flattened shapes:
        #      flat_agent_infos      = (batch_size * num_agents_max, num_agents_max, obs_dim)
        #      flat_network          = (batch_size * num_agents_max, num_agents_max)
        #      flat_local_pad_flags  = (batch_size * num_agents_max,)
        #
        #    And similarly replicate the global masks for neighbors, etc.
        # ------------------------------------------------------------
        # agent_infos: permute(1,0,2,3) => (i, b, neighbor, obs_dim),
        # then reshape => (i*b, neighbor, obs_dim).
        permuted_agent_infos = agent_infos.permute(1, 0, 2, 3)  # (num_agents_max, batch_size, num_agents_max, obs_dim)
        flat_agent_infos = permuted_agent_infos.reshape(num_agents_max * batch_size, num_agents_max, obs_dim)

        # network: permute(1,0,2) => (i, b, neighbor),
        # then reshape => (i*b, neighbor).
        permuted_network = network.permute(1, 0, 2)  # (num_agents_max, batch_size, num_agents_max)
        flat_network = permuted_network.reshape(num_agents_max * batch_size, num_agents_max)

        # padding_mask used for neighbors:
        # Expand so each agent i sees the same "who is padded" info for neighbors.
        # shape => (i, b, neighbor) => flatten => (i*b, neighbor).
        expanded_padding_mask = padding_mask.unsqueeze(0).expand(num_agents_max, -1, -1)
        flat_padding_mask_for_neighbors = expanded_padding_mask.reshape(num_agents_max * batch_size, num_agents_max)
        if self.respect_padding_mask:
            flat_network = flat_network.bool() & flat_padding_mask_for_neighbors.bool()

        # local_padding_flags = padding_mask[:, i], but we want the same flatten order (i,b).
        # So permute(1,0) => (i, b) => flatten => (i*b).
        permuted_local_padding = padding_mask.permute(1, 0)  # (num_agents_max, batch_size)
        flat_local_padding_flags = permuted_local_padding.reshape(num_agents_max * batch_size)

        # is_from_my_env: replicate for each agent i => (i, b) => flatten => (i*b).
        expanded_is_from_my_env = is_from_my_env.unsqueeze(0).expand(num_agents_max, -1)  # (num_agents_max, batch_size)
        flat_is_from_my_env = expanded_is_from_my_env.reshape(num_agents_max * batch_size)

        # global_stats: same (B,) -> (i*b) replication as is_from_my_env, but
        # feature-dim 4. Only consumed when gs_proj is configured.
        flat_global_stats = None
        if self.gs_proj is not None and "global_stats" in obs_dict:
            gs = obs_dict["global_stats"]  # (batch_size, 4)
            flat_global_stats = gs.unsqueeze(0).expand(num_agents_max, -1, -1) \
                .reshape(num_agents_max * batch_size, -1)

        # ------------------------------------------------------------
        # 3) Call local_forward once on the flattened data
        #
        #    local_forward signature (in your original loop) was:
        #       local_forward(
        #           local_agent_info, local_network, padding_mask, is_from_my_env, local_padding_flags
        #       )
        #
        #    So now we pass the flattened arguments:
        # ------------------------------------------------------------
        sub_att_scores_flat, _, h_c_N_flat, encoder_out_flat = self.local_forward(
            flat_agent_infos,  # (batch_size * num_agents_max, num_agents_max, obs_dim)
            flat_network,  # (batch_size * num_agents_max, num_agents_max)
            flat_padding_mask_for_neighbors,  # (batch_size * num_agents_max, num_agents_max)
            flat_is_from_my_env,  # (batch_size * num_agents_max,)
            flat_local_padding_flags,  # (batch_size * num_agents_max,)
            global_stats=flat_global_stats,  # (batch_size * num_agents_max, 4) or None
        )
        # sub_att_scores_flat  => (batch_size * num_agents_max, num_agents_max)
        # h_c_N_flat           => (batch_size * num_agents_max, 1, d_embed_context)
        # encoder_out_flat     => (batch_size * num_agents_max, num_agents_max, d_embed_input)

        # ------------------------------------------------------------
        # 4) Reshape back to the original (batch_size, num_agents_max, num_agents_max)
        #    ordering so that it matches the loop-based version exactly.
        # ------------------------------------------------------------
        # sub_att_scores_flat => (i*b, neighbor).  Reshape => (i, b, neighbor). Then
        # permute => (b, i, neighbor).
        sub_att_scores_reshaped = sub_att_scores_flat.view(num_agents_max, batch_size, num_agents_max)
        att_scores = sub_att_scores_reshaped.permute(1, 0, 2)  # => (batch_size, num_agents_max, num_agents_max)

        # ------------------------------------------------------------
        # 5) Sum up h_c_N across all i (like h_c_N_accumulator += h_c_N in the og loop).
        #    Then average across the (non-padded) agents at the end.
        # ------------------------------------------------------------
        # h_c_N_flat => (i*b, 1, d_embed_context).  Reshape => (i, b, 1, d_embed_context).
        # Then sum across the i dimension => (b, 1, d_embed_context).
        _, _, d_embed_context = h_c_N_flat.shape
        h_c_N_reshaped = h_c_N_flat.view(num_agents_max, batch_size, 1, d_embed_context)
        h_c_N_accumulator = h_c_N_reshaped.sum(dim=0)  # => (batch_size, 1, d_embed_context)

        # Now average across the actual number of agents (not counting padding).
        num_agents_per_sample = num_agents_per_sample.view(-1, 1, 1).float()  # (batch_size, 1, 1)
        if self.respect_padding_mask:
            # RLlib may construct an all-zero dummy observation while probing a
            # model. Real environment observations always contain active agents.
            num_agents_per_sample = num_agents_per_sample.clamp(min=1.0)
        average_h_c_N = h_c_N_accumulator / num_agents_per_sample  # (batch_size, 1, d_embed_context)

        # Per-agent context vectors (for auxiliary heads). Padded rows are exactly zero
        # (set in local_forward), so downstream consumers can mask via padding_mask.
        per_agent_h_c_N = h_c_N_reshaped.squeeze(2).permute(1, 0, 2)  # (batch_size, num_agents_max, d_embed_context)

        # Per-pair encoder embeddings (for the pair_embedding aux head). The flatten/permute
        # order in step 2 was permute(1,0,...).reshape -> (i*b, j, d_embed). Inverse: view -> permute.
        d_embed_input = encoder_out_flat.shape[-1]
        encoder_out_reshaped = encoder_out_flat.view(num_agents_max, batch_size, num_agents_max, d_embed_input)
        per_pair_encoder_out = encoder_out_reshaped.permute(1, 0, 2, 3)  # (batch_size, N_i, N_j, d_embed_input)

        # ------------------------------------------------------------
        # 6) Return results
        # ------------------------------------------------------------
        # att_scores:           (batch_size, num_agents_max, num_agents_max)
        # average_h_c_N:        (batch_size, 1, d_embed_context)
        # per_agent_h_c_N:      (batch_size, num_agents_max, d_embed_context)
        # per_pair_encoder_out: (batch_size, N_i, N_j, d_embed_input)
        return att_scores, average_h_c_N, per_agent_h_c_N, per_pair_encoder_out

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(tgt, encoder_out, tgt_mask, src_tgt_mask)

    def local_forward(
            self,
            local_agent_info,
            local_network,
            padding_mask,
            is_from_my_env,
            local_padding_flags,
            global_stats=None,
    ):
        """
        :param local_agent_info: (batch_size, num_agents_max, obs_dim)
        :param local_network:    (batch_size, num_agents_max)
        :param padding_mask:     (batch_size, num_agents_max)
        :param is_from_my_env:   (batch_size,)
        :param local_padding_flags: (batch_size,)  # Checks if the agent of each batch is padded or not in local_forward
        :param global_stats:     (batch_size, 4) or None — swarm summary for gs_proj conditioning
        :return: sub_att_scores: (batch_size, num_agents_max)
        """
        batch_size, num_agents_max, obs_dim = local_agent_info.shape

        # Initialize outputs for all sequences
        decoder_out = torch.zeros(batch_size, 1, self.d_embed_context, device=local_agent_info.device)
        h_c_N = torch.zeros(batch_size, 1, self.d_embed_context, device=local_agent_info.device)
        sub_att_scores = torch.full((batch_size, num_agents_max), -1e9, device=local_agent_info.device)
        # Per-query encoder output, zero-filled for padded query rows. Used by aux heads.
        encoder_out_full = torch.zeros(batch_size, num_agents_max, self.src_embed.d_embed,
                                       device=local_agent_info.device)

        # Get mask for non-padded sequences
        non_padded_mask = local_padding_flags.bool()  # (batch_size,)

        if non_padded_mask.any():  # If there is at least one non-padded sequence
            # Select non-padded sequences
            local_agent_info_np = local_agent_info[non_padded_mask]  # (n_non_padded, num_agents_max, obs_dim)
            local_network_np = local_network[non_padded_mask]  # (n_non_padded, num_agents_max)
            is_from_my_env_np = is_from_my_env[non_padded_mask]  # (n_non_padded,)

            # Masks for non-padded sequences
            src_mask_tokens = local_network_np.ne(0)  # (n_non_padded, num_agents_max)
            src_mask_idx = 0
            src_mask = self.make_src_mask(src_mask_tokens, mask_idx=src_mask_idx)
            tgt_mask = None
            context_mask_token = torch.ones_like(src_mask_tokens[:, 0:1], dtype=torch.bool)
            src_tgt_mask = self.make_src_tgt_mask(src_mask_tokens, context_mask_token, mask_idx=src_mask_idx)

            # Encoding
            encoder_out = self.encode(local_agent_info_np, src_mask.unsqueeze(1))

            # Context embedding
            h_c_N_np = self.get_context_vector(
                embeddings=encoder_out,
                pad_tokens=~src_mask_tokens,
                is_from_my_env=is_from_my_env_np,
                use_embeddings_mask=True,
                debug=True
            )  # (n_non_padded, 1, d_embed_context)

            # Global swarm-stats conditioning of the context (query) vector
            if self.gs_proj is not None and global_stats is not None:
                gs_np = global_stats[non_padded_mask].to(h_c_N_np.dtype)  # (n_non_padded, 4)
                h_c_N_np = self.gs_proj(
                    torch.cat([h_c_N_np, gs_np.unsqueeze(1)], dim=-1))  # (n_non_padded, 1, d_ctx)

            # Decoding
            decoder_out_np = self.decode(h_c_N_np, encoder_out, tgt_mask, src_tgt_mask.unsqueeze(1))

            # Generator
            sub_att_scores_np = self.generator(input_query=decoder_out_np, input_key=encoder_out,
                                               mask=src_tgt_mask).squeeze(1)  # (n_non_padded, num_agents_max)

            # Update results for non-padded sequences
            decoder_out[non_padded_mask] = decoder_out_np
            h_c_N[non_padded_mask] = h_c_N_np
            sub_att_scores[non_padded_mask] = sub_att_scores_np
            encoder_out_full[non_padded_mask] = encoder_out
        elif not self.respect_padding_mask:
            print("WARNING All samples are padded in the batch. If this is not expected such as "
                  "parallelized forward, check the padding mask.")

        return sub_att_scores, decoder_out, h_c_N, encoder_out_full

    def get_context_vector(self, embeddings, pad_tokens, is_from_my_env, use_embeddings_mask=True, debug=False):
        # embeddings: shape (batch_size, num_agents_max==seq_len_src, data_size==d_embed_input)
        # pad_tokens: shape (batch_size, num_agents_max==seq_len_src)
        # is_from_my_env: shape (batch_size,)

        # Obtain batch_size, num_agents, data_size from embeddings
        batch_size, num_agents_max, data_size = embeddings.shape

        if use_embeddings_mask:  #... Could be way simpler
            # Expand the dimensions of pad_tokens to match the shape of embeddings
            # # mask==1: padding, mask==0: non-padding
            mask = pad_tokens.unsqueeze(-1).expand_as(embeddings)  # (batch_size, num_agents_max, data_size)

            # Replace masked values with zero for the average computation
            # embeddings_masked: (batch_size, num_agents_max, data_size)
            embeddings_masked = torch.where(mask==0, embeddings, torch.zeros_like(embeddings))

            # Compute the sum and count non-zero elements
            embeddings_sum = torch.sum(embeddings_masked, dim=1, keepdim=True)  # (batch_size, 1, data_size)
            embeddings_count = torch.sum((mask==0), dim=1, keepdim=True).float()  # (batch_size, 1, data_size)

            # Check if there is any sample where all agents are padded
            if debug:
                if is_from_my_env.dtype == torch.bool:  # if it isn't in the env_check mode
                    if torch.any(embeddings_count == 0):  # is supposed to never happen due to self-loops
                        raise ValueError("All agents are padded in at least one sample.")
            # Compute the average embeddings, only for non-masked elements
            embeddings_avg = embeddings_sum / embeddings_count
        else:
            # Compute the average embeddings: shape (batch_size, 1, data_size)
            embeddings_avg = torch.mean(embeddings, dim=1, keepdim=True)  # num_agents_max dim is reduced

        # Construct context embedding: shape (batch_size, 1, d_embed_context)
        # The resulting tensor, h_c, will have shape (batch_size, 1, d_embed_context)
        # Concatenate the additional info to h_c, if you need more info for the context vector.
        h_c = embeddings_avg  # no concatenation in this project
        # This represents the graph embeddings.
        # It summarizes the information of all nodes in the graph.

        return h_c  # (batch_size, 1, d_embed_context)

    def make_src_mask(self, src, mask_idx=1):
        pad_mask = self.make_pad_mask(src, src, pad_idx=mask_idx)
        return pad_mask  # (batch_size, seq_len_src, seq_len_src)

    def make_src_tgt_mask(self, src, tgt, mask_idx=1):
        # src: key/value; tgt: query
        pad_mask = self.make_pad_mask(tgt, src, pad_idx=mask_idx)
        return pad_mask  # (batch_size, seq_len_tgt, seq_len_src)

    def make_pad_mask(self, query, key, pad_idx=1, dim_check=False):
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        # If input_token==pad_idx, then the mask value is 0, else 1
        # In the MHA layer, (no attention) == (attention_score: -inf) == (mask value is 0) == (input_token==pad_idx)
        # WARNING: Choose pad_idx carefully, particularly about the data type (e.g. float, int, ...)

        # Check if the query and key have the same dimension
        if dim_check:
            assert len(query.shape) == 2, "query must have 2 dimensions: (n_batch, query_seq_len)"
            assert len(key.shape) == 2, "key must have 2 dimensions: (n_batch, key_seq_len)"
            assert query.size(0) == key.size(0), "query and key must have the same batch size"

        query_seq_len, key_seq_len = query.size(1), key.size(1)

        key_mask = key.ne(pad_idx).unsqueeze(1)  # (n_batch, 1, key_seq_len); on the same device as key
        key_mask = key_mask.repeat(1, query_seq_len, 1)  # (n_batch, query_seq_len, key_seq_len)

        query_mask = query.ne(pad_idx).unsqueeze(2)  # (n_batch, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, key_seq_len)  # (n_batch, query_seq_len, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False
        return mask  # output shape: (n_batch, query_seq_len, key_seq_len)  # Keep in mind: 'NO HEADING DIM' here!!

    def forward_og(self, obs_dict: Dict[str, TensorType]):
        # Get data
        agent_infos = obs_dict["local_agent_infos"]  # (batch_size, num_agents_max, num_agents_max, obs_dim)
        network = obs_dict["neighbor_masks"]  # (batch_size, num_agents_max, num_agents_max); (:,i,:): i-th agent's net
        padding_mask = obs_dict["padding_mask"]  # (batch_size, num_agents_max); applies over all agents same
        is_from_my_env = obs_dict["is_from_my_env"]  # (batch_size,); 1: from my env, 0: under the env_check
        # Caution: masks are torch FLOAT tensors, not boolean tensors, which I don't like in RLlib (v2.1.0)

        batch_size, num_agents_max, _, obs_dim = agent_infos.shape

        """
        In forward_og, padding agents are not *fully* supported in the model (network: src_maskS, pad_mask: tgt_mask ?)
        """

        # Get sub-attention scores
        att_scores = torch.zeros_like(network, dtype=torch.float32)  # (batch_size, num_agents_max, num_agents_max)
        num_agents = padding_mask.sum(dim=1).int()  # (batch_size,); number of agents in each sample
        h_c_N_accumulator = torch.zeros(batch_size, 1, self.d_embed_context, device=agent_infos.device)

        for i in range(num_agents_max):  # In the latest ver, pushed agent dim onto batch dim to parallelize
            local_agent_info = agent_infos[:, i, :, :]
            local_network = network[:, i, :]
            local_padding_flags = padding_mask[:, i]

            # sub_att_scores: shape: (batch_size, num_agents_max)
            # h_c_N: shape: (batch_size, 1, d_embed_context)
            sub_att_scores, _, h_c_N = self.local_forward_og(local_agent_info, local_network, padding_mask, is_from_my_env, local_padding_flags)

            # Get the i-th row of the attention scores
            att_scores[:, i, :] = sub_att_scores

            # Accumulate h_c_N values
            h_c_N_accumulator += h_c_N  # In latest, 패딩 에이전트 안 더해짐.

        # Calculate average_h_c_N
        num_agents = num_agents.view(-1, 1, 1).float()  # (batch_size, 1, 1)
        average_h_c_N = h_c_N_accumulator / num_agents

        # att_scores: (batch_size, num_agents_max, num_agents_max)
        # average_h_c_N: (batch_size, 1, d_embed_context)
        return att_scores, average_h_c_N

    def local_forward_og(
            self,
            local_agent_info,
            local_network,
            padding_mask,
            is_from_my_env,
            local_padding_flags,
    ):
        """
        :param local_agent_info: (batch_size, num_agents_max, obs_dim)
        :param local_network:    (batch_size, num_agents_max)
        :param padding_mask:     (batch_size, num_agents_max)
        :param is_from_my_env:   (batch_size,)
        :param local_padding_flags: (batch_size,)
        :return: sub_att_scores: (batch_size, num_agents_max)
        """

        # Get data
        src = local_agent_info  # (batch_size, num_agents_max, d_v)

        # Get masks
        # local_network:
        # # 0: padding / disconnected, 1: connected
        # # 0: no attention,           1: attention
        # # 0: False in mask,          1: True in mask
        src_mask_tokens = local_network.ne(0)  # (batch_size, num_agents_max==seq_len_src)  bool tensor
        src_mask_idx = 0
        src_mask = self.make_src_mask(src_mask_tokens, mask_idx=src_mask_idx)  # (batch_size, seq_len_src, seq_len_src)
        tgt_mask = None  # No (masked) self-attention layer in the decoder block
        context_mask_token = torch.ones_like(src_mask_tokens[:, 0:1], dtype=torch.bool)  # (batch_size, 1); it's 2D
        # In the Cross-Attention, Q=tgt=context, K/V=src=enc_out
        src_tgt_mask = self.make_src_tgt_mask(src_mask_tokens, context_mask_token, mask_idx=src_mask_idx)

        # Embedding: in the encoder method

        # Encoder
        # encoder_out: shape: (batch_size, src_seq_len, d_embed) == (batch_size, num_agents_max, d_embed)
        # unsqueeze(1) has been applied to src_mask to broadcast over head dim in the MHA layer
        encoder_out = self.encode(src, src_mask.unsqueeze(1))

        # Context embedding, if needed
        h_c_N = self.get_context_vector(embeddings=encoder_out, pad_tokens=~src_mask_tokens,
                                        is_from_my_env=is_from_my_env,
                                        use_embeddings_mask=True, debug=True)  # (batch_size, 1, d_embed_context)

        # Decoder: Cross-Attention (glimpse); Q: context, K/V: encoder_out
        # decoder_out: shape: (batch_size, 1, d_embed_context)
        decoder_out = self.decode(h_c_N, encoder_out, tgt_mask, src_tgt_mask.unsqueeze(1))  # h_c_(N+1)

        # Generator: raw attention scores by CA; Q: rich-context (decoder_out), K/V: encoder_out
        sub_att_scores = self.generator(input_query=decoder_out, input_key=encoder_out, mask=src_tgt_mask).squeeze(1)  # kill q dim

        # sub_att_scores: (batch_size, num_agents_max)
        # decoder_out: (batch_size, 1, d_embed_context)
        # h_c_N: (batch_size, 1, d_embed_context)
        return sub_att_scores, decoder_out, h_c_N
