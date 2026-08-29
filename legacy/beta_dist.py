import numpy as np
import torch
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper


class TorchContinuousWeightDist(TorchDistributionWrapper):
    """Squashed Gaussian distribution for continuous [0,1] neighbor weights.

    Model outputs 2*N logits: first N are mean logits (pre-sigmoid),
    second N are log_std logits. Actions = sigmoid(Normal(mean, std).sample()).
    Output is reshaped to (batch, sqrt(N), sqrt(N)) to match the 2D action space.
    """

    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        N = inputs.shape[-1] // 2
        self._spatial_n = int(round(N ** 0.5))
        self.mean = torch.nan_to_num(inputs[..., :N], nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        self.log_std = torch.nan_to_num(inputs[..., N:], nan=-1.0).clamp(-5.0, 2.0)
        self.std = torch.exp(self.log_std)
        self.normal = torch.distributions.Normal(self.mean, self.std)

    def sample(self):
        z = self.normal.rsample()
        action = torch.sigmoid(z)
        self.last_sample = action.reshape(-1, self._spatial_n, self._spatial_n)
        return self.last_sample

    def deterministic_sample(self):
        action = torch.sigmoid(self.mean)
        self.last_sample = action.reshape(-1, self._spatial_n, self._spatial_n)
        return self.last_sample

    def logp(self, actions):
        actions = actions.reshape(actions.shape[0], -1).clamp(1e-6, 1.0 - 1e-6)
        z = torch.log(actions / (1.0 - actions))
        normal_logp = self.normal.log_prob(z)
        correction = torch.log(actions * (1.0 - actions) + 1e-8)
        return (normal_logp - correction).sum(-1)

    def entropy(self):
        return self.normal.entropy().sum(-1)

    def kl(self, other):
        return torch.distributions.kl_divergence(self.normal, other.normal).sum(-1)

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return int(np.prod(action_space.shape, dtype=np.int32)) * 2
